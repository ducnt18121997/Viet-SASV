import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.frontend.sinc_net.layers import SincConv, LayerNorm, act_fun
from models.frontend.sinc_net.config import SincNetConfig


class SincNet(nn.Module):

    def __init__(
        self, sample_rate: int, input_dim: int, config: Optional[SincNetConfig] = None
    ):
        super(SincNet, self).__init__()

        # initialize hparams
        self.sample_rate = sample_rate
        self.input_dim = input_dim
        self.config = config if config else SincNetConfig()

        # initialize model structure
        self.cnn_N_filt = self.config.cnn_N_filt
        self.cnn_len_filt = self.config.cnn_len_filt
        self.cnn_max_pool_len = self.config.cnn_max_pool_len

        self.cnn_act = self.config.cnn_act
        self.cnn_drop = self.config.cnn_drop

        self.cnn_use_laynorm = self.config.cnn_use_laynorm
        self.cnn_use_batchnorm = self.config.cnn_use_batchnorm
        self.cnn_use_laynorm_inp = self.config.cnn_use_laynorm_inp
        self.cnn_use_batchnorm_inp = self.config.cnn_use_batchnorm_inp

        self.N_cnn_lay = len(self.config.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(features=self.input_dim)
        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(num_features=[self.input_dim], momentum=0.05)

        current_input = self.input_dim
        for i in range(self.N_cnn_lay):
            N_filt = self.cnn_N_filt[i]
            len_filt = self.cnn_len_filt[i]

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(act_type=self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm(
                    features=[
                        N_filt,
                        int((current_input - len_filt + 1) / self.cnn_max_pool_len[i]),
                    ]
                )
            )

            self.bn.append(
                nn.BatchNorm1d(
                    N_filt,
                    int((current_input - len_filt + 1) / self.cnn_max_pool_len[i]),
                    momentum=0.05,
                )
            )

            if i == 0:
                self.conv.append(
                    SincConv(
                        out_channels=self.cnn_N_filt[0],
                        kernel_size=self.cnn_len_filt[0],
                        sample_rate=self.sample_rate,
                    )
                )
            else:
                self.conv.append(
                    nn.Conv1d(
                        in_channels=self.cnn_N_filt[i - 1],
                        out_channels=self.cnn_N_filt[i],
                        kernel_size=len_filt,
                    )
                )
            current_input = int(
                (current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]
            )

        self.flatten = self.config.is_flat
        if self.flatten is True:
            self.out_dim = (
                current_input * N_filt
            )  # NOTE: flatten output use this outdim
        else:
            self.out_dim = N_filt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        batch = x.shape[0]
        seq_len = x.shape[1]
        if self.cnn_use_laynorm_inp:
            x = self.ln0((x))
        if self.cnn_use_batchnorm_inp:
            x = self.bn0((x))
        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = torch.abs(self.conv[i](x))
                    x = F.max_pool1d(x, self.cnn_max_pool_len[i])
                    x = self.act[i](self.ln[i](x))
                    x = self.drop[i](x)
                else:
                    x = self.conv[i](x)
                    x = F.max_pool1d(x, self.cnn_max_pool_len[i])
                    x = self.act[i](self.ln[i](x))
                    x = self.drop[i](x)

            if self.cnn_use_batchnorm[i]:
                x = self.conv[i](x)
                x = F.max_pool1d(x, self.cnn_max_pool_len[i])
                x = self.act[i](self.bn[i](x))
                x = self.drop[i](x)

            if self.cnn_use_batchnorm[i] is False and self.cnn_use_laynorm[i] is False:
                x = self.conv[i](x)
                x = F.max_pool1d(x, self.cnn_max_pool_len[i])
                x = self.drop[i](self.act[i](x))

        if self.flatten is True:
            x = x.view(batch, -1)

        return x
