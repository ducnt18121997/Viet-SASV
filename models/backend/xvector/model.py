import torch
import torch.nn as nn
from typing import Optional
from models.backend.xvector.layers import StatsPool
from models.backend.xvector.augment import FbankAug
from models.backend.xvector.config import XVectorConfig


class XVector(nn.Module):

    def __init__(self, idims: int, config: Optional[XVectorConfig] = None):
        super().__init__()

        self.config = config if config else XVectorConfig()
        self.idims = idims
        self.specaug = FbankAug()  # Spec augmentation

        # initialize model configuration
        self.tdnns = nn.ModuleList()
        in_channel = self.idims

        for out_channel, kernel_size, dilation in zip(
            self.config.out_channels,
            self.config.kernel_sizes,
            self.config.dilations,
        ):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()
        self.odims = in_channel * 2

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor = None, aug: bool = False
    ) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            Batch of waveforms with shape (B, T_feats, aux_channels).
        weights : torch.Tensor, optional
            Batch of weights with shape (batch, frame).
        aug: bool, False
            Bool value to use SpecAugment, yes or no?
        """

        if aug == True:
            x = self.specaug(x)

        for block in self.tdnns:
            x = block(x)
        outputs = self.stats_pool(x, weights=weights)

        return outputs
