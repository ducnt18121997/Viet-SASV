import math
import torch
import torch.nn as nn
from typing import Optional
from models.backend.tdnn.layers import Bottle2neck
from models.backend.tdnn.augment import FbankAug
from models.backend.tdnn.config import ECAPA_TDNNConfig


class ECAPA_TDNN(nn.Module):

    def __init__(self, idims: int, config: Optional[ECAPA_TDNNConfig] = None):
        super(ECAPA_TDNN, self).__init__()

        self.config = config if config else ECAPA_TDNNConfig()
        self.specaug = FbankAug()  # Spec augmentation

        # initialize model structure
        self.idims = idims
        self.odims = self.config.embedding_size

        self.conv1 = nn.Conv1d(
            in_channels=self.idims,
            out_channels=self.config.C,
            kernel_size=self.config.kernel_size,
            stride=1,
            padding=2,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=self.config.C)
        self.layer1 = Bottle2neck(
            inplanes=self.config.C,
            planes=self.config.C,
            kernel_size=self.config.bottle_neck.kernel_size,
            dilation=self.config.bottle_neck.dilation[0],
            scale=self.config.bottle_neck.scale,
        )
        self.layer2 = Bottle2neck(
            inplanes=self.config.C,
            planes=self.config.C,
            kernel_size=self.config.bottle_neck.kernel_size,
            dilation=self.config.bottle_neck.dilation[1],
            scale=self.config.bottle_neck.scale,
        )
        self.layer3 = Bottle2neck(
            inplanes=self.config.C,
            planes=self.config.C,
            kernel_size=self.config.bottle_neck.kernel_size,
            dilation=self.config.bottle_neck.dilation[2],
            scale=self.config.bottle_neck.scale,
        )

        # fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(
            in_channels=3 * self.config.C,
            out_channels=self.config.fixed_C,
            kernel_size=1,
        )
        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels=self.config.fixed_C * 3,
                out_channels=self.config.attn_dims,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.config.attn_dims),
            nn.Tanh(),  # add this layer
            nn.Conv1d(
                in_channels=self.config.attn_dims,
                out_channels=self.config.fixed_C,
                kernel_size=1,
            ),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(num_features=self.config.fixed_C * 2)
        self.fc6 = nn.Linear(
            in_features=self.config.fixed_C * 2, out_features=self.odims
        )
        self.bn6 = nn.BatchNorm1d(num_features=self.odims)

    def forward(self, x: torch.Tensor, aug: bool) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Feature tensor (B, T_feats, aux_channels).
            aug (bool): Use spec augmentation

        Returns:
            Tensor: Feature tensor (B, embedding_size, T_feats).
        """

        if aug == True:
            x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(
                    1, 1, t
                ),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
