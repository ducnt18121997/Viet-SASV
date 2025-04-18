import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from models.backend.tdnn.modules.utils import Sparse, attn_fn
from models.backend.tdnn.modules.weight_init import trunc_normal_


class SparseDGF(nn.Module):
    def __init__(
        self,
        dim: int,
        T: int,
        ratios: float,
        K: int,
        temperature: float,
        dropout: float,
    ):
        super().__init__()
        self.K = K
        self.complex_weight = nn.Parameter(
            torch.randn(K, dim, (T // 2) + 1, 2, dtype=torch.float32) * 0.02
        )
        trunc_normal_(self.complex_weight, std=0.02)
        self.fn = attn_fn(dim, ratios, K, temperature)
        self.sparse = Sparse(sparse=dropout)

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        # Sparse forward#
        B, C, T = x.shape
        weight = self.complex_weight  # [K, C, T//2, 2]
        attn = self.fn(x)  # [B, K]
        x = torch.fft.rfft(x, dim=-1, norm="ortho")
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = (
                F.interpolate(
                    weight.permute(3, 0, 1, 2),
                    size=x.shape[1:3],
                    mode="bilinear",
                    align_corners=True,
                )
                .permute(1, 2, 3, 0)
                .contiguous()
            )

        agg_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2)
        agg_weight = torch.view_as_complex(agg_weight)
        sparse_w, mask = self.sparse(agg_weight)
        x = (x * mask) + (x * sparse_w)
        x = torch.fft.irfft(x, n=T, dim=-1, norm="ortho")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        weight = self.complex_weight  # [K, C, T//2, 2]
        attn = self.fn(x)  # [B, K]
        x = torch.fft.rfft(x, dim=-1, norm="ortho")
        if not weight.shape[2] == x.shape[-1]:
            # [2, K, C, T//2]
            weight = (
                F.interpolate(
                    weight.permute(3, 0, 1, 2),
                    size=x.shape[1:3],
                    mode="bilinear",
                    align_corners=True,
                )
                .permute(1, 2, 3, 0)
                .contiguous()
            )

        agg_weight = torch.mm(attn, weight.reshape(self.K, -1)).reshape(B, C, -1, 2)
        agg_weight = torch.view_as_complex(agg_weight)
        x = x * agg_weight
        x = torch.fft.irfft(x, n=T, dim=-1, norm="ortho")
        return x
