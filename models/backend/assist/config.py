from pydantic import BaseModel


class AssistConfig(BaseModel):
    filts: list[int] = [16, 32, 64, 128, 256]
    gat_dims: list[int] = [128, 128]
    pool_ratios: list[float] = [0.25, 0.25]
    temperatures: list[float] = [1.0, 1.0]
    first_conv: int = 3
