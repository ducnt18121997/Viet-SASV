from pydantic import BaseModel


class Bottle2neckConfig(BaseModel):
    kernel_size: int = 3
    dilation: list[int] = [2, 3, 4]
    scale: int = 8


class ECAPA_TDNNConfig(BaseModel):
    C: int = 1024
    fixed_C: int = 1536
    kernel_size: int = 5
    bottle_neck: Bottle2neckConfig = Bottle2neckConfig()
    attn_dims: int = 256
    embedding_size: int = 192
