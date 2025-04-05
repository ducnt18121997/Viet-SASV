from typing import List
from pydantic import BaseModel


class SincNetConfig(BaseModel):
    cnn_N_filt: List[int] = [80, 60, 60]
    cnn_len_filt: List[int] = [251, 5, 5]
    cnn_max_pool_len: List[int] = [3, 3, 3]
    cnn_use_laynorm_inp: bool = True
    cnn_use_batchnorm_inp: bool = False
    cnn_use_laynorm: List[bool] = [True, True, True]
    cnn_use_batchnorm: List[bool] = [False, False, False]
    cnn_act: List[str] = ["leaky_relu", "leaky_relu", "leaky_relu"]
    cnn_drop: List[float] = [0.0, 0.0, 0.0]
    is_flat: bool = False  # default: false (change depend on backend model)
