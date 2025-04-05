from pydantic import BaseModel


class XVectorConfig(BaseModel):
    out_channels = [512, 512, 512, 512, 1500]
    kernel_sizes = [5, 3, 3, 1, 1]
    dilations = [1, 2, 3, 1, 1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            len(self.out_channels) == len(self.kernel_sizes) == len(self.dilations)
        ), "Expected length of out_channels, kernel_sizes, and dilations must be the same. Given: out_channels - {}, kernel_sizes - {}, dilations - {}".format(
            len(self.out_channels), len(self.kernel_sizes), len(self.dilations)
        )
