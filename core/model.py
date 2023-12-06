from typing import Any, cast, List, Union
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, config, batch_norm):
        super().__init__()

        self.layers = make_layers(config, batch_norm)

    def forward(self, x):
        x = self.layers(x)

        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 12
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class FCN(nn.Module):
    def __init__(
        self,
        encoder_config,
        encoder_batch_norm,
    ):
        super().__init__()
        self.encoder = VGGEncoder(encoder_config, encoder_batch_norm)

        self.no2_head = nn.Conv2d(512, 1, kernel_size=1)
        # self.land_cover_head = nn.Conv2d(512, 11, 1)

    def forward(self, x):
        x = self.encoder(x)
        no2_output = self.no2_head(x)
        # land_cover_output = self.land_cover_head(x)
        return no2_output
