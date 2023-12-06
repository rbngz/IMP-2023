from typing import Any, cast, List, Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


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

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(32)
        self.no2_head = nn.Conv2d(32, 1, kernel_size=1)
        self.land_cover_head = nn.Conv2d(32, 11, 1)

    def forward(self, x):
        x = self.encoder(x)

        score = self.bn1(self.relu(self.deconv1(x)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        no2_output = self.no2_head(score)  # size=(N, n_class, x.H/1, x.W/1)
        land_cover_output = self.land_cover_head(score)
        return no2_output, land_cover_output
        # land_cover_output = self.land_cover_head(x)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, chs=(12, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            outputs.append(x)

            # No need to pool last block output
            if i != len(self.blocks) - 1:
                x = self.pool(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList(
            # Check stride in original U-Net paper
            [nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)]
        )
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_outputs):
        for i, block in enumerate(self.blocks):
            x = self.upconvs[i](x)
            cropped_output = self.crop(encoder_outputs[i], x)
            x = torch.cat([x, cropped_output], dim=1)
            x = block(x)
        return x

    def crop(self, encoder_output, x):
        _, _, H, W = x.shape
        cropped_output = torchvision.transforms.CenterCrop([H, W])(encoder_output)
        return cropped_output


class UNet(nn.Module):
    def __init__(
        self,
        enc_chs=(12, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.no2_head = nn.Conv2d(dec_chs[-1], 1, 1)
        self.land_cover_head = nn.Conv2d(dec_chs[-1], 11, 1)

    def get_output_dim(self, input_dim: int):
        output_dim = input_dim
        for _ in range(len(self.enc_chs) - 2):
            output_dim -= 4
            output_dim = output_dim // 2
        output_dim -= 4
        for _ in range(len(self.dec_chs) - 1):
            output_dim *= 2
            output_dim -= 4
        return output_dim

    def forward(self, x):
        encoder_outputs = self.encoder.forward(x)
        decoder_output = self.decoder.forward(
            encoder_outputs[-1], encoder_outputs[::-1][1:]
        )
        no2_output = self.no2_head(decoder_output)
        land_cover_output = self.land_cover_head(decoder_output)
        return no2_output, land_cover_output
