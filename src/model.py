import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


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
        num_class=1,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)

    def forward(self, x):
        encoder_outputs = self.encoder.forward(x)
        output = self.decoder.forward(
            encoder_outputs[::-1][0], encoder_outputs[::-1][1:]
        )
        output = self.head(output)
        return output
