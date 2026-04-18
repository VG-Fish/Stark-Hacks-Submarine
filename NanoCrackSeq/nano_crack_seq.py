import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, include_pool: bool = True) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DWConvBlock(in_ch, out_ch),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if include_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DWConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class NanoCrackSeg(nn.Module):
   class NanoCrackSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.enc1 = EncoderBlock(1, 8, include_pool=True)
        self.enc2 = EncoderBlock(8, 16, include_pool=True)
        self.enc3 = EncoderBlock(16, 32, include_pool=True)

        self.bottleneck = EncoderBlock(32, 32, include_pool=False)
        self.dec1 = DecoderBlock(32, 16)
        self.dec2 = DecoderBlock(16, 8)

        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d1 = self.dec1(b)
        d2 = self.dec2(d1)

        out = self.output(d2)
        return out