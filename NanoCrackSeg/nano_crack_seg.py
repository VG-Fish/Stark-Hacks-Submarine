import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw_conv(x)


class NanoCrackSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = DWConvBlock(1, 8)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DWConvBlock(8, 16)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DWConvBlock(16, 32)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DWConvBlock(32, 64)

        # Decoder (with skip connections)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DWConvBlock(64 + 32, 32)  # 64 (bottleneck) + 32 (skip from enc3)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DWConvBlock(32 + 16, 16)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DWConvBlock(16 + 8, 8)

        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with Concatenation (U-Net style)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


def rfkd_loss(student_logits, teacher_logits, true_masks, T=4.0, alpha=0.5):
    # 1. Hard Loss
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_masks)

    # 2. Soft Loss (Distillation)
    # Convert logits to soft probabilities via sigmoid with temperature
    s_p = torch.sigmoid(student_logits / T)
    t_p = torch.sigmoid(teacher_logits / T).detach()

    # We use BCE for the soft targets in binary tasks
    soft_loss = F.binary_cross_entropy(s_p, t_p) * (T**2)

    return (1 - alpha) * hard_loss + alpha * soft_loss
