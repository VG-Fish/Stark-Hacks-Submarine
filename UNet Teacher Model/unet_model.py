import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for skip connections — helps the model focus on crack regions."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class UNet(nn.Module):
    def __init__(
        self, in_channels: int = 1, base_filters: int = 64, dropout: float = 0.2
    ) -> None:
        super().__init__()
        f = base_filters

        self.enc1 = DoubleConv(in_channels, f, dropout=0.0)
        self.enc2 = DoubleConv(f, f * 2, dropout=dropout)
        self.enc3 = DoubleConv(f * 2, f * 4, dropout=dropout)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f * 4, f * 8, dropout=dropout * 2)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(gate_ch=f * 4, skip_ch=f * 4, inter_ch=f * 2)
        self.dec3 = DoubleConv(f * 8, f * 4, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(gate_ch=f * 2, skip_ch=f * 2, inter_ch=f)
        self.dec2 = DoubleConv(f * 4, f * 2, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(gate_ch=f, skip_ch=f, inter_ch=f // 2)
        self.dec1 = DoubleConv(f * 2, f, dropout=0.0)

        self.out = nn.Conv2d(f, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder path
        up3 = self.up3(b)
        d3 = self.dec3(torch.cat([up3, self.attn3(up3, e3)], dim=1))

        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat([up2, self.attn2(up2, e2)], dim=1))

        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat([up1, self.attn1(up1, e1)], dim=1))

        return self.out(d1)


def dice_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    alpha: float = 0.6,  # Weights the positive class (cracks) higher
    gamma: float = 2.0,  # Penalizes the model more for being confident but wrong
) -> torch.Tensor:
    prediction_probabilities: torch.Tensor = torch.sigmoid(pred_logits)

    intersection: torch.Tensor = (prediction_probabilities * target).sum(dim=(2, 3))
    dice: torch.Tensor = 1 - (2 * intersection + smooth) / (
        prediction_probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth
    )

    bce: torch.Tensor = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction="none"
    )
    pt: torch.Tensor = torch.exp(-bce)

    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    focal_loss = (alpha_t * (1 - pt) ** gamma * bce).mean()

    return dice.mean() + focal_loss
