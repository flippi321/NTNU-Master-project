import torch
import torch.nn as nn

# ----------------------------
# Residual 3D U-Net components
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # First convolutional layer
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            # Second convolutional layer
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2, groups=out_ch, bias=False)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.reduce(x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResidualUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        # --- Encoder ---
        self.e1 = ConvBlock(in_ch, base)
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)

        # --- Bottleneck ---
        self.bott = ConvBlock(base * 8, base * 16)
        self.bott_sec = ConvBlock(base * 16, base * 16)

        # --- Decoder ---
        self.u4 = Up(base * 16, base * 8)
        self.u3 = Up(base * 8, base * 4)
        self.u2 = Up(base * 4, base * 2)
        self.u1 = Out(base * 2, base)  # last layer: no upsample

        self.out = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)

        s2, x2 = self.d2(s1)
        s3, x3 = self.d3(x2)
        s4, x4 = self.d4(x3)

        b = self.bott_sec(self.bott(x4))

        d4 = self.u4(b, s4)
        d3 = self.u3(d4, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u1(d2, s1)

        delta = self.out(d1)
        return x + delta, delta
