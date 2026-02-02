import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
 
# ----------------------------
# Residual 3D U-Net components
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x): return self.block(x)
 
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # depthwise downsample (cheap) + conv block
        self.pool = nn.Conv3d(in_ch, in_ch, kernel_size=2, stride=2, groups=in_ch)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))
 
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # pad to match (handles odd shapes)
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = F.pad(x, [0, dx, 0, dy, 0, dz])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
 
class ResidualUNet3D(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base*2)
        self.e3 = Down(base*2, base*4)
        self.e4 = Down(base*4, base*8)
        self.bott = ConvBlock(base*8, base*16)
        self.bott_sec = ConvBlock(base*16, base*16)
 
        self.u4 = Up(base*16, base*8)
        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)
        self.out = nn.Conv3d(base, 1, 1)  # residual Δ
 
    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b  = self.bott_sec(self.bott(s4))
        d4 = self.u4(b, s4)
        d3 = self.u3(d4, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u1(d2, s1)
        delta = self.out(d1)
        return x + delta, delta  # predicted follow-up and residual
    
# ----------------------------
# 3D Discriminator for GANs
# ----------------------------
class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # You can make this PatchGAN-like: progressively downsample in 3D.
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
 
            nn.Conv3d(256, 1, 1, stride=1, padding=0),
        )
    def forward(self, x):
        return self.net(x).mean(dim=[2,3,4])  # (B,1,*,*,*) -> (B,1) score
 
class DoubleConv3D(nn.Module):
    """(Conv3D → IN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),   # or LeakyReLU if you prefer
 
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.net(x)
 
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),   # or LeakyReLU if you prefer
 
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.net(x)
 
# ----------------------------
# 3D U-Net Generator for GANs
# ----------------------------
 
class UpConvBlock3D(nn.Module):
    """Upsample + concat skip + DoubleConv3D (like 3D U-Net)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_channels * 2, out_channels)
 
    def forward(self, x, skip):
        x = self.up(x)
        # same padding/cropping logic you already have
        if x.shape[2:] != skip.shape[2:]:
            diffD = skip.size(2) - x.size(2)
            diffH = skip.size(3) - x.size(3)
            diffW = skip.size(4) - x.size(4)
            x = F.pad(x, [diffW // 2, diffW - diffW // 2,
                          diffH // 2, diffH - diffH // 2,
                          diffD // 2, diffD - diffD // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
 
 
class Generator3D(nn.Module):
    """3D U-Net Generator for continuous MRI prediction"""
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3D(in_channels, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock3D(base_ch * 4, base_ch * 8)
 
        # Downsample layers
        self.down = nn.MaxPool3d(2)
 
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_ch * 8, base_ch * 16)
 
        # Decoder
        self.up3 = UpConvBlock3D(base_ch * 16, base_ch * 8)
        self.up2 = UpConvBlock3D(base_ch * 8, base_ch * 4)
        self.up1 = UpConvBlock3D(base_ch * 4, base_ch * 2)
        self.up0 = UpConvBlock3D(base_ch * 2, base_ch)
 
        # Output
        self.out_conv = nn.Conv3d(base_ch, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()  # constrain to [0,1]
 
    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.down(s1))
        s3 = self.enc3(self.down(s2))
        s4 = self.enc4(self.down(s3))
 
        # Bottleneck
        b = self.bottleneck(self.down(s4))
 
        # Decoder with skips
        d3 = self.up3(b, s4)
        d2 = self.up2(d3, s3)
        d1 = self.up1(d2, s2)
        d0 = self.up0(d1, s1)
 
        out = self.out_conv(d0)
        out = self.activation(out)
        return out