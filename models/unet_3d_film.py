import torch
import torch.nn as nn

# ============================================================
# Simple FiLM layer (FiLM-Si papir)
# ============================================================
class FiLMSimple3D(nn.Module):
    """
    FiLM-Si (simple): y = gamma * x + beta, but gamma/beta are scalars
    per sample (shared across channels).
    Paper: "FiLM simple layer applies the same affine transformation to
    all the input feature maps".  :contentReference[oaicite:1]{index=1}
    """
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM-Si to 3D input. 
        x: [B, C, D, H, W]
        gamma, beta: [B, 1]
        
        Returns: [B, C, D, H, W]
        """
        gamma = gamma[:, :, None, None, None] 
        beta  = beta[:, :, None, None, None]
        return gamma * x + beta

class FiLMSimpleGenerator(nn.Module):
    """
    Generates (gamma, beta) for each encoder block, FiLM-Si style.
    Output per block is (gamma_i, beta_i) with shape [B,1] each.

    This matches the paper's "simple" FiLM where gamma/beta do NOT depend
    on channel c.  :contentReference[oaicite:2]{index=2}
    """
    def __init__(self, cond_dim: int, n_blocks: int, hidden: int = 128):
        super().__init__()
        self.n_blocks = n_blocks
        total = 2 * n_blocks  # 2 scalars per block

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, total),
        )

        # Identity-ish init: gamma = 1, beta = 0
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond):
        # cond: [B, cond_dim]
        v = self.mlp(cond)  # [B, 2*n_blocks]
        params = []
        idx = 0
        for _ in range(self.n_blocks):
            g = v[:, idx:idx+1]; idx += 1  # [B,1]
            b = v[:, idx:idx+1]; idx += 1  # [B,1]
            params.append((1.0 + g, b))
        return params


# ============================================================
# U-Net blocks
# ============================================================
class ConvBlock(nn.Module):
    """Conv -> IN -> (FiLM-Si) -> LeakyReLU twice"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.act2  = nn.LeakyReLU(0.1, inplace=True)

        self.film = FiLMSimple3D()

    def forward(self, x, film=None):
        # film: (gamma, beta) with gamma/beta shape [B,1] (FiLM-Si) or None
        x = self.conv1(x)
        x = self.norm1(x)
        if film is not None:
            x = self.film(x, *film)   # same affine for all channels
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if film is not None:
            x = self.film(x, *film)
        x = self.act2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.Conv3d(out_ch, out_ch, 2, stride=2, groups=out_ch, bias=False)

    def forward(self, x, film=None):
        skip = self.conv(x, film=film)
        down = self.pool(skip)
        return skip, down


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=None)  # decoder is NOT FiLMed


class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=None)  # decoder is NOT FiLMed


# ============================================================
# FiLM U-Net (encoder-only FiLM-Si)
# ============================================================
class FiLMUNet3D(nn.Module):
    """
    Encoder-only FiLM-Si 3D U-Net (simple FiLM per 1907.01277).
    forward(x, cond) where cond is [B, cond_dim].
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, cond_dim=6):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = Down(base, base * 2)
        self.e3 = Down(base * 2, base * 4)
        self.e4 = Down(base * 4, base * 8)

        # Bottleneck (NOT FiLMed)
        self.b1 = ConvBlock(base * 8, base * 16)
        self.b2 = ConvBlock(base * 16, base * 16)

        # Decoder
        self.u4 = Up(base * 16, base * 8)
        self.u3 = Up(base * 8, base * 4)
        self.u2 = Up(base * 4, base * 2)
        self.u1 = Out(base * 2, base)

        self.out = nn.Conv3d(base, out_ch, 1)

        # FiLM generator for encoder blocks only: 4 blocks => 4 (gamma,beta) pairs
        self.film_gen = FiLMSimpleGenerator(cond_dim=cond_dim, n_blocks=4, hidden=128)

    def forward(self, x, cond):
        films = self.film_gen(cond)  # [(g1,b1),(g2,b2),(g3,b3),(g4,b4)], each [B,1]

        s1 = self.e1(x,  film=films[0])
        s2, x2 = self.e2(s1, film=films[1])
        s3, x3 = self.e3(x2, film=films[2])
        s4, x4 = self.e4(x3, film=films[3])

        b = self.b2(self.b1(x4, film=None), film=None)

        d4 = self.u4(b, s4)
        d3 = self.u3(d4, s3)
        d2 = self.u2(d3, s2)
        d1 = self.u1(d2, s1)

        return self.out(d1)
