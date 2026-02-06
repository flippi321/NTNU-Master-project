import torch
import torch.nn as nn

# ============================================================
# FiLM layers
# ============================================================

class FiLMLayer(nn.Module):
    """
    FiLM taking a torch input (x) and transforming it with
    y = gamma * x + beta
    """
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        gamma = gamma[:, :, None, None, None]
        beta  = beta[:, :, None, None, None]
        return gamma * x + beta


# ============================================================
# FiLM generators
# ============================================================

class FiLMSimpleGenerator(nn.Module):
    """
    Generates (gamma, beta) for each encoder block, FiLM-Si style.
    Output per block: gamma,beta are [B,1].
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

        # Identity-ish init: gamma=1, beta=0
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor):
        v = self.mlp(cond)  # [B, 2*n_blocks]
        params = []
        idx = 0
        for _ in range(self.n_blocks):
            g = v[:, idx:idx+1]; idx += 1  # [B,1]
            b = v[:, idx:idx+1]; idx += 1  # [B,1]
            params.append((1.0 + g, b))
        return params


class FiLMComplexGenerator(nn.Module):
    """
    Generates (gamma, beta) for each encoder block, complex (per-channel) FiLM.
    Output per block: gamma,beta are [B,C_block] for that block.
    """
    def __init__(self, cond_dim: int, enc_channels: list[int], hidden: int = 128):
        super().__init__()
        self.enc_channels = enc_channels
        total = sum(2 * c for c in enc_channels)  # per-channel gamma+beta

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, total),
        )

        # identity-ish init: gamma=1, beta=0
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor):
        v = self.mlp(cond)  # [B, total]
        params = []
        idx = 0
        for C in self.enc_channels:
            g = v[:, idx:idx + C]; idx += C  # [B,C]
            b = v[:, idx:idx + C]; idx += C  # [B,C]
            params.append((1.0 + g, b))
        return params


# ============================================================
# U-Net blocks
# ============================================================

class ConvBlock(nn.Module):
    """Conv -> IN -> (FiLM) -> LeakyReLU twice"""
    def __init__(self, in_ch, out_ch, use_simple: bool = True):
        super().__init__()
        self.use_simple = use_simple

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.act2  = nn.LeakyReLU(0.1, inplace=True)

        self.film  = FiLMLayer()

    def _apply_film(self, x: torch.Tensor, film):
        if film is None:
            return x
        gamma, beta = film
        if self.use_simple:
            return self.film(x, gamma, beta)    # gamma/beta: [B,1]
        else:
            return self.film(x, gamma, beta)    # gamma/beta: [B,C]
        
    def forward(self, x, film=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self._apply_film(x, film)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self._apply_film(x, film)
        x = self.act2(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_simple: bool = True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, use_simple=use_simple)
        self.pool = nn.Conv3d(out_ch, out_ch, 2, stride=2, groups=out_ch, bias=False)

    def forward(self, x, film=None):
        skip = self.conv(x, film=film)
        down = self.pool(skip)
        return skip, down


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_simple: bool = True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch, use_simple=use_simple)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=None)  # decoder is NOT FiLMed


class Out(nn.Module):
    def __init__(self, in_ch, out_ch, use_simple: bool = True):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv = ConvBlock(out_ch * 2, out_ch, use_simple=use_simple)

    def forward(self, x, skip):
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=None)  # decoder is NOT FiLMed


# ============================================================
# FiLM U-Net (encoder-only FiLM, selectable simple/complex)
# ============================================================

class FiLMUNet3D(nn.Module):
    """
    Encoder-only FiLM 3D U-Net.
    If use_simple=True: FiLM-Si scalars per block ([B,1]).
    If use_simple=False: complex per-channel FiLM ([B,C_block]).
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, cond_dim=6, use_simple: bool = True):
        super().__init__()
        self.use_simple = use_simple

        # Encoder
        self.e1 = ConvBlock(in_ch, base, use_simple=use_simple)
        self.e2 = Down(base, base * 2, use_simple=use_simple)
        self.e3 = Down(base * 2, base * 4, use_simple=use_simple)
        self.e4 = Down(base * 4, base * 8, use_simple=use_simple)

        # Bottleneck (NOT FiLMed)
        self.b1 = ConvBlock(base * 8, base * 16, use_simple=use_simple)
        self.b2 = ConvBlock(base * 16, base * 16, use_simple=use_simple)

        # Decoder (NOT FiLMed)
        self.u4 = Up(base * 16, base * 8, use_simple=use_simple)
        self.u3 = Up(base * 8, base * 4, use_simple=use_simple)
        self.u2 = Up(base * 4, base * 2, use_simple=use_simple)
        self.u1 = Out(base * 2, base, use_simple=use_simple)

        self.out = nn.Conv3d(base, out_ch, 1)

        # FiLM generator (encoder blocks only)
        if use_simple:
            self.film_gen = FiLMSimpleGenerator(cond_dim=cond_dim, n_blocks=4, hidden=128)
        else:
            enc_channels = [base, base * 2, base * 4, base * 8]
            self.film_gen = FiLMComplexGenerator(cond_dim=cond_dim, enc_channels=enc_channels, hidden=128)

    def forward(self, x, cond):
        films = self.film_gen(cond)  # list of 4 (gamma,beta)

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
