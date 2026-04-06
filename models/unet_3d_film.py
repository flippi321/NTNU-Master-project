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
        # gamma/beta are [B,1] (simple) or [B,C] (complex) — reshape to broadcast over spatial dims
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1, 1)
        beta  = beta.view(beta.shape[0],   beta.shape[1],  1, 1, 1)
        return gamma * x + beta


# ============================================================
# FiLM generators
# ============================================================

class FiLMSimpleGenerator(nn.Module):
    """
    Generates (gamma, beta) for each conditioned block, FiLM-Si style.
    Output per block: gamma, beta are [B,1].
    """
    def __init__(self, cond_dim: int, n_blocks: int, hidden: int = 64):
        super().__init__()
        self.n_blocks = n_blocks
        total = 2 * n_blocks  # 2 scalars per block

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, hidden),
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
    Generates (gamma, beta) for each conditioned block, per-channel FiLM.
    Output per block: gamma, beta are [B,C_block] for that block.
    all_channels is an ordered list of channel counts for all conditioned blocks
    (encoder + bottleneck + decoder in forward order).
    """
    def __init__(self, cond_dim: int, all_channels: list[int], hidden: int = 64):
        super().__init__()
        self.all_channels = all_channels
        total = sum(2 * c for c in all_channels)  # per-channel gamma+beta

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden, total),
        )

        # Identity-ish init: gamma=1, beta=0
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, cond: torch.Tensor):
        v = self.mlp(cond)  # [B, total]
        params = []
        idx = 0
        for C in self.all_channels:
            g = v[:, idx:idx + C]; idx += C  # [B,C]
            b = v[:, idx:idx + C]; idx += C  # [B,C]
            params.append((1.0 + g, b))
        return params


# ============================================================
# U-Net blocks
# ============================================================

class ConvBlock(nn.Module):
    """
    Conv -> IN -> FiLM -> LeakyReLU -> Conv -> IN -> LeakyReLU

    FiLM is applied once, after the first norm and before the first activation.

    IMPORTANT: the activation after FiLM must NOT be inplace=True.
    The FiLM output tensor (gamma * x + beta) is a leaf in the autograd graph
    that needs to be read during the backward pass. An inplace op would
    overwrite it before the gradient can be computed through the multiplication,
    silently corrupting gamma/beta gradients. The second activation (no FiLM
    upstream) is safe to run inplace as in the baseline.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch)

        # Not inplace — this activation may follow FiLM, whose output tensor
        # must survive intact for the backward pass through gamma's multiplication
        self.act1 = nn.LeakyReLU(0.1, inplace=False)
        # Safe to be inplace — no FiLM node upstream in this half of the block
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        self.film = FiLMLayer()

    def forward(self, x, film=None):
        x = self.norm1(self.conv1(x))
        # FiLM after the first norm, before the first activation — single application
        if film is not None:
            gamma, beta = film
            x = self.film(x, gamma, beta)
        x = self.act1(x)  # out-of-place to protect FiLM's backward when film was applied

        x = self.act2(self.norm2(self.conv2(x)))
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
        self.up     = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv   = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip, film=None):
        x = self.up(x)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=film)


class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv3d(in_ch, out_ch, 1)
        self.conv   = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip, film=None):
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, film=film)


# ============================================================
# FiLM U-Net (full FiLM: encoder + bottleneck + decoder)
# ============================================================

class FiLMUNet3D(nn.Module):
    """
    Full FiLM 3D U-Net — conditioning applied to encoder, bottleneck, and decoder.
    If use_simple=True:  FiLM-Si scalars per block ([B,1]).
    If use_simple=False: complex per-channel FiLM ([B,C_block]).

    Conditioned blocks (10 total):
      Encoder:     e1, e2, e3, e4        channels: [B, 2B, 4B, 8B]
      Bottleneck:  b1, b2                channels: [16B, 16B]
      Decoder:     u4, u3, u2, u1        channels: [8B, 4B, 2B, B]

    GPU split mirrors the baseline UNet3D:
      GPU 0 — encoder + bottleneck + heavy decoder (u4, u3)
      GPU 1 — light decoder (u2, u1) + output conv
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, cond_dim=3, use_simple: bool = False):
        super().__init__()
        self.use_simple = use_simple
        self.dev0 = torch.device("cuda:0")
        self.dev1 = torch.device("cuda:1")

        # --- Encoder (GPU 0) ---
        self.e1 = ConvBlock(in_ch, base).to(self.dev0)
        self.e2 = Down(base, base * 2).to(self.dev0)
        self.e3 = Down(base * 2, base * 4).to(self.dev0)
        self.e4 = Down(base * 4, base * 8).to(self.dev0)

        # --- Bottleneck (GPU 0) — keeps the big memory spike on the same device ---
        self.b1 = ConvBlock(base * 8,  base * 16).to(self.dev0)
        self.b2 = ConvBlock(base * 16, base * 16).to(self.dev0)

        # --- Heavy decoder (GPU 0) ---
        self.u4 = Up(base * 16, base * 8).to(self.dev0)
        self.u3 = Up(base * 8,  base * 4).to(self.dev0)

        # --- Light decoder (GPU 1) ---
        self.u2  = Up(base * 4, base * 2).to(self.dev1)
        self.u1  = Out(base * 2, base).to(self.dev1)
        self.out = nn.Conv3d(base, out_ch, 1).to(self.dev1)

        # --- FiLM generator (GPU 0) ---
        # Block order: enc(e1,e2,e3,e4) + bott(b1,b2) + dec(u4,u3,u2,u1) = 10 blocks
        enc_channels  = [base, base * 2, base * 4, base * 8]
        bott_channels = [base * 16, base * 16]
        dec_channels  = [base * 8, base * 4, base * 2, base]
        all_channels  = enc_channels + bott_channels + dec_channels  # 10 blocks

        if use_simple:
            self.film_gen = FiLMSimpleGenerator(
                cond_dim=cond_dim, n_blocks=len(all_channels), hidden=64
            ).to(self.dev0)
        else:
            self.film_gen = FiLMComplexGenerator(
                cond_dim=cond_dim, all_channels=all_channels, hidden=64
            ).to(self.dev0)

    def forward(self, x, cond):
        films = self.film_gen(cond.to(self.dev0))
        # films[0..3]  → encoder  (e1, e2, e3, e4)
        # films[4..5]  → bottleneck (b1, b2)
        # films[6..9]  → decoder  (u4, u3, u2, u1)

        # --- Encoder + bottleneck + heavy decoder on GPU 0 ---
        x  = x.to(self.dev0)
        s1 = self.e1(x,   film=films[0])
        s2, x2 = self.e2(s1, film=films[1])
        s3, x3 = self.e3(x2, film=films[2])
        s4, x4 = self.e4(x3, film=films[3])

        b  = self.b2(self.b1(x4, film=films[4]), film=films[5])
        d4 = self.u4(b,  s4, film=films[6])
        d3 = self.u3(d4, s3, film=films[7])

        # --- Move only the light-decoder inputs to GPU 1 ---
        d3 = d3.to(self.dev1)
        s2 = s2.to(self.dev1)
        s1 = s1.to(self.dev1)
        f8 = (films[8][0].to(self.dev1), films[8][1].to(self.dev1))
        f9 = (films[9][0].to(self.dev1), films[9][1].to(self.dev1))

        # --- Light decoder on GPU 1 ---
        d2 = self.u2(d3, s2, film=f8)
        d1 = self.u1(d2, s1, film=f9)
        return self.out(d1)
