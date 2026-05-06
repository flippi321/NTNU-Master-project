import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# ============================================================
# 3D building blocks (replace 2D conv/BN/interp with 3D)
# ============================================================
class CBR3D(nn.Module):
    """Conv3d + BN3d + PReLU"""
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, d=1):
        super().__init__()
        padding = (kSize - 1) // 2
        self.conv3d = nn.Conv3d(
            nIn, nOut,
            kernel_size=kSize,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
            dilation=d
        )
        self.bn = nn.BatchNorm3d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(self.conv3d(x)))


class Mlp(nn.Module):
    """Token-space FFN"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ============================================================
# Efficient self-attention (GLOBAL) adapted to 3D volumes
# ============================================================
class Self_Attention3D(nn.Module):
    """
    Efficient self-attention for 3D:
      - tokens: (B, N, C) where N = D*H*W
      - Q from full tokens
      - K/V from strided Conv3d with kernel=stride=(rd,rh,rw) to reduce tokens
    """
    def __init__(self, dim, ratio_d=2, ratio_h=2, ratio_w=2,
                 num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.ratio_d = ratio_d
        self.ratio_h = ratio_h
        self.ratio_w = ratio_w
        self.s = int(ratio_d * ratio_h * ratio_w)  # token reduction factor (approx)

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # K/V spatial reducers in feature-map space
        self.ke = nn.Conv3d(dim, dim,
                            kernel_size=(ratio_d, ratio_h, ratio_w),
                            stride=(ratio_d, ratio_h, ratio_w),
                            bias=qkv_bias)
        self.ve = nn.Conv3d(dim, dim,
                            kernel_size=(ratio_d, ratio_h, ratio_w),
                            stride=(ratio_d, ratio_h, ratio_w),
                            bias=qkv_bias)

        # Linear projections for metadata tokens that share the K/V channel space.
        # Lazily created via add_module so the parameter cost is paid only when
        # the model is conditioned (callers that pass meta_tokens=None pay nothing).
        self.ke_meta = nn.Linear(dim, dim, bias=qkv_bias)
        self.ve_meta = nn.Linear(dim, dim, bias=qkv_bias)

        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, D, H, W, meta_tokens=None):
        """
        x: (B, N, C), N = D*H*W
        D,H,W must be provided (volume grid dims).
        meta_tokens: optional (B, M, C) tokens prepended to the reduced K/V.
        """
        B, N, C = x.shape
        assert N == D * H * W, "Provided D,H,W do not match token length"

        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Q: (B, heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # K/V: reduce in 3D map space then tokenize
        k_map = k.transpose(1, 2).reshape(B, C, D, H, W)
        v_map = v.transpose(1, 2).reshape(B, C, D, H, W)

        k_red = self.ke(k_map).flatten(2).transpose(1, 2)  # (B, N', C)
        v_red = self.ve(v_map).flatten(2).transpose(1, 2)  # (B, N', C)

        if meta_tokens is not None:
            # Re-project metadata through the same K/V channel space and prepend.
            k_meta = self.ke_meta(meta_tokens) if hasattr(self, "ke_meta") else meta_tokens
            v_meta = self.ve_meta(meta_tokens) if hasattr(self, "ve_meta") else meta_tokens
            k_red = torch.cat([k_meta, k_red], dim=1)
            v_red = torch.cat([v_meta, v_red], dim=1)

        Np = k_red.shape[1]
        k_red = k_red.reshape(B, Np, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_red = v_red.reshape(B, Np, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_red = self.norm_k(k_red)
        v_red = self.norm_v(v_red)

        attn = (q @ k_red.transpose(-2, -1)) * self.scale
        # Force softmax to fp32 — under autocast the inputs are fp16 and softmax
        # of large attention scores overflows to NaN.
        attn = self.attn_drop(attn.float().softmax(dim=-1).to(q.dtype))

        out = (attn @ v_red).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class ETransformer_block3D(nn.Module):
    """Efficient Transformer block (global branch) for 3D"""
    def __init__(self, dim, ratio_d=2, ratio_h=2, ratio_w=2, num_heads=8,
                 qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 out_features=None, mlp_ratio=4.0):
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention3D(
            dim, ratio_d, ratio_h, ratio_w, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x, D, H, W, meta_tokens=None):
        x = x + self.attn(self.norm1(x), D, H, W, meta_tokens=meta_tokens)
        if self.out_features:
            return self.mlp(self.norm2(x))
        return x + self.mlp(self.norm2(x))


# ============================================================
# Self-attention (LOCAL) adapted to 3D patch windows
# ============================================================
class Self_Attention_local3D(nn.Module):
    """
    Local attention over 3D patches:
      input tokens shaped (B, R, Np, C)
      where R = number of patches, Np = pd*ph*pw
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, meta_tokens=None):
        B, R, Np, C = x.shape
        qkv = self.qkv(x).reshape(B, R, Np, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: (B, R, heads, Np, head_dim)

        if meta_tokens is not None:
            # meta_tokens: (B, M, C). Reuse self.qkv and broadcast across patches so
            # every local window sees the same metadata K/V. Q from metadata is unused.
            M_ = meta_tokens.shape[1]
            m_qkv = self.qkv(meta_tokens) \
                .reshape(B, M_, 3, self.num_heads, C // self.num_heads) \
                .permute(2, 0, 3, 1, 4)
            # m_qkv: (3, B, heads, M, head_dim)
            m_k, m_v = m_qkv[1], m_qkv[2]
            # Broadcast to per-patch: (B, R, heads, M, head_dim)
            m_k = m_k.unsqueeze(1).expand(B, R, -1, -1, -1)
            m_v = m_v.unsqueeze(1).expand(B, R, -1, -1, -1)
            k = torch.cat([m_k, k], dim=3)
            v = torch.cat([m_v, v], dim=3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # fp32 softmax — see Self_Attention3D for rationale.
        attn = self.attn_drop(attn.float().softmax(dim=-1).to(q.dtype))

        # (B, R, heads, Np, head_dim) -> (B, R, Np, heads, head_dim) -> (B, R, Np, C)
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, R, Np, C)
        out = self.proj_drop(self.proj(out))
        return out


class ETransformer_block_local3D(nn.Module):
    """Efficient Transformer block (local branch) for 3D"""
    def __init__(self, dim, qkv_bias=False, qk_scale=None, num_heads=8,
                 drop=0.0, attn_drop=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 out_features=None, mlp_ratio=4.0):
        super().__init__()
        self.out_features = out_features
        self.norm1 = norm_layer(dim)
        self.attn = Self_Attention_local3D(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden, out_features=out_features, act_layer=act_layer, drop=drop)

    def forward(self, x, meta_tokens=None):
        x = x + self.attn(self.norm1(x), meta_tokens=meta_tokens)
        if self.out_features:
            return self.mlp(self.norm2(x))
        return x + self.mlp(self.norm2(x))


# ============================================================
# META module for 3D volumes
# ============================================================
class META3D(nn.Module):
    """
    3D META:
      - Local branch: attention within (pd,ph,pw) patches
      - Global branch: efficient attention with (ratio_d,ratio_h,ratio_w) reduction
      - Gate = sigmoid(loc + glo), output = x * gate
    """
    def __init__(self, dim, pd=2, ph=4, pw=4,
                 ratio_d=2, ratio_h=2, ratio_w=2,
                 num_heads=8, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.pd, self.ph, self.pw = pd, ph, pw
        self.loc_attn = ETransformer_block_local3D(dim=dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop)
        self.glo_attn = ETransformer_block3D(dim=dim, ratio_d=ratio_d, ratio_h=ratio_h, ratio_w=ratio_w,
                                             num_heads=num_heads, drop=drop, attn_drop=attn_drop)

    def forward(self, x, meta_tokens=None, feature=False):
        """
        x: (B, C, D, H, W)
        D,H,W must be divisible by pd,ph,pw respectively.
        meta_tokens: optional (B, M, C) — passed to both global and local branches
                     so the gate is shaped by the patient/segment metadata.
        """
        b, c, d, h, w = x.shape
        assert d % self.pd == 0 and h % self.ph == 0 and w % self.pw == 0, \
            "D,H,W must be divisible by (pd,ph,pw) for local patches"

        # Local: (B, R, Np, C) where R=(d/pd)*(h/ph)*(w/pw), Np=pd*ph*pw
        loc_x = rearrange(
            x, 'b c (nd pd) (nh ph) (nw pw) -> b (nd nh nw) (pd ph pw) c',
            pd=self.pd, ph=self.ph, pw=self.pw
        )
        loc_y = self.loc_attn(loc_x, meta_tokens=meta_tokens)
        loc_y = rearrange(
            loc_y, 'b (nd nh nw) (pd ph pw) c -> b c (nd pd) (nh ph) (nw pw)',
            nd=d // self.pd, nh=h // self.ph, nw=w // self.pw,
            pd=self.pd, ph=self.ph, pw=self.pw
        )

        # Global: tokens (B, N, C)
        glo_x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        glo_y = self.glo_attn(glo_x, d, h, w, meta_tokens=meta_tokens)
        glo_y = glo_y.transpose(1, 2).reshape(b, c, d, h, w)

        gate = torch.sigmoid(loc_y + glo_y)
        out = x * gate
        if feature:
            return loc_y, glo_y, out
        return out


# ============================================================
# Metadata tokenizer (patient + segment metadata -> transformer tokens)
# ============================================================
class MetadataTokenizer(nn.Module):
    """
    Maps a metadata feature vector cond ∈ ℝ^F to a small bank of M tokens
    in the META channel space (B, M, C). The tokens are prepended to K/V
    inside the META blocks; metadata never scales or shifts feature maps
    directly (no FiLM γ/β).
    """
    def __init__(self, cond_dim: int, n_tokens: int, dim: int, hidden: int = 64):
        super().__init__()
        self.n_tokens = n_tokens
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_tokens * dim),
        )
        # LayerNorm pins the token scale so meta tokens enter attention K/V at the
        # same statistical scale as the LayerNorm'd feature tokens — without this
        # the attention scores depend on whatever scale the MLP produces, which
        # makes softmax unstable.
        self.norm = nn.LayerNorm(dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, F) -> (B, M, C)
        v = self.mlp(cond).reshape(cond.shape[0], self.n_tokens, self.dim)
        return self.norm(v)


# ============================================================
# 3D Seg head (matches original pattern but 3D)
# ============================================================
class Seg_head3D(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv1 = CBR3D(nIn=nIn, nOut=nIn, kSize=3)
        self.conv2 = nn.Conv3d(nIn, nOut, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = x + self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = self.conv2(x)
        return x


# ============================================================
# Backbone note:
#   Your code uses a 2D resnet34 returning feat4/8/16/32.
#   For true 3D volumes, you need a 3D backbone.
#   Below is a minimal, self-contained 3D backbone that returns
#   4 scales similar to your resnet outputs.
# ============================================================
class SimpleBackbone3D(nn.Module):
    """
    Minimal 3D backbone producing 4 feature scales:
      feat4, feat8, feat16, feat32 (roughly)
    Strides: 4,8,16,32 relative to input if you start with stride-2 twice.
    """
    def __init__(self, in_ch=1, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        c0, c1, c2, c3, c4 = channels

        # stem -> /2
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, c0, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True),
        )
        # /4
        self.stage1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            CBR3D(c0, c1, 3),
            CBR3D(c1, c1, 3),
        )
        # /8
        self.stage2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            CBR3D(c2, c2, 3),
        )
        # /16
        self.stage3 = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
            CBR3D(c3, c3, 3),
        )
        # /32
        self.stage4 = nn.Sequential(
            nn.Conv3d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c4),
            nn.ReLU(inplace=True),
            CBR3D(c4, c4, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        feat4 = self.stage1(x)
        feat8 = self.stage2(feat4)
        feat16 = self.stage3(feat8)
        feat32 = self.stage4(feat16)
        return feat4, feat8, feat16, feat32


# ============================================================
# META-Unet for 3D volumes — Hunt-3 + metadata -> Hunt-4 regression
# ============================================================
class META_Unet3D(nn.Module):
    """
    Multi-scale efficient transformer attention U-Net (paper-faithful) adapted
    to a 3D regression task. Inputs:
        x    : (B, in_ch, D, H, W) — Hunt-3 volume
        cond : (B, F)              — patient + segment metadata vector
    Output:
        y_hat: (B, out_ch, D, H, W) — predicted Hunt-4 volume

    Metadata is consumed via MetadataTokenizer -> M tokens in channel space,
    prepended to the K/V sequences inside both branches of every META block.
    No FiLM γ/β.

    Runs entirely on `device` (defaults to cuda:0 if available, else cpu).
    """
    def __init__(self, in_ch=1, out_ch=1, base=32, cond_dim=3, n_meta_tokens=8,
                 p1=(2,4,4), p2=(2,4,4), p3=(2,4,4),
                 r1=(2,2,4), r2=(4,4,4), r3=(4,16,16),
                 num_heads=4, auto_pad=True, residual=True,
                 device: torch.device | str | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.auto_pad = auto_pad
        self.residual = residual

        channel = [base, base * 2, base * 4, base * 8, base * 16]

        self.backbone = SimpleBackbone3D(in_ch=in_ch, channels=tuple(channel)).to(self.device)

        # project all scales to channel[0]
        self.proj4  = CBR3D(nIn=channel[1], nOut=channel[0], kSize=1).to(self.device)
        self.proj8  = CBR3D(nIn=channel[2], nOut=channel[0], kSize=1).to(self.device)
        self.proj16 = CBR3D(nIn=channel[3], nOut=channel[0], kSize=1).to(self.device)
        self.proj32 = CBR3D(nIn=channel[4], nOut=channel[0], kSize=1).to(self.device)

        # Metadata path (single shared tokenizer, reused by all META blocks).
        self.tokenizer = MetadataTokenizer(
            cond_dim=cond_dim, n_tokens=n_meta_tokens, dim=channel[0]
        ).to(self.device)

        self.mstf32_16 = META3D(dim=channel[0], pd=p1[0], ph=p1[1], pw=p1[2],
                                ratio_d=r1[0], ratio_h=r1[1], ratio_w=r1[2],
                                num_heads=num_heads).to(self.device)
        self.mstf16_8  = META3D(dim=channel[0], pd=p2[0], ph=p2[1], pw=p2[2],
                                ratio_d=r2[0], ratio_h=r2[1], ratio_w=r2[2],
                                num_heads=num_heads).to(self.device)
        self.mstf8_4   = META3D(dim=channel[0], pd=p3[0], ph=p3[1], pw=p3[2],
                                ratio_d=r3[0], ratio_h=r3[1], ratio_w=r3[2],
                                num_heads=num_heads).to(self.device)

        self.seg_head = Seg_head3D(channel[0], out_ch).to(self.device)

        if self.residual:
            # Zero-init the final conv so delta ≈ 0 at start; recon ≈ x at iter 0,
            # which is much closer to Hunt-4 than a random output and gives the
            # SSIM-L1 loss a reasonable starting point.
            nn.init.zeros_(self.seg_head.conv2.weight)
            if self.seg_head.conv2.bias is not None:
                nn.init.zeros_(self.seg_head.conv2.bias)

    def _required_divisor(self):
        # The local rearrange in each META requires that the resolution at that scale
        # be divisible by the patch size, hence the input must be divisible by:
        #   /16 -> 16 * p1
        #   /8  ->  8 * p2
        #   /4  ->  4 * p3
        # Take the per-axis LCM so all three are satisfied with one pad.
        from math import lcm
        d = lcm(16 * self.p1[0],  8 * self.p2[0], 4 * self.p3[0])
        h = lcm(16 * self.p1[1],  8 * self.p2[1], 4 * self.p3[1])
        w = lcm(16 * self.p1[2],  8 * self.p2[2], 4 * self.p3[2])
        return d, h, w

    def forward(self, x, cond):
        x = x.to(self.device)
        cond = cond.to(self.device)

        # Pad to the smallest shape that satisfies divisibility at every META scale.
        _, _, D0, H0, W0 = x.shape
        if self.auto_pad:
            div_d, div_h, div_w = self._required_divisor()
            pad_d = (div_d - D0 % div_d) % div_d
            pad_h = (div_h - H0 % div_h) % div_h
            pad_w = (div_w - W0 % div_w) % div_w
            if pad_d or pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

        meta_tokens = self.tokenizer(cond)

        # Backbone -> 4 scales.
        feat4, feat8, feat16, feat32 = self.backbone(x)
        feat4  = self.proj4(feat4)
        feat8  = self.proj8(feat8)
        feat16 = self.proj16(feat16)
        feat32 = self.proj32(feat32)

        # Top-down fusion with metadata-aware META refinement.
        feat32 = F.interpolate(feat32, scale_factor=2, mode="trilinear", align_corners=True)
        feat16 = self.mstf32_16(feat16 + feat32, meta_tokens=meta_tokens)

        feat16 = F.interpolate(feat16, scale_factor=2, mode="trilinear", align_corners=True)
        feat8  = self.mstf16_8(feat8 + feat16, meta_tokens=meta_tokens)

        feat8 = F.interpolate(feat8, scale_factor=2, mode="trilinear", align_corners=True)

        feat4 = self.mstf8_4(feat4 + feat8, meta_tokens=meta_tokens)

        delta = self.seg_head(feat4)

        # Crop back to original input shape (auto-pad may have grown D/H/W).
        delta = delta[:, :, :D0, :H0, :W0]

        if self.residual:
            # Residual head: predict only the change from Hunt-3 to Hunt-4.
            # Same contract as ResidualUNet3D: returns (recon, delta).
            return x[:, :, :D0, :H0, :W0] + delta, delta
        return delta