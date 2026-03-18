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

        self.norm_k = nn.LayerNorm(head_dim)
        self.norm_v = nn.LayerNorm(head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, D, H, W):
        """
        x: (B, N, C), N = D*H*W
        D,H,W must be provided (volume grid dims).
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

        Np = k_red.shape[1]
        k_red = k_red.reshape(B, Np, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v_red = v_red.reshape(B, Np, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k_red = self.norm_k(k_red)
        v_red = self.norm_v(v_red)

        attn = (q @ k_red.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

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

    def forward(self, x, D, H, W):
        x = x + self.attn(self.norm1(x), D, H, W)
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

    def forward(self, x):
        B, R, Np, C = x.shape
        qkv = self.qkv(x).reshape(B, R, Np, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(-1, -2).reshape(B, R, Np, C)
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
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

    def forward(self, x, feature=False):
        """
        x: (B, C, D, H, W)
        D,H,W must be divisible by pd,ph,pw respectively.
        """
        b, c, d, h, w = x.shape
        assert d % self.pd == 0 and h % self.ph == 0 and w % self.pw == 0, \
            "D,H,W must be divisible by (pd,ph,pw) for local patches"

        # Local: (B, R, Np, C) where R=(d/pd)*(h/ph)*(w/pw), Np=pd*ph*pw
        loc_x = rearrange(
            x, 'b c (nd pd) (nh ph) (nw pw) -> b (nd nh nw) (pd ph pw) c',
            pd=self.pd, ph=self.ph, pw=self.pw
        )
        loc_y = self.loc_attn(loc_x)
        loc_y = rearrange(
            loc_y, 'b (nd nh nw) (pd ph pw) c -> b c (nd pd) (nh ph) (nw pw)',
            nd=d // self.pd, nh=h // self.ph, nw=w // self.pw,
            pd=self.pd, ph=self.ph, pw=self.pw
        )

        # Global: tokens (B, N, C)
        glo_x = x.flatten(2).transpose(1, 2)   # (B, N, C)
        glo_y = self.glo_attn(glo_x, d, h, w)  # (B, N, C)
        glo_y = glo_y.transpose(1, 2).reshape(b, c, d, h, w)

        gate = torch.sigmoid(loc_y + glo_y)
        out = x * gate
        if feature:
            return loc_y, glo_y, out
        return out


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
# META-Unet for 3D volumes (same fusion logic, trilinear upsample)
# ============================================================
class META_Unet3D(nn.Module):
    def __init__(self, nIn=1, classes=2, p1=(2,4,4), p2=(2,4,4), p3=(2,4,4)):
        super().__init__()
        channel = [32, 64, 128, 256, 512]
        num_heads = 4

        self.backbone = SimpleBackbone3D(in_ch=nIn, channels=tuple(channel))

        # project all scales to channel[0], like original
        self.proj4 = CBR3D(nIn=channel[1], nOut=channel[0], kSize=1)
        self.proj8 = CBR3D(nIn=channel[2], nOut=channel[0], kSize=1)
        self.proj16 = CBR3D(nIn=channel[3], nOut=channel[0], kSize=1)
        self.proj32 = CBR3D(nIn=channel[4], nOut=channel[0], kSize=1)

        # META blocks (now 3D). Ratios should be feasible at each scale.
        # You can tune ratio_* to control global token reduction (efficiency).
        self.mstf32_16 = META3D(dim=channel[0], pd=p1[0], ph=p1[1], pw=p1[2],
                                ratio_d=2, ratio_h=4, ratio_w=4, num_heads=num_heads, drop=0., attn_drop=0.)
        self.mstf16_8 = META3D(dim=channel[0], pd=p2[0], ph=p2[1], pw=p2[2],
                               ratio_d=2, ratio_h=4, ratio_w=4, num_heads=num_heads, drop=0., attn_drop=0.)
        self.mstf8_4 = META3D(dim=channel[0], pd=p3[0], ph=p3[1], pw=p3[2],
                              ratio_d=2, ratio_h=4, ratio_w=4, num_heads=num_heads, drop=0., attn_drop=0.)

        self.seg_head = Seg_head3D(channel[0], classes)

    def forward(self, x):
        # backbone features
        feat4, feat8, feat16, feat32 = self.backbone(x)

        # projection to common channel dim
        feat4 = self.proj4(feat4)
        feat8 = self.proj8(feat8)
        feat16 = self.proj16(feat16)
        feat32 = self.proj32(feat32)

        # top-down fusion with META refinement (use trilinear)
        feat32 = F.interpolate(feat32, scale_factor=2, mode="trilinear", align_corners=True)
        feat16 = self.mstf32_16(feat16 + feat32)

        feat16 = F.interpolate(feat16, scale_factor=2, mode="trilinear", align_corners=True)
        feat8 = self.mstf16_8(feat8 + feat16)

        feat8 = F.interpolate(feat8, scale_factor=2, mode="trilinear", align_corners=True)
        feat4 = self.mstf8_4(feat4 + feat8)

        # segmentation head upsamples x4 overall (2 then 2)
        out = self.seg_head(feat4)
        return out