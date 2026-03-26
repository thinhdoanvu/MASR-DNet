import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import ceil, gcd
#from torchvision.ops import deform_conv2d
####YOLO-LIGHT-v8

class MyGhost(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 ratio=2, dw_kernel_size=3, relu=True):
        super(MyGhost, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)   # số kênh ban đầu
        new_channels = out_channels - init_channels # số kênh "rẻ" sinh ra thêm

        # Conv chuẩn tạo feature maps gốc
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # Depthwise conv sinh feature maps "rẻ"
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, 
                      dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        #print(f"MyGhost input: {x.shape}")
        x1 = self.primary_conv(x)   # feature gốc
        x2 = self.cheap_operation(x1)  # feature sinh thêm
        out = torch.cat([x1, x2], dim=1)  # concat lại
        #print(f"MyGhost output: {out.shape}")
        return out[:, :self.out_channels, :, :]  # đảm bảo đúng số kênh out

class MyDCNv2(nn.Module):
    """
    Deformable Conv v2 (PyTorch version, offset + mask)
    - Nếu dùng torchvision >= 0.14: có deform_conv2d hỗ trợ mask
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deformable_groups=1):
        super(MyDCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.deformable_groups = deformable_groups

        # Conv để sinh offset + mask
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        #print(f"MyDCNv2 input: {x.shape}")
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)  # offset_x, offset_y, mask
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)

        # deformable conv2d với mask modulation
        out = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.conv.weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            mask=mask
        )
        #print(f"MyDCNv2 output: {self.act(self.bn(out)).shape}")
        return self.act(self.bn(out))

class MySRBlock(nn.Module):
    """
    Small-object Refinement Block
    - Parallel convs với dilation khác nhau
    - Channel attention (SE block)
    """
    def __init__(self, c1, c2):
        super(MySRBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, padding=1, dilation=1, groups=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=2, dilation=2, groups=1)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

        # Squeeze-and-Excitation
        reduction = max(1, c2 // 4)  # tránh bị 0
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, c2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #print(f"MyDCNv2 input: {x.shape}")
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        out = feat1 + feat2
        out = self.relu(self.bn(out))
        w = self.se(out)
        #print(f"MySRBlock output: {(out*w).shape}")
        return out * w

class DynamicGhost(nn.Module):
    """
    Dynamic Ghost Convolution
    - Kết hợp giữa real conv (primary_conv) và cheap operation (depthwise conv)
    - Tỉ lệ real/cheap channels được tính động dựa trên ratio
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 ratio=0.5, act=True):
        super(DynamicGhost, self).__init__()
        assert 0 < ratio <= 1, "ratio phai nam trong (0, 1]"

        self.in_channels = in_channels
        self.out_channels = out_channels

        # số kênh real conv
        self.init_channels = math.ceil(out_channels * ratio)
        # số kênh cheap op
        self.new_channels = out_channels - self.init_channels

        padding = kernel_size // 2

        # Real conv (primary)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.init_channels,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

        # Cheap operation (depthwise conv)
        if self.new_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.init_channels, self.new_channels,
                          kernel_size=3, stride=1, padding=1,
                          groups=self.init_channels, bias=False),
                nn.BatchNorm2d(self.new_channels),
                nn.ReLU(inplace=True) if act else nn.Identity()
            )
        else:
            self.cheap_operation = None

    def forward(self, x):
        #print(f"DynamicGhost input: {x.shape}")

        # real conv
        x1 = self.primary_conv(x)

        # cheap op
        if self.cheap_operation is not None:
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1

        # dam bao dung out_channels
        #print(f"DynamicGhost output: {out.shape}")
        return out[:, :self.out_channels, :, :]

class DynamicDCNv2(nn.Module):
    """
    Dynamic DCNv2 with gating
    - Học offset+mask như DCNv2
    - Gating alpha \in [0,1] từ input (GAP -> MLP nhỏ)
    - Điều biến: offset' = alpha * offset, mask' = alpha * mask
    - Trộn mềm: y = alpha * y_deform + (1 - alpha) * y_base
      (giúp khi alpha nhỏ, gần như chạy conv thường -> tiết kiệm FLOPs & regularize)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = None,
                 deformable_groups: int = 1,
                 gate_hidden: int = 128,
                 act: str = "relu"):
        super().__init__()
        assert kernel_size in (1, 3, 5, 7), "kernel_size nên là số lẻ nhỏ (1/3/5/7)"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size // 2) if padding is None else padding
        self.deformable_groups = deformable_groups

        # (Δx, Δy, m) cho mỗi vị trí kernel
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=True,
        )
        nn.init.constant_(self.offset_mask_conv.weight, 0.0)
        nn.init.constant_(self.offset_mask_conv.bias, 0.0)

        # Trọng số cho nhánh deformable
        self.weight_deform = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight_deform, a=math.sqrt(5))
        self.bias_deform = None  # DCNv2 thường không dùng bias ở nhánh deform

        # Nhánh chuẩn (base conv)
        self.base_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=self.padding, bias=False
        )

        # Chuẩn hoá + kích hoạt sau khi trộn
        self.bn = nn.BatchNorm2d(out_channels)
        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()

        # Gating: alpha \in [0,1] (per-sample scalar)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, gate_hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )
        # (tùy chọn) hệ số nhiệt để alpha sắc nét hơn
        self.register_buffer("gate_temp", torch.tensor(1.0))  # có thể set <1.0 để alpha nhạy hơn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, C_in, H, W]
        return: [N, C_out, H/stride, W/stride]
        """
        # ---- Gating alpha (per-sample) ----
        # alpha: [N,1,1,1]
        alpha = self.gate(self.gap(x))
        if self.gate_temp.item() != 1.0:
            alpha = torch.sigmoid(torch.logit(alpha.clamp(1e-6, 1-1e-6)) / self.gate_temp)

        # ---- Offset + mask ----
        # out_off: [N, dg*3*K*K, H_out, W_out] -> split 3 phần theo kênh
        out_off = self.offset_mask_conv(x)
        k2 = self.kernel_size * self.kernel_size
        dg = self.deformable_groups

        o1, o2, m = torch.chunk(out_off, 3, dim=1)  # mỗi phần: [N, dg*K*K, H_out, W_out]
        offset = torch.cat([o1, o2], dim=1)         # [N, dg*2*K*K, H_out, W_out]
        mask = torch.sigmoid(m)                      # [N, dg*K*K,   H_out, W_out]

        # Điều biến theo alpha
        # broadcast alpha: [N,1,1,1] -> scale toàn cục theo mẫu
        offset = offset * alpha
        mask = mask * alpha

        # ---- Nhánh deformable ----
        y_def = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight_deform,
            bias=self.bias_deform,
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            mask=mask
        )

        # ---- Nhánh base (chuẩn) ----
        y_base = self.base_conv(x)

        # ---- Trộn mềm theo alpha ----
        # alpha: [N,1,1,1] -> broadcast sang [N,C,H,W]
        y = alpha * y_def + (1.0 - alpha) * y_base

        # ---- BN + Act ----
        return self.act(self.bn(y))

# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        attn = self.fc1(avg_pool + max_pool)
        attn = F.silu(attn)
        attn = self.fc2(attn)
        return torch.sigmoid(attn) * x  

# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, groups=1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_pool, max_pool], dim=1)
        attn = self.conv(attn)
        return torch.sigmoid(attn) * x

# HAAM
class HAAM(nn.Module):
    def __init__(self, channels):
        super(HAAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.fusion_mlp = nn.Sequential(
            nn.Conv2d(2 * channels, 1, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        alpha = self.fusion_mlp(torch.cat([ca, sa], dim=1))
        M = alpha * ca + (1 - alpha) * sa
        M = M / (M.max() + 1e-6)  # tránh chia 0
        return M + x    

class GhostHAAM(nn.Module):
    """
    Ghost Convolution + HAAM
    - Primary conv tạo feature chính
    - Cheap operation (depthwise conv) tạo feature phụ
    - Ghép lại rồi qua HAAM để refine
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 ratio=2, dw_kernel_size=3, relu=True, use_haam=True):
        super(GhostHAAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_haam = use_haam

        # số kênh chính/phụ
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        # primary conv
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # cheap operation (depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1,
                      dw_kernel_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # HAAM
        if self.use_haam:
            self.haam = HAAM(out_channels)

    def forward(self, x):
        # primary feature
        x1 = self.primary_conv(x)
        # cheap feature
        x2 = self.cheap_operation(x1)
        # concat
        out = torch.cat([x1, x2], dim=1)
        # refine by HAAM
        if self.use_haam:
            out = self.haam(out)
        return out

class HAAMSRBlock(nn.Module):
    """
    Small-object Refinement Block
    - Parallel convs với dilation khác nhau
    - Attention bằng HAAM (thay cho SE)
    """
    def __init__(self, c1, c2):
        super(HAAMSRBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, padding=1, dilation=1, groups=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=2, dilation=2, groups=1)
        self.bn = nn.BatchNorm2d(c2)
        self.relu = nn.ReLU(inplace=True)

        # HAAM thay cho SE
        self.attn = HAAM(c2)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = self.bn(y1 + y2)
        y = self.relu(y)
        y = self.attn(y)  # dùng HAAM
        return y

class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, 
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # squeeze: (B, C, H, W) -> (B, C, 1)
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, C) -> (B, 1, C)
        y = self.conv(y)                                    # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1) # (B, C, 1, 1)
        return x * y.expand_as(x)

class ConvECA(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.eca = ECA(c2, k_size=k)

    def forward(self, x):
        #print(f"[ConvECA] Input: {x.shape}")
        x = self.act(self.bn(self.conv(x)))
        x = self.eca(x)
        #print(f"[ConvECA] After ECA: {x.shape}")
        return x
        
class ConvHAAM(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.haam = HAAM(c2)  # thay vì ECA
    def forward(self, x):
        #print(f"[ConvHAAM] Input: {x.shape}")
        x = self.act(self.bn(self.conv(x)))
        x = self.haam(x)
        #print(f"[ConvHAAM] After HAAM: {x.shape}")
        return x
        

# -------------------------
# Utils
# -------------------------
def _fit_num_heads(dim: int, desire: int) -> int:
    if desire <= 1:
        return 1
    for h in range(min(desire, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1

# -------------------------
# Coord positional embedding (light) - dtype-safe
# -------------------------
class CoordPositionalEmbedding(nn.Module):
    """
    Create 2D coordinate positional embeddings via a small MLP.
    This version is dtype/device-safe: you must pass device and dtype from input.
    Output: (B, H*W, pe_dim)
    """
    def __init__(self, pe_dim: int = 16):
        super().__init__()
        self.pe_dim = pe_dim
        self.proj = nn.Sequential(
            nn.Linear(2, pe_dim),
            nn.GELU(),
            nn.Linear(pe_dim, pe_dim)
        )

    @staticmethod
    def _make_grid(B: int, H: int, W: int, device, dtype):
        # create coords in [-1,1] with requested dtype
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        coords = torch.stack([xs, ys], dim=-1)          # [H,W,2], dtype as requested
        coords = coords.reshape(1, H * W, 2).repeat(B, 1, 1)  # [B, N, 2]
        return coords

    def forward(self, B: int, H: int, W: int, device, dtype):
        coords = self._make_grid(B, H, W, device, dtype)  # [B, N, 2], correct dtype
        pe = self.proj(coords)                             # [B, N, pe_dim]
        return pe
        
# -------------------------
# MultiheadAttention
# -------------------------
class MultiheadAttention(nn.Module):
    """
    Multihead Attention by thinhdv
    Input:
        query, key, value: (B, N, C)
    Output:
        (B, N, C)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N, C = query.shape
        dtype = query.dtype
        device = query.device

        # project Q/K/V
        Q = self.W_q(query).reshape(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)  # [B,H,N,D]
        K = self.W_k(key).reshape(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = self.W_v(value).reshape(B, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

        # attention scores
        attn_scores = (Q @ K.transpose(-2,-1)) / (self.head_dim ** 0.5)  # [B,H,N,M]
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ V  # [B,H,N,D]
        out = out.permute(0,2,1,3).reshape(B, N, C)  # [B,N,C]
        out = self.W_o(out).to(dtype=dtype, device=device)
        return out
        
class MyMHSA(nn.Module):

    """
    Multi-Head Self Attention block dùng sau C2f cuối.
    Input:  (B, C, H, W)  e.g. (B, 1024, 7, 7)
    Output: (B, c2, H, W) e.g. (B, 512, 7, 7)
    """
    def __init__(self, c1, c2, num_heads=2, dropout=0.0):
        super().__init__()
        self.c1 = c1              # input channels (từ C2f)
        self.c2 = c2              # output channels sau reduce
        self.d = c1               # embedding dim = số channel đầu vào

        # Multi-head attention custom
        self.attn = MultiheadAttention(embed_dim=self.d,
                                       num_heads=num_heads,
                                       dropout=dropout)

        # reduce sau concat (2*d -> c2)
        self.reduce = nn.Conv2d(2 * self.d, c2, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, H, W = x.size()           # (B, C, H, W)
        N = H * W                       # sequence length

        # Flatten sang (B, N, C)
        flat_x = x.view(B, C, N).permute(0, 2, 1)   # (B, N, C)

        # Self-attention
        attn_out = self.attn(flat_x, flat_x, flat_x)  # (B, N, C)

        # reshape lại (B, C, H, W)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # concat + reduce
        out = torch.cat((x, attn_out), dim=1)   # (B, 2*C, H, W)
        out = self.reduce(out)                  # (B, c2, H, W)

        return out
        

# Attention 

#Nếu dim=256 thì num_heads có thể là 1, 2, 4, 8, 16, 32.
#Nếu dim=192 thì num_heads có thể là 1, 2, 3, 6, 12, 24.
#Nếu dim=640 thì num_heads có thể là 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640.

# -------------------------
# Lightweight MLP (FFN)
# -------------------------
class FFN(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        #print(f"FFNInput: {x.shape},{x.dtype}")
        return self.net(x)

# -------------------------
# Global MHSA but keys/values are pooled (dtype-safe)
# -------------------------
class GlobalContextAttention(nn.Module):
    """
    Global attention: keys/values duoc pool => giam complexity.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = _fit_num_heads(dim, num_heads)
        if self.num_heads != num_heads:
            print(f"[GlobalContextAttention] adjust num_heads {num_heads} -> {self.num_heads} for dim={dim}")
        self.reduction = max(1, int(reduction))
        self.mha_qkv = MultiheadAttention(embed_dim=dim, num_heads=self.num_heads, dropout=dropout)

    def forward(self, x_seq, H, W):
        """
        x_seq: (B, N, C), dtype/device preserved
        """
        B, N, C = x_seq.shape
        assert N == H * W
        dtype = x_seq.dtype
        device = x_seq.device

        # Full attention
        if self.reduction == 1:
            return self.mha_qkv(x_seq, x_seq, x_seq)

        # reshape ve spatial
        feat = x_seq.transpose(1, 2).reshape(B, C, H, W).contiguous()  # [B,C,H,W]

        # pool keys/values
        kv = nn.functional.avg_pool2d(feat, kernel_size=self.reduction, stride=self.reduction)  # [B,C,H/r,W/r]

        # flatten lai sequence
        Hp, Wp = kv.shape[2], kv.shape[3]
        kv_seq = kv.reshape(B, C, Hp*Wp).permute(0, 2, 1).contiguous()  # [B,M,C]

        # MHA (Q full, K/V pooled)
        return self.mha_qkv(x_seq, kv_seq, kv_seq)

# -------------------------
# Window Local Attention (dtype-safe)
# -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = int(window_size)
        self.num_heads = _fit_num_heads(dim, num_heads)
        if self.num_heads != num_heads:
            print(f"[WindowAttention] adjust num_heads {num_heads} -> {self.num_heads} for dim={dim}")
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=self.num_heads, batch_first=True, dropout=dropout)

    def forward(self, x_seq, H, W):
        B, N, C = x_seq.shape
        assert N == H * W
        ws = max(1, min(self.window_size, H, W))

        x = x_seq.reshape(B, H, W, C)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[1], x.shape[2]
        x = x.reshape(B, Hp//ws, ws, Wp//ws, ws, C)
        x = x.permute(0,1,3,2,4,5).contiguous().reshape(-1, ws*ws, C).to(x_seq.dtype)

        out, _ = self.mha(x, x, x)
        out = out.reshape(B, Hp//ws, Wp//ws, ws, ws, C)
        out = out.permute(0,1,3,2,4,5).reshape(B, Hp, Wp, C)
        out = out[:, :H, :W, :].reshape(B, H*W, C)
        return out

# -------------------------
# Small Attention
# -------------------------
class SmallAttention(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 window_size: int = 7,
                 pe_dim: int = 16,
                 global_reduction: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pe_dim = pe_dim
        self.global_reduction = max(1, int(global_reduction))

        self.pre_proj = nn.Linear(in_channels + pe_dim, out_channels)
        self.pos_emb = CoordPositionalEmbedding(pe_dim)
        self.norm_in = nn.LayerNorm(out_channels)
        self.global_attn = GlobalContextAttention(out_channels, num_heads=num_heads, dropout=dropout, reduction=self.global_reduction)
        self.local_attn = WindowAttention(out_channels, window_size=window_size, num_heads=max(1, num_heads//2), dropout=dropout)
        self.fuse = nn.Linear(out_channels*2, out_channels)
        self.ffn = FFN(out_channels, hidden_mult=4, dropout=dropout)
        self.norm_mid = nn.LayerNorm(out_channels)
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C_in, H, W = x.shape
        dtype = x.dtype
        device = x.device
        N = H*W

        x_seq = x.flatten(2).transpose(1,2)  # B,N,C_in
        pe = self.pos_emb(B, H, W, device, dtype)
        x_cat = torch.cat([x_seq, pe], dim=-1)
        x_seq = self.pre_proj(x_cat)

        x_norm = self.norm_in(x_seq)
        g = self.global_attn(x_norm, H, W)
        l = self.local_attn(x_norm, H, W)

        fused = torch.cat([g, l], dim=-1)
        fused = self.fuse(fused)
        x_seq = x_seq + fused

        y = self.norm_mid(x_seq)
        y = self.ffn(y)
        x_seq = x_seq + y
        x_seq = self.norm_out(x_seq)

        out = x_seq.transpose(1,2).reshape(B, self.out_channels, H, W).to(dtype).to(device)
        return out

# -------------------------
# Large Attention (80x80x256) - chỉ local để tiết kiệm
# -------------------------
class LargeAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, window_size=10, pe_dim=16, dropout=0.0):
        super().__init__()
        self.pre_proj = nn.Linear(in_channels + pe_dim, out_channels)
        self.pos_emb = CoordPositionalEmbedding(pe_dim)
        self.norm_in = nn.LayerNorm(out_channels)
        self.local_attn = WindowAttention(out_channels, window_size=window_size, num_heads=num_heads, dropout=dropout)
        self.ffn = FFN(out_channels, hidden_mult=4, dropout=dropout)
        self.norm_mid = nn.LayerNorm(out_channels)
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        N = H*W

        x_seq = x.flatten(2).transpose(1,2)        # B,N,C
        pe = self.pos_emb(B, H, W, device, dtype)  # B,N,pe
        x_seq = self.pre_proj(torch.cat([x_seq, pe], dim=-1))

        x_norm = self.norm_in(x_seq)
        l = self.local_attn(x_norm, H, W)          # chỉ local

        x_seq = x_seq + l
        y = self.norm_mid(x_seq)
        y = self.ffn(y)
        x_seq = self.norm_out(x_seq + y)

        return x_seq.transpose(1,2).reshape(B, -1, H, W).to(dtype)

# -------------------------
# Medium Attention (40x40x512) - local + global
# -------------------------
class MediumAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, window_size=7, pe_dim=16, global_reduction=4, dropout=0.0):
        super().__init__()
        self.pre_proj = nn.Linear(in_channels + pe_dim, out_channels)
        self.pos_emb = CoordPositionalEmbedding(pe_dim)
        self.norm_in = nn.LayerNorm(out_channels)
        self.global_attn = GlobalContextAttention(out_channels, num_heads=num_heads, dropout=dropout, reduction=global_reduction)
        self.local_attn = WindowAttention(out_channels, window_size=window_size, num_heads=num_heads//2, dropout=dropout)
        self.fuse = nn.Linear(out_channels*2, out_channels)
        self.ffn = FFN(out_channels, hidden_mult=4, dropout=dropout)
        self.norm_mid = nn.LayerNorm(out_channels)
        self.norm_out = nn.LayerNorm(out_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device

        x_seq = x.flatten(2).transpose(1,2)
        pe = self.pos_emb(B, H, W, device, dtype)
        x_seq = self.pre_proj(torch.cat([x_seq, pe], dim=-1))

        x_norm = self.norm_in(x_seq)
        g = self.global_attn(x_norm, H, W)
        l = self.local_attn(x_norm, H, W)
        fused = self.fuse(torch.cat([g, l], dim=-1))

        x_seq = x_seq + fused
        y = self.norm_mid(x_seq)
        y = self.ffn(y)
        x_seq = self.norm_out(x_seq + y)

        return x_seq.transpose(1,2).reshape(B, -1, H, W).to(dtype)
        

