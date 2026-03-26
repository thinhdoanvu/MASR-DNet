# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
# from tests.networks.nets.test_dynunet import kernel_size
from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    #Toi se them:
    "MediumAttention",
    "LargeAttention",
    "SmallAttention",
    #"WindowAttention",
    #"GlobalContextAttention",
    #"FFN",
    #"MyMHSA",
    #"MultiheadAttention",
    #"CoordPositionalEmbedding",
    #"ConvECA",
    "ECA",
    #"SpatialAttention",
    #"ChannelAttention",
    #"DynamicDCNv2",
    #"DynamicGhost",
    #"MySRBlock",
    "MyGhost",
    "CBAM",
    "ASPP",
    "SEBlock",
    "ConvSE",
    "MyDCNv2",
    #HSDPA
    "C_Attention",
    "H_Attention",
    "W_Attention",
    "CBS",
    "Spatial_Attention",
    "ScaleDotProduct",
    "Contigous_Att",
    "CARAFE",
    "CoordAtt",
    "WAFU",
    "ROIAttentionBlock",
    "WAFU_ROIAttention",
    "MobileNetBlock"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        return y


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        #print("# Input Size: ", x.size())
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        #print(f"# Output of Aconv: {self.cv1(x).size()}")
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)#2,0 -> 4,1
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1) #3//2=1 -> 5//2=2
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=3): #default by 5
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2) # k=3
        self.cv3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2) # k=3
        self.cv4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2) # k = 3
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        # print(f"# Input SPP-ELAN: {x.shape}")
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        output = self.cv5(torch.cat(y, 1))
        # print(f"# Output SPP-ELAN: {output.shape}")
        return output


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        import torch
        from ultralytics.models.common import C2fPSA
        model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        x = torch.randn(1, 64, 128, 128)
        output = model(x)
        print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        import torch
        from ultralytics import SCDown
        model = SCDown(c1=64, c2=128, k=3, s=2)
        x = torch.randn(1, 64, 128, 128)
        y = model(x)
        print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
    #     from flash_attn.flash_attn_interface import flash_attn_func
    #     USE_FLASH_ATTN = True
    # else:
    #     from torch.nn.functional import scaled_dot_product_attention as sdpa
    #     logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    # logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        import torch
        from ultralytics.nn.modules import AAttn
        model = AAttn(dim=64, num_heads=2, area=4)
        x = torch.randn(2, 64, 128, 128)
        output = model(x)
        print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True): # Tí nữa chạy xong nhớ trả lại 0.5
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


# Cua toi ne 
import torch
import torch.nn as nn
import math
from math import ceil, gcd
from torchvision.ops import DeformConv2d
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
        # print(f"MyGhost input: {x.shape}")
        x1 = self.primary_conv(x)   # feature gốc
        x2 = self.cheap_operation(x1)  # feature sinh thêm
        out = torch.cat([x1, x2], dim=1)  # concat lại
        # print(f"MyGhost output: {out.shape}")
        return out[:, :self.out_channels, :, :]  # đảm bảo đúng số kênh out

class MyDCNv2(nn.Module):
    """
    Custom Deformable Conv2d (v2) tương thích mọi torchvision.
    - Dùng grayscale hoặc RGB
    - Kernel size cố định 3
    - Offset + mask modulation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MyDCNv2, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Conv để sinh offset + mask
        # Output channels = 2*kernel^2 + kernel^2 = 3*kernel^2
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding= kernel_size//2,#padding,
            bias=True
        )
        nn.init.constant_(self.conv_offset_mask.weight, 0.)
        nn.init.constant_(self.conv_offset_mask.bias, 0.)

        # DeformConv2d chuẩn
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size//2, bias=False #padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(f"MyDCNv2 input: {x.shape}")
        # Sinh offset + mask
        out = self.conv_offset_mask(x)
        # Chia thành offset_x, offset_y, mask
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        offset = offset * 0.5 # hoặc 0.25 tránh kernel nhảy quá xa
        mask = torch.sigmoid(mask)

        # Một số phiên bản torchvision chưa hỗ trợ mask modulation
        # => bỏ mask nếu không cần thiết
        try:
            y = self.deform_conv(x, offset, mask)
        except TypeError:
            # fallback: chỉ dùng offset
            y = self.deform_conv(x, offset)

        output = self.act(self.bn(y))
        # print(f"MyDCNv2 output: {output.shape}")
        return output

class MySRBlock(nn.Module):
    def __init__(self, c1, c2):  # c1: input, c2: output
        super().__init__()
        # 1. Nhánh nâng cấp độ phân giải (tương đương SR)
        # Giả sử chúng ta muốn làm nét đặc trưng lên gấp đôi rồi nén lại
        mid_c = c1 // 2
        self.conv1 = nn.Conv2d(c1, mid_c * 4, 3, padding=1, groups=mid_c)  # Chuẩn bị cho PixelShuffle
        self.pixel_shuffle = nn.PixelShuffle(2)  # Tăng kích thước H, W lên gấp 2

        # 2. Pixel Attention: Giúp mô hình biết pixel nào là cạnh của van tim
        self.pa = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, 1),
            nn.Sigmoid()
        )

        # 3. Downsample lại để khớp với kích thước của Head nhưng giữ được chi tiết đã khôi phục
        self.down = nn.Conv2d(mid_c, c2, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        # Nâng cấp đặc trưng lên không gian phân giải cao
        feat_high = self.pixel_shuffle(self.conv1(x))

        # Áp dụng Pixel Attention để làm sắc nét vùng quan trọng
        feat_high = feat_high * self.pa(feat_high)

        # Đưa về kích thước gốc nhưng với thông tin đã được "tái cấu trúc"
        return self.act(self.bn(self.down(feat_high)))

class DynamicGhost(nn.Module):
    """
    Dynamic Ghost Convolution
    - Kết hợp giữa real conv (primary_conv) và cheap operation (depthwise conv)
    - Tỉ lệ real/cheap channels được tính động dựa trên ratio
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ratio=0.25, act=True):
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
            nn.Conv2d(in_channels, self.init_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

        # Cheap operation (depthwise conv)
        if self.new_channels > 0:
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.init_channels, self.new_channels, kernel_size=3, stride=1, padding=1, groups=self.init_channels, bias=False),
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
    - Gating alpha in [0,1] từ input (GAP -> MLP nhỏ)
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
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, Channel_num):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = 0.6  # separation rate
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    def get_im_subim_channels(self, C_im, M):
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature

class ECA(nn.Module):
    def __init__(self, in_channel, out_channel=None, k_size=3):
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
        #x = self.act(self.bn(self.conv(x)))
        x = self.conv(x)
        x = self.eca(x)
        #print(f"[ConvECA] After ECA: {x.shape}")
        return x
        
class ConvSE(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        """
        c1: in_channels
        c2: out_channels
        reduction: reduction ratio trong SE block
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.se = SEBlock(c2, reduction)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.se(x)
        return x
# ---------------------- SE Block ----------------------
class SEBlock(nn.Module):
    def __init__(self, c1, c2=None, reduction=16):
        super(SEBlock, self).__init__()
        if c2 is None:
            c2 = c1
        self.in_channels = c1
        self.out_channels = c2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

        # Nếu c1 != c2 thì thêm conv để match kênh
        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_channels, 1, 1)
        return self.proj(x) * y

# ---------------------- CBAM ----------------------
class CBAM(nn.Module):
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        if c2 is None:
            c2 = c1
        self.in_channels = c1
        self.out_channels = c2

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x_proj = self.proj(x)

        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x_proj * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

# ---------------------- ASPP ----------------------
class ASPP(nn.Module):
    def __init__(self, c1, c2, dilation_rates=[1, 6, 12]):
        super().__init__()
        mid_c = c2 // 2
        self.cv1 = nn.Conv2d(c1, mid_c, 1, 1)

        # Nhánh 1: Global Context (Global Avg Pool)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cv_global = nn.Conv2d(mid_c, mid_c, 1, 1)

        # Nhánh 2: Strip Pooling (Cực tốt cho vật thể có hình vòng cung như van tim)
        self.strip_h = nn.AdaptiveAvgPool2d((None, 1))
        self.strip_w = nn.AdaptiveAvgPool2d((1, None))
        self.cv_strip = nn.Conv2d(mid_c, mid_c, 1, 1)

        # Nhánh 3: Multi-scale Atrous Conv với Dilation thấp để tránh Grid Effect
        self.atrous_blocks = nn.ModuleList([
            nn.Conv2d(mid_c, mid_c, 3, padding=d, dilation=d, groups=mid_c)
            for d in dilation_rates
        ])

        # Alpha để mô hình tự học trọng số cho từng tỷ lệ Dilation
        self.alpha = nn.Parameter(torch.ones(len(dilation_rates) + 2))

        self.cv2 = nn.Conv2d(mid_c * (len(dilation_rates) + 2), c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.cv1(x)
        weights = torch.softmax(self.alpha, dim=0)

        results = []

        # 1. Xử lý Global
        g = self.global_pool(x)
        results.append(self.cv_global(g).expand_as(x) * weights[0])

        # 2. Xử lý Strip (Quét ngang/dọc)
        s = self.strip_h(x) * self.strip_w(x)
        results.append(self.cv_strip(s) * weights[1])

        # 3. Xử lý Atrous (Đa quy mô)
        for i, block in enumerate(self.atrous_blocks):
            results.append(block(x) * weights[i + 2])

        # Concat và nén
        out = torch.cat(results, dim=1)
        return self.act(self.bn(self.cv2(out)))

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

class MyMHSA(nn.Module):

    """
    Multi-Head Self Attention block dùng sau C2f cuối.
    Input:  (B, C, H, W)  e.g. (B, 1024, 7, 7)
    Output: (B, c2, H, W) e.g. (B, 512, 7, 7)
    """
    def __init__(self, c1, c2, num_heads=1, dropout=0.0):
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
class C_Attention(nn.Module):
    """Constructs a Channel, Height, and Weight Attention module.
    Input: Batch size x Channel x Height x Weight: 1x96x20x20
    Args:
        kernel size - k: Adaptive selection of kernel size
        output: 1x96x1x1
    """
    def __init__(self, c2, k=3): # mac du khong dung c2 nhung ghi cho thong nhat ECA, G_A, W_A
        super(C_Attention, self).__init__()
        # For Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False) # [batch_size, channels, length]
        self.silu = nn.SiLU() # SiLU
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1x96x20x20
        y = self.avg_pool(x)    # y: 1x96x1x1

        # squeeze(-1): remove last dimension: 1x96x1
        # transpose(-1, -2): swap 2 last dimensions: 1x1x96
        # conv1d(1,1,3,1): kernel size=3 > size(h,w) cua 1x96x1, tuc h=w=1

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.silu(y)
        y = self.sigmoid(y)
        return y    # B x C x 1 x 1
    
class H_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 20 x 1
    """
    def __init__(self, c1, c2, k=3):
        super(H_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))  # Output size: [H, 1]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU() # SiLU
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, H, 1] 1 x 1024 x 20 x 1
        y = y.squeeze(-1)     # y: [B, C, H]: 1 x 1024 x 20
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, H'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        y = y.unsqueeze(-1)     # [B, C, H', 1] 1 x 2048 x 20 x 1
        
        return y  # Output shape: [B, C, H', 1]: 1 x 2048 x 20 x 1

class W_Attention(nn.Module):
    """Constructs a Height Attention module.
    Input: Batch size x Channel x Height x Width: 1x96x20x20
    Args:
        c2: Number of channels from the previous layer
        kernel_size - k: Adaptive selection of kernel size
        Output: 1 x 2048 x 1 x 20
    """
    def __init__(self, c1, c2, k=3):
        super(W_Attention, self).__init__()
        self.kernel_size = k
        
        # For Height Attention
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Output size: [1, W]
        self.conv = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=k, padding=(k - 1) // 2, bias=False)  # Convolution to maintain the number of channels
        self.silu = nn.SiLU() # SiLU
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input-x example: 1 x 1024 x 20 x 20
        B, C, H, W = x.shape
        
        # Adaptive Average Pooling to reduce width to 1
        y = self.avg_pool(x)  # y: [B, C, 1, W] 1 x 1024 x 1 x 20
        y = y.squeeze(-2)     # y: [B, C, W]: 1 x 1024 x 20
        
        # # Multi-scale information fusion
        y = self.conv(y)  # [B, C, W'] 1 x 2048 x 20
        y = self.silu(y)
        y = self.sigmoid(y)
        
        # Restore the shape [B, C, 1, W]
        y = y.unsqueeze(-2)     # [B, C, 1, W'] 1 x 2048 x 1 x 20
        
        return y  # Output shape: [B, C, 1, W']: 1 x 2048 x 1 x 20

class CBS(nn.Module):
    default_act = nn.SiLU()  # SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        # Gán padding nếu chưa có
        p = k // 2 if p is None else p
        self.p = p  # ✅ thêm dòng này để tránh lỗi
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        # print("# Input CBS: ", x.size())
        output = self.act(self.bn(self.conv(x)))
        # print("# Output CBS: ", output.size())
        return output

class ScaleDotProduct(nn.Module):    # Input CxHxW
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)
        self.ca = C_Attention(c2)
        
        
    def forward(self, x):
        # print("# Input Size: ", x.size())                   # B x C1 x H x W
        h_att = self.ha(x)                                  
        # print("# Height Output Size: ", h_att.size())       # B x C1 x H x 1
        w_att = self.wa(x)                                  
        # print("# Weight Output Size: ", w_att.size())       # B x C1 x 1 x W
        c_att = self.ca(x)                                  
        # print("# Channel Output Size: ", c_att.size())      # B x C1 x 1 x 1
        
        # matmul (H,W)
        BQ,CQ,HQ,WQ = h_att.size()
        Q = h_att.view(BQ,CQ,HQ*WQ)
        # print("# Q Output Size: ", Q.size())                # B x C1 x H*W
        
        # Interpolate w_att to have the same height and width as h_att
        w_att = nn.functional.interpolate(w_att, size=(h_att.size(2), h_att.size(3)), mode='bilinear', align_corners=False)
        
        
        # BK,CK,HK,WK = w_att.size()
        # K = w_att.view(BK,CK,HK*WK)
        K = w_att.view(BQ, CQ, HQ * WQ)
        # print("# K Output Size: ", K.size())                # B x C1 x H*W
        
        # Compute the attention scores by performing matrix multiplication of Q and K
        K_transpose = K.transpose(1,2)
        # print("# K_transpose Output Size: ", K_transpose.size())    # B x H*W x C1
        
        scores = torch.bmm(Q, K_transpose)
        # print("# scores Output Size: ", scores.size())              # B x C1 x C1
        
        # Scale the scores
        d_k = Q.size(-1)  # This is the depth (H*W)
        scores_scaled = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # Scale by sqrt(d_k)
        # print("# Scaled scores Output Size: ", scores_scaled.size())     # B x C1 x C1
        
        # Apply softmax to get the attention weights
        attention_weights = F.softmax(scores_scaled, dim=-1)  # Shape: [Batch, Channels, Channels]
        # print("# Attention weights Output Size: ", attention_weights.size())     # B x C1 x C1

        # Multiply the attention weights with the value vector V
        BV,CV,HV,WV = c_att.size()
        V = c_att.view(BV,CV,HV*WV)
        
        output = torch.bmm(attention_weights, V)    # Shape: [Batch, Channels, 1]
        # print("# Output Size: ", output.size())     # B x C1 x C1

        # Reshape output to match the original spatial dimensions
        output = output.view(BV, CV, 1, 1)
        # print("# Output Reshape Size: ", output.size())     # B x C1 x 1 x 1
        
        # trans from into input shape
        output = x * output
        # print("# Final Reshape Size: ", output.size())     # B x C1 x H x W
        
        return output


class GIN_Spatial_Attention(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)

        # Các lớp hạ mẫu đa quy mô
        self.mp3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=3//2)
        self.mp7 = nn.AvgPool2d(kernel_size=7, stride=2, padding=7//2)
        self.mp13 = nn.AvgPool2d(kernel_size=13, stride=2, padding=13//2)

        # Nhánh skip-connection để giữ thông tin gốc sau khi hạ mẫu
        self.mp_skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # Tổng cộng: 3 nhánh pooling (3*c1) + 1 nhánh skip (c1) = 4*c1
        self.conv = nn.Conv2d(c1 * 4, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()

    def forward(self, x):
        # Bước 1: Nhánh Attention (Tinh lọc thông tin không gian)
        # Tìm xem vật thể nằm ở đâu theo trục dọc và ngang
        h_att = self.ha(x)
        w_att = self.wa(x)

        # Áp dụng trọng số attention vào input gốc
        # Bây giờ x_focused sẽ chứa các vùng đặc trưng được nhấn mạnh
        x_focused = x * h_att * w_att

        # Bước 2: Hạ mẫu đa quy mô từ vùng đã được chú ý (Focused Features)
        mpl3 = self.mp3(x_focused)
        mpl7 = self.mp7(x_focused)
        mpl13 = self.mp13(x_focused)

        # Bước 3: Nhánh Skip-connection (Giữ lại đặc trưng gốc để tránh mất mát)
        skip = self.mp_skip(x)

        # Bước 4: Kết hợp tất cả lại
        # Sự kết hợp giữa đặc trưng "đã lọc" và đặc trưng "gốc" giúp mô hình ổn định hơn
        cat = torch.cat((mpl3, mpl7, mpl13, skip), dim=1)

        output = self.conv(cat)
        output = self.bn(output)
        output = self.sl(output)

        return output


# Hierarchical Spatial Attention

class Spatial_Attention(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # --- 1. Nhánh Coordinate Attention (X-Y Localization) ---
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // 16)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

        # --- 2. Nhánh Multi-Scale (Dilated Conv) ---
        ##############################################################################################
        # 3 x 5, 7, 9, 11, 15: dilation = 2, 3, 4, 5, 7
        ##############################################################################################
        # Kernel size = 3
        # self.d3 = nn.Conv2d(c1, c1, 3, padding=1, groups=c1)

        # Nhánh 2: Kernel 3, Dilation 3
        # Padding tính theo công thức: Dilation * (K - 1) / 2
        # self.d7 = nn.Conv2d(c1, c1, 3, padding=2, dilation=2, groups=c1)

        ##############################################################################################
        # 5 x 9, 13: dilation = 2, 3
        ##############################################################################################
        # Thay Kernel 3 thành 5
        # Padding = (5 - 1) / 2 = 2
        self.d3 = nn.Conv2d(c1, c1, 5, padding=2, groups=c1)

        # Nhánh 2: Kernel 5, Dilation 3
        # Padding tính theo công thức: Dilation * (K - 1) / 2
        # Padding = 3 * (5 - 1) / 2 = 6
        self.d7 = nn.Conv2d(c1, c1, 5, padding=4, dilation=2, groups=c1)

        ##############################################################################################
        # 7 x 13: dilation = 2
        ##############################################################################################

        # self.d3 = nn.Conv2d(c1, c1, 7, padding=3, groups=c1)  # padding = (7 - 1) / 2 = 3
        # self.d7 = nn.Conv2d(c1, c1, 7, padding=6, dilation=2, groups=c1)  # padding = 2 * (7 - 1) / 2 = 6


        # 3,2={3,5}=67.7%, 3,3={3,7}=66.0%, 3,4={3,9}=65.5%, 3,5={3,11}=66.1%, 3,7={3,15}=%
        # 5,2={5,9}=, 5,3 = {5,13}
        # 7,2 = {7,13}

        # kernel_size, dilation.
        # 3,2={3,5}=67.7% (95.4), 3,3={3,7}=66.0%, 3,4={3,9}=65.5%, 3,5={3,11}=66.1%, 3,7={3,15}=66.6%
        # 5,2={5,9}=66.8% (96.4), 5,3={5,13}=66.0%
        # 7,2={7,13}=%

        # Alpha tự học cho 3 nhánh (Dùng trước khi Concat)
        self.alpha = nn.Parameter(torch.ones(3))

        # --- 3. Concat Fusion Layer ---
        # Nén thông tin từ 3 nhánh (c1 * 3) về c2 (kích thước đầu ra mong muốn)
        self.fusion_conv = nn.Conv2d(c1 * 3, c2, kernel_size=1, stride=1, padding=0)
        self.bn_out = nn.BatchNorm2d(c2)
        self.act_out = nn.SiLU()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Bước 1: Coordinate Attention
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        x_coord = x * a_h * a_w

        # Bước 2: Trích xuất đặc trưng đa quy mô
        w_scaled = torch.softmax(self.alpha, dim=0)

        feat3 = self.d3(x_coord) * w_scaled[0]
        feat7 = self.d7(x_coord) * w_scaled[1]
        feat_id = identity * w_scaled[2]

        # Bước 3: Concat Fusion (Chìa khóa để tăng mAP@50-95)
        # Kết hợp tất cả lại để lớp fusion_conv tự học cách chọn lọc pixel
        out = torch.cat([feat3, feat7, feat_id], dim=1)

        return self.act_out(self.bn_out(self.fusion_conv(out)))

    @torch.no_grad()
    def get_branch_weights(self):
        w = torch.softmax(self.alpha, dim=0)
        return {
            "3x3": w[0].item(),
            "7x7": w[1].item(),
            "skip": w[2].item()
        }

class v3Spatial_Attention(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Giữ nguyên các lớp Attention dọc/ngang của bạn
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)

        # Các lớp hạ mẫu đa quy mô
        self.mp3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.mp7 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)
        self.mp13 = nn.AvgPool2d(kernel_size=13, stride=2, padding=6)

        # Nhánh skip-connection
        self.mp_skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # --- ĐIỂM MỚI: Trọng số tự học cho 4 nhánh ---
        # Khởi tạo bằng 1 để ban đầu mỗi nhánh có sức mạnh ngang nhau (0.25)
        self.alpha = nn.Parameter(torch.ones(4))

        # Tổng cộng: 4 nhánh kết hợp
        self.conv = nn.Conv2d(c1 * 4, c2, 1, 1)
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU()

    def forward(self, x):
        # Bước 1: Attention lọc không gian
        h_att = self.ha(x)
        w_att = self.wa(x)

        # Nhấn mạnh vùng đặc trưng quan trọng
        x_focused = x * h_att * w_att

        # Bước 2: Hạ mẫu đa quy mô
        mpl3 = self.mp3(x_focused)
        mpl7 = self.mp7(x_focused)
        mpl13 = self.mp13(x_focused)

        # Bước 3: Nhánh Skip
        skip = self.mp_skip(x)

        # --- ĐIỂM MỚI: Cơ chế Weighted Fusion ---
        # Dùng Softmax để đảm bảo tổng trọng số của 4 nhánh = 1 (100%)
        w = torch.softmax(self.alpha, dim=0)

        # Nhân trọng số học được vào từng nhánh trước khi nối lại
        # Điều này giúp mô hình "ưu tiên" nhánh tốt nhất
        cat = torch.cat((
            mpl3 * w[0],
            mpl7 * w[1],
            mpl13 * w[2],
            skip * w[3]
        ), dim=1)

        # Bước 4: Kết hợp và chuyển đổi kênh
        output = self.conv(cat)
        output = self.bn(output)
        output = self.sl(output)

        return output

    # Hàm hỗ trợ để bạn theo dõi mô hình đang "nghĩ" gì
    @torch.no_grad()
    def get_branch_weights(self):
        w = torch.softmax(self.alpha, dim=0)
        return {
            "3x3": w[0].item(),
            "7x7": w[1].item(),
            "13x13": w[2].item(),
            "skip": w[3].item()
        }

class NewSpatial_Attention(nn.Module):
    """
    Input : B x C x H x W
    Output: B x c2 x H' x W'
    """

    def __init__(
        self,
        c1,
        c2,
        kernel_sizes=(3, 5, 7, 9, 11, 13),
        topk=3
    ):
        super().__init__()

        # -----------------------------
        # Height & Width Attention
        # -----------------------------
        self.ha = H_Attention(c1, c1)
        self.wa = W_Attention(c1, c1)

        # -----------------------------
        # Multi-scale Avg Pooling
        # -----------------------------
        self.kernel_sizes = kernel_sizes
        self.topk = topk

        self.pools = nn.ModuleList([
            nn.AvgPool2d(kernel_size=k, stride=2, padding=k // 2)
            for k in kernel_sizes
        ])

        # -----------------------------
        # Learnable kernel importance
        # -----------------------------
        self.alpha = nn.Parameter(torch.ones(len(kernel_sizes)))

        # Mask kernel đang active (không có gradient)
        self.register_buffer(
            "active_mask",
            torch.ones(len(kernel_sizes), dtype=torch.bool)
        )

        # -----------------------------
        # Projection
        # -----------------------------
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    # ==================================================
    # Forward
    # ==================================================
    def forward(self, x):
        """
        x: B x C x H x W
        """

        # Height & Width attention
        h_att = self.ha(x)             # B x C x H x 1
        w_att = self.wa(x)             # B x C x 1 x W
        x_hw = x * h_att * w_att       # B x C x H x W

        # Pooling
        pooled_feats = [pool(x_hw) for pool in self.pools]

        # Softmax normalize alpha
        weights = F.softmax(self.alpha, dim=0)

        out = None
        base_size = None

        for i, (w, feat) in enumerate(zip(weights, pooled_feats)):
            if not self.active_mask[i]:
                continue

            if base_size is None:
                base_size = feat.shape[2:]

            feat = F.interpolate(
                feat,
                size=base_size,
                mode="nearest"
            )

            out = w * feat if out is None else out + w * feat

        # Projection
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)

        return out

    # ==================================================
    # Update top-k kernels (call AFTER each epoch)
    # ==================================================
    @torch.no_grad()
    def set_top_k(self, k=None):
        """
        Set top-k kernels to be active. If k is None, activate all kernels.
        """
        if k is None:
            # Activate all kernels (warm-up)
            self.active_mask.fill_(True)
        else:
            weights = torch.softmax(self.alpha, dim=0)
            topk_idx = torch.topk(weights, k=k).indices
            new_mask = torch.zeros_like(self.active_mask)
            new_mask[topk_idx] = True
            self.active_mask.copy_(new_mask)

    # ==================================================
    # Utils (optional – logging)
    # ==================================================
    def get_active_kernels(self):
        """
        Return list of active kernel sizes
        """
        return [
            k for k, m in zip(self.kernel_sizes, self.active_mask.tolist()) if m
        ]

    def get_kernel_weights(self):
        """
        Return softmax(alpha)
        """
        return torch.softmax(self.alpha, dim=0).detach().cpu()

class Contigous_Att(nn.Module):    # Input CxHxW
    
    def __init__(self, c1, c2):
        super().__init__()
        self.sdp = ScaleDotProduct(c1, c2)
        self.conv = nn.Conv2d(c1*5, c2, 1, 1)   # after concate 4 maxpoling
        self.bn = nn.BatchNorm2d(c2)
        self.sl = nn.SiLU() # SiLU
        self.se = SEBlock(c2)
        
    def forward(self, x):
        # print("# Input Contigous_Att: ", x.size())                   # B x C1 x H x W
        y1 = self.sdp(x)                                  
        # print("# y1 Output Size: ", y1.size())              # B x C1 x H x W
        y2 = self.sdp(y1)
        # print("# y2 Output Size: ", y2.size())              # B x C1 x H x W
        y3 = self.sdp(y2)
        # print("# y3 Output Size: ", y3.size())              # B x C1 x H x W
        y4 = self.sdp(y3)
        # print("# y4 Output Size: ", y4.size())              # B x C1 x H x W
        output = torch.cat((y1,y2,y3,y4,x),dim=1)
        # print("# Output Size: ", output.size())             # B x C1*5 x H x W
        output = self.conv(output)
        output = self.sl(self.bn(output))
        # output = self.se(output)
        # print("# Output Contigous_Att: ", output.size())        # B x C1*5 x H x W
        return output
        
class MyCBAM(nn.Module): # Viet lai de sau nay dung cac module cho thong nhat
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super(MyCBAM, self).__init__()
        if c2 is None:
            c2 = c1
        self.in_channels = c1
        self.out_channels = c2

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c2, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

        self.proj = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

    def forward(self, x):
        x_proj = self.proj(x)

        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x = x_proj * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

class CARAFE(nn.Module):
    
    #CARAFE: Content-Aware ReAssembly of FEatures       https://arxiv.org/pdf/1905.02188.pdf
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        # print("# Input Carafe: ", x.size())
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("# Output Carafe:",out_tensor.size())
        return out_tensor
        
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class CoordAtt(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out
        
class ConvCA(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.ca = CoordAtt(c1, c2, reduction=32)

    def forward(self, x):
        #print(f"[ConvCA] Input: {x.shape}")
        x = self.conv(x)
        x = self.ca(x)
        #print(f"[ConvCA] After CA: {x.shape}")
        return x

# ===========================
# Gamma
# ===========================
class GammaCorrection(nn.Module):
    def __init__(self, gamma=1.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, x):
        return torch.clamp(x ** self.gamma, 0, 1)

# ===========================
# Sobel
# ===========================
class SobelFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.channels = channels

    def forward(self, x):
        C = x.shape[1]
        kx = self.kx.expand(C, 1, 3, 3).to(x.device)
        ky = self.ky.expand(C, 1, 3, 3).to(x.device)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return torch.sqrt(gx*gx + gy*gy + 1e-6)

# ===========================
# Laplacian Fuse
# ===========================
class LaplacianFuse(nn.Module):
    def __init__(self, mode="max"):
        super().__init__()
        self.mode = mode

    def fuse(self, a, b):
        return torch.max(a, b) if self.mode == "max" else (a + b)*0.5

    def forward(self, x1, x2):
        low1 = F.avg_pool2d(x1, 2)
        low2 = F.avg_pool2d(x2, 2)

        up1 = F.interpolate(low1, x1.shape[2:])
        up2 = F.interpolate(low2, x2.shape[2:])

        lap1 = x1 - up1
        lap2 = x2 - up2

        fused_low = self.fuse(low1, low2)
        fused_lap = self.fuse(lap1, lap2)

        up_low = F.interpolate(fused_low, x1.shape[2:])
        return up_low + fused_lap

# ===========================
# WAFU BLOCK (residual)
# ===========================
class WAFU(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, gamma=1.2, fuse_mode="max"):
        super().__init__()

        self.stride = stride

        # Projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Residual path
        if in_channels != out_channels or stride != 1:
            self.res_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channels)
        else:
            self.res_proj = None

        # Enhancement modules
        self.gamma = GammaCorrection(gamma)
        self.sobel = SobelFilter(out_channels)
        self.fuse  = LaplacianFuse(fuse_mode)

    def forward(self, x):
        # print(f"# WAFU Input: {x.shape}")
        # ---- projection ----
        h = self.proj(x)
        h = self.bn(h)

        # ---- normalize to 0..1 ----
        minv = h.amin(dim=(2,3), keepdim=True)
        maxv = h.amax(dim=(2,3), keepdim=True)
        h_norm = (h - minv) / (maxv - minv + 1e-6)

        # ---- branches ----
        x1 = self.gamma(h_norm)
        x2 = self.sobel(h_norm)
        x2 = x2 / (x2.amax(dim=(2,3), keepdim=True) + 1e-6)

        # ---- fuse ----
        fused = self.fuse(x1, x2)

        # ---- denormalize ----
        fused = fused * (maxv - minv) + minv

        # ---- residual ----
        if self.res_proj is None:
            res = x
        else:
            res = self.res_bn(self.res_proj(x))

        out = fused + res
        # print(f"# WAFU Output: {out.shape}")
        return out

from torchvision.ops import roi_align

class ROIAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(14,14), hidden_dim=128):
        super().__init__()
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        # Channel attention (SE style)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels * output_size[0] * output_size[1], 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_channels)
        )
        self.output_size = output_size
        self.rois = None

    def set_rois(self, rois):
        self.rois = rois

    def forward(self, feature_map):
        if self.rois is None:
            return feature_map
        roi_features = roi_align(feature_map, self.rois, output_size=self.output_size)

        # Spatial + channel attention
        spatial_mask = self.attention(roi_features)
        channel_mask = self.channel_attn(roi_features)
        roi_features = roi_features * spatial_mask * channel_mask + roi_features

        roi_features = roi_features.view(roi_features.size(0), -1)
        return self.fc(roi_features)

class WAFU_ROIAttention(nn.Module):
    def __init__(self, in_channels, out_channels, output_size=(7,7), hidden_dim=256, center_bias=True, sigma=0.125):
        super().__init__()
        self.wafu = WAFU(in_channels, out_channels)
        self.roi_attn = ROIAttentionBlock(out_channels, out_channels, output_size, hidden_dim)
        self._cached_rois = None
        self.center_bias = center_bias
        self.sigma = sigma

    def set_rois(self, rois):
        self._cached_rois = rois
        self.roi_attn.set_rois(rois)

    def generate_roi_mask(self, rois, shape):
        mask = torch.zeros((shape[0], 1, shape[2], shape[3]), dtype=torch.float32, device=rois.device)
        for roi in rois:
            b, x1, y1, x2, y2 = roi.int()
            mask[b, 0, y1:y2, x1:x2] = 1.0
        return mask

    def forward(self, feature_map):
        x_wafu = self.wafu(feature_map)

        if self._cached_rois is None:
            return x_wafu

        roi_mask = self.generate_roi_mask(self._cached_rois, x_wafu.shape)

        # Tích hợp center bias
        if self.center_bias:
            bias_mask = center_bias_mask(x_wafu.shape, sigma=self.sigma).to(x_wafu.device)
            roi_mask = roi_mask * bias_mask  # ưu tiên vùng giữa trong ROI

        x_masked = x_wafu * roi_mask

        out = self.roi_attn(x_masked)
        return out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        # 1x1 pointwise conv (expand)
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        # 3x3 depthwise conv
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))

        # 1x1 pointwise conv (project)
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
