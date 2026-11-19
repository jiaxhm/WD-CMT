# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import math
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

# === Mamba & Wavelets Imports ===
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    # Pure-PyTorch equivalent of mamba_ssm.modules.mamba_simple.Mamba.  This
    # keeps previously trained checkpoints usable on machines without the
    # optional mamba_ssm CUDA package (albeit more slowly).
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=8, d_conv=4, expand=2):
            super().__init__()
            self.d_inner = d_model * expand
            self.d_state = d_state
            self.dt_rank = math.ceil(d_model / 16)
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                self.d_inner, self.d_inner, kernel_size=d_conv,
                padding=d_conv - 1, groups=self.d_inner, bias=True,
            )
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(self.d_inner, 1).float()))
            self.D = nn.Parameter(torch.ones(self.d_inner))
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        def forward(self, x):
            batch, length, _ = x.shape
            xz = self.in_proj(x)
            u, z = xz.chunk(2, dim=-1)
            u = F.silu(self.conv1d(u.transpose(1, 2))[..., :length])
            x_dbl = self.x_proj(u.transpose(1, 2))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            delta = F.softplus(self.dt_proj(dt).transpose(1, 2))
            A = -torch.exp(self.A_log.float())
            state = u.new_zeros(batch, self.d_inner, self.d_state)
            outputs = []
            for index in range(length):
                delta_i = delta[:, :, index]
                u_i = u[:, :, index]
                delta_A = torch.exp(delta_i.unsqueeze(-1) * A)
                delta_B_u = delta_i.unsqueeze(-1) * B[:, index].unsqueeze(1) * u_i.unsqueeze(-1)
                state = delta_A * state + delta_B_u
                outputs.append((state * C[:, index].unsqueeze(1)).sum(dim=-1) + self.D * u_i)
            y = torch.stack(outputs, dim=-1).transpose(1, 2)
            return self.out_proj(y * F.silu(z))

_CSMS6S_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "csms6s.py")
_CSMS6S_SPEC = importlib.util.spec_from_file_location("_local_csms6s", _CSMS6S_PATH)
_CSMS6S = importlib.util.module_from_spec(_CSMS6S_SPEC)
_CSMS6S_SPEC.loader.exec_module(_CSMS6S)
SelectiveScanMamba = _CSMS6S.SelectiveScanMamba
SelectiveScanCore = _CSMS6S.SelectiveScanCore
SelectiveScanOflex = _CSMS6S.SelectiveScanOflex
_HAS_SELECTIVE_SCAN = hasattr(_CSMS6S, "selective_scan_cuda")
from pytorch_wavelets import DWTForward, DWTInverse
try:
    from 小波绘图 import save_wavelet_visualizations
except ModuleNotFoundError:
    def save_wavelet_visualizations(*args, **kwargs):
        raise RuntimeError(
            "Wavelet visualization was requested, but the optional module "
            "'小波绘图' is not available."
        )

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class SE_3D_Attention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class LL_Context_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.dw3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, groups=out_channels, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        base = self.proj(x)
        local = self.mix(torch.cat([self.dw3(base), self.dw5(base)], dim=1))
        return base + local


class OverlapPatchCrossGatedDenoiser(nn.Module):
    def __init__(self, channels, dirs=3, patch_size=2, stride=1):
        super().__init__()
        assert channels % 2 == 0, "channels 必须为偶数，才能沿通道均分为双分支"
        assert dirs == 3, "当前实现默认三个高频方向：LH/HL/HH"
        assert patch_size > 0 and stride > 0
        self.channels = channels
        self.dirs = dirs
        self.patch_size = patch_size
        self.stride = stride
        self.branch_channels = channels // 2
        self.subband_linear = nn.Sequential(
            nn.Linear(channels * (dirs + 1), channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        patch_dim = channels * patch_size * patch_size
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        def make_branch():
            return nn.Sequential(
                nn.Conv2d(
                    self.branch_channels,
                    self.branch_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, self.branch_channels),
                nn.GELU(),
            )
        self.branch1 = make_branch()
        self.branch2 = make_branch()
        def make_gate():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(
                    self.branch_channels,
                    self.branch_channels,
                    kernel_size=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
        self.gate1 = make_gate()
        self.gate2 = make_gate()
        self.fused_norm = nn.GroupNorm(1, channels)
        self.patch_decoder = nn.Linear(
            channels,
            dirs * channels * patch_size * patch_size,
        )
        self.residual_scale = nn.Parameter(
            torch.full((1, 1, dirs, 1, 1), -2.0)
        )
    def forward(self, low, high):
        BT, C, Dirs, H, W = high.shape
        if low.shape != (BT, C, H, W):
            raise ValueError(f"low 与 high 尺寸不匹配：low={tuple(low.shape)}, high={tuple(high.shape)}")
        if C != self.channels:
            raise ValueError(f"期望通道数 {self.channels}，实际得到 {C}")
        if Dirs != self.dirs:
            raise ValueError(f"期望高频方向数 {self.dirs}，实际得到 {Dirs}")
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(f"空间尺寸 {(H, W)} 小于 patch_size={self.patch_size}")
        high_by_dir = high.permute(0, 2, 1, 3, 4).contiguous()
        high_by_dir = high_by_dir.view(BT, Dirs * C, H, W)
        subbands = torch.cat([low, high_by_dir], dim=1)  # [BT,4C,H,W]
        fused_map = subbands.permute(0, 2, 3, 1).contiguous()  # [BT,H,W,4C]
        fused_map = self.subband_linear(fused_map)  # [BT,H,W,C]
        fused_map = fused_map.permute(0, 3, 1, 2).contiguous()  # [BT,C,H,W]
        patches = F.unfold(
            fused_map,
            kernel_size=self.patch_size,
            stride=self.stride,
        ).transpose(1, 2)  # [BT,M,C*p*p]
        grid_h = (H - self.patch_size) // self.stride + 1
        grid_w = (W - self.patch_size) // self.stride + 1
        num_patches = grid_h * grid_w
        if patches.shape[1] != num_patches:
            raise RuntimeError(f"Patch 数量异常：计算得到 {num_patches}，实际为 {patches.shape[1]}")
        tokens = self.patch_encoder(patches)  # [BT,M,C]
        token_grid = tokens.transpose(1, 2).reshape(BT, C, grid_h, grid_w)  # [BT,C,grid_h,grid_w]
        branch1_in, branch2_in = token_grid.chunk(2, dim=1)
        q1 = self.branch1(branch1_in)
        q2 = self.branch2(branch2_in)
        g1 = self.gate1(q1)  # [BT,C/2,1,1]
        g2 = self.gate2(q2)  # [BT,C/2,1,1]
        z1 = q1 * g2
        z2 = q2 * g1
        fused_grid = torch.cat([z1, z2], dim=1)
        fused_grid = self.fused_norm(fused_grid + token_grid)
        fused_tokens = fused_grid.flatten(2).transpose(1, 2)  # [BT,M,C]
        delta_patches = self.patch_decoder(fused_tokens)  # [BT,M,3*C*p*p]
        delta_patches = delta_patches.transpose(1, 2)  # [BT,3*C*p*p,M]
        delta_high_by_dir = F.fold(
            delta_patches,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.stride,
        )  # [BT,3C,H,W]
        ones = torch.ones(
            (1, 1, H, W),
            dtype=delta_high_by_dir.dtype,
            device=delta_high_by_dir.device,
        )
        overlap_count = F.fold(
            F.unfold(
                ones,
                kernel_size=self.patch_size,
                stride=self.stride,
            ),
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.stride,
        )
        delta_high_by_dir = delta_high_by_dir / overlap_count.clamp_min(1.0)
        delta_high = delta_high_by_dir.view(BT, Dirs, C, H, W)
        delta_high = delta_high.permute(0, 2, 1, 3, 4).contiguous()
        scale = torch.sigmoid(self.residual_scale).to(
            dtype=high.dtype,
            device=high.device,
        )
        return high + scale * delta_high

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, mid_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)

class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"ChannelLayerNorm2d 期望输入 [N,C,T,H]，实际为 {tuple(x.shape)}"
            )
        x = x.permute(0, 2, 3, 1).contiguous()  # [N,T,H,C]
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()  # [N,C,T,H]


class MultiScaleSpectralConv2d(nn.Module):
    def __init__(self, channels, kernel_sizes):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正整数")
        if not kernel_sizes:
            raise ValueError("kernel_sizes 不能为空")
        if any(k <= 0 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("光谱卷积核尺寸必须为正奇数")
        self.channels = channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.num_scales = len(self.kernel_sizes)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(kernel_size, 1),
                    padding=(kernel_size // 2, 0),
                    groups=channels,
                    bias=False,
                ),
                ChannelLayerNorm2d(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            )
            for kernel_size in self.kernel_sizes
        ])
        selector_hidden = max(4, channels // 2)
        self.scale_selector = nn.Sequential(
            nn.Conv2d(channels, selector_hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(
                selector_hidden,
                channels * self.num_scales,
                kernel_size=1,
                bias=True,
            ),
        )
        self.out_norm = ChannelLayerNorm2d(channels)
        self.out_act = nn.GELU()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"MultiScaleSpectralConv2d 期望输入 [N,C,T,H]，实际为 {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"期望输入通道数 {self.channels}，实际为 {x.shape[1]}"
            )
        scale_features = torch.stack(
            [branch(x) for branch in self.branches],
            dim=1,
        )  # [N,S,C,T,H]
        descriptor = x.abs().mean(dim=2, keepdim=True)  # [N,C,1,H]
        logits = self.scale_selector(descriptor)
        N, _, _, H = logits.shape
        weights = logits.view(
            N,
            self.num_scales,
            self.channels,
            1,
            H,
        )
        weights = torch.softmax(weights, dim=1)
        fused = torch.sum(scale_features * weights, dim=1)
        return self.out_act(self.out_norm(fused))

class AdaptiveSpectralSparseShrinkage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正整数")

        self.channels = channels
        hidden = max(4, channels // 2)

        self.threshold_predictor = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        final_conv = self.threshold_predictor[2]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, -2.0)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"AdaptiveSpectralSparseShrinkage 期望输入 [N,C,T,H]，实际为 {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"期望输入通道数 {self.channels}，实际为 {x.shape[1]}"
            )

        spectral_energy = x.abs().mean(dim=2, keepdim=True)  # [N,C,1,H]
        threshold_ratio = self.threshold_predictor(spectral_energy)
        threshold = threshold_ratio * spectral_energy

        return torch.sign(x) * F.relu(x.abs() - threshold)


class MultiScaleSpectralWaveletShrinkageHT(nn.Module):
    def __init__(self, in_channels, out_channels, layer_scale_init=1e-2):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels 和 out_channels 必须为正整数")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ChannelLayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.low_spectral = MultiScaleSpectralConv2d(
            channels=out_channels,
            kernel_sizes=(3, 5, 7),
        )
        self.high_spectral = MultiScaleSpectralConv2d(
            channels=out_channels,
            kernel_sizes=(1, 3, 5),
        )
        self.high_shrinkage = AdaptiveSpectralSparseShrinkage(
            channels=out_channels
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            ChannelLayerNorm2d(out_channels),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )
        # 仅用于可视化；detach 避免保存训练计算图。
        self.last_low_1d = None
        self.last_high_1d = None

    def _haar_dwt_t(self, x):
        original_t = x.shape[2]
        if original_t % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='replicate')
        even = x[:, :, 0::2, :]
        odd = x[:, :, 1::2, :]
        low = (even + odd) * self.inv_sqrt2
        high = (even - odd) * self.inv_sqrt2
        return low, high, original_t

    def _haar_idwt_t(self, low, high, original_t):
        if low.shape != high.shape:
            raise ValueError(
                f"低频和高频尺寸必须一致，实际为 low={tuple(low.shape)}, high={tuple(high.shape)}"
            )
        even = (low + high) * self.inv_sqrt2
        odd = (low - high) * self.inv_sqrt2
        x = torch.stack((even, odd), dim=3).flatten(2, 3)
        return x[:, :, :original_t, :]
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"MultiScaleSpectralWaveletShrinkageHT 期望输入 [N,C,T,H]，实际为 {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"期望输入通道数 {self.in_channels}，实际为 {x.shape[1]}"
            )
        base = self.in_proj(x)
        low, high, original_t = self._haar_dwt_t(base)
        self.last_low_1d = low.detach()
        self.last_high_1d = high.detach()
        low_enhanced = self.low_spectral(low)
        high_enhanced = self.high_spectral(high)
        high_sparse = self.high_shrinkage(high_enhanced)
        reconstructed = self._haar_idwt_t(
            low_enhanced,
            high_sparse,
            original_t,
        )
        enhancement = self.out_proj(reconstructed - base)
        return base + self.layer_scale * enhancement

class W_GSAM_Tokenizer(nn.Module):
    def __init__(self, in_bands, out_bands, spatial_kernel, hidden_dim, out_channels):
        super().__init__()
        self.out_bands = out_bands
        self.out_channels = out_channels
        self.core_channels = 4
        self.crop_pad = spatial_kernel // 2
        self.ghost_total = self.out_channels - self.core_channels
        assert self.ghost_total % 2 == 0, "out_channels 减去 4 后必须是偶数"
        self.ghost_half = self.ghost_total // 2
        self.split_channels = self.core_channels // 2
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_bands, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=spatial_kernel, padding=0, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim)
        )
        self.compress_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_bands, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_bands),
            nn.GELU()
        )
        self.core_extract = nn.Sequential(
            nn.Conv2d(out_bands, self.core_channels * out_bands, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.core_channels * out_bands),
            nn.GELU()
        )
        self.slice_conv = MultiScaleSpectralWaveletShrinkageHT(
            in_channels=self.split_channels,
            out_channels=self.ghost_half,
            layer_scale_init=1e-2,
        )
        self.dwt_haar = DWTForward(J=1, wave='haar', mode='zero')
        self.dwt_db2 = DWTForward(J=1, wave='db2', mode='zero')
        self.dwt_sym2 = DWTForward(J=1, wave='sym2', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        self.lf_conv_haar = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                      groups=self.split_channels)
        self.lf_conv_db2 = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                     groups=self.split_channels)
        self.lf_conv_sym2 = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                      groups=self.split_channels)
        self.hf_conv_haar = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                      groups=self.split_channels)
        self.hf_conv_db2 = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                     groups=self.split_channels)
        self.hf_conv_sym2 = nn.Conv2d(self.split_channels, self.ghost_half, kernel_size=3, padding=1,
                                      groups=self.split_channels)
        self.branch_weights_lf = nn.Parameter(torch.ones(3))
        self.branch_weights_hf = nn.Parameter(torch.ones(3))
        self.channel_attention = ChannelAttention(self.ghost_half)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.hf_patch_cross_gate = OverlapPatchCrossGatedDenoiser(
            channels=self.ghost_half,
            dirs=3,
            patch_size=2,
            stride=1,
        )
        self.high_mamba_dim = 32
        self.high_proj_in = nn.Linear(self.ghost_half * 3, self.high_mamba_dim)
        self.mamba = Mamba(d_model=self.high_mamba_dim, d_state=8, d_conv=4, expand=2)
        self.high_proj_out = nn.Linear(self.high_mamba_dim, self.ghost_half * 3)
        self.attention = SE_3D_Attention(channels=self.out_channels)
        self.wavelet_visualization_enabled = False
        self.wavelet_visualization_done = False
        self.wavelet_visualization_dir = "new photo"

    def enable_wavelet_visualization(self, output_dir="new photo"):
        """在下一次经过小波分支的前向传播时保存 14 张系数图。"""
        self.wavelet_visualization_enabled = True
        self.wavelet_visualization_done = False
        self.wavelet_visualization_dir = output_dir

    def forward(self, x):
        B, _, _, H, W = x.shape
        x_2d = x.squeeze(1)
        x_exp = self.expand_conv(x_2d)
        x_spa = self.spatial_dw(x_exp)
        c = self.crop_pad
        x_exp_cropped = x_exp[:, :, c:-c, c:-c]
        x_fused = x_spa + x_exp_cropped
        x_comp = self.compress_conv(x_fused)
        H_out, W_out = H - 2 * c, W - 2 * c
        out_core_2d = self.core_extract(x_comp)
        out_core_5d = out_core_2d.view(B, self.core_channels, self.out_bands, H_out, W_out)
        core_folded = out_core_5d.permute(0, 2, 1, 3, 4).reshape(B * self.out_bands, self.core_channels, H_out, W_out)
        x_spatial_base, x_wavelet_base = core_folded.chunk(2, dim=1)
        x_spa_5d = x_spatial_base.view(B, self.out_bands, self.split_channels, H_out, W_out)
        x_slice = x_spa_5d.permute(0, 4, 2, 1, 3)
        x_slice = x_slice.reshape(B * W_out, self.split_channels, self.out_bands, H_out)
        ghost_spatial_slice = self.slice_conv(x_slice)
        ghost_spatial_slice = ghost_spatial_slice.view(B, W_out, self.ghost_half, self.out_bands, H_out)
        ghost_spatial_5d = ghost_spatial_slice.permute(0, 3, 2, 4, 1)
        ghost_spatial = ghost_spatial_5d.reshape(B * self.out_bands, self.ghost_half, H_out, W_out)
        x_wavelet_pad = F.pad(x_wavelet_base, (0, 1, 0, 1), mode='replicate')
        LL_haar, Highs_haar = self.dwt_haar(x_wavelet_pad)
        LL_db2, Highs_db2 = self.dwt_db2(x_wavelet_pad)
        LL_sym2, Highs_sym2 = self.dwt_sym2(x_wavelet_pad)
        H_haar = Highs_haar[0]
        BT, C_split, Dirs, H_h, W_h = H_haar.shape
        LL_db2 = LL_db2[:, :, :H_h, :W_h]
        H_db2 = Highs_db2[0][:, :, :, :H_h, :W_h]
        LL_sym2 = LL_sym2[:, :, :H_h, :W_h]
        H_sym2 = Highs_sym2[0][:, :, :, :H_h, :W_h]
        if self.wavelet_visualization_enabled and not self.wavelet_visualization_done:
            # 保存原始 DWT 系数（卷积、注意力和融合之前），使三种滤波器可直接比较。
            self.wavelet_visualization_done = True
            try:
                saved_paths = save_wavelet_visualizations(
                    low_1d=self.slice_conv.last_low_1d,
                    high_1d=self.slice_conv.last_high_1d,
                    batch_size=B,
                    width=W_out,
                    ll_by_filter={"haar": LL_haar, "db2": LL_db2, "sym2": LL_sym2},
                    high_by_filter={"haar": H_haar, "db2": H_db2, "sym2": H_sym2},
                    output_dir=self.wavelet_visualization_dir,
                )
                print(f"小波可视化已保存 {len(saved_paths)} 张图片至: {self.wavelet_visualization_dir}")
            except Exception:
                self.wavelet_visualization_done = False
                raise
        H_haar_flat = H_haar.permute(0, 2, 1, 3, 4).reshape(BT * Dirs, C_split, H_h, W_h)
        H_db2_flat = H_db2.permute(0, 2, 1, 3, 4).reshape(BT * Dirs, C_split, H_h, W_h)
        H_sym2_flat = H_sym2.permute(0, 2, 1, 3, 4).reshape(BT * Dirs, C_split, H_h, W_h)
        LL_haar_feat = self.lf_conv_haar(LL_haar)  # [BT, 14, H_h, W_h]
        LL_db2_feat = self.lf_conv_db2(LL_db2)
        LL_sym2_feat = self.lf_conv_sym2(LL_sym2)
        H_haar_feat = self.hf_conv_haar(H_haar_flat)  # [BT*Dirs, 14, H_h, W_h]
        H_db2_feat = self.hf_conv_db2(H_db2_flat)
        H_sym2_feat = self.hf_conv_sym2(H_sym2_flat)
        H_haar_feat = H_haar_feat.view(BT, Dirs, self.ghost_half, H_h, W_h).permute(0, 2, 1, 3, 4)
        H_db2_feat = H_db2_feat.view(BT, Dirs, self.ghost_half, H_h, W_h).permute(0, 2, 1, 3, 4)
        H_sym2_feat = H_sym2_feat.view(BT, Dirs, self.ghost_half, H_h, W_h).permute(0, 2, 1, 3, 4)
        weights_lf = F.softmax(self.branch_weights_lf, dim=0)
        weights_hf = F.softmax(self.branch_weights_hf, dim=0)
        LL_fused = (weights_lf[0] * LL_haar_feat +
                    weights_lf[1] * LL_db2_feat +
                    weights_lf[2] * LL_sym2_feat)
        H_fused = (weights_hf[0] * H_haar_feat +
                   weights_hf[1] * H_db2_feat +
                   weights_hf[2] * H_sym2_feat)
        LL_out_att = self.channel_attention(LL_fused)
        H_fused_flat = H_fused.reshape(BT, self.ghost_half * Dirs, H_h, W_h)
        H_out_att_flat = self.spatial_attention(H_fused_flat)
        H_comp = H_out_att_flat.reshape(BT, self.ghost_half, Dirs, H_h, W_h)
        LL_out = LL_out_att
        BT_after, C_in, Dirs_after, H_h_after, W_h_after = H_comp.shape
        seq_len = H_h_after * W_h_after
        H_comp = self.hf_patch_cross_gate(LL_out, H_comp)
        h_seq = H_comp.view(BT_after, C_in, Dirs_after, seq_len).permute(0, 3, 1, 2).reshape(BT_after, seq_len,
                                                                                             C_in * Dirs_after)
        h_seq = self.high_proj_in(h_seq)
        h_seq = self.mamba(h_seq)
        h_seq = self.high_proj_out(h_seq)
        H_out_tensor = h_seq.view(BT_after, seq_len, self.ghost_half, Dirs_after).permute(0, 2, 3, 1).reshape(BT_after,
                                                                                                              self.ghost_half,
                                                                                                              Dirs_after,
                                                                                                              H_h_after,
                                                                                                              W_h_after)
        ghost_wavelet_pad = self.idwt((LL_out, [H_out_tensor]))
        ghost_wavelet = ghost_wavelet_pad[:, :, 0:H_out, 0:W_out]
        ghost_folded = torch.cat([ghost_spatial, ghost_wavelet], dim=1)
        ghost_5d = ghost_folded.view(B, self.out_bands, self.ghost_total, H_out, W_out).permute(0, 2, 1, 3, 4)

        out_fused = torch.cat([out_core_5d, ghost_5d], dim=1)
        final_out = self.attention(out_fused)
        return final_out


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class Block(nn.Module, mamba_init):
    def __init__(self,
                 scan_type=None,
                 group_type=None,
                 k_group=None,
                 dim=None,
                 dt_rank=None,
                 d_inner=None,
                 d_state=None,
                 seq=False,
                 force_fp32=True,
                 dropout=0.0,
                 bimamba=False,
                 pos=None,
                 cls=None,
                 selective_scan_backend='cuda',
                 **kwargs):
        super().__init__()
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        self.force_fp32 = force_fp32
        self.seq = seq
        self.k_group = k_group
        self.group_type = group_type
        self.scan_type = scan_type
        self.selective_scan_backend = selective_scan_backend
        if selective_scan_backend not in ('cuda', 'torch', 'legacy_skip'):
            raise ValueError(f'Unknown selective_scan_backend: {selective_scan_backend}')
        self.in_proj = nn.Linear(dim, d_inner * 2, bias=bias, **kwargs)
        self.act: nn.Module = act_layer()
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True)
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, dim, bias=bias, **kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def flatten_spectral_spatial(self, x):
        x = rearrange(x, 'b c t h w -> b c (h w) t')
        x = rearrange(x, 'b c n m -> b c (n m)')
        return x

    def flatten_spatial_spectral(self, x):
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = rearrange(x, 'b c n m -> b c (n m)')
        return x

    def reshape_spectral_spatial(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, H * W, T, -1)
        y = rearrange(y, 'b o t c -> b t o c')
        y = y.view(B, T, H, W, -1)
        return y

    def reshape_spatial_spectral(self, y, B, H, W, T):
        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = y.view(B, T, H, W, -1)
        return y

    def scan(self, x, scan_type=None, group_type=None):
        if scan_type == 'Spectral-priority':
            x = self.flatten_spectral_spatial(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Spatial-priority':
            x = self.flatten_spatial_spectral(x)
            xs = torch.stack([x, torch.flip(x, dims=[-1])], dim=1)
        elif scan_type == 'Cross spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spa, dims=[-1])], dim=1)
        elif scan_type == 'Cross spatial-spectral':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spa, torch.flip(x_spe, dims=[-1])], dim=1)
        elif scan_type == 'Parallel spectral-spatial':
            x_spe = self.flatten_spectral_spatial(x)
            x_spa = self.flatten_spatial_spectral(x)
            xs = torch.stack([x_spe, torch.flip(x_spe, dims=[-1]), x_spa, torch.flip(x_spa, dims=[-1])], dim=1)
        return xs

    def forward(self, x: Tensor, SelectiveScan=SelectiveScanMamba):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(x)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            if self.selective_scan_backend == 'torch':
                from models.selective_scan_torch import selective_scan_torch
                return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus)
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, T, H, W = x.shape
        L = T * H * W
        if self.selective_scan_backend == 'legacy_skip':
            y = rearrange(x, 'b d t h w -> b t h w d')
            y = self.out_norm(y)
            y = y * z
            return self.dropout(self.out_proj(y))
        if self.selective_scan_backend == 'cuda' and not _HAS_SELECTIVE_SCAN:
            raise RuntimeError('selective_scan_cuda is unavailable. Refusing to skip Mamba. '
                               'Install a compatible CUDA extension or explicitly use selective_scan_backend="torch".')
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        xs = self.scan(x, scan_type=self.scan_type, group_type=self.group_type)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if self.force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)
        out_y = selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        if self.group_type == 'Cube':
            if self.scan_type == 'Spectral-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                y = self.reshape_spectral_spatial(y, B, H, W, T)
                y = self.out_norm(y)
            elif self.scan_type == 'Spatial-priority':
                y = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                y = self.reshape_spatial_spectral(y, B, H, W, T)
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spectral-spatial':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spectral_spatial(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spatial_spectral(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Cross spatial-spectral':
                y_fwd = out_y[:, 0]
                y_fwd = self.reshape_spatial_spectral(y_fwd, B, H, W, T)
                y_rvs = torch.flip(out_y[:, 1], dims=[-1])
                y_rvs = self.reshape_spectral_spatial(y_rvs, B, H, W, T)
                y = y_fwd + y_rvs
                y = self.out_norm(y)
            elif self.scan_type == 'Parallel spectral-spatial':
                ye = out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1])
                ye = self.reshape_spectral_spatial(ye, B, H, W, T)
                ya = out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1])
                ya = self.reshape_spatial_spectral(ya, B, H, W, T)
                y = ye + ya
                y = self.out_norm(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class TokenMoE(nn.Module):
    def __init__(self, dim, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        routing_weights = torch.softmax(self.router(x), dim=-1)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            out += expert_out * routing_weights[:, :, i:i + 1]
        return x + out


class WDCMT(nn.Module):
    """Wavelet Decoupling and Complementary Mamba-Transformer model (WD-CMT)."""

    def __init__(
            self,
            group_type=None,
            k_group=None,
            depth=None,
            embed_dim=None,
            dt_rank: int = None,
            d_inner: int = None,
            d_state: int = None,
            num_classes: int = None,
            drop_rate=0.,
            drop_path_rate=0.1,
            fused_add_norm=False,
            residual_in_fp32=True,
            bimamba=True,
            fc_drop_rate=0.,
            use_checkpoint=False,
            checkpoint_num=0,
            Pos_Cls=False,
            scan_type=None,
            conv3D_channel: int = None,
            conv3D_kernel: int = None,
            dim_patch: int = None,
            dim_linear: int = None,
            # 专家数量
            num_experts: int = 4,
            **kwargs,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.Pos_Cls = Pos_Cls
        self.scan_type = scan_type
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.k_group = k_group
        self.group_type = group_type
        original_bands = dim_linear + conv3D_kernel[0] - 1
        self.conv3d_features = W_GSAM_Tokenizer(
            in_bands=original_bands,
            out_bands=dim_linear,
            spatial_kernel=conv3D_kernel[1],
            hidden_dim=conv3D_channel * 2,
            out_channels=conv3D_channel
        )

        self.embedding_spatial_spectral = nn.Sequential(nn.Linear(conv3D_channel, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.trans_dim = conv3D_channel * dim_linear
        self.trans_seq_len = dim_patch * dim_patch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.trans_dim,
            nhead=8,
            dim_feedforward=self.trans_dim * 2,
            dropout=drop_rate,
            batch_first=True
        )
        self.transformer_branch = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.trans_pos_embed = nn.Parameter(torch.zeros(1, self.trans_seq_len, self.trans_dim))
        trunc_normal_(self.trans_pos_embed, std=.02)
        self.token_moe = TokenMoE(dim=self.trans_dim, num_experts=num_experts)
        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features * 2, num_classes) if num_classes > 0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layers = nn.ModuleList(
            [
                Block(
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=k_group,
                    dim=embed_dim,
                    d_state=d_state,
                    d_inner=d_inner,
                    dt_rank=dt_rank,
                    bimamba=bimamba,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

    def enable_wavelet_visualization(self, output_dir="new photo"):
        """将小波分支中下一次前向传播的中间系数保存为图片。"""
        self.conv3d_features.enable_wavelet_visualization(output_dir)

    def forward_features(self, x, inference_params=None):
        x_conv = self.conv3d_features(x)
        x_mamba = rearrange(x_conv, 'b c t h w -> b t h w c')
        x_mamba = self.embedding_spatial_spectral(x_mamba)
        x_mamba = self.pos_drop(x_mamba)
        for idx, layer in enumerate(self.layers):
            x_mamba = x_mamba + self.drop_path(layer(self.norm(x_mamba)))
        B, C, T, H, W = x_conv.shape
        x_trans = x_conv.view(B, C * T, H * W).permute(0, 2, 1)
        x_trans = x_trans + self.trans_pos_embed
        x_trans = self.token_moe(x_trans)
        x_trans = self.transformer_branch(x_trans)
        x_trans = x_trans.permute(0, 2, 1).view(B, C, T, H, W)
        x_trans = x_trans.permute(0, 2, 3, 4, 1)
        x_trans = self.embedding_spatial_spectral(x_trans)
        pool_fn = lambda feat: self.flatten(self.avgpool(feat.permute(0, 4, 1, 2, 3)).mean(dim=2))
        x_mamba_pooled = pool_fn(x_mamba)
        x_trans_pooled = pool_fn(x_trans)
        x_fused_pooled = torch.cat([x_mamba_pooled, x_trans_pooled], dim=1)
        return x_fused_pooled, x_mamba, x_trans

    def forward(self, x, inference_params=None):
        x_fused, x_mamba, x_trans = self.forward_features(x, inference_params)
        outputs = self.head(self.head_drop(x_fused))
        return outputs, x_fused, x_mamba, x_trans
