"""PNG 风格无损/近无损时域压缩：因果分块预测差分滤波。

物理本质：相邻 token 隐向量高度自相关，一阶差分 Δ = x_i - x_{i-1} 后
方差骤降、数值集中于 0 附近（极低信息熵）。块首存全精度锚点（I-Token），
块内其余存差分；cumsum 因果重建，keep_ratio=1.0 时逐位可逆（fp 舍入级）。
keep_ratio<1.0 时按 |Δ| 幅值每（块,特征）保留 Top-M 步，模拟近无损档位。

张量契约：输入 (..., S, D)，操作轴为倒数第二维（seq 轴，dim=-2）。
纯线性算子 + 离散选择，autograd 天然可微（梯度只流向被保留的差分步）。
"""

import math

import torch
import torch.nn.functional as F


def _pad_to_blocks(x, r):
    """把 seq 轴（-2）零填充到 r 的整数倍，返回 (块张量, 原始长度)。"""
    S = x.shape[-2]
    n = (S + r - 1) // r
    pad = n * r - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    blocks = x.reshape(*x.shape[:-2], n, r, x.shape[-1])
    return blocks, S


def png_filter(x, block_size, keep_ratio=1.0):
    """差分滤波：x (..., S, D) -> (deltas (..., n, r-1, D), anchor (..., n, D))。
    每块位置 0 强制全精度锚点；位置 i>=1 存 Δ_i = x_i - x_{i-1}。
    keep_ratio<1.0 时按 |Δ| 幅值每（块,特征）保留 Top-M 步
    （m = ceil((r-1)*keep_ratio)，m>=1），其余置 0。"""
    assert 0 < keep_ratio <= 1.0, "keep_ratio 必须在 (0, 1] 内"
    blocks, _ = _pad_to_blocks(x, block_size)
    anchor = blocks[..., 0, :]                                # (..., n, D)
    delta = blocks[..., 1:, :] - blocks[..., :-1, :]          # (..., n, r-1, D)
    if keep_ratio < 1.0:
        m = max(1, math.ceil(delta.shape[-2] * keep_ratio))
        topi = delta.abs().topk(m, dim=-2).indices
        mask = torch.zeros_like(delta, dtype=torch.bool)
        mask.scatter_(-2, topi, True)
        delta = delta * mask
    return delta, anchor


def png_reconstruct(deltas, anchor):
    """因果重建：锚点 + cumsum。位置 0 = 锚点，位置 i>=1 = 锚点 + 前 i 步差分。
    deltas (..., n, r-1, D)，anchor (..., n, D) -> x (..., n*r, D)。"""
    base = anchor.unsqueeze(-2)                               # (..., n, 1, D)
    rec = torch.cat([base, base + torch.cumsum(deltas, dim=-2)], dim=-2)
    return rec.reshape(*rec.shape[:-3], rec.shape[-3] * rec.shape[-2],
                       rec.shape[-1])


class PNGCodec:
    """PNG 风格预测滤波编解码器（无状态，可跨 cache 共享实例）。"""

    name = "png"

    def __init__(self, block_size=8, keep_ratio=1.0):
        assert 0 < keep_ratio <= 1.0, "keep_ratio 必须在 (0, 1] 内"
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def filter(self, x):
        return png_filter(x, self.block_size, self.keep_ratio)

    def reconstruct(self, deltas, anchor):
        return png_reconstruct(deltas, anchor)

    def storage_ratio(self):
        """稠密张量存储比 = 1.0（锚点 + (r-1) 步差分与原始块等大）。
        PNG 模式的价值在信息熵层面：差分方差骤降 ~30×、Top-M 裁剪后
        残差高度稀疏且近零——这是熵编码器（行程/算术编码）的压缩空间，
        本库不内置熵编码，如实按稠密张量计。"""
        return 1.0

    def __repr__(self):
        return (f"PNGCodec(block_size={self.block_size}, "
                f"keep_ratio={self.keep_ratio})")
