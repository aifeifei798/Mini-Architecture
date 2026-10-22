"""JPEG 风格有损频域压缩：正交分块 1D DCT-II。

物理本质：相邻 token 的隐向量构成强自相关时序信号，能量集中在低频。
分块投影到前 K 个正交 DCT-II 基底（DC 均值 + AC 线性趋势...），滤除高频
随机毛刺，实现 r/K 几何压缩比。基底行正交归一，keep=r 时逐块无损。

张量契约：输入 (..., S, D)，压缩/滤波操作轴为倒数第二维（seq 轴，
dim=-2）。全部为批量矩阵乘（GEMM），无显式 Python 循环。
"""

import math

import torch
import torch.nn.functional as F

_BASIS_CACHE = {}


def dct_basis(r, k, device=None, dtype=None):
    """预计算正交归一化 DCT-II 基底 M_enc (k, r)：
    M[j, n] = c_j · cos(π/r · (n+½)·j)，c_0 = sqrt(1/r)，c_j = sqrt(2/r)。
    行正交归一（M @ M.T = I），k == r 时为完备正交基（重构无损）。
    模块级常量缓存，按 (r, k) 键控。"""
    key = (r, k)
    if key not in _BASIS_CACHE:
        n = torch.arange(r, dtype=torch.float32)
        j = torch.arange(k, dtype=torch.float32)
        c = torch.where(j == 0, math.sqrt(1.0 / r), math.sqrt(2.0 / r))
        m = c[:, None] * torch.cos(math.pi / r * (n[None, :] + 0.5) * j[:, None])
        _BASIS_CACHE[key] = m
    m = _BASIS_CACHE[key]
    if device is not None or dtype is not None:
        m = m.to(device=device, dtype=dtype or m.dtype)
    return m


def _pad_to_blocks(x, r):
    """把 seq 轴（-2）零填充到 r 的整数倍，返回 (块张量, 原始长度)。"""
    S = x.shape[-2]
    n = (S + r - 1) // r
    pad = n * r - S
    if pad:
        x = F.pad(x, (0, 0, 0, pad))          # (..., S, D) 只 pad -2 轴
    blocks = x.reshape(*x.shape[:-2], n, r, x.shape[-1])
    return blocks, S


def dct_compress(x, r, k):
    """分块 DCT 投影：x (..., S, D) -> coeffs (..., n_blocks, k, D)。
    seq 轴按 r 分块（尾部零填充），每块沿块内轴投影到前 k 个 DCT-II 基底。
    压缩存储比 = r/k（外加尾部 padding 开销）。"""
    blocks, _ = _pad_to_blocks(x, r)
    basis = dct_basis(r, k, device=x.device, dtype=torch.float32)
    hp = blocks.float().transpose(-2, -1)              # (..., n, D, r)
    coef = torch.matmul(hp, basis.t())                 # (..., n, D, k)
    return coef.transpose(-2, -1)                      # (..., n, k, D)


def dct_decompress(coeffs, r, out_len=None):
    """分块 DCT 重构：coeffs (..., n_blocks, k, D) -> x (..., n_blocks*r, D)。
    低频系数乘基底重构块内波形；out_len 给定时裁掉尾部 padding。"""
    n, k = coeffs.shape[-3], coeffs.shape[-2]
    basis = dct_basis(r, k, device=coeffs.device, dtype=torch.float32)
    hp = torch.matmul(coeffs.transpose(-2, -1).float(), basis)   # (...,n,D,r)
    rec = hp.transpose(-2, -1).reshape(
        *coeffs.shape[:-3], n * r, coeffs.shape[-1])
    if out_len is not None:
        rec = rec[..., :out_len, :]
    return rec


class JPEGCodec:
    """JPEG 风格有损频域编解码器（无状态，可跨 cache 共享实例）。"""

    name = "jpeg"

    def __init__(self, block_size=8, keep_freqs=2):
        assert 1 <= keep_freqs <= block_size, \
            f"keep_freqs 必须在 [1, {block_size}] 内"
        self.block_size = block_size
        self.keep_freqs = keep_freqs

    def compress(self, x):
        return dct_compress(x, self.block_size, self.keep_freqs)

    def decompress(self, coeffs, out_len=None):
        return dct_decompress(coeffs, self.block_size, out_len)

    def storage_ratio(self):
        """归档区张量元素数 / 原始元素数（不含尾部 raw）。"""
        return self.keep_freqs / self.block_size

    def __repr__(self):
        return (f"JPEGCodec(block_size={self.block_size}, "
                f"keep_freqs={self.keep_freqs})")
