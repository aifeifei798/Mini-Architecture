"""重构质量压测：不同模式/档位下的 PSNR 与余弦相似度。

信号：合成自相关隐向量（随机游走，模拟真实 KV 的强时序相关）。
用法：.venv/bin/python benchmarks/bench_reconstruct.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

torch.manual_seed(0)
B, H, S, D = 2, 8, 2048, 64
x = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)   # 自相关隐向量


def psnr(ref, hat):
    rmse = ((ref - hat) ** 2).mean().sqrt().item()
    peak = ref.abs().max().item()
    return 20 * math.log10(peak / max(rmse, 1e-12))


def cos(ref, hat):
    return torch.nn.functional.cosine_similarity(
        ref.flatten(), hat.flatten(), dim=0).item()


CONFIGS = [
    ("jpeg", dict(mode="jpeg", keep_freqs=1)),
    ("jpeg", dict(mode="jpeg", keep_freqs=2)),
    ("jpeg", dict(mode="jpeg", keep_freqs=4)),
    ("png", dict(mode="png", keep_ratio=0.25)),
    ("png", dict(mode="png", keep_ratio=0.5)),
    ("png", dict(mode="png", keep_ratio=1.0)),
]

print(f"signal: random-walk B={B} H={H} S={S} D={D}（自相关隐向量模拟）\n")
header = f"{'config':<16} | {'PSNR(dB)':>9} | {'cosine':>8} | {'storage':>8}"
print(header)
print("-" * len(header))
st = cache = None
for _, kw in CONFIGS:
    cache = CodecKVCache(block_size=8, **kw)
    hat, _ = cache.forward(x, x)
    s = cache.stats(0)
    name = (f"{kw['mode']} r=8"
            + (f" k={kw['keep_freqs']}" if kw["mode"] == "jpeg"
               else f" keep={kw['keep_ratio']}"))
    print(f"{name:<16} | {psnr(x, hat):>9.2f} | {cos(x, hat):>8.5f} | "
          f"{s['storage_bytes'] / s['raw_bytes']:>7.2%}")
