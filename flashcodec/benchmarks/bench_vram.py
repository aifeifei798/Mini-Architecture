"""显存/存储压测：对比 4K/16K/64K 序列下原始 KV 与各编解码档位的
实际缓存占用（tensor storage bytes）与填充/读取耗时。

用法：.venv/bin/python benchmarks/bench_vram.py [--cuda]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

DEVICE = "cuda" if (torch.cuda.is_available() and "--cuda" in sys.argv) else "cpu"
B, H, D = 1, 8, 64
MODES = [
    ("raw (fp32)", None),
    ("jpeg k=2 (4x)", dict(mode="jpeg", keep_freqs=2)),
    ("jpeg k=4 (2x)", dict(mode="jpeg", keep_freqs=4)),
    ("png keep=0.5", dict(mode="png", keep_ratio=0.5)),
    ("png keep=1.0", dict(mode="png", keep_ratio=1.0)),
]


def mb(n):
    return n / 1024 / 1024


print(f"device={DEVICE}  B={B} H={H} D={D}  block=8  fp32\n")
header = f"{'seq':>6} | {'mode':<14} | {'storage':>10} | {'ratio':>6} | {'fill':>8} | {'read':>8}"
print(header)
print("-" * len(header))
for S in (4096, 16384, 65536):
    torch.manual_seed(0)
    k = torch.randn(B, H, S, D, device=DEVICE)
    v = torch.randn(B, H, S, D, device=DEVICE)
    raw_bytes = (k.numel() + v.numel()) * k.element_size()
    for name, kw in MODES:
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        if kw is None:
            _ = (k, v)
            fill_s = read_s = 0.0
            store, ratio = raw_bytes, 1.0
        else:
            cache = CodecKVCache(block_size=8, **kw)
            cache.forward(k, v)
            t1 = time.time()
            view = cache.key_cache[0]
            _ = view.sum().item()          # 强制物化
            t2 = time.time()
            fill_s, read_s = t1 - t0, t2 - t1
            st = cache.stats(0)
            store, ratio = st["storage_bytes"], st["storage_bytes"] / st["raw_bytes"]
        peak = (torch.cuda.max_memory_allocated() / 1024 / 1024
                if DEVICE == "cuda" else float("nan"))
        print(f"{S:>6} | {name:<14} | {mb(store):>8.2f}MB | {ratio:>6.2f} | "
              f"{fill_s * 1e3:>6.1f}ms | {read_s * 1e3:>6.1f}ms"
              + (f"  (peak {peak:.0f}MB)" if DEVICE == "cuda" else ""))
    print()
