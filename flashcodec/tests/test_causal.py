"""因果安全与 Cache 一致性测试：
1) 未来扰动零泄露（已归档块张量逐位不变，全量/增量两种模式）
2) 增量解码（lossless png）与全量前向的注意力输出对齐 ≤1e-4
3) forward 填充与逐步填充的缓存视图逐位一致
4) jpeg 有损档位的注意力输出质量（余弦报告）
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import CodecKVCache

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 96, 32, 8
k_seq = torch.randn(B, H, S, D)
v_seq = torch.randn(B, H, S, D)


def toy_attn(q, k, v):
    """单段因果注意力（B,H,S_q,D）：q 与全量 KV 的缩放点积 + causal mask。"""
    Sq, Sk = q.shape[-2], k.shape[-2]
    logits = q @ k.transpose(-2, -1) / math.sqrt(D)
    mask = torch.ones(Sq, Sk, dtype=torch.bool, device=q.device).tril(
        diagonal=Sk - Sq)
    return torch.softmax(logits.masked_fill(~mask, float("-inf")),
                         dim=-1) @ v


def _fill_incr(mode, **kw):
    cache = CodecKVCache(mode=mode, **kw)
    for t in range(S):
        cache.decode_step(k_seq[..., t:t + 1, :], v_seq[..., t:t + 1, :])
    return cache


def _archive_tensors(archives):
    out = []
    for ar in archives:
        out.extend(ar if isinstance(ar, tuple) else (ar,))
    return out


def test_future_perturbation_zero_leak_decode():
    c = CodecKVCache(mode="png", block_size=R, keep_ratio=0.5)
    cut = 2 * R                                  # 已归档 2 个整块
    for t in range(cut):
        c.decode_step(k_seq[..., t:t + 1, :], v_seq[..., t:t + 1, :])
    snap = [[t.clone() for t in _archive_tensors(st.archives)]
            for st in (c._keys[0], c._values[0])]
    view_before = c.key_cache[0].clone()
    pert = torch.randn_like(k_seq[..., cut:, :])
    for t in range(cut, S):                      # 扰动未来 token
        c.decode_step(pert[..., t - cut:t - cut + 1, :],
                      pert[..., t - cut:t - cut + 1, :])
    for st, snap_i in zip((c._keys[0], c._values[0]), snap):
        for ar, ar0 in zip(_archive_tensors(st.archives), snap_i):
            assert torch.equal(ar, ar0), "未来扰动改写了已归档张量"
    assert torch.equal(c.key_cache[0][..., :cut, :], view_before[..., :cut, :]), \
        "未来扰动改写了历史视图"


def test_future_perturbation_zero_leak_forward():
    c1 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    c2 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    cut = 3 * R                                  # 块对齐的截断点
    k2 = k_seq.clone()
    k2[..., cut:, :] = torch.randn_like(k2[..., cut:, :])
    c1.forward(k_seq, v_seq)
    c2.forward(k2, v_seq)
    # 全量模式下所有完整块在一次 append 中批量归档为单个 chunk
    # (B,H,n_blocks,k,D)；块维在 -3
    a1, a2 = c1._keys[0].archives[0], c2._keys[0].archives[0]
    assert torch.equal(a1[..., :3, :, :], a2[..., :3, :, :]), \
        "全量模式下未来扰动影响了历史归档"
    assert torch.equal(c1.key_cache[0][..., :cut, :],
                       c2.key_cache[0][..., :cut, :])


def test_incremental_vs_full_attention_lossless():
    """lossless png cache 下，逐步解码注意力 == 全量前向注意力（≤1e-4）。"""
    q = torch.randn(B, H, S, D)
    c_full = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    kf, vf = c_full.forward(k_seq, v_seq)
    out_full = toy_attn(q, kf, vf)

    c = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    outs = []
    for t in range(S):
        kt, vt = c.decode_step(k_seq[..., t:t + 1, :],
                               v_seq[..., t:t + 1, :])
        outs.append(toy_attn(q[..., t:t + 1, :], kt, vt))
    out_incr = torch.cat(outs, dim=-2)
    d = (out_full - out_incr).abs().max().item()
    assert d < 1e-4, f"增量解码与全量前向不一致: maxdiff={d:.2e}"


def test_forward_fill_vs_incremental_view_bitwise():
    c1 = CodecKVCache(mode="png", block_size=R, keep_ratio=1.0)
    k1, v1 = c1.forward(k_seq, v_seq)
    c2 = _fill_incr("png", block_size=R, keep_ratio=1.0)
    assert torch.equal(k1, c2.key_cache[0]) and torch.equal(v1, c2.value_cache[0])
    c3 = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    k3, _ = c3.forward(k_seq, v_seq)
    c4 = _fill_incr("jpeg", block_size=R, keep_freqs=2)
    assert torch.equal(k3, c4.key_cache[0]), "jpeg 全量/增量视图应逐位一致"


def test_jpeg_attention_quality_report():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
    q = torch.randn(B, H, S, D)
    out_exact = toy_attn(q, smooth, smooth)
    c = CodecKVCache(mode="jpeg", block_size=R, keep_freqs=2)
    kq, vq = c.forward(smooth, smooth)
    out_q = toy_attn(q, kq, vq)
    cos = torch.nn.functional.cosine_similarity(
        out_exact.flatten(), out_q.flatten(), dim=0).item()
    print(f"   jpeg(r=8,k=2) 注意力输出余弦 = {cos:.4f}")
    assert cos > 0.99, f"jpeg 档位注意力质量过低: cos={cos:.4f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = True
    for fn in fns:
        try:
            fn()
            print(f"✅ {fn.__name__}")
        except AssertionError as e:
            ok = False
            print(f"❌ {fn.__name__}: {e}")
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
