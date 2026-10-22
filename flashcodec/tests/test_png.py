"""PNG（时域无损/近无损）编解码测试：逐位无损基线、方差压缩比、梯度。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import PNGCodec, png_filter, png_reconstruct

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 64, 32, 8
x = torch.randn(B, H, S, D)
smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)


def _roundtrip(sig, keep=1.0):
    delta, anchor = png_filter(sig, R, keep)
    rec = png_reconstruct(delta, anchor)
    return rec[..., :sig.shape[-2], :], delta, anchor


def test_lossless_bitwise_baseline():
    rec, _, _ = _roundtrip(x, 1.0)
    d = (rec - x).abs().max().item()
    assert d < 1e-6, f"keep_ratio=1.0 应逐位无损, maxdiff={d:.2e}"


def test_variance_collapse():
    var_x = smooth.var().item()
    var_d = (smooth[..., 1:, :] - smooth[..., :-1, :]).var().item()
    ratio = var_d / var_x
    print(f"   方差比 Var(Δ)/Var(X) = {ratio:.4f}（方差压缩 {1 / ratio:.1f}×）")
    assert ratio < 0.5, f"差分方差应骤降, got ratio={ratio:.4f}"


def test_keep_ratio_monotonic():
    e1 = (_roundtrip(smooth, 1.0)[0] - smooth).abs().mean().item()
    e05 = (_roundtrip(smooth, 0.5)[0] - smooth).abs().mean().item()
    assert e05 > e1, f"Top-M 裁剪应有损: err1.0={e1:.2e} err0.5={e05:.4f}"


def test_autograd_differentiable():
    xr = torch.randn(B, H, S, D, requires_grad=True)
    rec, _, _ = _roundtrip(xr, 0.5)
    rec.pow(2).mean().backward()
    assert xr.grad is not None and torch.isfinite(xr.grad).all()


def test_padding_and_shapes():
    odd = torch.randn(B, H, 50, D)
    rec, delta, anchor = _roundtrip(odd, 1.0)
    assert rec.shape == odd.shape
    assert anchor.shape == (B, H, (50 + R - 1) // R, D)
    assert delta.shape == (B, H, (50 + R - 1) // R, R - 1, D)


def test_codec_class_api():
    c = PNGCodec(block_size=R, keep_ratio=1.0)
    delta, anchor = c.filter(x)
    rec = c.reconstruct(delta, anchor)[..., :S, :]
    assert (rec - x).abs().max().item() < 1e-6
    assert c.storage_ratio() == 1.0, "稠密存储应如实计为 1.0（稀疏性留给熵编码）"


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
