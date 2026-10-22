"""JPEG（频域有损）编解码测试：正交归一、重构保真、K 阶单调、梯度。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from flashcodec import JPEGCodec, dct_basis, dct_compress, dct_decompress

torch.manual_seed(0)

B, H, S, D, R = 2, 4, 64, 32, 8
x = torch.randn(B, H, S, D)


def test_basis_orthonormal():
    m = dct_basis(R, R)
    assert torch.allclose(m @ m.t(), torch.eye(R), atol=1e-6), "基底应行正交归一"


def test_lossless_at_k_equals_r():
    coef = dct_compress(x, R, R)
    rec = dct_decompress(coef, R, out_len=S)
    d = (rec - x).abs().max().item()
    assert d < 1e-5, f"keep=r 应无损（fp32 GEMM 往返级）, maxdiff={d:.2e}"


def test_low_freq_fidelity_cosine_gt_0_99():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)  # 自相关信号
    rec = dct_decompress(dct_compress(smooth, R, 2), R, out_len=S)
    cos = torch.nn.functional.cosine_similarity(
        rec.flatten(), smooth.flatten(), dim=0).item()
    assert cos > 0.99, f"r=8,k=2 平滑信号余弦相似度应>0.99, got {cos:.4f}"


def test_k_monotonicity():
    smooth = torch.cumsum(torch.randn(B, H, S, D) * 0.1, dim=-2)
    e2 = (dct_decompress(dct_compress(smooth, R, 2), R, S) - smooth).abs().mean()
    e4 = (dct_decompress(dct_compress(smooth, R, 4), R, S) - smooth).abs().mean()
    assert e4 < e2, f"误差应随 K 单调下降: err2={e2:.4f} err4={e4:.4f}"


def test_padding_and_shapes():
    odd = torch.randn(B, H, 50, D)                     # 非整块序列
    rec = dct_decompress(dct_compress(odd, R, 2), R, out_len=50)
    assert rec.shape == odd.shape, "尾部 padding 应被 out_len 裁掉"
    flat = torch.randn(100, D)                          # 任意前置批次维
    assert dct_decompress(dct_compress(flat, R, 2), R, 100).shape == flat.shape


def test_autograd_differentiable():
    xr = torch.randn(B, H, S, D, requires_grad=True)
    rec = dct_decompress(dct_compress(xr, R, 2), R, out_len=S)
    rec.pow(2).mean().backward()
    assert xr.grad is not None and torch.isfinite(xr.grad).all()


def test_codec_class_api():
    c = JPEGCodec(block_size=R, keep_freqs=2)
    coef = c.compress(x)
    rec = c.decompress(coef, out_len=S)
    assert rec.shape == x.shape
    assert abs(c.storage_ratio() - 0.25) < 1e-9       # k/r = 2/8


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
