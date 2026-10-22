"""flashcodec —— 把时序 KV Cache 张量当作"高维多媒体图像信号"来压缩。

两大经典压缩理论迁移至 Transformer KV Cache 管理：
- mode="jpeg"：正交分块 1D DCT-II，有损频域压缩（低频骨架）；
- mode="png"：因果分块预测差分，无损/近无损时域压缩（锚点 + Δ）。

张量契约：KV 布局 (B, H, S, D)，压缩轴 dim=-2（seq 轴）。
"""

from .cache import CodecKVCache, make_codec
from .codecs import JPEGCodec, PNGCodec, dct_basis, dct_compress, \
    dct_decompress, png_filter, png_reconstruct
from .patch import apply_flash_codec, find_attention_modules, undo_flash_codec

__version__ = "0.1.0"

__all__ = [
    "apply_flash_codec", "undo_flash_codec", "find_attention_modules",
    "CodecKVCache", "make_codec", "JPEGCodec", "PNGCodec",
    "dct_basis", "dct_compress", "dct_decompress",
    "png_filter", "png_reconstruct",
    "__version__",
]
