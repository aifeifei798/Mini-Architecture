"""统一编解码 KV Cache 管理器。

设计：
- 每条 KV 流（key/value × layer）内部维护
  ``raw_buffer``（未满一个块的尾部 token，保持原始高保真）与
  ``compressed_storage``（完成块的压缩归档：jpeg 存低频系数 / png 存
  锚点+差分）；
- 一旦累积满 block_size 个 token，触发批量归档压缩（一次 GEMM，无循环）；
- 因果安全：只归档"完整块"，当前未满块永远是 raw——增量解码时任意时刻
  的未来扰动都不会改写已归档张量；
- 双接口：``forward(key_seq, value_seq)`` 全量分块平铺；
  ``decode_step(new_k, new_v)`` 单步追加、跨块边界自动归档；
- 鸭子类型兼容 HuggingFace ``Cache``（update / get_seq_length /
  key_cache / value_cache），可直接注入 generate 流程。

张量契约：KV 布局 (batch, num_heads, seq_len, head_dim)，压缩轴 dim=-2。
"""

import torch

from .codecs import JPEGCodec, PNGCodec, dct_decompress, png_reconstruct


def make_codec(mode="jpeg", block_size=8, keep_freqs=2, keep_ratio=1.0):
    """按模式名构造编解码器实例。"""
    if mode == "jpeg":
        return JPEGCodec(block_size=block_size, keep_freqs=keep_freqs)
    if mode == "png":
        return PNGCodec(block_size=block_size, keep_ratio=keep_ratio)
    raise ValueError(f"未知 mode: {mode!r}（可选 'jpeg' | 'png'）")


class _CodecStream:
    """单条 (B,H,S,D) 流：raw 尾巴 + 压缩归档列表。"""

    def __init__(self, codec):
        self.codec = codec
        self.raw = None          # (B,H,t,D)，t < block_size
        self.archives = []       # jpeg: coeffs (B,H,n_i,k,D)；png: (delta, anchor)
        self.n_archived = 0      # 已归档 token 总数
        self.meta = None         # (B, H, D, elem_size)

    @property
    def block_size(self):
        return self.codec.block_size

    def append(self, tokens):
        """追加 (B,H,m,D)；满块即批量压缩归档（一次批量 GEMM）。"""
        r = self.block_size
        if self.meta is None:
            self.meta = (tokens.shape[0], tokens.shape[1], tokens.shape[-1],
                         tokens.element_size())
        buf = tokens if self.raw is None else torch.cat(
            [self.raw, tokens], dim=-2)
        total = buf.shape[-2]
        n_full = total // r
        if n_full > 0:
            head = buf[..., :n_full * r, :]
            if self.codec.name == "jpeg":
                self.archives.append(self.codec.compress(head))
            else:
                self.archives.append(self.codec.filter(head))
            self.n_archived += n_full * r
        self.raw = (buf[..., n_full * r:, :].contiguous()
                    if total > n_full * r else None)

    def read(self):
        """重构视图 (B,H,S,D)：逐块解压归档 + raw 尾巴拼接。"""
        parts = []
        for ar in self.archives:
            if self.codec.name == "jpeg":
                parts.append(dct_decompress(ar, self.block_size))
            else:
                delta, anchor = ar
                parts.append(png_reconstruct(delta, anchor))
        if self.raw is not None:
            parts.append(self.raw)
        return torch.cat(parts, dim=-2) if parts else None

    def reset(self):
        self.raw = None
        self.archives = []
        self.n_archived = 0

    @property
    def seq_len(self):
        return self.n_archived + (0 if self.raw is None else self.raw.shape[-2])

    def storage_bytes(self):
        total = 0
        for ar in self.archives:
            if self.codec.name == "jpeg":
                total += ar.numel() * ar.element_size()
            else:
                delta, anchor = ar
                total += (delta.numel() * delta.element_size()
                          + anchor.numel() * anchor.element_size())
        if self.raw is not None:
            total += self.raw.numel() * self.raw.element_size()
        return total


class CodecKVCache:
    """编解码 KV Cache（多 layer，HF Cache 鸭子类型）。

    用法::

        cache = CodecKVCache(mode="jpeg", block_size=8, keep_freqs=2)
        k, v = cache.update(k_states, v_states, layer_idx=0)   # 增量
        k, v = cache.forward(k_seq, v_seq, layer_idx=0)        # 全量
    """

    def __init__(self, mode="jpeg", block_size=8, keep_freqs=2,
                 keep_ratio=1.0):
        self.mode = mode
        self.codec = make_codec(mode, block_size, keep_freqs, keep_ratio)
        self._keys = {}
        self._values = {}

    # ---------- 内部 ----------
    def _stream(self, store, layer_idx):
        st = store.get(layer_idx)
        if st is None:
            st = _CodecStream(self.codec)
            store[layer_idx] = st
        return st

    # ---------- 双接口 ----------
    def update(self, key_states, value_states, layer_idx=0):
        """增量自回归接口：追加新 (B,H,m,D)，跨块边界自动压缩归档；
        返回该层重构后的 (key_view, value_view)。"""
        self._stream(self._keys, layer_idx).append(key_states)
        self._stream(self._values, layer_idx).append(value_states)
        return (self._keys[layer_idx].read(),
                self._values[layer_idx].read())

    decode_step = update

    def forward(self, key_seq, value_seq, layer_idx=0):
        """全量前向接口：在序列维度直接分块平铺（重置该层后一次性填充），
        返回重构后的 (key_view, value_view)。"""
        self.reset(layer_idx)
        return self.update(key_seq, value_seq, layer_idx)

    def reset(self, layer_idx=None):
        if layer_idx is None:
            self._keys.clear()
            self._values.clear()
        else:
            self._keys.pop(layer_idx, None)
            self._values.pop(layer_idx, None)

    # ---------- HF Cache 鸭子类型 ----------
    @property
    def key_cache(self):
        mx = max(self._keys, default=-1)
        return [self._keys[i].read() if i in self._keys else None
                for i in range(mx + 1)]

    @property
    def value_cache(self):
        mx = max(self._values, default=-1)
        return [self._values[i].read() if i in self._values else None
                for i in range(mx + 1)]

    def get_seq_length(self, layer_idx=0):
        st = self._keys.get(layer_idx)
        return 0 if st is None else st.seq_len

    def get_max_length(self):
        return None                     # 无上限

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    # ---------- 诊断 ----------
    def storage_bytes(self, layer_idx=None):
        layers = ([layer_idx] if layer_idx is not None
                  else sorted(set(self._keys) | set(self._values)))
        return sum(self._keys[i].storage_bytes() for i in layers if i in self._keys
                   ) + sum(self._values[i].storage_bytes()
                           for i in layers if i in self._values)

    def stats(self, layer_idx=0):
        ks = self._keys.get(layer_idx)
        if ks is None or ks.meta is None:
            return {"tokens": 0, "archived": 0, "raw": 0,
                    "storage_bytes": 0, "raw_bytes": 0}
        B, H, D, elem = ks.meta
        tokens = ks.seq_len
        return {"tokens": tokens, "archived": ks.n_archived,
                "raw": tokens - ks.n_archived,
                "storage_bytes": self.storage_bytes(layer_idx),
                "raw_bytes": tokens * B * H * D * elem * 2}

    def __repr__(self):
        return (f"CodecKVCache(mode={self.mode!r}, codec={self.codec}, "
                f"layers={sorted(set(self._keys) | set(self._values))})")
