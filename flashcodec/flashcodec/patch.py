"""一键热补丁：把 CodecKVCache 注入 HuggingFace 因果 LM 的生成流程。

机制说明（为什么在 cache 层而不是动注意力内核）：
标准 HF 模型的 KV 写入/读取统一收敛在 Cache 接口上——注意力模块计算出
key/value 后调用 ``past_key_value.update(k, v, layer_idx)``，随后从
``key_cache[layer_idx] / value_cache[layer_idx]`` 读回全历史。因此把
Cache 对象换成 CodecKVCache（鸭子类型兼容），即完成"KV 写入与读取逻辑
的替换"，无需针对每种注意力实现（GQA/MQA/RoPE 变体）做模型手术，
对模型结构变化的鲁棒性远好于逐模块包装。

apply_flash_codec 仍会遍历 model.modules() 识别标准 KV 投影注意力模块
（带 k_proj/v_proj 的 LLaMA/Qwen/Mistral 风格）做校验与报告；若模型
未走标准 Cache 接口会给出警告。
"""

import functools
import warnings

from .cache import CodecKVCache, make_codec

__all__ = ["apply_flash_codec", "undo_flash_codec", "find_attention_modules"]


def find_attention_modules(model):
    """识别标准 Self-Attention / KV 投影模块（含 k_proj 与 v_proj）。"""
    return [m for m in model.modules()
            if hasattr(m, "k_proj") and hasattr(m, "v_proj")]


def apply_flash_codec(model, mode="jpeg", block_size=8, **kwargs):
    """一行热补丁：为 model 注入编解码 KV Cache。

    支持 mode="jpeg"（kwargs: keep_freqs，默认 2）与
    mode="png"（kwargs: keep_ratio，默认 1.0）。
    包装 model.forward：当调用方请求 use_cache 且未显式传 past_key_value
    时，自动注入 CodecKVCache。返回 model（原地打补丁，undo_flash_codec
    可还原）。"""
    if getattr(model, "_flashcodec", None) is not None:
        warnings.warn("model 已打过 flashcodec 补丁，先 undo_flash_codec")
        return model

    attns = find_attention_modules(model)
    codec = make_codec(mode, block_size,
                       keep_freqs=kwargs.get("keep_freqs", 2),
                       keep_ratio=kwargs.get("keep_ratio", 1.0))

    def cache_factory():
        return CodecKVCache(mode=mode, block_size=block_size,
                            keep_freqs=kwargs.get("keep_freqs", 2),
                            keep_ratio=kwargs.get("keep_ratio", 1.0))

    base_forward = model.forward

    @functools.wraps(base_forward)
    def wrapped(*args, **kw):
        if kw.get("use_cache") and kw.get("past_key_value") is None:
            kw["past_key_value"] = cache_factory()
        return base_forward(*args, **kw)

    model.forward = wrapped
    model._flashcodec = {
        "mode": mode, "codec": codec,
        "attention_modules": attns, "undo": lambda: _restore(model, base_forward)}
    if not attns:
        warnings.warn("未找到带 k_proj/v_proj 的注意力模块——若模型使用非标准 "
                      "KV 接口，补丁可能不生效")
    return model


def _restore(model, base_forward):
    model.forward = base_forward
    model._flashcodec = None


def undo_flash_codec(model):
    """撤销 apply_flash_codec 打的补丁。"""
    info = getattr(model, "_flashcodec", None)
    if info is not None:
        info["undo"]()
