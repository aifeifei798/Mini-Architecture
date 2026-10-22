"""V4.1 长上下文数据管道：minimind jsonl → token 化打包 .bin。

- pretrain：从 pretrain_t2t_mini.jsonl 流式 reservoir 采样 N_PRETRAIN 条
  {"text"}，逐条 tokenize + eos，串成一条大 token 流，切 SEQ_LEN 窗口，
  末尾 VAL_FRAC 作验证集。全标签 LM（预训练语义，无 prompt masking）。
- sft：从 sft_t2t_mini.jsonl 采样 N_SFT 条多轮对话，转「问：… 答：…」
  模板（与 chat_v41 兼容），同样打包。
- vocab 129280 > 65536，token 用 uint32 存 .bin（numpy memmap 直读）。

用法：.venv/bin/python data_long.py   （读 ./minimind_dataset/*.jsonl，
产出 ./data_long/{train,val,sft_train}.bin + meta.json，全量约 1 分钟）
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

PRETRAIN_JSONL = "./minimind_dataset/pretrain_t2t_mini.jsonl"
SFT_JSONL = "./minimind_dataset/sft_t2t_mini.jsonl"
OUT_DIR = Path("./data_long")
N_PRETRAIN = 100_000        # 采样文档数（~2500 万 token）
N_SFT = 50_000              # 采样对话数
SEQ_LEN = 512
VAL_FRAC = 0.05
SEED = 42


def reservoir_lines(path, n, seed=SEED):
    """流式读 jsonl 前 n 条被 reservoir 采样的行（一遍读完，内存 O(n)）。"""
    rng = random.Random(seed)
    res = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < n:
                res.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    res[j] = line
    return res


def pack_docs(token_ids_list, seq_len, val_frac):
    """token 列表列表 → (train, val) 定长窗口 [seq_len]。"""
    stream = []
    for ids in token_ids_list:
        stream.extend(ids)
    n_val = int(len(stream) * val_frac / seq_len) * seq_len
    n_train = (len(stream) - n_val) // seq_len * seq_len
    train = np.asarray(stream[:n_train], dtype=np.uint32)
    val = np.asarray(stream[n_train:n_train + n_val], dtype=np.uint32)
    return (train.reshape(-1, seq_len), val.reshape(-1, seq_len))


def save_bin(path, arr):
    arr.tofile(path)
    return path, arr.shape


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("./tokenizer")
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer 无 eos_token_id"
    OUT_DIR.mkdir(exist_ok=True)

    # ---- 1. pretrain ----
    print(f"📥 采样 pretrain（{N_PRETRAIN} 条，流式读 {PRETRAIN_JSONL}）...")
    lines = reservoir_lines(PRETRAIN_JSONL, N_PRETRAIN)
    docs = [tok.encode(json.loads(l)["text"]) + [eos] for l in lines]
    train, val = pack_docs(docs, SEQ_LEN, VAL_FRAC)
    save_bin(OUT_DIR / "train.bin", train)
    save_bin(OUT_DIR / "val.bin", val)
    n_tok = (train.size + val.size)
    print(f"✅ pretrain: train {train.shape[0]} × {SEQ_LEN}，"
          f"val {val.shape[0]} × {SEQ_LEN}，共 {n_tok/1e6:.1f}M token")

    # ---- 2. sft ----
    print(f"📥 采样 sft（{N_SFT} 条，流式读 {SFT_JSONL}）...")
    lines = reservoir_lines(SFT_JSONL, N_SFT, seed=SEED + 1)
    docs = []
    for l in lines:
        convs = json.loads(l)["conversations"]
        pairs = [(m["content"], convs[i + 1]["content"])
                 for i, m in enumerate(convs[:-1])
                 if m["role"] == "user" and convs[i + 1]["role"] == "assistant"]
        if not pairs:
            continue
        text = "".join(f"问：{q} 答：{a}\n" for q, a in pairs)
        docs.append(tok.encode(text) + [eos])
    sft_train, sft_val = pack_docs(docs, SEQ_LEN, val_frac=VAL_FRAC)
    save_bin(OUT_DIR / "sft_train.bin", sft_train)
    save_bin(OUT_DIR / "sft_val.bin", sft_val)
    print(f"✅ sft: train {sft_train.shape[0]} × {SEQ_LEN}，"
          f"val {sft_val.shape[0]} × {SEQ_LEN}，共 {(sft_train.size+sft_val.size)/1e6:.1f}M token")

    meta = {"vocab_size": len(tok), "seq_len": SEQ_LEN,
            "dtype": "uint32", "n_train": int(train.shape[0]),
            "n_val": int(val.shape[0]), "n_sft_train": int(sft_train.shape[0]),
            "n_sft_val": int(sft_val.shape[0])}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                  indent=2))
    print(f"📄 meta: {meta}")
