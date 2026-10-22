"""V4.1 最小架构交互式对话 + 每层路由/稀疏诊断。

用法：.venv/bin/python chat_v41.py   （先跑 train_v41.py）
"""

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_v41 import MiniV41ForCausalLM   # 注册 deepseek_v41_mini（必须先于加载）

MODEL_DIR = os.environ.get("MODEL_DIR", "./my_deepseek_v41_mini")
device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")


def print_stats(model):
    usage = model.expert_usage().cpu()
    dens = model.index_density()
    print("\n" + "=" * 65)
    print("📊 \033[1;36mMoE 专家使用 + indexer 稀疏密度\033[0m")
    print("=" * 65)
    for i, row in enumerate(usage):
        bars = " | ".join(
            f"\033[1;32m{v:4.0%}\033[0m" if v > 0.4 else
            (f"\033[0;36m{v:4.0%}\033[0m" if v > 0 else "  ·  ")
            for v in row)
        print(f"  L{i} [E0..E7] {bars}   idx密度 \033[1;35m{dens[i]:.2f}\033[0m")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    print(f"🚀 设备: {device}")
    print(f"📦 加载 {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    print("=" * 65)
    print("🎉 V4.1 最小架构 · 终端对话 + 路由诊断已启动")
    print("💡 输入问题（如 '中国的首都是哪里？'），q 退出")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("User ❯ ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "exit", "quit"):
                print("再见！👋")
                break

            prompt = (user_input if user_input.startswith("问：") and
                      "答：" in user_input else f"问：{user_input} 答：")

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            model.reset_stats()
            print("DeepSeek-V4.1-Mini ❯ ", end="", flush=True)

            gen = []
            state = {}
            with torch.no_grad():
                out = model(input_ids, state=state)      # prefill：全 prompt 一次
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                for _ in range(30):                      # 之后每 token 增量解码
                    if nxt.item() == tokenizer.eos_token_id:
                        break
                    gen.append(nxt.item())
                    sys.stdout.write(tokenizer.decode([nxt.item()],
                                                      skip_special_tokens=True))
                    sys.stdout.flush()
                    time.sleep(0.04)
                    nxt = model.decode_step(nxt, state)[
                        :, -1, :].argmax(dim=-1, keepdim=True)
            print()

            if gen:
                print_stats(model)

        except KeyboardInterrupt:
            print("\n再见！👋")
            break
