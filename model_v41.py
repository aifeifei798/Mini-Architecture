"""Mini DeepSeek V4.1 — 最小架构研究实现（纯文本，无 vision）。

v2（本轮）：用 transformers 5.20.2 内置的官方 deepseek_v4 源码逐组件校准，
并实现 V4.1 独有字段中可落地的部分。

官方对照结论（transformers/models/deepseek_v4/ 为 V4.0，与 V4.1 共享组件）：
  - Hyper-Connections 官方叫 mHC：每个子层（attn/ffn）各一个模块，单个 fn
    参数产 pre/post/comb 三组权重；pre 在模块内收拢流、post 缩放子层输出、
    comb（softmax + Sinkhorn 行列归一双随机）混合旧流，消费方向为转置
    （new_k = Σ_j comb[j,k]·old_j）。流初始为 embedding 的 N 份拷贝（无缩放），
    末端由 HyperHead（只含 pre 型权重）收拢。v2 完全按此重写。
  - 注意力：单 KV 头直接投影 hidden→head_dim（**无 kv_lora 低秩**——V4 弃用了
    V3 的 MLA 压缩），K=V 共用张量；部分 RoPE 只旋转头部尾部 rope 维
    （[nope|rope] 布局、interleaved 相邻成对配对、fp32 计算）；输出对 rope 维
    做共轭旋转（-sin）；每头可学习 attention sink（进 softmax 分母、不进加权和）。
  - o_groups：分组低秩输出投影——每组独立投影到 o_lora_rank（官方该字段是
    每组的秩），拼接后单个 o_b 线性混合回 hidden。
  - Router：选择用 scores+bias，权重取**未加偏置**的原始分数归一化后乘
    routed_scaling_factor。
  - SwiGLU clamp：作用在激活前的 gate(max=limit) 与 up(±limit)。
  - indexer 打分官方为 ∑_h w_h·ReLU(q·K)·d^-0.5，w=weights_proj(x)·H^-0.5
    （可学习逐头权重）。v2 按此替换 v1 的 max-pool 打分。
  - compress_rope_theta：官方有 main/compress 两套 rope，滑窗层用
    main(θ=10000)，压缩层与 indexer 用 compress(θ=160000)。mini 映射：
    KV 存储层 ratio>0 用 compress，否则 main；复用层 q 跟随 source 的存储
    类型（保证 q·k 同 θ）。
  - 官方 V4.0 的 compress_rates dict + layer_types（CSA/HCA 层内压缩自身
    KV）与 V4.1 的 compress_ratios + kv_source_layer_ids（跨层 KV 复用）是
    两套机制；后者官方无源码，仍为 [INFERRED]。v2 细化：source 存 raw roped
    KV，各层按自身 ratio 池化读取（只池化一层，修正 v1 的双重池化）。

V4.1 独有字段实现（官方亦无源码，[INFERRED]）：
  - MTP（num_nextn_predict_layers）：训练期辅助损失——fuse(h_t, emb(t+1))
    经 1 个轻量块（单头注意力 + SwiGLU）用共享 lm_head 预测 t+2，
    loss 权重 mtp_loss_coef。生成时不启用（labels=None），不影响主路径。
  - engram（engram_layer_ids）：n-gram（1..max_ngram）多项式哈希查表记忆，
    加权求和经零初始化投影注入 hidden。官方 3.84 亿行大表缩为
    engram_table_size 行。
  - dspark_*：仍为占位字段（未实现）。它是推理侧块推测解码头（noise token
    起步、Markov 低秩转移、独立 MoE），S<=48 的训练研究价值低且语义推断
    空间最大，留 v3。

设计取舍（相对原 config 的有意缩小，均已验证可训）：
  - compress_ratios 尾部 3 个 0 改 1，使第 7 层复用第 6 层
  - candidate_topk_blocks 2048 -> 4（S=48 时大 topk 退化成近稠密）
  - hc_sinkhorn_iters 保留官方 20（矩阵只有 4x4，开销可忽略）
  - rope_scaling(yarn) 未实现——S<=48 不需要，纯 RoPE
  - 玩具式 eager 实现，非官方缓存/压缩器流水线
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, GenerationMixin,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast


# ============================================================
# 工具函数（与官方对齐）
# ============================================================
def rope_cache(seq_len, rot_dim, theta, device, dtype):
    """RoPE 半尺寸频率缓存（interleaved：每个交错对一个 θ）。"""
    inv_freq = 1.0 / (theta**(torch.arange(0, rot_dim, 2, device=device,
                                            dtype=torch.float32) / rot_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def rotate_half(x):
    """interleaved 旋转：相邻两通道成对 (x0,x1) -> (-x1,x0)。"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    """x: (B,H,S,hd)，头部布局 [nope|rope]，只旋转尾部 rope 维（官方 V4 语义）。
    cos/sin: (S, rope_dim/2)，先 repeat_interleave 展开再旋转，fp32 计算。"""
    cos = cos.repeat_interleave(2, dim=-1)[None, None]
    sin = sin.repeat_interleave(2, dim=-1)[None, None]
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def unweighted_rms_norm(x, eps):
    """官方 mHC 的输入归一化：无权重 RMSNorm。"""
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)


def sinkhorn_stochastic(m, iters, eps):
    """官方 mHC Sinkhorn：输入已是正权重（softmax 后），列归一化起步，
    之后行/列交替 iters-1 轮，收敛到近似双随机。"""
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def causal_pool_read(latent, r):
    """[INFERRED] 因果安全的 KV 池化读取（r>1 时）。

    块 k 覆盖位置 [k*r, (k+1)*r-1]。读取规则（位置 p 只看到 <= p 的信息）：
      p 是块尾 (p % r == r-1) -> 读本块（恰好截止 p）
      否则                    -> 读上一块（截止 < p），p<r 时退回原始 latent
    """
    if r <= 1:
        return latent
    B, S = latent.shape[:2]
    n_full = S // r
    if n_full == 0:
        return latent
    pooled = latent[:, :n_full * r].reshape(B, n_full, r, -1).mean(dim=2)
    pooled = pooled.view(B, n_full, *latent.shape[2:])
    pos = torch.arange(S, device=latent.device)
    k = torch.div(pos, r, rounding_mode='floor')
    is_tail = (pos % r) == (r - 1)
    src = torch.where(is_tail, k, k - 1)
    valid = src >= 0
    gathered = pooled[:, src.clamp(min=0)]
    return torch.where(valid.view(1, S, *([1] * (latent.dim() - 2))), gathered,
                       latent)


# ============================================================
# Config
# ============================================================
class MiniV41Config(PretrainedConfig):
    model_type = "deepseek_v41_mini"

    def __init__(self,
                 vocab_size=129280,
                 hidden_size=128,
                 num_hidden_layers=8,
                 num_attention_heads=4,
                 num_key_value_heads=1,
                 head_dim=32,
                 qk_rope_head_dim=8,
                 q_lora_rank=32,
                 o_lora_rank=16,          # 官方语义：每组的秩（非总秩）
                 o_groups=2,
                 rms_norm_eps=1e-6,
                 initializer_range=0.02,
                 rope_theta=10000.0,
                 compress_rope_theta=16000.0,   # 官方 160000，mini 等比缩小
                 # 稀疏注意力
                 sliding_window=16,
                 index_n_heads=4,
                 index_head_dim=16,
                 index_topk=8,
                 candidate_source_layer_id=2,
                 candidate_block_size=4,
                 candidate_topk_blocks=4,
                 # MoE
                 n_routed_experts=8,
                 num_experts_per_tok=2,
                 n_shared_experts=1,
                 moe_intermediate_size=64,
                 swiglu_limit=10.0,
                 routed_scaling_factor=1.5,
                 # Hyper-Connections（mHC）
                 hc_mult=4,
                 hc_sinkhorn_iters=20,    # 官方值（4x4 矩阵，开销可忽略）
                 hc_eps=1e-6,
                 # 跨层 KV 压缩复用 [INFERRED]
                 compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1),
                 kv_source_layer_ids=(2, 6),
                 index_source_layer_ids=(2, 6),
                 # MTP [INFERRED]（mini 用 1 层，官方 3）
                 num_nextn_predict_layers=1,
                 mtp_loss_coef=0.3,
                 # engram [INFERRED]（mini 缩小版）
                 engram_layer_ids=(1, 3),
                 engram_max_ngram_size=4,
                 engram_vocab_size=4096,        # 哈希空间（官方 16000000）
                 engram_table_size=1024,        # 每层表行数（官方 ~3.84e8）
                 # 保留占位（未实现）
                 dspark_block_size=5,
                 dspark_target_layer_ids=(),
                 **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.sliding_window = sliding_window
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_block_size = candidate_block_size
        self.candidate_topk_blocks = candidate_topk_blocks
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.swiglu_limit = swiglu_limit
        self.routed_scaling_factor = routed_scaling_factor
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.compress_ratios = list(compress_ratios)
        self.kv_source_layer_ids = list(kv_source_layer_ids)
        self.index_source_layer_ids = list(index_source_layer_ids)
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_coef = mtp_loss_coef
        self.engram_layer_ids = list(engram_layer_ids)
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_table_size = engram_table_size
        self.dspark_block_size = dspark_block_size
        self.dspark_target_layer_ids = list(dspark_target_layer_ids)


AutoConfig.register("deepseek_v41_mini", MiniV41Config)


# ============================================================
# mHC（官方语义：pre 收拢 / post 缩放 / comb 双随机混合）
# ============================================================
class HyperConnection(nn.Module):
    """官方 mHC 子模块（每个 attn/ffn 子层各一个）。

    单个 fn 参数 ((2+N)*N, N*hidden) 从 N 路流（摊平+无权重 RMSNorm）线性
    产出三组权重：
      pre  (N,)  : sigmoid(w*pre_scale + base) + eps —— 模块内收拢流
      post (N,)  : 2*sigmoid(...) —— 子层输出的逐流缩放
      comb (N,N) : softmax(...) + eps，经 Sinkhorn 行列归一成双随机 —— 旧流混合
    comb 以转置方向消费：new_k = Σ_j comb[j,k] * old_j（对第一根流轴求和）。
    权重 fp32 计算（官方 _keep_in_fp32_modules 含 attn_hc）。
    """

    def __init__(self, hidden, n, iters, eps):
        super().__init__()
        self.n = n
        self.iters = iters
        self.eps = eps
        self.fn = nn.Parameter(torch.empty((2 + n) * n, n * hidden))
        self.base = nn.Parameter(torch.empty((2 + n) * n))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, streams):
        """streams: (B,S,N,H) -> (post (B,S,N), comb (B,S,N,N), collapsed (B,S,H))"""
        B, S = streams.shape[:2]
        n = self.n
        flat = unweighted_rms_norm(streams.flatten(2), self.eps)
        w = F.linear(flat.float(), self.fn.float())
        pre_w, post_w, comb_w = w.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([n, n, n * n])
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)

        comb_w = comb_w.view(B, S, n, n)
        comb_b = comb_b.view(n, n)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.eps
        post = 2.0 * torch.sigmoid(post_w * post_scale + post_b)
        comb = torch.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.eps
        comb = sinkhorn_stochastic(comb, self.iters, self.eps)

        collapsed = (pre.unsqueeze(-1) * streams.float()).sum(dim=2)
        return post, comb, collapsed.to(streams.dtype)


class HyperHead(nn.Module):
    """官方末端收拢（HyperHead）：只有 pre 型权重，把 N 路流折成 1 路。"""

    def __init__(self, hidden, n, eps):
        super().__init__()
        self.n = n
        self.eps = eps
        self.fn = nn.Parameter(torch.empty(n, n * hidden))
        self.base = nn.Parameter(torch.empty(n))
        self.scale = nn.Parameter(torch.empty(1))

    def forward(self, streams):
        flat = unweighted_rms_norm(streams.flatten(2), self.eps)
        w = F.linear(flat.float(), self.fn.float())
        pre = torch.sigmoid(w * self.scale.float()
                            + self.base.float()) + self.eps
        return (pre.unsqueeze(-1) * streams.float()).sum(dim=2).to(streams.dtype)


# ============================================================
# 稀疏注意力 Indexer（官方 Lightning-Indexer 打分 + 玩具版选择结构）
# ============================================================
class SparseIndexer(nn.Module):
    """打分对齐官方：∑_h w_h · ReLU(q·K)·d^-0.5，w=weights_proj(x)·H^-0.5。
    选择结构保持玩具版：滑窗 ∪ topk token ∪ topk block（V4.1 块候选字段
    candidate_* 语义 [INFERRED]），强制因果。

    同时输出打分作为注意力加性 bias（topk 不可导，bias 保证梯度通路）。
    模块按 index_source_layer_ids 跨层共享（source 层持有参数，其余层引用）。
    last_density 由 Block 记录，避免共享时相互覆盖。
    """

    def __init__(self, hidden, n_heads, head_dim, topk, block_size, topk_blocks,
                 window):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.topk = topk
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.window = window
        self.q_proj = nn.Linear(hidden, n_heads * head_dim)
        self.k_proj = nn.Linear(hidden, n_heads * head_dim)
        self.weights_proj = nn.Linear(hidden, n_heads)
        self.softmax_scale = head_dim ** -0.5
        self.weights_scaling = n_heads ** -0.5

    def forward(self, x):
        """x: (B,S,H) -> (mask (B,1,S,S) bool, bias (B,1,S,S) float)"""
        B, S, H = x.shape
        device = x.device
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # 官方打分：ReLU 逐头内积 + 可学习逐头权重求和
        scores = F.relu(q.float() @ k.float().transpose(-2, -1)) * self.softmax_scale
        weights = self.weights_proj(x).float() * self.weights_scaling  # (B,S,H)
        weights = weights.transpose(1, 2).unsqueeze(-1)                # (B,H,S,1)
        idx_scores = (scores * weights).sum(dim=1)                     # (B,S,S)

        causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
        idx_scores = idx_scores.masked_fill(~causal, float('-inf'))

        # token topk（钳到 S 防早期序列过短）
        kk = min(self.topk, S)
        tok = idx_scores.topk(kk, dim=-1).indices
        tok_mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        tok_mask.scatter_(-1, tok, True)

        # block topk：按 key 维切块取均值分
        bs = self.block_size
        n_real = (S + bs - 1) // bs
        pad_len = n_real * bs - S
        padded = F.pad(idx_scores, (0, pad_len), value=float('-inf'))
        block_scores = padded.view(B, S, n_real, bs).mean(dim=-1)
        # 只有包含 <= i 位置的块才可选
        pos_b = torch.arange(n_real, device=device) * bs
        block_ok = pos_b.view(1, -1) <= torch.arange(S, device=device).view(-1, 1)
        block_scores = block_scores.masked_fill(~block_ok, float('-inf'))
        kb = min(self.topk_blocks, n_real)
        blk = block_scores.topk(kb, dim=-1).indices
        blk_mask = torch.zeros(B, S, n_real, dtype=torch.bool, device=device)
        blk_mask.scatter_(-1, blk, True)
        blk_mask = blk_mask.unsqueeze(-1).repeat(1, 1, 1, bs)
        blk_mask = blk_mask.view(B, S, n_real * bs)[..., :S]

        # 滑窗
        pos = torch.arange(S, device=device)
        diff = pos.view(1, -1) - pos.view(-1, 1)                 # i - j
        win_mask = (diff >= 0) & (diff < self.window)

        mask = (win_mask | tok_mask | blk_mask) & causal
        mask = mask.unsqueeze(1)                                  # (B,1,S,S)
        bias = idx_scores.masked_fill(~causal, 0.0).clamp(-10.0, 10.0)
        bias = bias.unsqueeze(1)
        return mask, bias


# ============================================================
# 注意力（官方 V4 语义：单头 K=V 直投 + 部分 interleaved RoPE +
# 共轭旋转 + attention sink + 分组低秩输出）
# ============================================================
class MiniMLA(nn.Module):
    """V4 注意力（迷你）：
    - q 低秩（q_lora_rank），q_a_norm/q_b_norm 双归一
    - kv 直接投影 hidden→head_dim（单头，无 kv_lora——V4 弃用 V3 的 MLA 低秩），
      K=V 共用；RoPE 只旋转尾部 rope 维，输出做共轭旋转
    - 每头可学习 attention sink：softmax 分母多一列 exp(sink)，不参与加权和
    - o 分组低秩：每组投影到 o_lora_rank，拼接后 o_b 混合回 hidden
    - owns_kv=False 的层不带 kv_proj，复用 source 层 KV（跨层复用 [INFERRED]）
    """

    def __init__(self, config, owns_kv=True):
        super().__init__()
        c = config
        self.heads = c.num_attention_heads
        self.head_dim = c.head_dim
        self.q_down = nn.Linear(c.hidden_size, c.q_lora_rank)
        self.q_a_norm = nn.RMSNorm(c.q_lora_rank, eps=c.rms_norm_eps)
        self.q_up = nn.Linear(c.q_lora_rank, self.heads * c.head_dim)
        self.q_b_norm = nn.RMSNorm(c.head_dim, eps=c.rms_norm_eps)
        self.kv_proj = (nn.Linear(c.hidden_size, c.head_dim)
                        if owns_kv else None)
        self.kv_norm = nn.RMSNorm(c.head_dim, eps=c.rms_norm_eps)
        self.sinks = nn.Parameter(torch.zeros(self.heads))
        # 分组输出投影：每组 -> o_lora_rank，拼接 -> o_b -> hidden（官方结构）
        self.groups = c.o_groups
        assert self.heads % self.groups == 0
        gdim = (self.heads // self.groups) * c.head_dim
        self.o_down = nn.ModuleList(
            [nn.Linear(gdim, c.o_lora_rank) for _ in range(self.groups)])
        self.o_b = nn.Linear(c.o_groups * c.o_lora_rank, c.hidden_size)

    def project_kv(self, x, cos, sin):
        """source 层调用：(B,S,H) -> roped KV (B,S,1,hd)（K=V 共用）。"""
        kv = self.kv_norm(self.kv_proj(x)).view(*x.shape[:-1], 1, self.head_dim)
        return apply_rope(kv.transpose(1, 2), cos, sin).transpose(1, 2)

    def forward(self, x, kv, cos, sin, mask, bias):
        """x: (B,S,H) 收拢后的查询流；kv: (B,S,1,hd) 已含 rope 的共享 KV。
        mask/bias: (B,1,S,S)。返回 (B,S,H)。"""
        B, S, H = x.shape

        q = self.q_up(self.q_a_norm(self.q_down(x)))
        q = q.view(B, S, self.heads, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rope(q, cos, sin)                              # (B,Hd,S,hd)

        k = kv.transpose(1, 2).expand(B, self.heads, S, self.head_dim)
        v = k                                                     # K=V

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + bias
        scores = scores.masked_fill(~mask, float('-inf'))
        # attention sink：分母多一列 exp(sink)，该质量不进入输出
        sink_col = self.sinks.view(1, self.heads, 1, 1).expand(
            B, self.heads, S, 1)
        attn = F.softmax(torch.cat([scores.float(), sink_col.float()],
                                   dim=-1), dim=-1)
        out = (attn[..., :-1] @ v.float()).to(q.dtype)            # (B,Hd,S,hd)

        # 共轭旋转：V 带 key 侧 rope，在查询位置转回来（官方 -sin）
        out = apply_rope(out, cos, -sin)
        out = out.transpose(1, 2).reshape(B, S, self.heads * self.head_dim)

        # 分组低秩输出投影
        out_g = out.view(B, S, self.groups, -1)
        parts = [self.o_down[g](out_g[:, :, g]) for g in range(self.groups)]
        return self.o_b(torch.cat(parts, dim=-1))


# ============================================================
# MoE
# ============================================================
class BoundedSwiGLU(nn.Module):
    """官方语义：clamp 在激活前——gate clamp(max=limit)，up clamp(±limit)。"""

    def __init__(self, in_features, hidden_features, limit=10.0):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features)
        self.w_up = nn.Linear(in_features, hidden_features)
        self.w_down = nn.Linear(hidden_features, in_features)
        self.limit = limit

    def forward(self, x):
        gate = self.w_gate(x).clamp(max=self.limit)
        up = self.w_up(x).clamp(min=-self.limit, max=self.limit)
        return self.w_down(F.silu(gate) * up)


class MiniMoE(nn.Module):
    """sqrtsoftplus 打分 + noaux_tc 修正偏置 + norm_topk + routed_scaling。
    官方语义修正：bias 只参与选择，权重取未加偏置的原始分数。
    _sel_counts 记录累计被选次数（诊断用，不入 checkpoint）。
    """

    def __init__(self, config):
        super().__init__()
        c = config
        self.n_experts = c.n_routed_experts
        self.top_k = c.num_experts_per_tok
        self.scale = c.routed_scaling_factor
        self.gate = nn.Linear(c.hidden_size, c.n_routed_experts)
        self.experts = nn.ModuleList([
            BoundedSwiGLU(c.hidden_size, c.moe_intermediate_size,
                          c.swiglu_limit) for _ in range(c.n_routed_experts)
        ])
        self.shared_expert = BoundedSwiGLU(
            c.hidden_size, c.moe_intermediate_size * c.n_shared_experts,
            c.swiglu_limit)
        self.register_buffer('e_score_correction_bias',
                             torch.zeros(c.n_routed_experts))
        self._sel_counts = torch.zeros(c.n_routed_experts)

    def _ensure_counts(self):
        """from_pretrained 在 meta 设备构造时 _sel_counts 会是 meta 张量，
        这里懒重建为真实 CPU 张量。"""
        cnt = getattr(self, "_sel_counts", None)
        if cnt is None or cnt.is_meta:
            cnt = torch.zeros(self.n_experts)
            self._sel_counts = cnt
        return cnt

    def forward(self, x):
        shape = x.shape
        x_flat = x.view(-1, shape[-1])

        scores = torch.sqrt(F.softplus(self.gate(x_flat)) + 1e-6)
        # 选择用 scores+bias；权重用原始 scores（官方 TopKRouter 语义）
        indices = torch.topk(scores + self.e_score_correction_bias,
                             self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        with torch.no_grad():
            counts = self._ensure_counts()
            cnt = torch.zeros(self.n_experts, device=x.device)
            cnt.scatter_add_(0, indices.view(-1),
                             torch.ones_like(indices, dtype=cnt.dtype).view(-1))
            counts += cnt.cpu()

        routed = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(self.n_experts):
                m = indices[:, k] == e
                if m.any():
                    routed[m] += self.experts[e](x_flat[m]) * weights[m, k, None]

        out = routed * self.scale + self.shared_expert(x_flat)
        return out.view(shape)

    @torch.no_grad()
    def update_balance_bias(self, lr=0.01, target=None):
        """[INFERRED] noaux_tc：按被选频次与均匀目标的偏差修正偏置。
        与 DeepSeek V3 的无辅助损失负载均衡思路一致（简化实现）。"""
        if target is None:
            target = 1.0 / self.n_experts
        counts = self._ensure_counts()
        total = counts.sum()
        if total < 1:
            return
        freq = counts / total
        self.e_score_correction_bias.add_(
            (freq - target).to(self.e_score_correction_bias.device) * lr)
        counts.zero_()

    def usage(self):
        counts = self._ensure_counts()
        total = counts.sum()
        if total < 1:
            return torch.zeros(self.n_experts)
        return counts / total


# ============================================================
# Engram 记忆（n-gram 哈希查表注入）[INFERRED]
# ============================================================
class EngramMemory(nn.Module):
    """对每个位置，取结尾 1..max_ngram 的 n-gram 做多项式哈希 -> 查本层小表，
    按可学习权重求和，经零初始化投影注入 hidden（起点为零注入，训练中长出）。
    哈希只用 <= 当前位置的 token（不足长度的位置以 0 填充前缀），因果安全。
    官方是 3.84 亿行大表 + 多头读取；这里缩为 engram_table_size 行加性注入。
    """

    def __init__(self, hidden, max_ngram, hash_mod, table_size):
        super().__init__()
        self.max_ngram = max_ngram
        self.hash_mod = hash_mod
        self.tables = nn.ModuleList(
            [nn.Embedding(table_size, hidden) for _ in range(max_ngram)])
        self.mix = nn.Parameter(torch.zeros(max_ngram))     # 各 n-gram 权重
        self.proj = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.proj.weight)                    # 零注入起步
        nn.init.zeros_(self.proj.bias)

    def forward(self, input_ids):
        """input_ids: (B,S) -> delta (B,S,H)"""
        B, S = input_ids.shape
        device = input_ids.device
        prime = 1000003
        w = torch.softmax(self.mix, dim=0)
        read = None
        for n in range(1, self.max_ngram + 1):
            # 结尾 n-gram 的多项式哈希：h[p] = Σ_i input_ids[p-n+i]·prime^(n-1-i)
            # 前 n-1 个位置用 0 填充（退化 gram，确定性且无未来信息）
            h = torch.zeros(B, S, dtype=torch.long, device=device)
            for i in range(n):
                L = max(S - (n - 1 - i), 0)
                shifted = F.pad(input_ids[:, :L], (S - L, 0), value=0)
                h = (h * prime + shifted) % self.hash_mod
            idx = h % self.tables[n - 1].num_embeddings
            emb_n = w[n - 1] * self.tables[n - 1](idx)
            read = emb_n if read is None else read + emb_n
        return self.proj(read)


# ============================================================
# MTP 模块（训练期辅助预测 t+2）[INFERRED]
# ============================================================
class MTPModule(nn.Module):
    """DeepSeek 式 MTP 迷你版：fuse(h_t, emb(t+1)) -> 1 个轻量块（单头注意力 +
    SwiGLU）-> 共享 lm_head 预测 t+2。只在 labels 存在时参与训练损失。"""

    def __init__(self, config):
        super().__init__()
        c = config
        self.fuse = nn.Linear(2 * c.hidden_size, c.hidden_size)
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.q = nn.Linear(c.hidden_size, c.hidden_size)
        self.k = nn.Linear(c.hidden_size, c.hidden_size)
        self.v = nn.Linear(c.hidden_size, c.hidden_size)
        self.o = nn.Linear(c.hidden_size, c.hidden_size)
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.ffn = BoundedSwiGLU(c.hidden_size, c.hidden_size,
                                 c.swiglu_limit)
        self.norm3 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)

    def forward(self, h, emb_next, cos, sin, lm_head):
        """h: (B,S,H) 最终 hidden；emb_next: (B,S,H) t+1 真值 embedding。
        返回 (B,S,V) logits。"""
        x = self.fuse(torch.cat([h, emb_next], dim=-1))
        B, S, H = x.shape
        res = x
        x = self.norm1(x)
        q = self.q(x).view(B, S, 1, H).transpose(1, 2)
        k = self.k(x).view(B, S, 1, H).transpose(1, 2)
        v = self.v(x).view(B, S, 1, H).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        causal = torch.ones(S, S, dtype=torch.bool,
                            device=x.device).tril().view(1, 1, S, S)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(H)
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        x = res + self.o((attn @ v).squeeze(1))
        x = x + self.ffn(self.norm2(x))
        return lm_head(self.norm3(x))


# ============================================================
# Block
# ============================================================
class V41Block(nn.Module):
    """官方结构：attn_hc 收拢流 -> attn -> post/comb 混合 -> ffn_hc -> moe ->
    混合。块内部无残差（mHC 统一持有）。own 层在块内算 KV 并写入 latents；
    reuse 层从 latents 读 source 的 KV 按自身 ratio 池化。indexer 跨层共享。
    """

    def __init__(self, config, kv_plan, indexer):
        super().__init__()
        c = config
        self.kv_plan = kv_plan                    # ('own', i, r) | ('reuse', j, r)
        self.attn_hc = HyperConnection(c.hidden_size, c.hc_mult,
                                       c.hc_sinkhorn_iters, c.hc_eps)
        self.ffn_hc = HyperConnection(c.hidden_size, c.hc_mult,
                                      c.hc_sinkhorn_iters, c.hc_eps)
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.attn = MiniMLA(c, owns_kv=(kv_plan[0] == 'own'))
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.moe = MiniMoE(c)
        # 共享 indexer 不在此注册（否则 state_dict 出现重复张量），由模型顶层
        # self.indexers 唯一注册，这里仅持引用
        self._indexer_ref = [indexer]
        self.last_density = 1.0   # 诊断：上次前向可见 token 比例

    def forward(self, X, cos, sin, cos_kv, sin_kv, latents):
        """X: (B,S,N,H) 流；cos/sin 为本层 q 的 rope；cos_kv/sin_kv 为 KV 存储
        时的 rope 类型；latents: dict，own 层写入 raw KV。返回新流 (B,S,N,H)。"""
        B, S, N, H = X.shape
        kind, j, ratio = self.kv_plan

        post, comb, collapsed = self.attn_hc(X)
        if kind == 'own':
            raw_kv = self.attn.project_kv(collapsed, cos_kv, sin_kv)
            latents[j] = raw_kv
            kv = causal_pool_read(raw_kv, ratio)
        else:
            kv = causal_pool_read(latents[j], ratio)

        mask, bias = self._indexer_ref[0](collapsed)
        self.last_density = mask.float().mean().item()
        a = self.attn(self.norm1(collapsed), kv, cos, sin, mask, bias)
        X = (post.to(X.dtype).unsqueeze(-1) * a.unsqueeze(-2)
             + torch.matmul(comb.to(X.dtype).transpose(-1, -2), X))

        post, comb, collapsed = self.ffn_hc(X)
        m = self.moe(self.norm2(collapsed))
        X = (post.to(X.dtype).unsqueeze(-1) * m.unsqueeze(-2)
             + torch.matmul(comb.to(X.dtype).transpose(-1, -2), X))
        return X


# ============================================================
# Model
# ============================================================
class MiniV41ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniV41Config
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        c = config
        assert len(c.compress_ratios) == c.num_hidden_layers
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)

        # KV 复用规划 [INFERRED]：own 层自己算，reuse 层取最近 source
        sources = set(c.kv_source_layer_ids)
        kv_plans = []
        for i in range(c.num_hidden_layers):
            r = c.compress_ratios[i]
            if r == 0 or i in sources:
                kv_plans.append(('own', i, r))
            else:
                kv_plans.append(('reuse', max(s for s in sources if s < i), r))
        # rope 类型跟随 KV 存储层的 ratio（复用层 q 与 source 存储同 θ，保证
        # q·k 同一 rope 族；官方语义为滑窗层 main、压缩层 compress）
        self.rope_is_compress = [
            (r if kind == 'own' else c.compress_ratios[j]) > 0
            for kind, j, r in kv_plans]

        # indexer 跨层共享：ratio=0 或 index source 层持有模块，其余层引用
        # 最近的持有层（index_source_layer_ids 语义 [INFERRED]）
        idx_sources = set(c.index_source_layer_ids)
        built = {}
        for i in range(c.num_hidden_layers):
            if i in idx_sources or c.compress_ratios[i] == 0:
                built[i] = SparseIndexer(
                    c.hidden_size, c.index_n_heads, c.index_head_dim,
                    c.index_topk, c.candidate_block_size,
                    c.candidate_topk_blocks, c.sliding_window)
        indexer_of = [built[i] if i in built
                      else built[max(s for s in built if s < i)]
                      for i in range(c.num_hidden_layers)]
        # 唯一注册点（跨层共享的 indexer 只在此注册一份，block 内持引用，
        # 避免 state_dict 重复张量）
        self.indexers = nn.ModuleList([built[i] for i in sorted(built)])

        self.blocks = nn.ModuleList([
            V41Block(c, kv_plans[i], indexer_of[i])
            for i in range(c.num_hidden_layers)
        ])

        self.hc_head = HyperHead(c.hidden_size, c.hc_mult, c.hc_eps)
        self.norm_f = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size)

        # MTP / Engram
        self.mtp = (MTPModule(c) if c.num_nextn_predict_layers > 0 else None)
        self.engrams = nn.ModuleList([
            EngramMemory(c.hidden_size, c.engram_max_ngram_size,
                         c.engram_vocab_size, c.engram_table_size)
            for _ in range(len(c.engram_layer_ids))])
        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (HyperConnection, HyperHead)):
            nn.init.normal_(module.fn, mean=0.0, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)

    def forward(self, input_ids=None, labels=None, **kwargs):
        c = self.config
        B, S = input_ids.shape
        device = input_ids.device
        dtype = self.embed.weight.dtype

        # 两套 rope 缓存：main / compress（官方双 rope 语义）
        caches = {name: rope_cache(S, c.qk_rope_head_dim, theta, device, dtype)
                  for name, theta in (("main", c.rope_theta),
                                      ("compress", c.compress_rope_theta))}

        x = self.embed(input_ids)
        engram_map = dict(zip(c.engram_layer_ids, self.engrams))
        X = x.unsqueeze(2).expand(-1, -1, c.hc_mult, -1).contiguous()

        latents = {}
        for i, block in enumerate(self.blocks):
            name = "compress" if self.rope_is_compress[i] else "main"
            cos, sin = caches[name]
            if i in engram_map:
                X = X + engram_map[i](input_ids).unsqueeze(2)
            X = block(X, cos, sin, *caches[name], latents)

        h = self.norm_f(self.hc_head(X))
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, c.vocab_size).float(),
                labels.view(-1),
                ignore_index=-100,
            )
            # MTP 辅助损失：fuse(h_t, emb(t+1)) 预测 t+2（S>=2 才有 shift 目标）
            if self.mtp is not None and S >= 2:
                next_lab = labels.masked_fill(labels == -100, 0)
                emb_next = self.embed(next_lab)
                mtp_logits = self.mtp(h, emb_next, *caches["main"],
                                      self.lm_head)
                mtp_loss = F.cross_entropy(
                    mtp_logits[:, :-1].reshape(-1, c.vocab_size).float(),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss = loss + c.mtp_loss_coef * mtp_loss
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @torch.no_grad()
    def update_expert_balance(self, lr=0.01):
        for block in self.blocks:
            block.moe.update_balance_bias(lr=lr)

    @torch.no_grad()
    def expert_usage(self):
        """返回 (num_layers, n_experts) 各层专家使用频率。"""
        return torch.stack([b.moe.usage() for b in self.blocks])

    @torch.no_grad()
    def index_density(self):
        """返回各层上次前向的可见 token 比例（共享 indexer 也不互相覆盖）。"""
        return [b.last_density for b in self.blocks]

    @torch.no_grad()
    def reset_stats(self):
        for block in self.blocks:
            block.moe._ensure_counts().zero_()


AutoModelForCausalLM.register(MiniV41Config, MiniV41ForCausalLM)
