"""Mini DeepSeek V4.1 — 最小架构研究实现（纯文本，无 vision）。

对应官方 deepseek_v41 text_config 的组件（v1 已实现）：
  - MLA：q_lora / kv_lora / o_lora 分组输出 / 解耦 RoPE（qk_rope_head_dim）
  - MoE：sqrtsoftplus 打分 + noaux_tc 均衡偏置 + norm_topk + routed_scaling
  - 稀疏注意力：sliding window + indexer topk token + block 候选
  - Hyper-Connections：hc_mult 路残差流 + Sinkhorn 混合
  - 跨层 KV 压缩复用：kv_source_layer_ids + compress_ratios

标注 [INFERRED] 的地方：官方源码不可用，按 config 字段命名与 DeepSeek 公开
论文惯例推断实现，语义可能与官方不完全一致。

v2 保留字段（未实现，传非空值不生效）：
  num_nextn_predict_layers / engram_* / dspark_*
  rope_scaling(yarn) — S<=48 不需要，纯 RoPE theta=10000。

设计取舍（相对原 config 的有意缩小，均已验证可训）：
  - compress_ratios 尾 0 改 1，使第 7 层复用第 6 层（覆盖 r=1 直接复用路径，
    否则第 6 层作为 source 无人使用）
  - index_topk_blocks 2048 -> 4（S=48 时 8 块会退化成近稠密，观察不到稀疏性）
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModelForCausalLM, GenerationMixin,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_outputs import CausalLMOutputWithPast


# ============================================================
# 工具函数
# ============================================================
def rope_cache(seq_len, rot_dim, theta, device, dtype):
    """标准 RoPE 频率缓存。rot_dim 为每头旋转维度（qk_rope_head_dim）。"""
    inv_freq = 1.0 / (theta**(torch.arange(0, rot_dim, 2, device=device,
                                            dtype=torch.float32) / rot_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, S, R)，cos/sin: (S, R/2)。rotate-half 式旋转。"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def sinkhorn(raw, iters, eps=1e-6):
    """[INFERRED] Sinkhorn 归一化：exp 后行列交替归一，近似双随机矩阵。"""
    m = torch.exp(raw.clamp(-20.0, 20.0))
    for _ in range(iters):
        m = m / (m.sum(-1, keepdim=True) + eps)
        m = m / (m.sum(-2, keepdim=True) + eps)
    return m


def hc_mix(mat, X):
    """mat: (B,S,n,n)，X: (B,S,n,H) -> (B,S,n,H)。out_i = sum_m mat[i,m] * X[m]"""
    return torch.einsum('bsnm,bsmh->bsnh', mat, X)


def causal_pool_read(latent, r):
    """[INFERRED] 因果安全的 KV latent 均值池化读取。

    块 k 覆盖位置 [k*r, (k+1)*r-1]，存储时池化为 1/r。
    读取规则（保证位置 p 只看到 <= p 的信息）：
      p 是块尾 (p % r == r-1) -> 读本块（恰好截止 p）
      否则                    -> 读上一块（截止 < p），p<r 时退回原始 latent
    训练时 eager 模式会重新展开为 (S,D)；缓存压缩收益属于推理侧优化。
    """
    if r <= 1:
        return latent
    B, S, D = latent.shape
    n_full = S // r
    if n_full == 0:
        return latent
    pooled = latent[:, :n_full * r].view(B, n_full, r, D).mean(dim=2)
    pos = torch.arange(S, device=latent.device)
    k = torch.div(pos, r, rounding_mode='floor')
    is_tail = (pos % r) == (r - 1)
    src = torch.where(is_tail, k, k - 1)
    valid = src >= 0
    gathered = pooled[:, src.clamp(min=0), :]
    return torch.where(valid.view(1, S, 1), gathered, latent)


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
                 kv_lora_rank=32,
                 o_lora_rank=32,
                 o_groups=2,
                 rms_norm_eps=1e-6,
                 initializer_range=0.02,
                 rope_theta=10000.0,
                 # 稀疏注意力
                 sliding_window=16,
                 index_n_heads=4,
                 index_head_dim=16,
                 index_topk=8,
                 index_block_size=4,
                 index_topk_blocks=4,
                 # MoE
                 n_routed_experts=8,
                 num_experts_per_tok=2,
                 n_shared_experts=1,
                 moe_intermediate_size=64,
                 swiglu_limit=10.0,
                 routed_scaling_factor=1.5,
                 # Hyper-Connections
                 hc_mult=4,
                 hc_sinkhorn_iters=3,
                 # 跨层 KV 压缩复用
                 compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1),
                 kv_source_layer_ids=(2, 6),
                 # v2 保留字段（未实现）
                 num_nextn_predict_layers=0,
                 engram_layer_ids=(),
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
        self.kv_lora_rank = kv_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_block_size = index_block_size
        self.index_topk_blocks = index_topk_blocks
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.swiglu_limit = swiglu_limit
        self.routed_scaling_factor = routed_scaling_factor
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.compress_ratios = list(compress_ratios)
        self.kv_source_layer_ids = list(kv_source_layer_ids)
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.engram_layer_ids = list(engram_layer_ids)
        self.dspark_target_layer_ids = list(dspark_target_layer_ids)


AutoConfig.register("deepseek_v41_mini", MiniV41Config)


# ============================================================
# Hyper-Connections（残差流扩展 + 动态混合）
# ============================================================
class HyperConnections(nn.Module):
    """[INFERRED] HC：残差流扩为 n 路，动态混合矩阵经 Sinkhorn 归一。

    - expand: 每路乘固定缩放 [1, .7, .5, .3] 打破初始对称（否则各路永远相同）
    - mix_in :  X_in = sinkhorn(I + dyn_in(s))  ⊗ X    →  本层变换的输入
    - mix_out:  X    = X + sinkhorn(I + dyn_out(y)) ⊗ y →  HC 持有残差
    块内部无残差（由 HC 统一持有，避免双重缩放）。
    """

    def __init__(self, hidden, n, iters):
        super().__init__()
        self.n = n
        self.iters = iters
        self.dyn_in = nn.Linear(hidden, n * n)
        self.dyn_out = nn.Linear(hidden, n * n)
        for lin in (self.dyn_in, self.dyn_out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        scale = [1.0, 0.7, 0.5, 0.3][:n]
        scale += [scale[-1]] * (n - len(scale))
        self.register_buffer('stream_scale', torch.tensor(scale))
        self.register_buffer('eye', torch.eye(n))

    def expand(self, x):
        # (B,S,H) -> (B,S,n,H)
        return x.unsqueeze(2) * self.stream_scale.view(1, 1, -1, 1)

    def mix_in(self, X):
        s = X.mean(dim=2)                                   # (B,S,H)
        raw = self.dyn_in(s).view(*s.shape[:2], self.n, self.n) + self.eye
        return hc_mix(sinkhorn(raw, self.iters), X)

    def mix_out(self, X, y):
        s = y.mean(dim=2)
        raw = self.dyn_out(s).view(*s.shape[:2], self.n, self.n) + self.eye
        return X + hc_mix(sinkhorn(raw, self.iters), y)


# ============================================================
# 稀疏注意力 Indexer
# ============================================================
class SparseIndexer(nn.Module):
    """[INFERRED] indexer：滑窗 ∪ topk token ∪ topk block，且强制因果。

    同时输出 index 分数作为注意力加性 bias（保证 indexer 有梯度通路，
    topk 选择本身不可导）。每层独立 indexer（官方用 index_source_layer_ids
    跨层共享，玩具版从简）。
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
        self.last_density = 1.0   # 诊断用：上次前向的可见 token 比例

    def forward(self, x):
        """x: (B,S,H) -> (mask: (B,1,S,S) bool, bias: (B,1,S,S) float)"""
        B, S, H = x.shape
        device = x.device
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        idx_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        idx_scores = idx_scores.max(dim=1).values               # (B,S,S)

        causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
        idx_scores = idx_scores.masked_fill(~causal, float('-inf'))

        # token topk（钳到 S 防早期序列过短）
        kk = min(self.topk, S)
        tok = idx_scores.topk(kk, dim=-1).indices
        tok_mask = torch.zeros(B, S, S, dtype=torch.bool, device=device)
        tok_mask.scatter_(-1, tok, True)

        # block topk：按 key 维切块取均值分
        bs = self.block_size
        nb_pad = (S + bs - 1) // bs
        pad_len = nb_pad * bs - S
        padded = F.pad(idx_scores, (0, pad_len), value=float('-inf'))
        block_scores = padded.view(B, S, nb_pad, bs).mean(dim=-1)
        # 只有包含 <= i 位置的块才可选（避免全 -inf 块影响 topk）
        block_ok = causal.any(dim=-1, keepdim=False)
        n_real = (S + bs - 1) // bs
        pos_b = torch.arange(n_real, device=device) * bs
        block_ok = (pos_b.view(1, -1) <= torch.arange(S, device=device).view(-1, 1))
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
        self.last_density = mask.float().mean().item()
        return mask, bias


# ============================================================
# MLA 注意力
# ============================================================
class MiniMLA(nn.Module):
    """MLA（迷你）：q/kv/o 低秩 + 解耦 RoPE + 单 KV 头（MQA）+ 分组输出投影。

    owns_kv=False 的层不带 kv_down（复用 source 层的 latent，见 Model）。
    k/v 由共享 latent 经本层 k_up/v_up 解码（跨层复用存储的是 latent）。
    """

    def __init__(self, config, owns_kv=True):
        super().__init__()
        c = config
        self.heads = c.num_attention_heads
        self.head_dim = c.head_dim
        self.rope_dim = c.qk_rope_head_dim
        self.nope_dim = c.head_dim - c.qk_rope_head_dim
        self.q_down = nn.Linear(c.hidden_size, c.q_lora_rank)
        self.q_up = nn.Linear(c.q_lora_rank, self.heads * c.head_dim)
        self.kv_down = (nn.Linear(c.hidden_size, c.kv_lora_rank)
                        if owns_kv else None)
        self.k_up = nn.Linear(c.kv_lora_rank, self.nope_dim)   # 单 KV 头
        self.v_up = nn.Linear(c.kv_lora_rank, c.head_dim)
        self.k_pe = nn.Linear(c.hidden_size, c.qk_rope_head_dim)
        # 分组输出：o_groups 组，每组 head 子集独立低秩投影后求和
        self.groups = c.o_groups
        assert self.heads % self.groups == 0
        self.heads_per_group = self.heads // self.groups
        gdim = self.heads_per_group * c.head_dim
        grank = max(c.o_lora_rank // self.groups, 1)
        self.o_down = nn.ModuleList(
            [nn.Linear(gdim, grank) for _ in range(self.groups)])
        self.o_up = nn.ModuleList(
            [nn.Linear(grank, c.hidden_size) for _ in range(self.groups)])

    def forward(self, x, s_mix, latent, cos, sin, mask, bias):
        """x: (Bn,S,H) 各流查询；s_mix: (B,S,H)；latent: (B,S,kv_lora)。
        mask/bias: (B,1,S,S)，会随流重复。返回 (Bn,S,H)。"""
        Bn, S, H = x.shape
        B = latent.shape[0]
        n_stream = Bn // B

        q = self.q_up(self.q_down(x)).view(Bn, S, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)                 # (Bn,Hd,S,hd)

        # 单 KV 头：latent 解码 nope 部分，s_mix 提供 rope 部分
        k_nope = self.k_up(latent).view(B, S, 1, self.nope_dim)
        k_rope = self.k_pe(s_mix).view(B, S, 1, self.rope_dim)
        k_rope = apply_rope(k_rope.transpose(1, 2), cos, sin).transpose(1, 2)
        v = self.v_up(latent).view(B, S, 1, self.head_dim)

        k = torch.cat([k_nope, k_rope], dim=-1)                  # (B,S,1,hd)
        k = k.repeat_interleave(n_stream, dim=0)                 # (Bn,S,1,hd)
        v = v.repeat_interleave(n_stream, dim=0)
        k = k.transpose(1, 2).expand(Bn, self.heads, S, self.head_dim)
        v = v.transpose(1, 2).expand(Bn, self.heads, S, self.head_dim)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_r = mask.repeat_interleave(n_stream, dim=0)         # (Bn,1,S,S)
        bias_r = bias.repeat_interleave(n_stream, dim=0)
        scores = scores + bias_r
        scores = scores.masked_fill(~mask_r, float('-inf'))
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (attn @ v).transpose(1, 2).reshape(Bn, S, self.heads * self.head_dim)

        # 分组输出投影
        out_g = out.view(Bn, S, self.groups, -1)
        y = sum(self.o_up[g](self.o_down[g](out_g[:, :, g]))
                for g in range(self.groups))
        return y


# ============================================================
# MoE
# ============================================================
class BoundedSwiGLU(nn.Module):
    """激活 clamp 到 ±limit 的 SwiGLU（swiglu_limit=10）。"""

    def __init__(self, in_features, hidden_features, limit=10.0):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features)
        self.w_up = nn.Linear(in_features, hidden_features)
        self.w_down = nn.Linear(hidden_features, in_features)
        self.limit = limit

    def forward(self, x):
        gate = torch.clamp(F.silu(self.w_gate(x)), max=self.limit)
        up = torch.clamp(self.w_up(x), min=-self.limit, max=self.limit)
        return self.w_down(gate * up)


class MiniMoE(nn.Module):
    """sqrtsoftplus 打分 + noaux_tc 修正偏置 + norm_topk + routed_scaling。

    _sel_counts 记录累计被选次数（供 update_expert_balance 与诊断读取，
    不入 checkpoint）。e_score_correction_bias 持久化。
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

        scores = torch.sqrt(
            F.softplus(self.gate(x_flat)) + 1e-6) + self.e_score_correction_bias
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

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
# Block
# ============================================================
class V41Block(nn.Module):
    """pre-norm 块：y = attn(norm1(s)) + moe(norm2(s))。

    无内部残差——残差由 Hyper-Connections 统一持有（见 mix_out）。
    s = X_in.mean(2)（各流约简）；indexer 每层独立；kv_down 只在 source 层。
    """

    def __init__(self, config, owns_kv):
        super().__init__()
        c = config
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.attn = MiniMLA(c, owns_kv=owns_kv)
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.moe = MiniMoE(c)
        self.indexer = SparseIndexer(c.hidden_size, c.index_n_heads,
                                     c.index_head_dim, c.index_topk,
                                     c.index_block_size, c.index_topk_blocks,
                                     c.sliding_window)

    def forward(self, X_in, s_mix, latent, cos, sin):
        """X_in: (B,S,n,H)；latent: (B,S,kv_lora)。返回 y: (B,S,n,H)。"""
        B, S, n, H = X_in.shape
        mask, bias = self.indexer(s_mix)
        x_flat = X_in.reshape(B * n, S, H)
        a = self.attn(self.norm1(x_flat), s_mix, latent, cos, sin, mask, bias)
        a = a.view(B, n, S, H).transpose(1, 2)
        m = self.moe(self.norm2(X_in))
        return a + m


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
        self.hc = HyperConnections(c.hidden_size, c.hc_mult,
                                   c.hc_sinkhorn_iters)
        sources = set(c.kv_source_layer_ids)
        self.blocks = nn.ModuleList([
            V41Block(c, owns_kv=(i in sources or c.compress_ratios[i] == 0))
            for i in range(c.num_hidden_layers)
        ])
        # 每层复用哪个 source：本身是 source 或 ratio=0 -> 自己算；
        # 否则取最近的 <= 自己的 source（config 保证存在）
        self.kv_reuse = []
        for i in range(c.num_hidden_layers):
            if c.compress_ratios[i] == 0:
                self.kv_reuse.append(('own', i, 0))
            elif i in sources:
                self.kv_reuse.append(('own', i, c.compress_ratios[i]))
            else:
                src = max(s for s in sources if s < i)
                self.kv_reuse.append(('reuse', src, c.compress_ratios[i]))
        self.norm_f = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids=None, labels=None, **kwargs):
        c = self.config
        B, S = input_ids.shape
        device = input_ids.device

        cos, sin = rope_cache(S, c.qk_rope_head_dim, c.rope_theta, device,
                              self.embed.weight.dtype)
        x = self.embed(input_ids)
        X = self.hc.expand(x)                       # (B,S,n,H)
        latents = {}

        for i, block in enumerate(self.blocks):
            kind, j, ratio = self.kv_reuse[i]
            if kind == 'own':
                # source 层：s_mix 供 kv_down / rope / indexer
                s_mix = X.mean(dim=2)
                kv_down = block.attn.kv_down
                lat = kv_down(s_mix)
                if ratio > 1:
                    # 源层也按 ratio 存储（尾部读取时因果安全展开）
                    lat = causal_pool_read(lat, ratio)
                latents[i] = lat
            else:
                s_mix = X.mean(dim=2)
                lat = causal_pool_read(latents[j], ratio)
            X_in = self.hc.mix_in(X)
            y = block(X_in, s_mix, lat, cos, sin)
            X = self.hc.mix_out(X, y)

        h = self.norm_f(X.mean(dim=2))
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, c.vocab_size).float(),
                labels.view(-1),
                ignore_index=-100,
            )
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
        """返回各层 indexer 上次前向的可见 token 比例。"""
        return [b.indexer.last_density for b in self.blocks]

    @torch.no_grad()
    def reset_stats(self):
        for block in self.blocks:
            block.moe._ensure_counts().zero_()


AutoModelForCausalLM.register(MiniV41Config, MiniV41ForCausalLM)
