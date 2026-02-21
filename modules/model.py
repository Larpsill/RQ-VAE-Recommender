"""
================================================================================
生成式推荐模型 (Generative Retrieval Model) 实现
================================================================================

这是TIGER的第二阶段模型，使用Transformer架构基于用户历史序列生成下一个物品的语义ID。

核心思想：
1. 将推荐问题建模为序列到序列的生成问题
2. Encoder处理用户历史序列上下文
3. Decoder自回归地生成目标物品的语义ID

模型架构：
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EncoderDecoderRetrievalModel                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  输入: 用户历史语义ID序列                                                    │
│  [[id₁¹, id₂¹, id₃¹, d¹], [id₁², id₂², id₃², d²], ...]                      │
│                           ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Embedding层:                                                            │ │
│  │   - 语义ID Embedding: 将每个ID映射为向量                               │ │
│  │   - 位置Embedding (WPE): 编码token位置                                  │ │
│  │   - Token类型Embedding (TTE): 区分语义ID的不同分量                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Transformer Encoder:                                                    │ │
│  │   处理历史序列，生成上下文表示                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           ↓                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ Transformer Decoder (自回归):                                           │ │
│  │   基于上下文，逐位生成目标语义ID                                        │ │
│  │   [BOS] → id₁ → id₂ → id₃ → dedup_id                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                           ↓                                                  │
│  输出: 下一个物品的语义ID预测                                               │
└─────────────────────────────────────────────────────────────────────────────┘

训练目标：
- 使用交叉熵损失，最大化正确语义ID的预测概率
- 损失 = -log P(id₁|context) - log P(id₂|context,id₁) - ...

推理过程：
- 使用约束解码，只生成有效的语义ID组合
- 通过前缀匹配验证，确保生成的ID对应真实物品
================================================================================
"""

import gin
import torch

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F

# 允许torch.compile继续（即使有小错误）
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


class ModelOutput(NamedTuple):
    """模型输出结构"""
    loss: Tensor      # 交叉熵损失
    logits: Tensor    # 预测logits [batch, seq_len, vocab_size]
    loss_d: Tensor    # 按位置分解的损失（用于分析）


class GenerationOutput(NamedTuple):
    """生成输出结构"""
    sem_ids: Tensor      # 生成的语义ID
    log_probas: Tensor   # 对应的log概率


class EncoderDecoderRetrievalModel(nn.Module):
    """
    ============================================================================
    Encoder-Decoder 生成式推荐模型
    ============================================================================
    
    这是TIGER第二阶段的核心模型，实现基于Transformer的序列推荐。
    
    模型特点：
    1. 使用Encoder处理用户历史序列
    2. 使用Decoder自回归生成目标语义ID
    3. 支持约束解码（只生成有效的语义ID）
    4. 支持jagged tensor优化（处理变长序列）
    
    输入输出：
    - 输入：用户历史物品的语义ID序列
    - 输出：下一个物品的语义ID预测分布
    """
    def __init__(
        self,
        embedding_dim,         # Embedding维度
        attn_dim,              # 注意力维度
        dropout,               # Dropout概率
        num_heads,             # 注意力头数
        n_layers,              # Transformer总层数（Encoder和Decoder各一半）
        num_embeddings,        # 词表大小（= 码本大小）
        sem_id_dim,            # 语义ID维度（n_layers + 1）
        inference_verifier_fn, # 约束解码验证函数
        max_pos=2048,          # 最大位置编码长度
        jagged_mode: bool = True,  # 是否使用jagged tensor优化
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        # ========================================================================
        # Embedding层
        # ========================================================================
        # BOS (Beginning of Sequence) embedding
        self.bos_emb = nn.Parameter(torch.rand(embedding_dim))
        
        # 归一化层
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        # 语义ID Embedder：将语义ID映射为向量
        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        
        # 用户ID Embedder（用于个性化）
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        # 位置编码 (WPE - Word Position Embedding)
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        
        # Token类型编码 (TTE - Token Type Embedding)
        # 区分语义ID的不同分量（第1位、第2位、第3位、去重位）
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        # ========================================================================
        # Transformer架构
        # ========================================================================
        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,   # Encoder层数
            decoder_layers=n_layers // 2    # Decoder层数
        ) if self.jagged_mode else nn.Transformer(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )

        # 投影层
        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias=False)  # 输出到词表空间
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        """
        ========================================================================
        模型前向预测核心逻辑
        ========================================================================
        
        处理流程：
        1. 将语义ID序列转换为embedding
        2. 添加位置编码和token类型编码
        3. Encoder处理历史序列上下文
        4. Decoder自回归生成目标语义ID
        
        参数：
            batch: 包含语义ID序列的批次数据
        
        返回：
            Transformer输出
        """
        # 获取用户embedding
        user_emb = self.user_id_embedder(batch.user_ids)
        
        # 获取语义ID的embedding
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut  # 历史序列和目标序列
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
        # pos = torch.arange(pos_max, device=batch.sem_ids.device).repeat_interleave(self.sem_id_dim)

        # ====================================================================
        # 构建Encoder输入：用户embedding + 历史序列embedding + 位置编码
        # ====================================================================
        pos = torch.arange(N, device=sem_ids_emb.device).unsqueeze(0)
        wpe = self.wpe(pos)  # 位置编码

        # 拼接用户embedding和历史序列embedding
        input_embedding = torch.cat([user_emb, wpe + sem_ids_emb], axis=1)
        
        # ====================================================================
        # 构建Decoder输入：BOS + 目标序列embedding（用于teacher forcing）
        # ====================================================================
        input_embedding_fut = self.bos_emb.repeat(B, 1, 1)  # BOS token
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)  # Token类型编码
            input_embedding_fut = torch.cat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        # ====================================================================
        # 处理变长序列（jagged mode优化）
        # ====================================================================
        if self.jagged_mode:
            # 转换为jagged tensor格式（去除padding，提高效率）
            input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

            seq_lengths_fut = torch.tensor(input_embedding_fut.shape[1], device=input_embedding_fut.device, dtype=torch.int64).repeat(B)
            input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
        else:
            # 标准padding mask处理
            mem_mask = torch.cat([
                torch.ones(B, 1, dtype=torch.bool, device=batch.seq_mask.device),
                batch.seq_mask
            ], axis=1)
            f_mask = torch.zeros_like(mem_mask, dtype=torch.float32)
            f_mask[~mem_mask] = float("-inf")
        
        # ====================================================================
        # 投影并通过Transformer
        # ====================================================================
        # 上下文（Encoder输入）
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        # Decoder输入
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        
        if self.jagged_mode:
            transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        else:
            # 标准Transformer处理
            causal_mask = nn.Transformer.generate_square_subsequent_mask(transformer_input.shape[1])
            transformer_output = self.transformer(src=transformer_context, tgt=transformer_input, tgt_is_causal=True, tgt_mask=causal_mask, src_key_padding_mask=f_mask, memory_key_padding_mask=f_mask)

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @torch.no_grad
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        """
        ========================================================================
        自回归生成下一个物品的语义ID（推理阶段使用）
        ========================================================================
        
        这是约束解码的核心实现：
        1. 逐位生成语义ID
        2. 每一步验证生成的前缀是否有效
        3. 保留top-k个有效候选
        
        参数：
            batch: 包含用户历史的批次数据
            temperature: 采样温度（越高越随机）
            top_k: 是否使用top-k采样
        
        返回：
            生成的语义ID和对应的log概率
        
        约束解码流程：
        1. 生成第1位id₁，验证[id₁]是否是有效前缀
        2. 生成第2位id₂，验证[id₁, id₂]是否是有效前缀
        3. ... 直到生成完整的语义ID
        4. 只保留对应真实物品的语义ID
        """
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated, log_probas = None, 0
        k = 64 if top_k else 1                    # 保留的候选数
        n_top_k_candidates = 256 if top_k else 1  # 初始采样数

        # 准备输入批次（不包含目标序列）
        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        # ====================================================================
        # 逐位生成语义ID
        # ====================================================================
        for i in range(self.sem_id_dim):
            # 获取当前位置的预测分布
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, dim=-1)
            
            # 多项式采样
            samples_batched = torch.multinomial(probas_batched, num_samples=n_top_k_candidates)

            # ----------------------------------------------------------------
            # 约束验证：检查生成的前缀是否对应真实物品
            # ----------------------------------------------------------------
            if generated is None:
                # 第一位：验证单个ID
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                # 后续位：验证完整前缀
                prefix = torch.cat([generated.flatten(0,1).unsqueeze(1).repeat_interleave(n_top_k_candidates, axis=1), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            sampled_log_probas = torch.log(torch.gather(probas_batched, 1, samples_batched)).reshape(B, -1)
            samples = samples_batched.reshape(B, -1)

            # ----------------------------------------------------------------
            # 选择top-K个有效候选（无效候选的概率被设为极小值）
            # ----------------------------------------------------------------
            sorted_log_probas, sorted_indices = (
                -10000*(~is_valid_prefix) +  # 无效前缀的惩罚 ⭐
                sampled_log_probas +
                maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            ).sort(-1, descending=True)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = torch.gather(samples, 1, top_k_indices)
            
            if generated is not None:
                parent_id = torch.gather(generated, 1, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand(-1,-1,i))
                top_k_samples = torch.cat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.arange(next_sem_ids.shape[1], device=next_sem_ids.device).repeat(next_sem_ids.shape[0], 1),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = torch.clone(top_k_samples.detach())
                log_probas = torch.clone(top_k_log_probas.detach())
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)
                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions 
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    cache = torch.zeros(input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim, device=input_batch.sem_ids.device)
                    cache_mask = torch.cat([torch.ones(input_batch.sem_ids.shape[0], 1, dtype=bool, device=input_batch.seq_mask.device), input_batch.seq_mask], axis=1)
                    cache[cache_mask] = self.transformer.cached_enc_output.values()
                    lengths = self.transformer.cached_enc_output.offsets().diff().repeat_interleave(k)
                    cache = cache.repeat_interleave(k, dim=0)
                    self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids.repeat_interleave(k, dim=0),
                    sem_ids=input_batch.sem_ids.repeat_interleave(k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=torch.zeros_like(next_sem_ids),
                    seq_mask=input_batch.seq_mask.repeat_interleave(k, dim=0),
                    token_type_ids=input_batch.token_type_ids.repeat_interleave(k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = torch.clone(top_k_log_probas.detach())
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    @torch.compile
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(out, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
