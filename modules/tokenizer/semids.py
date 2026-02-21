"""
================================================================================
语义ID Tokenizer (SemanticIdTokenizer)
================================================================================

这是连接RQ-VAE和Decoder模型的桥梁，负责：
1. 将物品转换为语义ID序列
2. 预计算所有物品的语义ID缓存
3. 处理语义ID冲突（多个物品可能映射到相同ID）
4. 提供前缀匹配验证（用于约束解码）

核心功能示意：
┌─────────────────────────────────────────────────────────────────────┐
│ 预计算阶段 (precompute_corpus_ids):                                  │
│   物品库 [item₁, item₂, ..., itemₙ]                                 │
│         ↓ RQ-VAE                                                     │
│   语义ID缓存: {                                                      │
│       item₁ → [id₁¹, id₂¹, id₃¹, 0],   # 第4位是去重位              │
│       item₂ → [id₁², id₂², id₃², 0],                                │
│       item₃ → [id₁¹, id₂¹, id₃¹, 1],   # 与item₁冲突，去重位=1      │
│       ...                                                            │
│   }                                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 推理阶段 (forward):                                                  │
│   用户序列 [itemₐ, itemᵦ, itemᵧ]                                     │
│         ↓ 查表                                                       │
│   语义ID序列 [[id₁ᵃ, id₂ᵃ, id₃ᵃ, 0], [id₁ᵇ, id₂ᵇ, id₃ᵇ, 0], ...]   │
├─────────────────────────────────────────────────────────────────────┤
│ 约束解码 (exists_prefix):                                            │
│   前缀 [id₁, id₂] → 检查是否存在以此为前缀的物品                      │
│   用于生成时约束只生成有效的语义ID                                   │
└─────────────────────────────────────────────────────────────────────┘
================================================================================
"""

import math
import torch

from data.processed import ItemData
from data.processed import SeqData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.rqvae import RqVae
from typing import List
from typing import Optional
from torch import nn
from torch import Tensor
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler

BATCH_SIZE = 16

class SemanticIdTokenizer(nn.Module):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    ============================================================================
    语义ID Tokenizer - 将物品序列转换为语义ID序列
    ============================================================================
    
    主要功能：
    1. 封装RQ-VAE，提供简洁的tokenize接口
    2. 预计算并缓存所有物品的语义ID（加速推理）
    3. 处理ID冲突：多个物品可能有相同的语义ID，通过添加去重位解决
    4. 提供前缀匹配：验证给定的语义ID前缀是否对应真实物品
    
    为什么需要去重？
    - 码本大小有限（如256³ = 16M种组合）
    - 物品数量可能超过组合数，导致冲突
    - 去重位扩展了有效的ID空间
    """
    def __init__(
        self,
        input_dim: int,           # RQ-VAE输入维度
        output_dim: int,          # RQ-VAE输出维度（embed_dim）
        hidden_dims: List[int],   # MLP隐藏层
        codebook_size: int,       # 码本大小
        n_layers: int = 3,        # 量化层数
        n_cat_feats: int = 18,    # 类别特征数
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,  # 预训练RQ-VAE路径
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False
    ) -> None:
        super().__init__()

        # ========================================================================
        # 内部RQ-VAE模型（用于生成语义ID）
        # ========================================================================
        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,  # 不需要KMeans，因为会加载预训练权重
            codebook_normalize=rqvae_codebook_normalize,
            codebook_sim_vq=rqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
        )
        
        # 加载预训练RQ-VAE权重
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()  # RQ-VAE在第二阶段固定不训练

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        """检查query中的ID是否与key中的ID匹配"""
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        """重置缓存"""
        self.cached_ids = None
    
    @property
    def sem_ids_dim(self):
        """语义ID的维度（包含去重位）"""
        return self.n_layers + 1  # n_layers个量化ID + 1个去重ID
    
    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        """
        ========================================================================
        预计算所有物品的语义ID（核心方法）
        ========================================================================
        
        这是一个重要的预处理步骤：
        1. 遍历所有物品，使用RQ-VAE计算其语义ID
        2. 检测并处理ID冲突（添加去重位）
        3. 将结果缓存起来，后续直接查表
        
        参数：
            movie_dataset: 物品数据集
        
        返回：
            cached_ids: 所有物品的语义ID [n_items, sem_ids_dim]
        
        去重处理：
        - 如果物品A和物品B有相同的语义ID [1, 2, 3]
        - 则A的完整ID变为 [1, 2, 3, 0]
        - B的完整ID变为 [1, 2, 3, 1]
        """
        cached_ids = None
        dedup_dim = []  # 存储每个物品的去重位
        
        # 分批处理物品（避免OOM）
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        
        for batch in dataloader:
            # 使用RQ-VAE获取语义ID
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            # Detect in-batch duplicates 检测批内冲突
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)  # 统计每个ID之前有多少相同的ID
            assert hits.min() >= 0
            
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates 检测批-缓存冲突
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)  # 加上与已缓存ID的冲突数
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids 拼接去重位，形成最终的语义ID
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        """
        ========================================================================
        检查语义ID前缀是否存在于缓存中
        ========================================================================
        
        这个方法用于约束解码（Constrained Decoding）：
        - 在自回归生成时，验证生成的前缀是否对应真实物品
        - 可以过滤掉不存在的语义ID组合，提高生成质量
        
        参数：
            sem_id_prefix: 语义ID前缀 [batch, ..., prefix_len]
        
        返回：
            bool张量，表示每个前缀是否存在
        
        例子：
            前缀 [1, 2] → True（如果存在物品的语义ID以[1,2]开头）
            前缀 [999, 999] → False（不存在这样的物品）
        """
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]  # 缓存中的对应前缀
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # 分批处理避免OOM
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            # 检查是否有任何缓存的前缀与给定前缀匹配
            matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        """从缓存中获取物品的语义ID"""
        return rearrange(self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])
    
    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        """
        ========================================================================
        将物品序列转换为语义ID序列
        ========================================================================
        
        参数：
            batch: 包含物品ID序列的批次数据
        
        返回：
            TokenizedSeqBatch: 包含语义ID序列的批次数据
        
        工作流程：
        1. 如果缓存不存在：直接使用RQ-VAE计算（用于预计算阶段）
        2. 如果缓存存在：从缓存查表获取（用于训练/推理阶段）
        """
        # ----------------------------------------------------------------
        # 情况1：缓存不存在，需要直接计算（预计算阶段）
        # ----------------------------------------------------------------
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids  # 直接用RQ-VAE
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        # ----------------------------------------------------------------
        # 情况2：缓存存在，从缓存查表（训练/推理阶段）
        # ----------------------------------------------------------------
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape  # D = n_layers + 1 (包含去重位)
            
            # 从缓存获取历史序列的语义ID
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            
            # 处理序列mask（每个物品的ID展开为D个token）
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1  # 无效位置设为-1

            # 获取目标物品的语义ID
            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
        
        # Token类型ID：标识每个位置属于语义ID的哪一位
        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,              # 历史序列的语义ID
            sem_ids_fut=sem_ids_fut,      # 目标物品的语义ID
            seq_mask=seq_mask,            # 有效位置mask
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )


if __name__ == "__main__":
    # 测试代码
    dataset = ItemData("dataset/ml-1m-movie")
    tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
    tokenizer.precompute_corpus_ids(dataset)
    
    seq_data = SeqData("dataset/ml-1m")
    batch = seq_data[:10]
    tokenized = tokenizer(batch)
    import pdb; pdb.set_trace()
