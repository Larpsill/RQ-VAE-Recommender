"""
================================================================================
向量量化层 (Vector Quantization Layer) 实现
================================================================================

这是RQ-VAE中每一层量化的核心实现。

核心思想：将连续的向量映射到有限的码本(Codebook)中最接近的离散向量。

问题：离散化操作不可微分，无法反向传播梯度
解决方案：使用以下三种技术之一：
1. Gumbel-Softmax：添加Gumbel噪声，使用软化的softmax近似one-hot
2. STE (Straight-Through Estimator)：前向用离散值，反向直接传梯度
3. Rotation Trick：通过旋转变换保持梯度流动

码本可视化：
┌────────────────────────────────────────────────┐
│ Codebook (码本)：256个32维向量                  │
│                                                 │
│   e₀ = [0.1, -0.2, 0.3, ...]  ← 码本向量0      │
│   e₁ = [0.5, 0.1, -0.4, ...]  ← 码本向量1      │
│   ...                                           │
│   e₂₅₅ = [0.2, 0.4, 0.1, ...]  ← 码本向量255   │
│                                                 │
│ 量化过程：                                      │
│   输入z → 找最近的码本向量eᵢ → 输出id=i和embedding=eᵢ │
└────────────────────────────────────────────────┘
================================================================================
"""

import gin
import torch

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from torch import nn
from torch import Tensor
from torch.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    """量化前向传播模式 - 解决离散化不可微的问题"""
    GUMBEL_SOFTMAX = 1    # Gumbel-Softmax重参数化
    STE = 2               # Straight-Through Estimator
    ROTATION_TRICK = 3    # 旋转技巧（最新方法）


class QuantizeDistance(Enum):
    """距离计算方式"""
    L2 = 1       # L2距离（欧氏距离）
    COSINE = 2   # 余弦相似度


class QuantizeOutput(NamedTuple):
    """量化层输出结构"""
    embeddings: Tensor  # 量化后的embedding（来自码本）
    ids: Tensor         # 选中的码本索引（即语义ID的一个分量）
    loss: Tensor        # 量化损失（commitment loss）


def efficient_rotation_trick_transform(u, q, e):
    """
    ============================================================================
    Rotation Trick变换 - 一种更优雅的梯度传递方法
    ============================================================================
    
    来源：论文 "Restructuring Vector Quantization with the Rotation Trick"
    (https://arxiv.org/abs/2410.06424) Section 4.2
    
    核心思想：
    - 使用Householder反射，将梯度从量化后的向量传递到原始向量
    - 比STE更精确地保持梯度方向
    
    参数：
        u: 归一化的原始输入向量 [batch, embed_dim]
        q: 归一化的量化后向量 [batch, embed_dim]
        e: 原始输入向量（未归一化）[batch, embed_dim]
    
    返回：
        变换后的向量，保持与q相同的值，但梯度可以流回e
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, dim=1, eps=1e-6).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()


class Quantize(nn.Module):
    """
    ============================================================================
    向量量化层 (Vector Quantize Layer)
    ============================================================================
    
    这是RQ-VAE的核心组件，负责将连续向量离散化。
    
    工作流程：
    1. 计算输入向量与所有码本向量的距离
    2. 选择最近的码本向量
    3. 使用可微分技术（Gumbel-Softmax/STE/Rotation）传递梯度
    
    关键概念：
    - 码本(Codebook)：存储n_embed个embed_dim维的可学习向量
    - 量化：找到最近的码本向量，用其代替输入
    - Commitment Loss：让编码器输出接近被选中的码本向量
    """
    def __init__(
        self,
        embed_dim: int,           # 每个码本向量的维度
        n_embed: int,             # 码本大小（可选向量数量）
        do_kmeans_init: bool = True,      # 是否使用KMeans初始化
        codebook_normalize: bool = False,  # 是否L2归一化码本
        sim_vq: bool = False,     # 是否使用SimVQ变体
        commitment_weight: float = 0.25,   # commitment loss权重
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,  # 前向模式
        distance_mode: QuantizeDistance = QuantizeDistance.L2  # 距离计算模式
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        # ========================================================================
        # 码本：nn.Embedding本质上就是一个可学习的矩阵 [n_embed, embed_dim]
        # 每一行代表一个可选的"原型向量"
        # ========================================================================
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        # 输出投影层（可选的归一化或SimVQ变换）
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )

        # 量化损失函数
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        """返回码本权重矩阵"""
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        """权重初始化：均匀分布"""
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad
    def _kmeans_init(self, x) -> None:
        """
        使用KMeans聚类初始化码本
        ========================================================================
        
        为什么使用KMeans初始化？
        - 随机初始化的码本可能导致大部分向量从未被使用（码本崩塌）
        - KMeans确保码本向量均匀分布在数据空间中
        - 加速收敛，提高码本利用率
        """
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        """根据ID从码本中获取对应的向量"""
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature) -> QuantizeOutput:
        """
        ========================================================================
        量化层前向传播
        ========================================================================
        
        核心流程：
        1. 计算输入向量与所有码本向量的距离
        2. 选择距离最近的码本向量（得到离散ID）
        3. 使用可微分技术处理梯度传递
        
        参数：
            x: 输入向量 [batch_size, embed_dim]
            temperature: Gumbel-Softmax温度参数（仅GUMBEL_SOFTMAX模式使用）
        
        返回：
            QuantizeOutput: 包含量化embedding、ID和损失
        """
        assert x.shape[-1] == self.embed_dim

        # Step 0: 首次调用时使用KMeans初始化码本
        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        # 获取（可能经过投影/归一化的）码本
        codebook = self.out_proj(self.embedding.weight)
        
        # Step 1: 计算距离矩阵
        if self.distance_mode == QuantizeDistance.L2:
            # L2距离: ||x - e||² = ||x||² + ||e||² - 2*x·e
            # 这是一种高效的向量化计算方式
            dist = (
                (x**2).sum(axis=1, keepdim=True) +        # ||x||²: [B, 1]
                (codebook.T**2).sum(axis=0, keepdim=True) - # ||e||²: [1, N]
                2 * x @ codebook.T                         # -2*x·e: [B, N]
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            # 余弦相似度（取负作为距离）
            dist = -(
                x / x.norm(dim=1, keepdim=True) @
                (codebook.T) / codebook.T.norm(dim=0, keepdim=True)
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")

        # Step 2: 选择最近的码本向量（硬选择，用于推理和ID生成）
        _, ids = (dist.detach()).min(axis=1)  # ids: [batch_size]

        # Step 3: 训练时使用可微分技术处理梯度
        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                # ----------------------------------------------------------------
                # Gumbel-Softmax：添加Gumbel噪声后做软化的softmax
                # 在低温时近似one-hot，但保持可微分
                # ----------------------------------------------------------------
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                # 软加权：所有码本向量的加权和（权重由softmax得到）
                emb = weights @ codebook
                emb_out = emb
                
            elif self.forward_mode == QuantizeForwardMode.STE:
                # ----------------------------------------------------------------
                # Straight-Through Estimator：
                # 前向用离散值，反向直接传梯度
                # x + (emb - x).detach() 在前向等于emb，但梯度直接流向x
                # ----------------------------------------------------------------
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
                
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                # ----------------------------------------------------------------
                # Rotation Trick：使用Householder变换
                # 比STE更精确地保持梯度方向
                # ----------------------------------------------------------------
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (x.norm(dim=-1, keepdim=True) + 1e-8),    # 归一化输入
                    emb / (emb.norm(dim=-1, keepdim=True) + 1e-8), # 归一化量化结果
                    x  # 原始输入
                )
                # 恢复正确的范数
                emb_out = emb_out * (
                    torch.norm(emb, dim=1, keepdim=True) / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                ).detach()
            else:
                raise Exception("Unsupported Quantize forward mode.")
            
            # 计算量化损失（commitment loss）
            loss = self.quantize_loss(query=x, value=emb)
        
        else:
            # 推理时直接使用硬量化
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss
        )
