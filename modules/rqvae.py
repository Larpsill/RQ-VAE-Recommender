"""
================================================================================
RQ-VAE (Residual Quantization Variational AutoEncoder) 核心实现
================================================================================

这是TIGER方法的第一阶段核心模块，负责将物品特征编码为语义ID。

核心思想：
1. 使用MLP Encoder将高维物品特征压缩到低维空间
2. 使用多层残差量化(Residual Quantization)将连续向量离散化为语义ID
3. 使用MLP Decoder重建原始特征，通过重建损失训练模型

残差量化流程示意：
    输入embedding z → [量化层1] → id₁, emb₁
                      ↓ (残差 r₁ = z - emb₁)
                     [量化层2] → id₂, emb₂  
                      ↓ (残差 r₂ = r₁ - emb₂)
                     [量化层3] → id₃, emb₃
    
    最终语义ID: [id₁, id₂, id₃]
    量化后embedding: emb₁ + emb₂ + emb₃
================================================================================
"""

import torch

from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List
from typing import NamedTuple
from torch import nn
from torch import Tensor

torch.set_float32_matmul_precision('high')


class RqVaeOutput(NamedTuple):
    """RQ-VAE get_semantic_ids方法的输出结构"""
    embeddings: Tensor      # 每层量化后的embedding [batch, embed_dim, n_layers]
    residuals: Tensor       # 每层的残差 [batch, embed_dim, n_layers]
    sem_ids: Tensor         # 语义ID [batch, n_layers]
    quantize_loss: Tensor   # 量化损失（commitment loss）


class RqVaeComputedLosses(NamedTuple):
    """RQ-VAE forward方法的输出结构（包含所有损失）"""
    loss: Tensor              # 总损失 = reconstruction_loss + rqvae_loss
    reconstruction_loss: Tensor  # 重建损失
    rqvae_loss: Tensor        # 量化损失
    embs_norm: Tensor         # embedding范数（用于监控）
    p_unique_ids: Tensor      # 唯一ID比例（用于监控码本利用率）


class RqVae(nn.Module, PyTorchModelHubMixin):
    """
    ============================================================================
    RQ-VAE (Residual Quantization VAE) 主模型类
    ============================================================================
    
    模型架构：
    ┌─────────────────────────────────────────────────────────────┐
    │  输入: 物品特征向量 x (768维，来自预训练embedding)           │
    │                          ↓                                   │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ Encoder (MLP): 768 → 512 → 256 → 128 → 32维             │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                          ↓                                   │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ 残差量化层 (3层，每层码本大小256)                        │ │
    │  │   - Layer 1: 量化原始embedding → id₁                    │ │
    │  │   - Layer 2: 量化残差 → id₂                             │ │
    │  │   - Layer 3: 量化残差 → id₃                             │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                          ↓                                   │
    │  输出: 语义ID [id₁, id₂, id₃] + 量化后的embedding            │
    │                          ↓                                   │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ Decoder (MLP): 32 → 128 → 256 → 512 → 768维             │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                          ↓                                   │
    │  输出: 重建的物品特征 x̂                                      │
    └─────────────────────────────────────────────────────────────┘
    
    训练目标：
    - 重建损失：min ||x - x̂||² (让重建尽可能接近原始输入)
    - 量化损失：commitment loss (让encoder输出接近码本向量)
    """
    def __init__(
        self,
        input_dim: int,           # 输入特征维度（如768，来自预训练模型）
        embed_dim: int,           # 量化embedding维度（如32）
        hidden_dims: List[int],   # MLP隐藏层维度（如[512, 256, 128]）
        codebook_size: int,       # 码本大小（如256，即每层有256个可选向量）
        codebook_kmeans_init: bool = True,    # 是否使用KMeans初始化码本
        codebook_normalize: bool = False,     # 是否L2归一化码本向量
        codebook_sim_vq: bool = False,        # 是否使用SimVQ变体
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,  # 量化前向传播模式
        n_layers: int = 3,        # 残差量化层数（决定语义ID长度）
        commitment_weight: float = 0.25,  # commitment loss的权重
        n_cat_features: int = 18,  # 类别特征数量（用于特殊处理）
    ) -> None:
        self._config = locals()
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        # ========================================================================
        # 核心组件1：多层量化层 (Residual Quantization Layers)
        # 每一层都有独立的码本，用于量化上一层的残差
        # ========================================================================
        self.layers = nn.ModuleList(modules=[
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,          # 码本大小
                forward_mode=codebook_mode,      # Gumbel-Softmax/STE/Rotation Trick
                do_kmeans_init=codebook_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,  # 只对第一层归一化
                sim_vq=codebook_sim_vq,
                commitment_weight=commitment_weight
            ) for i in range(n_layers)
        ])

        # ========================================================================
        # 核心组件2：Encoder - 将高维特征压缩到低维空间
        # ========================================================================
        self.encoder = MLP(
            input_dim=input_dim,        # 768维输入
            hidden_dims=hidden_dims,    # [512, 256, 128]
            out_dim=embed_dim,          # 32维输出
            normalize=codebook_normalize
        )

        # ========================================================================
        # 核心组件3：Decoder - 从量化embedding重建原始特征
        # ========================================================================
        self.decoder = MLP(
            input_dim=embed_dim,              # 32维输入
            hidden_dims=hidden_dims[-1::-1],  # [128, 256, 512] (反向)
            out_dim=input_dim,                # 768维输出
            normalize=True
        )

        # ========================================================================
        # 重建损失函数
        # 如果有类别特征，使用BCE loss；否则使用MSE loss
        # ========================================================================
        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
    
    @cached_property
    def config(self) -> dict:
        return self._config
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        """加载预训练的RQ-VAE权重"""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        """编码：将输入特征映射到低维空间"""
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        """解码：从低维表示重建原始特征"""
        return self.decoder(x)

    def get_semantic_ids(
        self,
        x: Tensor,
        gumbel_t: float = 0.001
    ) -> RqVaeOutput:
        """
        ========================================================================
        核心方法：获取语义ID
        ========================================================================
        
        这是RQ-VAE最核心的方法，实现残差量化流程：
        
        流程：
        1. 将输入编码到低维空间
        2. 对编码结果进行多层残差量化
        3. 每层输出一个离散的语义ID
        
        参数：
            x: 输入特征 [batch_size, input_dim]
            gumbel_t: Gumbel-Softmax的温度参数（越低越接近硬采样）
        
        返回：
            语义ID、量化embedding、残差、量化损失
        """
        # Step 1: 编码 - 将高维输入压缩到低维空间
        res = self.encode(x)  # res: [batch_size, embed_dim] 初始残差就是编码输出
        
        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        # Step 2: 多层残差量化 - 每层处理上一层的残差
        for layer in self.layers:
            residuals.append(res)  # 保存当前残差
            
            # 量化当前残差，得到最接近的码本向量
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss  # 累加量化损失
            
            emb, id = quantized.embeddings, quantized.ids
            
            # 计算新残差 = 当前残差 - 量化后的embedding
            # 这就是"残差量化"的核心：每层只需要量化上一层未能表示的部分
            res = res - emb
            
            sem_ids.append(id)   # 保存语义ID
            embs.append(emb)     # 保存量化后的embedding

        # embs是list，被视为 [n_layers, batch_size, embed_dim]
        # rearrange模式中: b=n_layers, h=batch_size, d=embed_dim
        # "b h d -> h d b" 变换后: [batch_size, embed_dim, n_layers]
        return RqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),       # [batch, embed_dim, n_layers]
            residuals=rearrange(residuals, "b h d -> h d b"),   # [batch, embed_dim, n_layers]
            sem_ids=rearrange(sem_ids, "b d -> d b"),           # [batch, n_layers]
            quantize_loss=quantize_loss
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        """
        ========================================================================
        前向传播 - 计算所有损失
        ========================================================================
        
        完整的RQ-VAE前向流程：
        1. 获取语义ID和量化embedding
        2. 解码重建输入
        3. 计算重建损失和量化损失
        
        参数：
            batch: 包含输入特征x的批次数据
            gumbel_t: Gumbel-Softmax温度
        
        返回：
            总损失、重建损失、量化损失等
        """
        x = batch.x
        
        # Step 1: 获取语义ID和量化embedding
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        
        # Step 2: 解码重建 - 将所有层的量化embedding相加后解码
        # embs.sum(axis=-1) 是所有层量化embedding的和
        # 因为之前后续层的embedding都是在逼近残差，所以求和以逼近x
        x_hat = self.decode(embs.sum(axis=-1))
        
        # L2归一化连续特征部分（类别特征保持不变）
        x_hat = torch.cat([l2norm(x_hat[...,:-self.n_cat_feats]), x_hat[...,-self.n_cat_feats:]], axis=-1)

        # Step 3: 计算损失
        # 重建损失：输入与重建之间的差异
        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        # 量化损失：encoder输出与码本向量之间的差异
        rqvae_loss = quantized.quantize_loss
        # 总损失 = 重建损失 + 量化损失
        loss = (reconstuction_loss + rqvae_loss).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            # embedding范数
            embs_norm = embs.norm(dim=1)
            # 唯一ID比例 - 用于监控码本利用率，越高说明利用越充分
            p_unique_ids = (~torch.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]
            # 两两关系 / 总数，需要两个item的sem_ids完全相同才算重复

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids
        )
