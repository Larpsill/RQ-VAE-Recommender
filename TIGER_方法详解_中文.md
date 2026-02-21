# TIGER方法详解 (RQ-VAE Recommender)

## 📚 概述

这个项目是论文 **"Recommender Systems with Generative Retrieval"** (TIGER) 的PyTorch实现。该方法使用基于RQ-VAE的语义ID来构建生成式检索推荐系统。

## 🏗️ 核心思想

TIGER的核心思想是将推荐问题转化为一个**序列生成问题**：

1. **语义ID生成**：使用RQ-VAE将每个物品编码成一个语义ID元组（例如 [c1, c2, c3]）
2. **序列预测**：训练一个Transformer模型，基于用户的历史交互序列，自回归地生成下一个物品的语义ID

## 📁 项目结构与重点文件

```
RQ-VAE-Recommender/
├── train_rqvae.py          ⭐ 第一阶段：训练RQ-VAE tokenizer
├── train_decoder.py        ⭐ 第二阶段：训练推荐模型
├── modules/
│   ├── rqvae.py           ⭐⭐⭐ RQ-VAE核心实现
│   ├── quantize.py        ⭐⭐⭐ 向量量化层实现
│   ├── model.py           ⭐⭐ Decoder推荐模型
│   ├── encoder.py         MLP编码器/解码器
│   ├── loss.py            损失函数定义
│   └── tokenizer/
│       └── semids.py      ⭐⭐ 语义ID Tokenizer
├── data/
│   ├── processed.py       数据集处理
│   ├── schemas.py         数据结构定义
│   └── amazon.py/ml1m.py  具体数据集实现
├── init/
│   └── kmeans.py          ⭐ KMeans初始化码本
├── distributions/
│   └── gumbel.py          ⭐ Gumbel-Softmax采样
└── configs/
    ├── rqvae_amazon.gin   RQ-VAE配置
    └── decoder_amazon.gin Decoder配置
```

## 🔄 完整训练流程

### 阶段一：RQ-VAE训练 (train_rqvae.py)

```
物品特征(768维embedding) 
    ↓ 
Encoder (MLP)
    ↓
latent embedding (32维)
    ↓
┌─────────────────────────────────────┐
│     Residual Quantization (3层)      │
│  ┌─────────────────────────────────┐ │
│  │ Layer 1: emb₁ = quantize(res₀) │ │
│  │ res₁ = res₀ - emb₁             │ │
│  ├─────────────────────────────────┤ │
│  │ Layer 2: emb₂ = quantize(res₁) │ │
│  │ res₂ = res₁ - emb₂             │ │
│  ├─────────────────────────────────┤ │
│  │ Layer 3: emb₃ = quantize(res₂) │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓
语义ID: [id₁, id₂, id₃] (每个id ∈ [0, 255])
量化embedding: emb₁ + emb₂ + emb₃
    ↓
Decoder (MLP)
    ↓
重建物品特征
```

**训练目标**：
- **重建损失**：最小化输入特征与重建特征的差异
- **量化损失**：让encoder输出接近码本向量（commitment loss）

### 阶段二：Decoder训练 (train_decoder.py)

```
用户历史序列: [item₁, item₂, ..., itemₙ]
    ↓
RQ-VAE Tokenizer (冻结)
    ↓
语义ID序列: [[id₁¹, id₂¹, id₃¹], [id₁², id₂², id₃²], ...]
    ↓
Embedding + Position Encoding
    ↓
Transformer Encoder-Decoder
    ↓
自回归生成下一个物品的语义ID: [id₁ⁿ⁺¹, id₂ⁿ⁺¹, id₃ⁿ⁺¹]
```

## 🔑 核心模块详解

### 1. RQ-VAE (modules/rqvae.py)

RQ-VAE是**残差量化变分自编码器**，核心流程：
1. **编码**：将高维物品特征压缩到低维空间
2. **残差量化**：多层量化，每层处理上一层的残差
3. **解码**：从量化后的embedding重建原始特征

### 2. Quantize层 (modules/quantize.py)

单层量化的核心步骤：
1. 计算输入embedding与码本中所有向量的距离
2. 选择最近的码本向量作为量化结果
3. 使用Gumbel-Softmax / STE / Rotation Trick进行可微分训练

### 3. SemanticIdTokenizer (modules/tokenizer/semids.py)

将物品转换为语义ID：
1. 预计算所有物品的语义ID
2. 处理ID冲突（多个物品可能映射到相同ID）
3. 提供前缀匹配验证（用于生成时的约束解码）

### 4. EncoderDecoderRetrievalModel (modules/model.py)

生成式推荐模型：
1. 使用Transformer Encoder处理历史序列上下文
2. 使用Transformer Decoder自回归生成语义ID
3. 支持约束解码（只生成存在的语义ID前缀）

## 🎯 关键技术点

### Gumbel-Softmax重参数化

使离散的量化操作变得可微分，允许梯度反向传播：
```python
# 从Gumbel(0,1)采样并添加到logits
y = logits + sample_gumbel(shape)
# 温度控制的softmax
sample = softmax(y / temperature)
```

### KMeans初始化码本

使用KMeans聚类初始化码本向量，加速收敛：
```python
# 对encoder输出进行KMeans聚类
centroids = kmeans(encoder_outputs, k=codebook_size)
codebook.weight = centroids
```

### 残差量化 (Residual Quantization)

多层量化捕获不同层次的信息：
```python
res = encoder(x)  # 初始残差
for layer in quantize_layers:
    emb = layer(res)      # 量化当前残差
    res = res - emb       # 计算新残差
```

## 💻 运行命令

```bash
# 1. 训练RQ-VAE (生成物品的语义ID)
python train_rqvae.py configs/rqvae_amazon.gin

# 2. 训练Decoder (序列推荐模型)
python train_decoder.py configs/decoder_amazon.gin
```

## 📊 配置参数说明

### RQ-VAE配置 (configs/rqvae_amazon.gin)
```
vae_input_dim=768        # 输入特征维度
vae_embed_dim=32         # 量化embedding维度  
vae_codebook_size=256    # 码本大小(每层ID的取值范围)
vae_n_layers=3           # 量化层数(语义ID的长度)
```

### Decoder配置
```
decoder_embed_dim=64     # Decoder embedding维度
attn_heads=8             # 注意力头数
attn_layers=4            # Transformer层数
```

## 🔗 参考论文

1. [Recommender Systems with Generative Retrieval](https://arxiv.org/pdf/2305.05065) - TIGER原论文
2. [Categorical Reparametrization with Gumbel-Softmax](https://openreview.net/pdf?id=rkE3y85ee) - Gumbel-Softmax技术
3. [Restructuring Vector Quantization with the Rotation Trick](https://arxiv.org/abs/2410.06424) - Rotation Trick优化
