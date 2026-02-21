"""
================================================================================
TIGER 第一阶段：RQ-VAE 训练脚本
================================================================================

这个脚本训练RQ-VAE模型，将物品的高维特征向量编码为离散的语义ID。

训练目标：
1. 最小化重建损失：让解码器能够从语义ID恢复原始特征
2. 最小化量化损失：让编码器输出接近码本向量

使用方法：
    python train_rqvae.py configs/rqvae_amazon.gin

训练完成后：
- 每个物品都可以被编码为一个语义ID元组，如 [123, 45, 67]
- 这些语义ID将在第二阶段用于训练序列推荐模型
================================================================================
"""

import gin
import os
import torch
import numpy as np
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm


@gin.configurable
def train(
    # ========================================================================
    # 训练超参数
    # ========================================================================
    iterations=50000,           # 总训练迭代次数
    batch_size=64,              # 批次大小
    learning_rate=0.0001,       # 学习率
    weight_decay=0.01,          # 权重衰减（L2正则化）
    
    # ========================================================================
    # 数据集配置
    # ========================================================================
    dataset_folder="dataset/ml-1m",   # 数据集根目录
    dataset=RecDataset.ML_1M,         # 数据集类型
    dataset_split="beauty",           # 数据集子集（如amazon的beauty子集）
    
    # ========================================================================
    # 模型保存与加载
    # ========================================================================
    pretrained_rqvae_path=None,  # 预训练RQ-VAE路径（用于继续训练）
    save_dir_root="out/",         # 模型保存目录
    save_model_every=1000000,     # 每N次迭代保存一次模型
    
    # ========================================================================
    # 训练设置
    # ========================================================================
    use_kmeans_init=True,        # 是否使用KMeans初始化码本（强烈建议开启）
    split_batches=True,          # 是否在多GPU间分割批次
    amp=False,                   # 是否使用混合精度训练
    mixed_precision_type="fp16", # 混合精度类型
    gradient_accumulate_every=1, # 梯度累积步数
    
    # ========================================================================
    # 日志与评估
    # ========================================================================
    wandb_logging=False,         # 是否使用WandB记录训练过程
    do_eval=True,                # 是否进行验证
    eval_every=50000,            # 每N次迭代进行一次验证
    force_dataset_process=False, # 是否强制重新处理数据集
    
    # ========================================================================
    # RQ-VAE模型配置 (最重要的超参数!)
    # ========================================================================
    commitment_weight=0.25,           # commitment loss权重
    vae_n_cat_feats=18,               # 类别特征数量
    vae_input_dim=18,                 # 输入特征维度 (如768来自预训练模型)
    vae_embed_dim=16,                 # 量化embedding维度 (压缩后的维度)
    vae_hidden_dims=[18, 18],         # MLP隐藏层维度
    vae_codebook_size=32,             # 码本大小 (每层ID的取值范围，如256表示0-255)
    vae_codebook_normalize=False,     # 是否归一化码本
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,  # 量化模式
    vae_sim_vq=False,                 # 是否使用SimVQ
    vae_n_layers=3                    # 残差量化层数 (决定语义ID长度)
):
    if wandb_logging:
        params = locals()

    # ========================================================================
    # 初始化分布式训练加速器（支持多GPU）
    # ========================================================================
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    # ========================================================================
    # 数据集准备
    # ========================================================================
    # 训练集：物品特征数据
    train_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=force_dataset_process, train_test_split="train" if do_eval else "all", split=dataset_split)
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=None, collate_fn=lambda batch: batch)
    train_dataloader = cycle(train_dataloader)  # 无限循环

    if do_eval:
        eval_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="eval", split=dataset_split)
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=None, collate_fn=lambda batch: batch)

    index_dataset = ItemData(root=dataset_folder, dataset=dataset, force_process=False, train_test_split="all", split=dataset_split) if do_eval else train_dataset
    
    train_dataloader = accelerator.prepare(train_dataloader)

    # ========================================================================
    # 创建RQ-VAE模型
    # ========================================================================
    model = RqVae(
        input_dim=vae_input_dim,           # 输入特征维度（如768）
        embed_dim=vae_embed_dim,           # 量化embedding维度（如32）
        hidden_dims=vae_hidden_dims,       # MLP隐藏层
        codebook_size=vae_codebook_size,   # 码本大小（如256）
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,             # 量化层数（如3，生成3元组语义ID）
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # WandB日志初始化
    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="rq-vae-training",
            config=params
        )

    # 加载预训练权重（如果有）
    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(pretrained_rqvae_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"]+1

    # 分布式准备
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    # ========================================================================
    # Tokenizer（用于生成语义ID，共享RQ-VAE模型）
    # ========================================================================
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer.rq_vae = model

    # ========================================================================
    # 训练主循环
    # ========================================================================
    with tqdm(initial=start_iter, total=start_iter+iterations,
              disable=not accelerator.is_main_process) as pbar:
        losses = [[], [], []]  # [总损失, 重建损失, 量化损失]
        
        for iter in range(start_iter, start_iter+1+iterations):
            model.train()
            total_loss = 0
            t = 0.2  # Gumbel-Softmax温度（固定值，较低的温度使分布更尖锐）
            
            # ----------------------------------------------------------------
            # 首次迭代时执行KMeans初始化
            # 这一步很重要：用训练数据初始化码本，避免码本崩塌
            # ----------------------------------------------------------------
            if iter == 0 and use_kmeans_init:
                kmeans_init_data = batch_to(train_dataset[torch.arange(min(20000, len(train_dataset)))], device)
                model(kmeans_init_data, t)

            optimizer.zero_grad()
            
            # ----------------------------------------------------------------
            # 梯度累积循环
            # ----------------------------------------------------------------
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                with accelerator.autocast():
                    # 前向传播：计算重建损失和量化损失
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            # 反向传播
            accelerator.backward(total_loss)

            # 记录损失（滑动窗口平均）
            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]
            if iter % 100 == 0:
                print_loss = np.mean(losses[0])
                print_rec_loss = np.mean(losses[1])
                print_vae_loss = np.mean(losses[2])

            pbar.set_description(f'loss: {print_loss:.4f}, rl: {print_rec_loss:.4f}, vl: {print_vae_loss:.4f}')

            accelerator.wait_for_everyone()

            # 参数更新
            optimizer.step()
            
            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process and wandb_logging:
                # Compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item() for i in range(vae_n_layers)
                }
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss.cpu().item(),
                    "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                    "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                    **emb_norms_avg_log,
                }

            if do_eval and ((iter+1) % eval_every == 0 or iter+1 == iterations):
                model.eval()
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=True) as pbar_eval:
                    eval_losses = [[], [], []]
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        with torch.no_grad():
                            eval_model_output = model(data, gumbel_t=t)

                        eval_losses[0].append(eval_model_output.loss.cpu().item())
                        eval_losses[1].append(eval_model_output.reconstruction_loss.cpu().item())
                        eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())
                    
                    eval_losses = np.array(eval_losses).mean(axis=-1)
                    id_diversity_log["eval_total_loss"] = eval_losses[0]
                    id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                    id_diversity_log["eval_rqvae_loss"] = eval_losses[2]
                    
            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if (iter+1) % eval_every == 0 or iter+1 == iterations:
                    tokenizer.reset()
                    model.eval()

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:,-1].max() / corpus_ids.shape[0]
                    
                    _, counts = torch.unique(corpus_ids[:,:-1], dim=0, return_counts=True)
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p*torch.log(p)).sum()

                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:,cid], return_counts=True)
                        id_diversity_log[f"codebook_usage_{cid}"] = len(counts) / vae_codebook_size

                    id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                    id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()
                
                if wandb_logging:
                    wandb.log({
                        **train_log,
                        **id_diversity_log
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
