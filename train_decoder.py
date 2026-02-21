"""
================================================================================
TIGER 第二阶段：Decoder (序列推荐模型) 训练脚本
================================================================================

这个脚本训练生成式推荐模型，基于用户的历史交互序列预测下一个物品的语义ID。

前置条件：
- 需要先完成第一阶段的RQ-VAE训练
- RQ-VAE用于将物品转换为语义ID

训练目标：
- 学习用户行为序列到下一个物品语义ID的映射
- 使用交叉熵损失训练Transformer模型

使用方法：
    python train_decoder.py configs/decoder_amazon.gin

推理流程：
1. 获取用户历史序列 [item₁, item₂, ..., itemₙ]
2. 使用RQ-VAE将每个物品转换为语义ID
3. Transformer自回归生成下一个物品的语义ID
4. 根据语义ID检索对应的物品
================================================================================
"""

import argparse
import os
import gin
import torch
import wandb

from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
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
    iterations=500000,          # 总训练迭代次数（比RQ-VAE更长）
    batch_size=64,              # 批次大小
    learning_rate=0.001,        # 学习率
    weight_decay=0.01,          # 权重衰减
    
    # ========================================================================
    # 数据集配置
    # ========================================================================
    dataset_folder="dataset/ml-1m",
    dataset=RecDataset.ML_1M,
    dataset_split="beauty",
    
    # ========================================================================
    # 模型路径配置
    # ========================================================================
    save_dir_root="out/",
    pretrained_rqvae_path=None,     # ⭐ 预训练RQ-VAE路径（必须提供！）
    pretrained_decoder_path=None,    # 预训练Decoder路径（用于继续训练）
    
    # ========================================================================
    # 训练设置
    # ========================================================================
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,    # 部分评估频率
    full_eval_every=10000,      # 完整评估频率
    
    # ========================================================================
    # RQ-VAE配置（必须与第一阶段一致！）
    # ========================================================================
    vae_input_dim=18,           # 输入特征维度
    vae_embed_dim=16,           # 量化embedding维度
    vae_hidden_dims=[18, 18],   # MLP隐藏层
    vae_codebook_size=32,       # 码本大小
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,             # 语义ID长度
    
    # ========================================================================
    # Decoder Transformer配置
    # ========================================================================
    decoder_embed_dim=64,       # Decoder embedding维度
    dropout_p=0.1,              # Dropout概率
    attn_heads=8,               # 注意力头数
    attn_embed_dim=64,          # 注意力维度
    attn_layers=4,              # Transformer层数
    
    # ========================================================================
    # 其他配置
    # ========================================================================
    wandb_logging=False,
    force_dataset_process=False,
    push_vae_to_hf=False,           # 是否推送到HuggingFace
    train_data_subsample=True,      # 是否对训练数据采样
    model_jagged_mode=True,         # 是否使用jagged tensor优化
    vae_hf_model_name="edobotta/rqvae-amazon-beauty"
):  
    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    # ========================================================================
    # 初始化分布式训练加速器
    # ========================================================================
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else 'no'
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        wandb.login()
        run = wandb.init(
            project="gen-retrieval-decoder-training",
            config=params
        )
    
    # ========================================================================
    # 数据集准备
    # ========================================================================
    # 物品数据：用于RQ-VAE预计算语义ID
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split
    )
    
    # 训练集：用户行为序列数据
    train_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=True, 
        subsample=train_data_subsample,  # 可选采样加速训练
        split=dataset_split
    )
    
    # 验证集：用户行为序列数据
    eval_dataset = SeqData(
        root=dataset_folder, 
        dataset=dataset, 
        is_train=False, 
        subsample=False, 
        split=dataset_split
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader = cycle(train_dataloader)  # 无限循环
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

    # ========================================================================
    # 创建语义ID Tokenizer（使用预训练的RQ-VAE）
    # ========================================================================
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,  # ⭐ 加载预训练RQ-VAE
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq
    )
    tokenizer = accelerator.prepare(tokenizer)
    
    # ⭐ 预计算所有物品的语义ID（推理时用于快速查找）
    tokenizer.precompute_corpus_ids(item_dataset)
    
    # 可选：将RQ-VAE推送到HuggingFace Hub
    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    # ========================================================================
    # 创建生成式推荐模型（Encoder-Decoder Transformer）
    # ========================================================================
    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,  # 词表大小 = 码本大小
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),  # 约束解码验证函数
        sem_id_dim=tokenizer.sem_ids_dim,  # 语义ID维度 (n_layers + 1)
        max_pos=train_dataset.max_seq_len*tokenizer.sem_ids_dim,  # 最大位置
        jagged_mode=model_jagged_mode
    )

    # ========================================================================
    # 优化器和学习率调度器
    # ========================================================================
    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 逆平方根学习率调度（带warmup）
    lr_scheduler = InverseSquareRootScheduler(
        optimizer=optimizer,
        warmup_steps=10000
    )
    
    # 加载预训练Decoder（如果有）
    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(pretrained_decoder_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    # 评估指标累加器
    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}")
    
    # ========================================================================
    # 训练主循环
    # ========================================================================
    with tqdm(initial=start_iter, total=start_iter + iterations,
              disable=not accelerator.is_main_process) as pbar:
        for iter in range(iterations):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            
            # ----------------------------------------------------------------
            # 梯度累积循环
            # ----------------------------------------------------------------
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                
                # ⭐ 使用Tokenizer将物品序列转换为语义ID序列
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    # 前向传播：预测下一个语义ID
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every
                    total_loss += loss
                
                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data, model_output)

                # 反向传播
                accelerator.backward(total_loss)
                assert model.sem_id_embedder.emb.weight.grad is not None

            pbar.set_description(f'loss: {total_loss.item():.4f}')

            accelerator.wait_for_everyone()

            # 参数更新
            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter+1) % partial_eval_every == 0:
                model.eval()
                model.enable_generation = False
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        model_output_eval = model(tokenized_data)

                    if wandb_logging and accelerator.is_main_process:
                        eval_debug_metrics = compute_debug_metrics(tokenized_data, model_output_eval, "eval")
                        eval_debug_metrics["eval_loss"] = model_output_eval.loss.detach().cpu().item()
                        wandb.log(eval_debug_metrics)

            if (iter+1) % full_eval_every == 0:
                model.eval()
                model.enable_generation = True
                with tqdm(eval_dataloader, desc=f'Eval {iter+1}', disable=not accelerator.is_main_process) as pbar_eval:
                    for batch in pbar_eval:
                        data = batch_to(batch, device)
                        tokenized_data = tokenizer(data)

                        generated = model.generate_next_sem_id(tokenized_data, top_k=True, temperature=1)
                        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids

                        metrics_accumulator.accumulate(actual=actual, top_k=top_k)

                        if accelerator.is_main_process and wandb_logging:
                            wandb.log(eval_debug_metrics)
                
                eval_metrics = metrics_accumulator.reduce()
                
                print(eval_metrics)
                if accelerator.is_main_process and wandb_logging:
                    wandb.log(eval_metrics)
                
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter+1) % save_model_every == 0 or iter+1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict()
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    torch.save(state, save_dir_root + f"checkpoint_{iter}.pt")
                
                if wandb_logging:
                    wandb.log({
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss.cpu().item(),
                        **train_debug_metrics
                    })

            pbar.update(1)
    
    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
