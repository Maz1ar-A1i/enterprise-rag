"""
Phase 4 — Generator LLM QLoRA Fine-Tuning on Kaggle (OOM-Proof + PagedAdamW8bit + Live TQDM)
=============================================================================================
Optimized for 15GB Kaggle GPUs (Tesla T4 / P100):
  - Micro-batch size = 1 with gradient_accumulation_steps = 16 (Effective batch = 16)
  - Paged 8-bit AdamW optimizer (PagedAdamW8bit) to slash optimizer memory by 75%
  - PyTorch expandable_segments memory management to eliminate CUDA fragmentation
  - Max sequence length = 768 (fits enterprise QA perfectly)
  - Completion-only loss masking (-100 on prompts)
  - Real-time tqdm progress bars and graphical analytics
"""

import os
import sys
import json
import math
import time
import random
import zipfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# Prevent CUDA memory fragmentation & suppress warning noise
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. ENVIRONMENT & GPU DIAGNOSTICS
# ============================================================================
print("=" * 80, flush=True)
print("  PHASE 4: OOM-SAFE GENERATOR LLM QLoRA FINE-TUNING", flush=True)
print("=" * 80, flush=True)

print(f"Python Version:         {sys.version.split()[0]}", flush=True)
print(f"PyTorch Version:        {torch.__version__}", flush=True)
print(f"CUDA Available:         {torch.cuda.is_available()}", flush=True)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device Name:        {gpu_name} ({gpu_count} GPU(s))", flush=True)
    print(f"Available VRAM:         {vram_gb:.2f} GB", flush=True)
    bf16_supported = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"Optimal Compute Dtype:  {compute_dtype}", flush=True)
else:
    device = torch.device("cpu")
    compute_dtype = torch.float32
    print("[WARNING] CUDA not detected. Enable GPU in Kaggle settings.", flush=True)

print("=" * 80, flush=True)

# ============================================================================
# 2. CONFIGURATION & HYPERPARAMETERS (OOM-PROOF FOR 15GB T4)
# ============================================================================
@dataclass
class LLMFinetuningConfig:
    # Target LLM (Default: Qwen2.5-7B-Instruct; open weights, no gated token)
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Context & Sequence Length (768 fits complete RAG prompt and prevents OOM)
    max_seq_length: int = 768
    max_context_chunks: int = 2
    
    # QLoRA 4-bit Settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Training Parameters (Micro batch 1 + 16 accumulation = 16 effective batch)
    num_train_epochs: int = 4
    learning_rate: float = 2e-4
    batch_size: int = 1                  # Micro-batch 1 to prevent OOM
    gradient_accumulation_steps: int = 16 # Effective batch size = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Evaluation & Checkpointing
    eval_every_steps: int = 10           # Evaluate validation loss every 10 optimizer steps
    save_dir: str = "./phase4_llm_adapter"
    output_zip: str = "./phase4_llm_adapter.zip"

cfg = LLMFinetuningConfig()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# ============================================================================
# 3. DATASET RESOLUTION & PREPARATION
# ============================================================================
def resolve_file(filename: str, fallback_paths: List[str] = None) -> Path:
    candidates = [Path(filename)]
    if fallback_paths:
        candidates.extend([Path(p) for p in fallback_paths])
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for found in kaggle_input.rglob(filename):
            if found.is_file():
                return found
    for found in Path(".").rglob(filename):
        if found.is_file():
            return found
    raise FileNotFoundError(f"Could not locate '{filename}' in local workspace or Kaggle inputs.")

print("\n--- Resolving Dataset Files ---", flush=True)
try:
    QUESTIONS_FILE = resolve_file("questions_visible.jsonl", [
        "datasets/benchmark/questions_visible.jsonl",
        "/kaggle/input/enterprise-rag-dataset/questions_visible.jsonl",
        "/kaggle/input/datasets/mazhar99/question-visible/questions_visible.jsonl"
    ])
    print(f"✔ Questions File:  {QUESTIONS_FILE}", flush=True)
except Exception as e:
    QUESTIONS_FILE = None
    print(f"✘ Questions File:  {e}", flush=True)

try:
    CHUNKS_FILE = resolve_file("chunks.jsonl", [
        "corpus/processed/chunks.jsonl",
        "/kaggle/input/enterprise-rag-dataset/chunks.jsonl",
        "/kaggle/input/datasets/mazhar99/chunks/chunks.jsonl"
    ])
    print(f"✔ Chunks File:     {CHUNKS_FILE}", flush=True)
except Exception as e:
    CHUNKS_FILE = None
    print(f"✘ Chunks File:     {e}", flush=True)

# ============================================================================
# 4. RAG PROMPT FORMATTING & DATASET BUILDER
# ============================================================================
SYSTEM_PROMPT = """You are a precise technical documentation assistant for DevOps and cloud infrastructure.
Your knowledge comes ONLY from the context passages provided below.

RULES:
1. Answer ONLY using information present in the provided context passages.
2. If the context does not contain enough information to fully answer the question, respond with:
   "I cannot find sufficient information in the provided documentation to answer this question."
   Do NOT speculate, guess, or use general knowledge.
3. Cite the source of each factual claim inline using the format: [Source: <URL>]
   where <URL> is the exact URL listed in the passage header.
4. If the question asks about a specific version, only answer from context matching that version.
5. Be concise, precise, and technical."""

def format_rag_prompt(question_text: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for chunk in context_chunks:
        url = chunk.get("url", "https://docs.enterprise.internal")
        source = chunk.get("source", "docs")
        text = chunk.get("text", "")
        context_blocks.append(f"--- Document: {source} | URL: {url} ---\n{text}")
    context_str = "\n\n".join(context_blocks) if context_blocks else "No relevant documentation available."
    return f"Context Passages:\n{context_str}\n\nUser Question:\n{question_text}"

class RAGInstructDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_seq_length: int = 768):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        user_prompt = item["user_prompt"]
        assistant_response = item["assistant_response"]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_messages = messages + [{"role": "assistant", "content": assistant_response}]
            full_str = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        else:
            prompt_str = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            full_str = f"{prompt_str}{assistant_response}<|im_end|>"

        prompt_tokens = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_tokens = self.tokenizer.encode(full_str, add_special_tokens=False)
        
        if self.tokenizer.eos_token_id and (len(full_tokens) == 0 or full_tokens[-1] != self.tokenizer.eos_token_id):
            full_tokens.append(self.tokenizer.eos_token_id)

        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[:self.max_seq_length]

        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        # Loss Masking on prompt tokens
        labels = input_ids.clone()
        prompt_len = min(len(prompt_tokens), len(labels))
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": item
        }

def collate_fn(batch, pad_token_id: int):
    input_ids_list = [item["input_ids"] for item in batch]
    attention_mask_list = [item["attention_mask"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    metadata_list = [item["metadata"] for item in batch]

    max_len = max(x.size(0) for x in input_ids_list)
    padded_input_ids, padded_attention_mask, padded_labels = [], [], []

    for inp, att, lab in zip(input_ids_list, attention_mask_list, labels_list):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            padded_input_ids.append(torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
            padded_attention_mask.append(torch.cat([att, torch.zeros(pad_len, dtype=torch.long)]))
            padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        else:
            padded_input_ids.append(inp)
            padded_attention_mask.append(att)
            padded_labels.append(lab)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
        "metadata": metadata_list
    }

# ============================================================================
# 5. DATA INGESTION & STRATIFIED SPLITS
# ============================================================================
def load_and_prepare_data(
    questions_path: Path,
    chunks_path: Path,
    max_context_chunks: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    print("\n--- Ingesting Chunks & Benchmark Questions ---", flush=True)
    chunks_map = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                chunks_map[c["chunk_id"]] = c
    print(f"Loaded {len(chunks_map):,} total corpus chunks.", flush=True)

    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    print(f"Loaded {len(questions):,} benchmark questions.", flush=True)

    samples = []
    all_chunk_ids = list(chunks_map.keys())

    for q in questions:
        relevant_ids = q.get("relevant_chunk_ids", [])
        gold_chunks = [chunks_map[cid] for cid in relevant_ids if cid in chunks_map]
        is_refusal = q.get("expected_refusal", False) or q.get("category") == "unanswerable"
        
        if is_refusal:
            distractor_ids = random.sample(all_chunk_ids, min(2, len(all_chunk_ids)))
            context_chunks = [chunks_map[cid] for cid in distractor_ids]
            assistant_response = (
                "I cannot find sufficient information in the provided documentation to answer this question."
            )
        else:
            context_chunks = gold_chunks[:max_context_chunks]
            assistant_response = q.get("answer", "")

        user_prompt = format_rag_prompt(q["question"], context_chunks)
        samples.append({
            "id": q.get("id", ""),
            "category": q.get("category", "direct"),
            "topic": q.get("topic", "general"),
            "difficulty": q.get("difficulty", "medium"),
            "expected_refusal": is_refusal,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response
        })

    categories = {}
    for s in samples:
        categories.setdefault(s["category"], []).append(s)

    rng = random.Random(cfg.seed)
    train_samples, val_samples, test_samples = [], [], []
    for cat, items in categories.items():
        rng.shuffle(items)
        n = len(items)
        n_tr, n_v = int(n * 0.8), int(n * 0.1)
        train_samples.extend(items[:n_tr])
        val_samples.extend(items[n_tr:n_tr + n_v])
        test_samples.extend(items[n_tr + n_v:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    print(f"Train samples:      {len(train_samples):,} (80%)", flush=True)
    print(f"Validation samples: {len(val_samples):,} (10%)", flush=True)
    print(f"Test samples:       {len(test_samples):,} (10%)", flush=True)
    return train_samples, val_samples, test_samples

# ============================================================================
# 6. MODEL INITIALIZATION & 4-BIT QLoRA SETUP (OOM-SAFE)
# ============================================================================
def setup_model_and_tokenizer(config: LLMFinetuningConfig):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"\n--- Loading Model: {config.base_model_name} in 4-bit QLoRA ---", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer

# ============================================================================
# 7. TRAINING ENGINE WITH PAGED 8-BIT ADAMW & LIVE PROGRESS
# ============================================================================
def evaluate_loss(model, dataloader) -> Tuple[float, float]:
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(min(avg_loss, 20.0))
    return avg_loss, perplexity

def train_qlora_llm(
    model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LLMFinetuningConfig
) -> Dict[str, Any]:
    # Use Paged 8-bit AdamW to save 75% optimizer VRAM and prevent fragmentation OOM
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_training_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.num_train_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    print("\n" + "=" * 80, flush=True)
    print("  STARTING OOM-PROOF QLoRA TRAINING", flush=True)
    print(f"  Total Epochs:          {config.num_train_epochs}", flush=True)
    print(f"  Total Optimizer Steps: {total_training_steps}", flush=True)
    print(f"  Micro-Batch:           {config.batch_size} (1 sample/batch -> No OOM)", flush=True)
    print(f"  Effective Batch Size:  {config.batch_size * config.gradient_accumulation_steps}", flush=True)
    print(f"  Optimizer:             bitsandbytes PagedAdamW8bit", flush=True)
    print("=" * 80 + "\n", flush=True)

    history = {
        "epoch": [], "step": [], "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [], "overfitting_gap": [], "learning_rate": []
    }

    best_val_loss = float("inf")
    os.makedirs(config.save_dir, exist_ok=True)
    global_step = 0

    print("Evaluating initial baseline validation loss...", flush=True)
    init_val_loss, init_val_ppl = evaluate_loss(model, val_loader)
    print(f"✔ Initial Pre-Train Validation Loss: {init_val_loss:.4f} (PPL: {init_val_ppl:.2f})\n", flush=True)

    for epoch in range(1, config.num_train_epochs + 1):
        model.train()
        accumulated_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{config.num_train_epochs}",
            unit="batch",
            dynamic_ncols=True
        )

        for step, batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            step_loss = loss.item() * config.gradient_accumulation_steps
            accumulated_loss += step_loss

            vram_use = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            pbar.set_postfix({
                "loss": f"{step_loss:.3f}",
                "opt_step": f"{global_step}/{total_training_steps}",
                "vram": f"{vram_use:.1f}GB"
            })

            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.eval_every_steps == 0 or global_step == total_training_steps:
                    val_loss, val_ppl = evaluate_loss(model, val_loader)
                    model.train()

                    cur_tr_loss = accumulated_loss / config.gradient_accumulation_steps
                    cur_tr_ppl = math.exp(min(cur_tr_loss, 20.0))
                    gap = val_loss - cur_tr_loss
                    cur_lr = scheduler.get_last_lr()[0]

                    history["epoch"].append(round(epoch - 1 + (step + 1) / len(train_loader), 2))
                    history["step"].append(global_step)
                    history["train_loss"].append(cur_tr_loss)
                    history["val_loss"].append(val_loss)
                    history["train_ppl"].append(cur_tr_ppl)
                    history["val_ppl"].append(val_ppl)
                    history["overfitting_gap"].append(gap)
                    history["learning_rate"].append(cur_lr)

                    status_flag = "✔ Healthy Generalized" if gap <= 0.5 else "⚠️ Possible Overfitting"
                    tqdm.write(
                        f"📊 [Step {global_step:03d}/{total_training_steps}] "
                        f"Train Loss: {cur_tr_loss:.4f} (PPL: {cur_tr_ppl:.2f}) | "
                        f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
                        f"Gap: {gap:+.4f} | {status_flag}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        tqdm.write(f"  ★ New Best Val Loss ({best_val_loss:.4f}) -> Saving Adapter Checkpoint...")
                        model.save_pretrained(config.save_dir)
                        tokenizer.save_pretrained(config.save_dir)

                    accumulated_loss = 0.0

        pbar.close()
        elapsed = time.time() - t0
        print(f"✔ Completed Epoch {epoch} in {elapsed:.1f}s ({elapsed/60:.1f} min)\n", flush=True)

    print(f"\n✔ Training Finished! Best Validation Loss: {best_val_loss:.4f}", flush=True)
    return history

# ============================================================================
# 8. VISUALIZATION OF TRAINING PROGRESS & OVERFITTING METRICS
# ============================================================================
def plot_training_analytics(history: Dict[str, Any], output_path: str = "training_analytics.png"):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    # 1. Train Loss vs. Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(history["step"], history["train_loss"], label="Train Loss (Completion-Only)", color="#1f77b4", lw=2.5)
    ax1.plot(history["step"], history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2.5, marker="o")
    ax1.set_title("Loss Dynamics: Train vs. Validation (Overfitting Monitor)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Optimization Steps", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Perplexity Curve
    ax2 = axes[0, 1]
    ax2.plot(history["step"], history["train_ppl"], label="Train Perplexity", color="#2ca02c", lw=2)
    ax2.plot(history["step"], history["val_ppl"], label="Validation Perplexity", color="#d62728", lw=2, marker="s")
    ax2.set_title("Perplexity Evolution (Lower = Better Prediction)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Optimization Steps", fontsize=11)
    ax2.set_ylabel("Perplexity (exp(Loss))", fontsize=11)
    ax2.legend(loc="upper right", frameon=True)

    # 3. Overfitting Gap
    ax3 = axes[1, 0]
    ax3.plot(history["step"], history["overfitting_gap"], label="Overfitting Gap (Val - Train)", color="#9467bd", lw=2.5)
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.7)
    max_gap = max(max(history["overfitting_gap"]), 1.0) if history["overfitting_gap"] else 1.0
    ax3.axhspan(0.5, max_gap + 0.2, color="red", alpha=0.1, label="Overfitting Warning Zone (>0.5)")
    ax3.set_title("Generalization / Overfitting Gap", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Optimization Steps", fontsize=11)
    ax3.set_ylabel("Delta Loss (Val - Train)", fontsize=11)
    ax3.legend(loc="upper left", frameon=True)

    # 4. Learning Rate Schedule
    ax4 = axes[1, 1]
    ax4.plot(history["step"], history["learning_rate"], label="Cosine Learning Rate", color="#8c564b", lw=2)
    ax4.set_title("Learning Rate Schedule with Warmup", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Optimization Steps", fontsize=11)
    ax4.set_ylabel("Learning Rate", fontsize=11)
    ax4.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\n✔ Training analytics saved to: {output_path}", flush=True)
    plt.close()

# ============================================================================
# 9. COMPREHENSIVE GENERATION & ACCURACY BENCHMARK ON TEST SET
# ============================================================================
def calculate_rouge_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    from collections import Counter
    import re

    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def get_ngrams(tokens: List[str], n: int):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def lcs(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[m][n]

    r1_list, r2_list, rl_list, bleu_list = [], [], [], []
    for pred, ref in zip(predictions, references):
        p_toks = tokenize(pred)
        r_toks = tokenize(ref)
        if not r_toks or not p_toks:
            r1_list.append(0.0); r2_list.append(0.0); rl_list.append(0.0); bleu_list.append(0.0)
            continue
        p_1, r_1 = Counter(p_toks), Counter(r_toks)
        overlap_1 = sum((p_1 & r_1).values())
        rec_1 = overlap_1 / len(r_toks)
        prec_1 = overlap_1 / len(p_toks)
        r1_list.append(2 * rec_1 * prec_1 / (rec_1 + prec_1 + 1e-9))

        p_2, r_2 = Counter(get_ngrams(p_toks, 2)), Counter(get_ngrams(r_toks, 2))
        overlap_2 = sum((p_2 & r_2).values())
        rec_2 = overlap_2 / max(1, len(r_toks) - 1)
        prec_2 = overlap_2 / max(1, len(p_toks) - 1)
        r2_list.append(2 * rec_2 * prec_2 / (rec_2 + prec_2 + 1e-9))

        lcs_len = lcs(p_toks, r_toks)
        rec_l = lcs_len / len(r_toks)
        prec_l = lcs_len / len(p_toks)
        rl_list.append(2 * rec_l * prec_l / (rec_l + prec_l + 1e-9))

        bleu_p = []
        for n in range(1, 5):
            pn, rn = Counter(get_ngrams(p_toks, n)), Counter(get_ngrams(r_toks, n))
            bleu_p.append((sum((pn & rn).values()) + 1e-9) / max(1, len(p_toks) - n + 1))
        gm = math.exp(sum(math.log(p) for p in bleu_p) / 4)
        bp = 1.0 if len(p_toks) > len(r_toks) else math.exp(1 - len(r_toks) / max(1, len(p_toks)))
        bleu_list.append(bp * gm)

    return {
        "rouge_1": float(np.mean(r1_list)),
        "rouge_2": float(np.mean(r2_list)),
        "rouge_l": float(np.mean(rl_list)),
        "bleu_4": float(np.mean(bleu_list))
    }

def evaluate_test_set_generations(
    model,
    tokenizer,
    test_samples: List[Dict[str, Any]],
    max_eval_count: int = 50
) -> Dict[str, Any]:
    print("\n" + "=" * 80, flush=True)
    print(f"  RUNNING GENERATION BENCHMARK ON {min(len(test_samples), max_eval_count)} TEST QUESTIONS", flush=True)
    print("=" * 80, flush=True)

    model.eval()
    test_subset = test_samples[:max_eval_count]
    predictions, gold_references = [], []
    category_scores = {}
    citation_correct_count, refusal_correct_count, refusal_total_count = 0, 0, 0

    t_start = time.time()
    for idx, item in enumerate(tqdm(test_subset, desc="Evaluating Test Set", unit="query")):
        user_prompt = item["user_prompt"]
        gold_answer = item["assistant_response"]
        cat = item["category"]
        is_refusal = item["expected_refusal"]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_str = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        gen_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        predictions.append(gen_text)
        gold_references.append(gold_answer)

        if "[Source:" in gen_text or "[source:" in gen_text.lower():
            citation_correct_count += 1

        if is_refusal:
            refusal_total_count += 1
            if "cannot find sufficient information" in gen_text.lower() or "insufficient information" in gen_text.lower():
                refusal_correct_count += 1

        category_scores.setdefault(cat, {"preds": [], "refs": []})
        category_scores[cat]["preds"].append(gen_text)
        category_scores[cat]["refs"].append(gold_answer)

    total_time = time.time() - t_start
    overall_metrics = calculate_rouge_bleu(predictions, gold_references)
    citation_rate = citation_correct_count / max(1, len(test_subset))
    refusal_acc = (refusal_correct_count / refusal_total_count) if refusal_total_count > 0 else 1.0

    category_summary = {}
    for cat, data in category_scores.items():
        cat_metrics = calculate_rouge_bleu(data["preds"], data["refs"])
        category_summary[cat] = {
            "count": len(data["preds"]),
            "rouge_l": round(cat_metrics["rouge_l"], 4),
            "bleu_4": round(cat_metrics["bleu_4"], 4)
        }

    results = {
        "rouge_1": round(overall_metrics["rouge_1"], 4),
        "rouge_2": round(overall_metrics["rouge_2"], 4),
        "rouge_l": round(overall_metrics["rouge_l"], 4),
        "bleu_4": round(overall_metrics["bleu_4"], 4),
        "citation_format_rate": round(citation_rate, 4),
        "refusal_accuracy": round(refusal_acc, 4),
        "eval_time_seconds": round(total_time, 2),
        "category_breakdown": category_summary,
        "sample_comparisons": [
            {
                "id": test_subset[i]["id"],
                "category": test_subset[i]["category"],
                "generated": predictions[i][:300],
                "ground_truth": gold_references[i][:300]
            }
            for i in range(min(4, len(test_subset)))
        ]
    }

    plot_category_performance(category_summary)
    return results

def plot_category_performance(cat_summary: Dict[str, Any], output_path: str = "category_performance.png"):
    cats = list(cat_summary.keys())
    rouge_l = [cat_summary[c]["rouge_l"] for c in cats]
    bleu_4 = [cat_summary[c]["bleu_4"] for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width/2, rouge_l, width, label="ROUGE-L Score", color="#2ca02c")
    plt.bar(x + width/2, bleu_4, width, label="BLEU-4 Score", color="#1f77b4")

    plt.xlabel("Question Category", fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    plt.title("Fine-Tuned LLM Accuracy Breakdown by Question Category", fontsize=13, fontweight="bold")
    plt.xticks(x, cats, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✔ Category performance chart saved to: {output_path}", flush=True)
    plt.close()

# ============================================================================
# 10. MAIN EXECUTION PIPELINE
# ============================================================================
def main():
    if QUESTIONS_FILE is None or CHUNKS_FILE is None:
        print("\n[ERROR] Missing dataset files. Please ensure questions_visible.jsonl and chunks.jsonl are uploaded.", flush=True)
        return

    train_samples, val_samples, test_samples = load_and_prepare_data(
        QUESTIONS_FILE, CHUNKS_FILE, max_context_chunks=cfg.max_context_chunks
    )

    model, tokenizer = setup_model_and_tokenizer(cfg)

    train_dataset = RAGInstructDataset(train_samples, tokenizer, max_seq_length=cfg.max_seq_length)
    val_dataset = RAGInstructDataset(val_samples, tokenizer, max_seq_length=cfg.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    history = train_qlora_llm(model, tokenizer, train_loader, val_loader, cfg)
    plot_training_analytics(history, output_path="training_analytics.png")

    eval_results = evaluate_test_set_generations(model, tokenizer, test_samples, max_eval_count=40)

    results_payload = {
        "hyperparameters": {
            "base_model": cfg.base_model_name,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "epochs": cfg.num_train_epochs,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size * cfg.gradient_accumulation_steps
        },
        "training_history": history,
        "test_benchmark": eval_results
    }

    results_file = Path(cfg.save_dir) / "phase4_eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n✔ Full evaluation results saved to: {results_file}", flush=True)

    print(f"\n--- Packaging Final Adapter Archive: {cfg.output_zip} ---", flush=True)
    with zipfile.ZipFile(cfg.output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cfg.save_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, cfg.save_dir)
                zipf.write(full_p, rel_p)
        if os.path.exists("training_analytics.png"):
            zipf.write("training_analytics.png", "training_analytics.png")
        if os.path.exists("category_performance.png"):
            zipf.write("category_performance.png", "category_performance.png")

    print(f"✔ Phase 4 LoRA Adapter successfully packaged into: {cfg.output_zip}", flush=True)
    print("=" * 80, flush=True)
    print("  PHASE 4 LLM FINE-TUNING PIPELINE FINISHED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
