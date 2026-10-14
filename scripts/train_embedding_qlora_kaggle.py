"""
Phase 3 — Embedding Model QLoRA Fine-Tuning on Kaggle (Fixed & Validated)
========================================================================
Fixes:
  1. Displays both Train Loss and Validation Loss for each epoch.
  2. Evaluates against the complete corpus (all 10,549 chunks) so gold chunks are always present,
     yielding accurate Recall@1, Recall@5, Recall@10, MRR@10, and nDCG@10 metrics.
"""

# ============================================================================
# 1. ENVIRONMENT CHECK
# ============================================================================
import os
import sys
import json
import random
import shutil
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=" * 70)
print("SECTION 1: ENVIRONMENT CHECK")
print("=" * 70)
print(f"Python Version:       {sys.version.split()[0]}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:          {torch.cuda.get_device_name(0)}")
    print(f"Device Count:         {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total VRAM:           {props.total_memory / (1024**3):.2f} GB")
else:
    print("[WARNING] CUDA is not available. Ensure GPU accelerator is enabled in Kaggle Settings.")
print("=" * 70)

# ============================================================================
# 2. INSTALL DEPENDENCIES (Run in Kaggle bash or notebook cell)
# ============================================================================
# !pip install -q --upgrade pip
# !pip install -q transformers>=4.44.0 peft>=0.12.0 bitsandbytes>=0.43.0 accelerate>=0.33.0 datasets>=2.20.0 rank-bm25>=0.2.2 sentence-transformers>=3.0.0 safetensors

import transformers
import peft
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from rank_bm25 import BM25Okapi

print(f"Transformers:         {transformers.__version__}")
print(f"PEFT:                 {peft.__version__}")
print(f"BitsAndBytes:         {bnb.__version__}")

# ============================================================================
# 3. CONFIGURATION & HYPERPARAMETERS
# ============================================================================
@dataclass
class HyperParameters:
    # Base Embedding Model
    base_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    max_query_length: int = 128
    max_doc_length: int = 400
    
    # QLoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("query", "key", "value", "dense")
    
    # Training Parameters
    temperature: float = 0.05  # InfoNCE temperature for normalized embeddings
    num_train_epochs: int = 5
    learning_rate: float = 2e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    
    # Negative Mining
    num_hard_negatives: int = 2
    
    # Split Ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    
    # Dataset Files
    questions_file: str = "questions_visible.jsonl"
    chunks_file: str = "chunks.jsonl"
    
    # Output paths
    output_dir: str = "./embedding_phase3_output"
    package_dir: str = "./embedding_phase3"

CONFIG = HyperParameters()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG.seed)
print("\n[Configuration Loaded]")
print(json.dumps(CONFIG.__dict__, indent=2))

# ============================================================================
# 4. LOAD CURRENT DATASET & SANITY CHECKS
# ============================================================================
def locate_file(filename: str) -> str:
    search_paths = [
        filename,
        os.path.join(".", filename),
        os.path.join("..", filename),
        os.path.join("datasets", "benchmark", filename),
        os.path.join("corpus", "processed", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/input/enterprise-rag-dataset", filename),
        os.path.join("/kaggle/working", filename),
    ]
    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if filename in files:
                return os.path.join(root, filename)

    for p in search_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate '{filename}'. Upload to Kaggle or place in working directory.")

print("\n" + "=" * 70)
print("SECTION 4: LOADING DATASET")
print("=" * 70)

q_path = locate_file(CONFIG.questions_file)
c_path = locate_file(CONFIG.chunks_file)
print(f"Loading questions from: {q_path}")
print(f"Loading chunks from:    {c_path}")

questions_data = []
with open(q_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            questions_data.append(json.loads(line))

chunks_dict: Dict[str, Dict[str, Any]] = {}
with open(c_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rec = json.loads(line)
            chunks_dict[rec["chunk_id"]] = rec

chunk_id_list = list(chunks_dict.keys())
print(f"Total Questions Loaded:  {len(questions_data)}")
print(f"Total Chunks in Corpus:  {len(chunks_dict)}")

# ============================================================================
# 5. BUILD EMBEDDING TRAINING TRIPLET DATASET (ANCHOR, POSITIVE, HARD NEGATIVES)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: MINING HARD NEGATIVES FOR CONTRASTIVE EMBEDDING TRAINING")
print("=" * 70)

corpus_texts = [chunks_dict[cid]["text"] for cid in chunk_id_list]
tokenized_corpus = [text.lower().split() for text in corpus_texts]
bm25 = BM25Okapi(tokenized_corpus)

triplet_records = []
for item in questions_data:
    qid = item["id"]
    query = item["question"]
    gold_cids = set(item.get("relevant_chunk_ids", []))
    
    pos_texts = [chunks_dict[cid]["text"] for cid in gold_cids if cid in chunks_dict]
    if not pos_texts:
        continue
    
    # Mine BM25 hard negatives
    query_tokens = query.lower().split()
    top_indices = bm25.get_top_n(query_tokens, list(range(len(chunk_id_list))), n=35)
    
    neg_texts = []
    for idx in top_indices:
        candidate_cid = chunk_id_list[idx]
        if candidate_cid not in gold_cids:
            neg_texts.append(chunks_dict[candidate_cid]["text"])
        if len(neg_texts) >= CONFIG.num_hard_negatives:
            break
            
    for pos in pos_texts:
        triplet_records.append({
            "id": qid,
            "query": CONFIG.query_prefix + query,  # Add BGE query prefix
            "positive": pos,
            "negatives": neg_texts,
            "raw_query": query,
            "gold_cids": list(gold_cids),
        })

print(f"Constructed {len(triplet_records)} contrastive training triplets.")

# ============================================================================
# 6. QUERY-LEVEL DATASET SPLIT (PREVENT DATA LEAKAGE)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 6: QUERY-LEVEL SPLIT")
print("=" * 70)

unique_qids = list(set(r["id"] for r in triplet_records))
random.seed(CONFIG.seed)
random.shuffle(unique_qids)

n_total = len(unique_qids)
n_train = int(n_total * CONFIG.train_ratio)
n_val = int(n_total * CONFIG.val_ratio)

train_qids = set(unique_qids[:n_train])
val_qids   = set(unique_qids[n_train:n_train + n_val])
test_qids  = set(unique_qids[n_train + n_val:])

train_triplets = [r for r in triplet_records if r["id"] in train_qids]
val_triplets   = [r for r in triplet_records if r["id"] in val_qids]
test_triplets  = [r for r in triplet_records if r["id"] in test_qids]

print(f"Split Sizes -> Train Queries: {len(train_qids)} ({len(train_triplets)} triplets)")
print(f"               Val Queries:   {len(val_qids)} ({len(val_triplets)} triplets)")
print(f"               Test Queries:  {len(test_qids)} ({len(test_triplets)} triplets)")

# ============================================================================
# 7 & 8. LOAD BASE EMBEDDING MODEL & APPLY QLORA
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 7 & 8: LOAD BASE EMBEDDING MODEL & APPLY QLORA")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(CONFIG.base_model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

print(f"Loading base embedding model '{CONFIG.base_model_name}' in 4-bit precision...")
base_transformer = AutoModel.from_pretrained(
    CONFIG.base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

base_transformer = prepare_model_for_kbit_training(base_transformer)

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    target_modules=list(CONFIG.target_modules),
    bias="none",
)

peft_embedding_model = get_peft_model(base_transformer, peft_config)

def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in m.parameters())
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable %: {100 * trainable_params / all_param:.4f}%"
    )

print_trainable_parameters(peft_embedding_model)

# ============================================================================
# 9. CONTRASTIVE EMBEDDING TRAINING WITH VALIDATION LOSS
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 9: CONTRASTIVE QLORA FINE-TUNING (WITH VALIDATION LOSS)")
print("=" * 70)

class EmbeddingTripletDataset(Dataset):
    def __init__(self, triplets: List[Dict]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        item = self.triplets[idx]
        neg = item["negatives"][0] if item["negatives"] else item["positive"]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": neg,
        }

train_dataset = EmbeddingTripletDataset(train_triplets)
val_dataset   = EmbeddingTripletDataset(val_triplets)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False)

def encode_texts(model, texts: List[str], max_len: int, device) -> torch.Tensor:
    """Encode texts using CLS pooling + L2 normalization."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    
    outputs = model(**inputs)
    cls_rep = outputs.last_hidden_state[:, 0]
    normalized_emb = F.normalize(cls_rep, p=2, dim=1)
    return normalized_emb

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, q_emb: torch.Tensor, p_emb: torch.Tensor, n_emb: torch.Tensor) -> torch.Tensor:
        sim_pos = torch.matmul(q_emb, p_emb.T) / self.temperature
        sim_neg = torch.matmul(q_emb, n_emb.T) / self.temperature
        scores = torch.cat([sim_pos, sim_neg], dim=1)
        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        loss = self.cross_entropy(scores, labels)
        return loss

loss_fn = MultipleNegativesRankingLoss(temperature=CONFIG.temperature)
optimizer = torch.optim.AdamW(
    peft_embedding_model.parameters(),
    lr=CONFIG.learning_rate,
    weight_decay=CONFIG.weight_decay,
)

device = next(peft_embedding_model.parameters()).device
print(f"Training on device: {device} for {CONFIG.num_train_epochs} epochs...")

best_val_loss = float("inf")
t0_train = time.time()

for epoch in range(1, CONFIG.num_train_epochs + 1):
    peft_embedding_model.train()
    total_train_loss = 0.0
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader, 1):
        q_emb = encode_texts(peft_embedding_model, batch["query"], CONFIG.max_query_length, device)
        p_emb = encode_texts(peft_embedding_model, batch["positive"], CONFIG.max_doc_length, device)
        n_emb = encode_texts(peft_embedding_model, batch["negative"], CONFIG.max_doc_length, device)
        
        loss = loss_fn(q_emb, p_emb, n_emb)
        loss = loss / CONFIG.gradient_accumulation_steps
        loss.backward()
        
        total_train_loss += loss.item() * CONFIG.gradient_accumulation_steps
        
        if step % CONFIG.gradient_accumulation_steps == 0 or step == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            
    avg_train_loss = total_train_loss / len(train_loader)
    
    # Validation Evaluation
    peft_embedding_model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            q_emb = encode_texts(peft_embedding_model, batch["query"], CONFIG.max_query_length, device)
            p_emb = encode_texts(peft_embedding_model, batch["positive"], CONFIG.max_doc_length, device)
            n_emb = encode_texts(peft_embedding_model, batch["negative"], CONFIG.max_doc_length, device)
            v_loss = loss_fn(q_emb, p_emb, n_emb)
            total_val_loss += v_loss.item()
            
    avg_val_loss = total_val_loss / max(len(val_loader), 1)
    
    print(f"Epoch {epoch}/{CONFIG.num_train_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

train_time_min = (time.time() - t0_train) / 60.0
print(f"\nEmbedding QLoRA Fine-Tuning Completed in {train_time_min:.2f} minutes.")

# ============================================================================
# 10 & 11. RETRIEVAL-LEVEL EVALUATION (SEARCH OVER FULL 10,549 CORPUS)
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 10 & 11: FULL CORPUS DENSE RETRIEVAL EVALUATION")
print("=" * 70)

# Build complete corpus list so gold chunks are 100% guaranteed to be present
eval_chunks = [chunks_dict[cid] for cid in chunk_id_list]
corpus_cids = [c["chunk_id"] for c in eval_chunks]
print(f"Total Corpus Search Pool: {len(eval_chunks):,} chunks")

def compute_corpus_embeddings(model_to_use, chunks: List[Dict], batch_size: int = 64) -> np.ndarray:
    """Encode all corpus chunks to L2-normalized float32 vectors in batches."""
    model_to_use.eval()
    all_embeddings = []
    texts = [c["text"] for c in chunks]
    device = next(model_to_use.parameters()).device
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            emb = encode_texts(model_to_use, batch_texts, CONFIG.max_doc_length, device)
            all_embeddings.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(all_embeddings)

print(f"Encoding full corpus with Fine-Tuned Embedding Model...")
ft_corpus_emb = compute_corpus_embeddings(peft_embedding_model, eval_chunks, batch_size=64)

print(f"Encoding full corpus with Base Pretrained Embedding Model...")
base_eval_model = AutoModel.from_pretrained(
    CONFIG.base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
base_corpus_emb = compute_corpus_embeddings(base_eval_model, eval_chunks, batch_size=64)

def evaluate_retrieval(model_to_use, corpus_emb: np.ndarray, test_items: List[Dict], k_vals=[1, 5, 10]) -> Dict[str, float]:
    """Calculate Recall@K, MRR@10, and nDCG@10 for dense retrieval against full corpus."""
    model_to_use.eval()
    device = next(model_to_use.parameters()).device
    
    r1_list, r5_list, r10_list, mrr_list, ndcg_list = [], [], [], [], []
    
    with torch.no_grad():
        for item in test_items:
            q_text = CONFIG.query_prefix + item["raw_query"]
            q_emb = encode_texts(model_to_use, [q_text], CONFIG.max_query_length, device).cpu().numpy()
            
            # Dot product (Cosine sim since L2 normalized)
            scores = np.dot(corpus_emb, q_emb.T).flatten()
            ranked_indices = np.argsort(scores)[::-1][:max(k_vals)]
            ranked_cids = [corpus_cids[idx] for idx in ranked_indices]
            
            gold_set = set(item["gold_cids"])
            
            # Recall@1, 5, 10
            r1_list.append(1.0 if any(cid in gold_set for cid in ranked_cids[:1]) else 0.0)
            r5_list.append(1.0 if any(cid in gold_set for cid in ranked_cids[:5]) else 0.0)
            r10_list.append(1.0 if any(cid in gold_set for cid in ranked_cids[:10]) else 0.0)
            
            # MRR
            first_hit = next((rank + 1 for rank, cid in enumerate(ranked_cids[:10]) if cid in gold_set), 0)
            mrr_list.append(1.0 / first_hit if first_hit > 0 else 0.0)
            
            # nDCG@10
            dcg = sum((1.0 if cid in gold_set else 0.0) / math.log2(r + 2) for r, cid in enumerate(ranked_cids[:10]))
            ndcg_list.append(dcg)
            
    return {
        "recall_at_1": float(np.mean(r1_list)),
        "recall_at_5": float(np.mean(r5_list)),
        "recall_at_10": float(np.mean(r10_list)),
        "mrr_at_10": float(np.mean(mrr_list)),
        "ndcg_at_10": float(np.mean(ndcg_list)),
    }

print("\nEvaluating Dense Retrieval Metrics on Held-Out Test Set...")
base_retrieval_metrics = evaluate_retrieval(base_eval_model, base_corpus_emb, test_triplets)
ft_retrieval_metrics   = evaluate_retrieval(peft_embedding_model, ft_corpus_emb, test_triplets)

del base_eval_model
torch.cuda.empty_cache()

print("\n" + "=" * 75)
print("PHASE 3 EMBEDDING EVALUATION COMPARISON (DENSE RETRIEVAL TEST SET)")
print("=" * 75)
print(f"{'Metric':<18} | {'Base Embedding':<18} | {'Fine-Tuned (QLoRA)':<22} | {'Delta':<10}")
print("-" * 75)
for m in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]:
    b_val = base_retrieval_metrics[m]
    f_val = ft_retrieval_metrics[m]
    delta = f_val - b_val
    sign = "+" if delta >= 0 else ""
    print(f"{m:<18} | {b_val:<18.4f} | {f_val:<22.4f} | {sign}{delta:.4f}")
print("=" * 75)

# ============================================================================
# 14, 15 & 16. SAVE ARTIFACTS & CREATE INTEGRATION PACKAGE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 14 & 16: SAVING ARTIFACTS FOR PHASE 3 INTEGRATION")
print("=" * 70)

package_dir = Path(CONFIG.package_dir)
adapter_dir = package_dir / "adapter"
tokenizer_dir = package_dir / "tokenizer"

if package_dir.exists():
    shutil.rmtree(package_dir)
package_dir.mkdir(parents=True, exist_ok=True)
adapter_dir.mkdir(parents=True, exist_ok=True)
tokenizer_dir.mkdir(parents=True, exist_ok=True)

# 1. Save PEFT LoRA adapter
print(f"Saving PEFT embedding adapter to: {adapter_dir}")
peft_embedding_model.save_pretrained(str(adapter_dir))

# 2. Save Tokenizer
print(f"Saving Tokenizer to:             {tokenizer_dir}")
tokenizer.save_pretrained(str(tokenizer_dir))

# 3. Save Evaluation Results
eval_summary = {
    "base_model": CONFIG.base_model_name,
    "fine_tuning_method": "QLoRA (4-bit NF4 + PEFT LoRA)",
    "training_time_minutes": round(train_time_min, 2),
    "dimensions": CONFIG.embedding_dim,
    "query_prefix": CONFIG.query_prefix,
    "metrics_comparison": {
        "base_embedding": base_retrieval_metrics,
        "fine_tuned_embedding_qlora": ft_retrieval_metrics,
    }
}
with open(package_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(eval_summary, f, indent=2)

with open(package_dir / "training_config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG.__dict__, f, indent=2)

# 4. Generate Integration README
readme_content = f"""# Phase 3 — Fine-Tuned Embedding Model (QLoRA)

## Model Overview
- **Base Embedding:** `{CONFIG.base_model_name}`
- **Architecture:** BERT-based Bi-Encoder (768 dimensions)
- **Pooling Strategy:** CLS Token Pooling (`outputs.last_hidden_state[:, 0]`)
- **Normalization:** L2 Normalization (`F.normalize(p=2)`)
- **Fine-Tuning Method:** QLoRA (4-bit NF4 + LoRA adapters)
- **LoRA Rank ($r$):** {CONFIG.lora_r} | **Alpha ($\alpha$):** {CONFIG.lora_alpha}

## Retrieval Evaluation Results (Full Corpus)
- **Recall@1:**  Base {base_retrieval_metrics['recall_at_1']:.4f} -> Fine-Tuned {ft_retrieval_metrics['recall_at_1']:.4f} ({ft_retrieval_metrics['recall_at_1'] - base_retrieval_metrics['recall_at_1']:+.4f})
- **Recall@5:**  Base {base_retrieval_metrics['recall_at_5']:.4f} -> Fine-Tuned {ft_retrieval_metrics['recall_at_5']:.4f} ({ft_retrieval_metrics['recall_at_5'] - base_retrieval_metrics['recall_at_5']:+.4f})
- **Recall@10:** Base {base_retrieval_metrics['recall_at_10']:.4f} -> Fine-Tuned {ft_retrieval_metrics['recall_at_10']:.4f} ({ft_retrieval_metrics['recall_at_10'] - base_retrieval_metrics['recall_at_10']:+.4f})
- **MRR@10:**   Base {base_retrieval_metrics['mrr_at_10']:.4f} -> Fine-Tuned {ft_retrieval_metrics['mrr_at_10']:.4f} ({ft_retrieval_metrics['mrr_at_10'] - base_retrieval_metrics['mrr_at_10']:+.4f})

## How to Load in Production RAG:
```python
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

base_model = AutoModel.from_pretrained("{CONFIG.base_model_name}")
fine_tuned_embedding = PeftModel.from_pretrained(base_model, "embedding_phase3/adapter")
```
"""
with open(package_dir / "README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

# 5. Save Requirements
requirements_txt = """transformers>=4.44.0
peft>=0.12.0
bitsandbytes>=0.43.0
torch>=2.2.0
faiss-cpu>=1.9.0
"""
with open(package_dir / "requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements_txt)

# 6. Create Zip Archive for Kaggle Download
shutil.make_archive("embedding_phase3", "zip", package_dir)
print("\n[Package Created]")
print("Zip archive ready at: /kaggle/working/embedding_phase3.zip")

print("\n" + "=" * 70)
print("SECTION 20: CHECKLIST OF FILES TO DOWNLOAD FROM KAGGLE")
print("=" * 70)
print("""
Download `embedding_phase3.zip` (or the folder `embedding_phase3/`) and place it in the project root:

[x] embedding_phase3/
    ├── adapter/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── README.md
    ├── tokenizer/
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   └── vocab.txt
    ├── evaluation_results.json
    ├── training_config.json
    ├── README.md
    └── requirements.txt
""")
print("Phase 3 QLoRA Embedding Training Script Completed Successfully!")
