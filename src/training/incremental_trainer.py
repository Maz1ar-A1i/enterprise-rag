"""
src/training/incremental_trainer.py
===================================
Native PyTorch Continual Fine-Tuner for SentenceTransformer Embedding Adapter.

Design & Safety Guarantees:
  1. Pure Native PyTorch: Uses DataLoader + model.smart_batching_collate directly,
     eliminating any third-party dataset package namespace conflicts.
  2. Multi-Threaded CPU Control: Uses max(1, CPU_COUNT - 2) threads to protect
     the running web server and keep live queries responsive.
  3. Real-Time Telemetry & Console Logs: Emits timestamped log lines and loss
     metrics per batch into `continual_training_status.json`.
  4. Anti-Catastrophic Forgetting: Mixes 25% experience replay anchors from the
     benchmark dataset with new knowledge pairs.
  5. Staging Validation: Saves to staging first, validates 768-dim vectors,
     creates a safety backup, and deploys.
"""

import os
import json
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

PROJECT_ROOT = Path(__file__).parent.parent.parent
ACTIVE_ADAPTER_DIR = PROJECT_ROOT / "embedding_adapter"
BACKUP_ADAPTER_DIR = PROJECT_ROOT / "embedding_adapter_backup"
STAGING_ADAPTER_DIR = PROJECT_ROOT / "embedding_adapter_staging"
STATUS_FILE = PROJECT_ROOT / "corpus" / "processed" / "continual_training_status.json"

log = logging.getLogger(__name__)


def write_status(
    status: str,
    progress: float = 0.0,
    epoch: int = 0,
    total_epochs: int = 1,
    current_step: int = 0,
    total_steps: int = 0,
    loss: Optional[float] = None,
    message: str = "",
    error: Optional[str] = None,
    log_line: Optional[str] = None,
) -> None:
    """Save real-time training progress and streaming console logs."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing logs if available
    existing_logs: List[str] = []
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                existing_logs = data.get("logs", [])
        except Exception:
            existing_logs = []

    timestamp = time.strftime("%H:%M:%S")
    if log_line:
        existing_logs.append(f"[{timestamp}] {log_line}")
        # Keep last 50 log lines
        existing_logs = existing_logs[-50:]

    payload = {
        "status": status,
        "progress": round(progress, 1),
        "epoch": epoch,
        "total_epochs": total_epochs,
        "current_step": current_step,
        "total_steps": total_steps,
        "loss": round(loss, 5) if loss is not None else None,
        "message": message,
        "error": error,
        "logs": existing_logs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_status() -> Dict[str, Any]:
    """Read the latest training status."""
    if not STATUS_FILE.exists():
        return {
            "status": "idle",
            "progress": 0.0,
            "message": "Ready to fine-tune on queued knowledge.",
            "logs": ["System ready. Ingest documents and click Start Background Fine-Tuning."],
        }
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"status": "idle", "progress": 0.0, "logs": []}


class IncrementalTrainer:
    """
    Native PyTorch Continual Trainer for the BGE-base embedding model.
    """

    def __init__(
        self,
        base_model_name: str = "BAAI/bge-base-en-v1.5",
        adapter_dir: Optional[Path] = None,
        batch_size: int = 4,
        lr: float = 2e-5,
    ):
        self.base_model_name = base_model_name
        self.adapter_dir = adapter_dir or ACTIVE_ADAPTER_DIR
        self.batch_size = batch_size
        self.lr = lr
        self._is_training = False

    def train_epoch(
        self,
        epochs: int = 1,
        replay_ratio: float = 0.25,
        clear_buffer_on_finish: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute native PyTorch continual training loop on CPU with streaming telemetry.
        """
        from src.training.continual_buffer import ContinualBuffer

        buffer = ContinualBuffer()
        examples = buffer.prepare_dataset(replay_ratio=replay_ratio)

        if not examples:
            write_status(
                status="idle",
                progress=0.0,
                message="No training pairs in buffer to train on.",
                log_line="Buffer is empty. Ingest documents first.",
            )
            return {"status": "skipped", "message": "Buffer is empty."}

        self._is_training = True
        t_start = time.perf_counter()

        # Step 1: Thread allocation for zero-downtime query serving
        n_threads = max(1, (os.cpu_count() or 4) - 2)
        torch.set_num_threads(n_threads)

        write_status(
            status="running",
            progress=5.0,
            epoch=0,
            total_epochs=epochs,
            message="Preparing dataset and allocating CPU worker threads...",
            log_line=f"Initialized CPU worker with {n_threads} OpenMP threads. Reserved cores for live web queries.",
        )

        try:
            # Step 2: Load warm-start model
            model_source = str(self.adapter_dir) if self.adapter_dir.exists() else self.base_model_name
            write_status(
                status="running",
                progress=15.0,
                epoch=0,
                total_epochs=epochs,
                message="Loading warm-start model weights...",
                log_line=f"Loading active model weights from {model_source}...",
            )

            model = SentenceTransformer(model_source, device="cpu")
            model.max_seq_length = 384

            write_status(
                status="running",
                progress=25.0,
                epoch=0,
                total_epochs=epochs,
                message="Configuring native PyTorch loader & loss...",
                log_line=f"Loaded SentenceTransformer ({len(examples)} total training samples including replay anchors).",
            )

            # Step 3: Configure DataLoader & native loss
            train_dataloader = DataLoader(
                examples,
                shuffle=True,
                batch_size=self.batch_size,
                collate_fn=model.smart_batching_collate,
            )
            loss_fn = losses.MultipleNegativesRankingLoss(model=model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

            total_batches = len(train_dataloader)
            total_steps = total_batches * epochs
            global_step = 0
            last_loss_val = None

            # Step 4: Native PyTorch Training Loop
            model.train()
            write_status(
                status="running",
                progress=30.0,
                epoch=1,
                total_epochs=epochs,
                message="Starting CPU backpropagation...",
                log_line=f"Starting training: {epochs} epoch(s), {total_batches} batches/epoch, batch_size={self.batch_size}.",
            )

            for epoch_idx in range(1, epochs + 1):
                for batch_idx, (batch_features, labels) in enumerate(train_dataloader, 1):
                    global_step += 1
                    optimizer.zero_grad()
                    loss = loss_fn(batch_features, labels)
                    loss.backward()
                    optimizer.step()

                    last_loss_val = float(loss.item())
                    progress_pct = 30.0 + ((global_step / total_steps) * 55.0)

                    write_status(
                        status="running",
                        progress=progress_pct,
                        epoch=epoch_idx,
                        total_epochs=epochs,
                        current_step=global_step,
                        total_steps=total_steps,
                        loss=last_loss_val,
                        message=f"Epoch {epoch_idx}/{epochs} - Batch {batch_idx}/{total_batches} - Loss: {last_loss_val:.4f}",
                        log_line=f"Epoch {epoch_idx}/{epochs} | Batch {batch_idx}/{total_batches} | Step {global_step}/{total_steps} | Loss: {last_loss_val:.4f}",
                    )

            # Step 5: Save to Staging Directory
            write_status(
                status="running",
                progress=88.0,
                epoch=epochs,
                total_epochs=epochs,
                loss=last_loss_val,
                message="Saving model to staging directory...",
                log_line="Training loop complete. Saving checkpoint to staging directory...",
            )

            if STAGING_ADAPTER_DIR.exists():
                shutil.rmtree(STAGING_ADAPTER_DIR, ignore_errors=True)
            STAGING_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
            model.save(str(STAGING_ADAPTER_DIR))

            # Step 6: Validate Staging Model Output
            write_status(
                status="running",
                progress=92.0,
                epoch=epochs,
                total_epochs=epochs,
                loss=last_loss_val,
                message="Validating trained vector output...",
                log_line="Running probe vector validation...",
            )

            validated_model = SentenceTransformer(str(STAGING_ADAPTER_DIR), device="cpu")
            test_emb = validated_model.encode(["Sanity verification probe"], normalize_embeddings=True)
            if test_emb.shape[-1] != 768:
                raise ValueError(f"Vector dimension mismatch: got {test_emb.shape[-1]}, expected 768.")

            # Step 7: Atomic Backup & Deployment
            write_status(
                status="running",
                progress=96.0,
                epoch=epochs,
                total_epochs=epochs,
                loss=last_loss_val,
                message="Creating safe backup and deploying updated weights...",
                log_line="Validated 768-dim embeddings. Creating backup and deploying new adapter...",
            )

            if self.adapter_dir.exists():
                if BACKUP_ADAPTER_DIR.exists():
                    shutil.rmtree(BACKUP_ADAPTER_DIR, ignore_errors=True)
                shutil.copytree(self.adapter_dir, BACKUP_ADAPTER_DIR)

            for item in STAGING_ADAPTER_DIR.iterdir():
                target = self.adapter_dir / item.name
                if item.is_dir():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)

            # Step 8: Clear Buffer
            if clear_buffer_on_finish:
                buffer.clear(archive=True)

            total_elapsed = round(time.perf_counter() - t_start, 1)

            write_status(
                status="completed",
                progress=100.0,
                epoch=epochs,
                total_epochs=epochs,
                current_step=total_steps,
                total_steps=total_steps,
                loss=last_loss_val,
                message=f"Fine-tuning completed in {total_elapsed}s (Final Loss: {last_loss_val:.4f}). Ready to hot-reload.",
                log_line=f"[SUCCESS] Fine-tuning finished in {total_elapsed}s! Adapter deployed. Ready for hot-reload.",
            )

            return {
                "status": "success",
                "samples_trained": len(examples),
                "epochs": epochs,
                "elapsed_seconds": total_elapsed,
                "final_loss": last_loss_val,
                "adapter_path": str(self.adapter_dir),
            }

        except Exception as e:
            err_msg = str(e)
            log.error(f"[Continual Trainer] Training failed: {err_msg}", exc_info=True)
            write_status(
                status="failed",
                progress=0.0,
                message=f"Training failed: {err_msg}",
                error=err_msg,
                log_line=f"[ERROR] Training aborted: {err_msg}",
            )
            return {"status": "error", "message": err_msg}

        finally:
            self._is_training = False
            if STAGING_ADAPTER_DIR.exists():
                shutil.rmtree(STAGING_ADAPTER_DIR, ignore_errors=True)

    def start_background_training(
        self,
        epochs: int = 1,
        replay_ratio: float = 0.25,
    ) -> threading.Thread:
        """
        Launches training in a separate background daemon thread.
        Returns immediately with zero blocking to caller.
        """
        if self._is_training:
            raise RuntimeError("A training run is already in progress.")

        thread = threading.Thread(
            target=self.train_epoch,
            kwargs={
                "epochs": epochs,
                "replay_ratio": replay_ratio,
                "clear_buffer_on_finish": True,
            },
            daemon=True,
            name="ContinualTrainingWorker",
        )
        thread.start()
        return thread
