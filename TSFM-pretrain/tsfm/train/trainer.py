from __future__ import annotations

import functools
import logging
import os
import random
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, optim
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsfm.distributed import setup_logging
from tsfm.utils import Timer, generate_run_name

logger = logging.getLogger(__name__)


def worker_init_fn(worker_id):
    # Get the Global Rank (GPU ID)
    # If using torchrun/DDP, 'RANK' is in the environment variables
    rank = int(os.environ.get("RANK", 0))

    # We use a large multiplier to ensure seeds don't overlap between ranks
    global_seed = (rank * 10000) + worker_id

    # Seed all RNGs
    np.random.seed(global_seed)
    random.seed(global_seed)
    torch.manual_seed(global_seed)


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.master_process and self.mlflow_run is not None:
                import mlflow

                mlflow.end_run()
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TSFMs, including:

    - Environment setup and distributed training configuration
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    model : nn.Module
        The TSFM model to be trained.
    dataloader : DataLoader
        DataLoader providing training data batches.
    val_dataloader : DataLoader, optional
        DataLoader providing validation data batches. If None, validation is skipped.
    """

    def __init__(self, config, model, dataloader, val_dataloader=None):
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.model_config = config.model
        # Batch size per GPU
        self.batch_size = config.micro_batch_size * config.gradient_accumulation_steps
        self.configure_ddp()
        self.build_model(model)
        self.configure_wandb()
        self.configure_mlflow()
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()

        self.context_length = config.data["context_length"]
        self.points_per_step = self.context_length * self.batch_size
        self.iter_dataloader = iter(self.dataloader)

        self._validation_setup()

    def log(
        self,
        level: int,
        message: str,
        *args,
        main_process_only: bool = True,
        **kwargs,
    ) -> None:
        if main_process_only and not self.master_process:
            return
        logger.log(level, message, *args, **kwargs)

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.batch_size
            self.batch_size = self.batch_size * self.ddp_world_size
            setup_logging()

            self.log(logging.INFO, "DDP training with %s processes", self.ddp_world_size)
            self.log(logging.INFO, "Per-GPU batch size: %s", original_batch_size)
            self.log(logging.INFO, "Effective batch size: %s", self.batch_size)
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            setup_logging()
            self.log(logging.INFO, "No DDP training")
            self.log(logging.INFO, "Effective batch size: %s", self.batch_size)

        self.curr_step = 0  # Initialize current step for training

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def build_model(self, model):
        """Build and initialize the TSFM."""

        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log(logging.INFO, "Model has %s parameters.", num_params)

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            self.log(logging.INFO, "Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            self.model = DDP(model, device_ids=[self.ddp_local_rank], broadcast_buffers=False)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            import wandb

            if self.config.wandb_name is None:
                self.config.wandb_name = generate_run_name()

            only_load_model = getattr(self.config, "only_load_model", False)
            resume_training = getattr(self.config, "resume_training", True)
            checkpoint_dir = getattr(self.config, "checkpoint_dir", None)
            if only_load_model:
                # Keep old checkpoint_dir for loading, but save new checkpoints under the new run name
                self.config.load_checkpoint_dir = checkpoint_dir
                self.config.checkpoint_dir = os.path.join("./checkpoints", self.config.wandb_name)
            else:
                self.config.checkpoint_dir = self._resolve_run_checkpoint_dir(
                    checkpoint_dir,
                    self.config.wandb_name,
                    resume_training=resume_training,
                )
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            wandb_id = getattr(self.config, "wandb_id", None)
            if wandb_id is None and resume_training and not only_load_model:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def configure_mlflow(self):
        """Set up MLflow logging (Simplified)."""
        if not (self.config.mlflow_log and self.master_process):
            self.mlflow_run = None
            return

        import mlflow

        mlflow.set_tracking_uri(f"file://{os.path.abspath(self.config.mlflow_dir)}")
        mlflow.set_experiment(self.config.mlflow_experiment)

        run_name = (
            self.config.mlflow_run_name
            or getattr(self.config, "wandb_name", None)
            or generate_run_name()
        )

        checkpoint_dir = getattr(self.config, "checkpoint_dir", None)
        resume_training = getattr(self.config, "resume_training", True)
        self.config.checkpoint_dir = self._resolve_run_checkpoint_dir(
            checkpoint_dir, run_name, resume_training=resume_training
        )
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        id_path = os.path.join(self.config.checkpoint_dir, "mlflow_id.txt")
        run_id = None

        resume_training = getattr(self.config, "resume_training", True)
        if os.path.exists(id_path) and resume_training:
            with open(id_path, "r") as f:
                run_id = f.read().strip()
            self.log(logging.INFO, "Resuming MLflow run: %s", run_id)

        # If run_id is None, a new run will be created
        self.mlflow_run = mlflow.start_run(
            run_id=run_id,
            run_name=run_name,
            tags=self.config.mlflow_tags if hasattr(self.config, "mlflow_tags") else None,
        )

        if not run_id:
            with open(id_path, "w") as f:
                f.write(self.mlflow_run.info.run_id)

            # Log params only on new runs to avoid "Changing param values is not allowed" errors
            flat_params = {}
            for key, value in vars(self.config).items():
                if not key.startswith("_") and not callable(value):
                    # Handle SimpleNamespace if present, otherwise just log value
                    if isinstance(value, SimpleNamespace):
                        for k, v in vars(value).items():
                            flat_params[f"{key}.{k}"] = v
                    else:
                        flat_params[key] = value
            mlflow.log_params(flat_params)

    def configure_optimizer(self):
        """Configure optimizer and scheduler."""

        self.optimizer = optim.AdamW(
            params=self.raw_model.parameters(),
            lr=self.config.optim["lr"],
            weight_decay=self.config.optim["weight_decay"],
            fused="cuda" in self.config.device,
            betas=tuple(self.config.optim.get("betas", (0.9, 0.999))),
        )
        self.scheduler = self.build_scheduler()

    def build_scheduler(self, last_epoch: int = -1):
        return OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.config.optim["lr"],
            total_steps=self.config.max_steps,
            pct_start=self.config.optim["scheduler"]["warmup_pct"],
            anneal_strategy="cos",
            div_factor=self.config.optim["scheduler"]["lr_div_factor"],
            final_div_factor=self.config.optim["scheduler"]["lr_final_div_factor"],
        )

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        if self.amp:
            amp_dtype = (
                torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16
            )
            self.log(
                logging.INFO,
                "Automatic Mixed Precision is enabled (%s).",
                self.config.dtype,
            )
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
            # GradScaler is only needed for float16; bfloat16 shares float32's
            # exponent range and must not be loss-scaled.
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(amp_dtype == torch.float16)
            )
        else:
            self.amp_ctx = nullcontext()
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")
        ]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(
                checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
            )[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            self.log(logging.WARNING, "Error parsing checkpoint filenames: %s", e)
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        Respects the `resume_training` config flag - if False, skips checkpoint loading.
        """

        # Check if we should resume from checkpoint
        resume_training = getattr(self.config, "resume_training", True)
        if not resume_training:
            self.log(logging.INFO, "resume_training is False, starting from scratch.")
            return
        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        else:
            # When only_load_model, configure_wandb stashes the source dir in load_checkpoint_dir
            load_dir = getattr(self.config, "load_checkpoint_dir", None) or getattr(self.config, "checkpoint_dir", None)
            if load_dir:
                orig = self.config.checkpoint_dir
                self.config.checkpoint_dir = load_dir
                checkpoint_path = self.get_latest_checkpoint()
                self.config.checkpoint_dir = orig

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            self.log(logging.INFO, "No checkpoint found, starting from scratch.")
            return

        self.log(logging.INFO, "Loading checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        sd = checkpoint["state_dict"]
        # Fix _orig_mod. prefix mismatch when model_compile changed between runs
        sample_model_key = next(iter(self.raw_model.state_dict()))
        sample_ckpt_key = next(iter(sd))
        if sample_model_key.startswith("_orig_mod.") and not sample_ckpt_key.startswith("_orig_mod."):
            sd = {"_orig_mod." + k: v for k, v in sd.items()}
        elif not sample_model_key.startswith("_orig_mod.") and sample_ckpt_key.startswith("_orig_mod."):
            sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        # Drop keys with shape mismatch (e.g. pos_embed when context_length changed)
        model_sd = self.raw_model.state_dict()
        shape_skipped = [k for k, v in sd.items() if k in model_sd and v.shape != model_sd[k].shape]
        if shape_skipped:
            print(f"WARNING: skipping {len(shape_skipped)} keys with shape mismatch (re-initializing): {shape_skipped[:5]}")
            sd = {k: v for k, v in sd.items() if k not in shape_skipped}
        self.raw_model.load_state_dict(sd, strict=False)

        # Optionally load optimizer and scheduler state
        only_load_model = getattr(self.config, "only_load_model", False)
        if only_load_model:
            self.log(logging.INFO, "Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.curr_step = checkpoint["curr_step"]

            if self.curr_step > self.config.max_steps:
                raise ValueError(
                    f"Checkpoint step {self.curr_step} exceeds configured max_steps "
                    f"{self.config.max_steps}. Increase max_steps or disable resume_training."
                )

            scheduler_state = checkpoint["scheduler_state"]
            saved_total_steps = scheduler_state.get("total_steps", self.config.max_steps)
            if saved_total_steps == self.config.max_steps:
                self.scheduler.load_state_dict(scheduler_state)
            else:
                self.scheduler = self.build_scheduler(last_epoch=self.curr_step - 1)
                self.log(
                    logging.INFO,
                    "Rebuilt scheduler for resumed training: "
                    "checkpoint total_steps=%s, config max_steps=%s",
                    saved_total_steps,
                    self.config.max_steps,
                )

            self.log(logging.INFO, "Resuming training at step %s", self.curr_step)

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [
            f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")
        ]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:  # noqa: E722
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    self.log(logging.WARNING, "Error removing checkpoint %s: %s", ckpt_path, e)

    def _validation_setup(self):
        self.best_val_loss = float("inf")
        self.val_every = getattr(self.config, "val_every", self.config.max_steps + 1)
        self.val_batches = getattr(self.config, "val_batches", None)

        if self.val_every > self.config.max_steps:
            self.log(
                logging.WARNING,
                "val_every (%s) is greater than max_steps (%s). "
                "Validation will not be performed.",
                self.val_every,
                self.config.max_steps,
            )
            self.validate = None
            return

        if self.val_dataloader is not None and self.val_batches is not None:
            self.log(
                logging.INFO,
                "Validation will be performed every %s steps on %s batches "
                "of the validation dataloader.",
                self.val_every,
                self.val_batches,
            )
            self.validate = self.validate_dataset

        elif self.val_dataloader is not None:
            self.log(
                logging.INFO,
                "Validation will be performed every %s steps on the entire validation dataloader.",
                self.val_every,
            )
            self.val_batches = len(self.val_dataloader)
            self.validate = self.validate_dataset

        elif self.val_batches is not None:
            self.log(
                logging.INFO,
                "Validation will be performed every %s steps on %s batches "
                "of the training dataloader.",
                self.val_every,
                self.val_batches,
            )
            self.val_dataloader = self.iter_dataloader
            self.validate = self.validate_batches

    @torch.no_grad()
    def validate_dataset(self) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0

        for batch_cpu in self.val_dataloader:
            batch = {
                k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch_cpu.items()
            }
            with self.amp_ctx:
                output = self.model(**batch)
            total_loss += output.loss.item()
            total_mse += output.mse_loss.item()
            total_mae += output.mae_loss.item()
            n_batches += 1

            if self.val_batches and n_batches >= self.val_batches:
                break

        self.model.train()
        n = max(n_batches, 1)
        return {
            "val_loss": total_loss / n,
            "val_mse_loss": total_mse / n,
            "val_mae_loss": total_mae / n,
        }

    @staticmethod
    def _looks_like_run_checkpoint_dir(checkpoint_dir):
        if not checkpoint_dir:
            return False

        if not os.path.isdir(checkpoint_dir):
            return False

        run_markers = {"wand_id.txt", "mlflow_id.txt"}
        for entry in os.listdir(checkpoint_dir):
            if entry in run_markers:
                return True
            if entry.startswith("step-") and entry.endswith(".ckpt"):
                return True

        return False

    @classmethod
    def _resolve_run_checkpoint_dir(cls, checkpoint_dir, run_name, resume_training=False):
        if not checkpoint_dir:
            checkpoint_dir = "./checkpoints"

        if cls._looks_like_run_checkpoint_dir(checkpoint_dir):
            if resume_training:
                return checkpoint_dir
            checkpoint_dir = os.path.dirname(os.path.normpath(checkpoint_dir))

        return os.path.join(checkpoint_dir, run_name)

    @torch.no_grad()
    def validate_batches(self) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0

        while n_batches < self.val_batches:
            batch_cpu = next(self.val_dataloader)
            batch = {
                k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch_cpu.items()
            }
            with self.amp_ctx:
                output = self.model(**batch)
            total_loss += output.loss.item()
            total_mse += output.mse_loss.item()
            total_mae += output.mae_loss.item()
            n_batches += 1

            if self.val_batches and n_batches >= self.val_batches:
                break

        self.model.train()
        n = max(n_batches, 1)
        return {
            "val_loss": total_loss / n,
            "val_mse_loss": total_mse / n,
            "val_mae_loss": total_mae / n,
        }

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(
                range(self.curr_step, self.config.max_steps),
                total=self.config.max_steps,
                initial=self.curr_step,
                desc="Step",
                leave=True,
            )
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        print_every = getattr(self.config, "print_every", None)
        if print_every is None and self.master_process:
            self.log(logging.WARNING, "print_every not set in config, defaulting to 100 steps.")
            print_every = 100

        global_results = {}

        for step in step_progress:
            # Train the model on the batch
            with Timer() as train_timer:
                results, dataloader_time = self.run_batch(self.iter_dataloader)
            train_time = train_timer.elapsed - dataloader_time

            # Clear CUDA cache to free memory
            # torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"dataloader_time": dataloader_time, "train_time": train_time})

                # Update progress bar with rounded values for cleaner display
                step_progress.set_postfix(
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()}
                )

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

                if self.validate is not None and self.curr_step % self.val_every == 0:
                    with Timer() as val_timer:
                        val_results = self.validate()
                    val_results["val_time"] = val_timer.elapsed
                    results.update(val_results)

                # if val_results["val_loss"] < self.best_val_loss:
                #     self.best_val_loss = val_results["val_loss"]
                #     self.save_checkpoint(name="best.ckpt")

            global_results.update(results)
            global_results["points_trained"] = self.curr_step * self.points_per_step

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                import wandb

                # Add learning rate to results
                global_results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)

            # Logging to MLflow
            if self.mlflow_run is not None:
                import mlflow

                # Add learning rate to results
                global_results["lr"] = self.scheduler.get_last_lr()[0]
                mlflow.log_metrics(results, step=self.curr_step)

            # Fallback logging to console
            if (
                self.wandb_run is None
                and self.mlflow_run is None
                and self.master_process
                and step % print_every == 0
            ):
                tqdm.write(str({**{"step": self.curr_step}, **global_results}))

    def run_micro_batch(
        self,
        micro_batch_cpu: dict,
        micro_batch_idx: int = 0,
        num_micro_batches: int = 1,
    ) -> dict:
        micro_batch = {
            k: v.to(self.config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch_cpu.items()
        }

        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        with self.amp_ctx:
            output = self.model(**micro_batch)
            loss = output.loss

        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        return {
            "loss": scaled_loss.item(),
            "mse_loss": (output.mse_loss / num_micro_batches).item(),
            "mae_loss": (output.mae_loss / num_micro_batches).item(),
        }

    def run_batch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Trains the model on a batch of time series. Handles gradient accumulation by
        splitting the batch into micro-batches. Skips micro-batches on CUDA OOM errors.
        Updates model parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        dataloader: DataLoader
            DataLoader for the batch.

        Returns
        ------
        batch_loss: float
            Total loss for the batch.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Split the batch into micro-batches along the first dimension
        micro_batches_cpu = []

        with Timer() as dataloader_timer:
            for _ in range(self.config.gradient_accumulation_steps):
                micro_batches_cpu.append(next(dataloader))

        dataloader_time = dataloader_timer.elapsed

        results = {"loss": 0.0, "mse_loss": 0.0, "mae_loss": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches_cpu):
            try:
                micro_results = self.run_micro_batch(
                    micro_batch, idx, self.config.gradient_accumulation_steps
                )
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                self.log(
                    logging.WARNING,
                    "OOM error in micro-batch %s/%s at step %s. Skipping.",
                    idx + 1,
                    self.config.gradient_accumulation_steps,
                    self.curr_step,
                    main_process_only=False,
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / self.config.gradient_accumulation_steps
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due "
                f"to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Clip the gradient
        if self.config.optim["gradient_clipping"] > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optim["gradient_clipping"]
            )

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.scheduler.step()

        return results, dataloader_time
