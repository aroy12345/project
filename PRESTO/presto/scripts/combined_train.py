#!/usr/bin/env python3

from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass, replace
from tqdm.auto import tqdm
import os
import pickle
import numpy as np
from icecream import ic
from contextlib import nullcontext
import time
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Assuming data_temp.py is in the same directory or accessible via PYTHONPATH
try:
    from data_temp import generate_dummy_datapoint, create_batch_from_datapoints
except ImportError:
    print("Error: Could not import from data_temp.py. Make sure it's accessible.")
    # Define dummy functions if import fails, to allow script structure analysis
    def generate_dummy_datapoint(**kwargs): return {}
    def create_batch_from_datapoints(datapoints, device): return {}


# Assuming franka_util provides the fk function
try:
    from presto.data.franka_util import franka_fk
except ImportError:
    print("Warning: Could not import franka_fk. Using placeholder.")
    def franka_fk(q):
        # Placeholder: returns zeros matching expected output shape [B, S, 3]
        return th.zeros(q.shape[0], q.shape[1], 3, device=q.device)

# Assuming network factory provides the model
from presto.network.combined import PrestoGIGA
# Assuming network factory provides scheduler getter
from presto.network.factory import get_scheduler, DiffusionConfig, ModelConfig

# Assuming diffusion utils are available
from presto.diffusion.util import pred_x0, diffusion_loss

# Assuming cost functions are available
try:
    from presto.cost.curobo_cost import CuroboCost
    # Add imports for cached cost components
    from presto.cost.cached_curobo_cost import CachedCuroboCost, cached_curobo_cost_with_ng
except ImportError:
    print("Warning: CuroboCost or CachedCuroboCost components not found. Collision/Distance losses and reweighting will be disabled.")
    CuroboCost = None
    CachedCuroboCost = None
    # Define a placeholder if the specific function is missing
    def cached_curobo_cost_with_ng(u_pred_sd, cost_fn_lambda):
        print("Warning: cached_curobo_cost_with_ng not available. Returning 0.0")
        # Return a tensor to avoid downstream errors
        return th.tensor(0.0, device=u_pred_sd.device)

# Assuming normalization utils are available
try:
    from presto.data.normalize import Normalize
except ImportError:
    print("Warning: Normalize not found. Using placeholder identity normalization.")
    class Normalize: # Basic placeholder
        def __init__(self, mean, std, device='cpu'):
            self.mean = th.tensor(mean, device=device, dtype=th.float32)
            self.std = th.tensor(std, device=device, dtype=th.float32)
            self.device = device
        def __call__(self, x): return (x - self.mean) / self.std
        def unnormalize(self, x): return x * self.std + self.mean
        def to(self, device):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            self.device = device
            return self
        @classmethod
        def identity(cls, dim, device='cpu'):
            return cls(mean=th.zeros(dim), std=th.ones(dim), device=device)


# Assuming checkpointing utils are available
from presto.util.ckpt import (load_ckpt, save_ckpt, last_ckpt)
# Assuming path utils are available (using basic os.path now)
# from presto.util.path import RunPath
import datetime

# Assuming logger utils are available
try:
    from presto.util.wandb_logger import WandbLogger
except ImportError:
    print("Warning: WandbLogger not found. Logging will be printed to console.")
    WandbLogger = None

import types

# --- Helper Function ---
def dict_to_namespace(d):
    """Recursively converts a dictionary to SimpleNamespace."""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return types.SimpleNamespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

# --- GIGA Loss Helper Functions ---
def _qual_loss_fn(pred_logits, target):
    """Calculates BCE loss from logits."""
    # pred_logits shape: [B, M]
    # target shape: [B, M]
    # Ensure target is float for BCEWithLogitsLoss
    return F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")

def _rot_loss_fn(pred, target):
    # Target shape: [B, M, 4] (assuming single GT from dummy data)
    return _quat_loss_fn(pred, target) # Directly compute loss

def _quat_loss_fn(pred, target):
    # pred/target shape: [B, M, 4]
    # Ensure normalization for safety, although model output should be normalized
    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    # Return 1 - |dot product|, shape [B, M]
    return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=-1))

def _width_loss_fn(pred, target):
    # Scale width prediction/target before MSE as in GIGA
    # Shape: [B, M]
    return F.mse_loss(40 * pred, 40 * target, reduction="none")

def _tsdf_loss_fn(pred_logits, target):
    """Calculates BCE loss from logits for TSDF."""
    # pred_logits shape: [B, N_tsdf]
    # target shape: [B, N_tsdf]
    # Using BCEWithLogitsLoss, ensure target is float
    return F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")

def train_loop(
    log_dir: str,
    model: nn.Module,
    sched: SchedulerMixin,
    optimizer: th.optim.Optimizer,
    lr_scheduler, # Add type hint if available
    scaler: th.cuda.amp.GradScaler, # Or torch.amp.GradScaler
    writer, # Add type hint if available (e.g., WandbLogger or None)
    cost, # Add type hint if available (e.g., CuroboCost or None)
    normalizer, # Add type hint if available (e.g., Normalize)
    fk_fn, # Add type hint if available (e.g., Callable)
    device: str,
    # --- Hyperparameters ---
    num_epochs: int,
    batch_size: int,
    use_amp: bool,
    diffusion_coef: float,
    grasp_coef: float,
    tsdf_coef: float,
    collision_coef: float,
    distance_coef: float,
    euclidean_coef: float,
    reweight_loss_by_coll: bool,
    x0_type: str,
    x0_iter: int,
    cost_margin: float, # Assuming cost object has margin internally if needed
    log_by: str, # 'epoch' or 'step'
    step_max: Optional[int],
    save_epoch: int,
    # --- Dummy Data Params ---
    seq_len: int,
    obs_dim: int,
    cond_dim: int,
    tsdf_dim: int,
    num_grasp_points: int,
    num_tsdf_points: int,
    # Add any other parameters train_loop might need from train() call
):
    """
    Main training loop for the combined PrestoGIGA model using dummy data.
    """
    global_step = 0
    start_time = time.time()

    # Determine max diffusion steps
    diff_step_max: int = sched.config.num_train_timesteps
    if step_max is not None:
        diff_step_max = step_max

    # Precompute iterative diffusion steps if needed
    substepss = None
    if x0_type == 'iter':
        # Cache all possible substeps shorter than `num_train_timesteps`.
        # Note: Using sched.config.num_train_timesteps here, adjust if needed
        substepss_list = [
            th.round(
                th.arange(i, 0, -i / x0_iter, device=device)).long()
            for i in range(1, sched.config.num_train_timesteps + 1)]
        # Pad shorter sequences to the length of the longest (x0_iter)
        max_len = x0_iter
        padded_substepss = [F.pad(s, (0, max_len - len(s))) for s in substepss_list]
        substepss = th.stack(padded_substepss, dim=0).to(
            device=device,
            dtype=th.long).sub_(1).clamp_(min=0) # Subtract 1 for 0-based indexing, clamp

    ic("Starting training loop...")

    for epoch in range(num_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_grasp_loss = 0.0
        epoch_tsdf_loss = 0.0
        epoch_collision_loss = 0.0
        epoch_distance_loss = 0.0
        epoch_euclidean_loss = 0.0

        # === Placeholder for data loading ===
        # Simulate a dataset size for progress bar
        dummy_dataset_size = 100 # Example: Simulate 100 datapoints
        num_batches = math.ceil(dummy_dataset_size / batch_size)
        # ===================================

        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx in progress_bar:
            optimizer.zero_grad()

            # === Generate Dummy Batch ===
            batch_list = [generate_dummy_datapoint(seq_len, obs_dim, cond_dim, tsdf_dim, num_grasp_points, num_tsdf_points) for _ in range(batch_size)]
            batch = create_batch_from_datapoints(batch_list, device)
            # ===========================

            # Extract ground truth data from batch
            # Transpose immediately to [B, C, T] format for diffusion
            true_action_ds = batch['trajectory'].swapaxes(-1, -2) # Shape: [B, Obs, Seq] -> [B, C, T]
            cond_data = batch['env-label']   # Shape: [B, Cond]
            tsdf_grid = batch['tsdf']        # Shape: [B, 1, D, D, D]
            grasp_query_points = batch['grasp_query_points'] # Shape: [B, N_grasp, 3]
            tsdf_query_points = batch['tsdf_query_points']   # Shape: [B, N_tsdf, 3]

            # Ground truth for GIGA/TSDF parts
            gt_grasp_label = batch['grasp_qual_labels'] # Shape: [B, N_grasp, 1] -> Squeeze later if needed
            gt_grasp_rot = batch['grasp_rot_labels']    # Shape: [B, N_grasp, 4] -> Adjust loss fn if needed for 2 candidates
            gt_grasp_width = batch['grasp_width_labels']# Shape: [B, N_grasp, 1] -> Squeeze later if needed
            gt_tsdf_value = batch['tsdf_labels']        # Shape: [B, N_tsdf, 1] -> Squeeze later if needed


            # --- Diffusion Process ---
            steps = th.randint(
                0, diff_step_max,
                (batch_size,), device=device
            ).long()
            # Create noise with the same shape, device, and dtype as the target tensor
            noise = th.randn_like(true_action_ds)

            noisy_actions = sched.add_noise(true_action_ds, noise, steps)

            # --- Model Forward Pass ---
            with th.cuda.amp.autocast(enabled=use_amp):
                # Transpose noisy_actions from [B, C, T] to [B, T, C] for the model's PatchEmbed
                model_input_sample = noisy_actions.swapaxes(-1, -2)
                model_output = model(
                    sample=model_input_sample, # Pass the transposed tensor
                    timestep=steps,
                    class_labels=cond_data,
                    tsdf=tsdf_grid,
                    p=grasp_query_points,
                    p_tsdf=tsdf_query_points,
                    mode="joint",
                    return_dict=True # Ensure we get a dictionary back
                )

                # Extract predictions
                # Model output 'sample' is expected to be [B, C, T] from forward_diffusion
                pred_noise_or_x0 = model_output['sample']
                pred_grasp_qual = model_output['qual']
                pred_grasp_rot = model_output['rot']
                pred_grasp_width = model_output['width']
                pred_tsdf = model_output.get('tsdf') # Use .get() for optional TSDF output

                # --- Loss Calculation ---
                total_loss = 0.0
                loss_dict = {}

                # 1. Diffusion Loss (PRESTO)
                # Expects preds [B, C, T], trajs [B, C, T], noise [B, C, T]
                loss_ddpm = diffusion_loss(
                    pred_noise_or_x0, true_action_ds,
                    noise, noisy_actions, steps, # noisy_actions is still [B, C, T] here
                    sched=sched,
                    reduction=('none' if reweight_loss_by_coll else 'mean')
                )
                # Note: Reweighting requires calculating collision cost first

                # Initialize auxiliary losses
                loss_coll = th.zeros_like(loss_ddpm.mean())
                loss_dist = th.zeros_like(loss_ddpm.mean())
                loss_eucd = th.zeros_like(loss_ddpm.mean())
                pred_traj_ds = None # Predicted trajectory at t=0

                # Calculate predicted trajectory x0 if needed for aux losses or reweighting
                needs_traj = collision_coef > 0 or distance_coef > 0 or euclidean_coef > 0 or reweight_loss_by_coll
                if needs_traj:
                    if x0_type == 'step':
                        pred_traj_ds = pred_x0(sched, steps, pred_noise_or_x0, noisy_actions)
                    elif x0_type == 'iter':
                        # Iterative refinement to get x0
                        pred_traj_ds = noisy_actions
                        if isinstance(sched, DDIMScheduler) and substepss is not None:
                             # Ensure substeps indices are valid for the current steps
                            current_substeps = substepss[steps] # Shape [B, x0_iter]
                            for i in range(x0_iter):
                                iter_steps = current_substeps[..., i]
                                # Need to re-run the diffusion part of the model for intermediate steps
                                iter_preds = model(x=pred_traj_ds, timestep=iter_steps, cond=cond_data, mode="diffusion")['diffusion']
                                # Use scheduler step function
                                step_output = sched.step(
                                     iter_preds, iter_steps[0].item(), pred_traj_ds # Assuming step takes single timestep value
                                     # Pass other necessary args for sched.step if any
                                 )
                                pred_traj_ds = step_output.prev_sample
                        else:
                             print(f"Warning: x0_type='iter' requires DDIMScheduler and precomputed substeps. Falling back to single step prediction.")
                             pred_traj_ds = pred_x0(sched, steps, pred_noise_or_x0, noisy_actions)
                    else: # 'legacy' or other unsupported
                         raise ValueError(f"Unsupported x0_type: {x0_type}")

                    # Unnormalize predicted trajectory for cost functions
                    # Assuming normalizer has 'unnormalize' method and works on shape [B, Seq, Obs]
                    u_pred_sd = normalizer.unnormalize(pred_traj_ds.swapaxes(-1, -2)) # Shape [B, Seq, Obs]

                    # 2. Collision Loss (PRESTO Aux)
                    if collision_coef > 0 and cost is not None:
                        # Assuming cost function takes [B, Seq, Obs] and condition dict
                        # The dummy batch needs a 'col_label' field compatible with the cost function
                        col_label = batch.get('col_label', None) # Get collision geometry info if available
                        if col_label is not None:
                             loss_coll = cost(u_pred_sd, col_label).mean()
                        else:
                             print("Warning: Collision coefficient > 0 but no 'col_label' found in dummy batch for cost function.")
                        total_loss += collision_coef * loss_coll
                        loss_dict['collision_loss'] = loss_coll.item()
                        epoch_collision_loss += loss_coll.item()


                    # 3. Distance Loss (PRESTO Aux)
                    if distance_coef > 0:
                        loss_dist = F.mse_loss(u_pred_sd[:, 1:, :], u_pred_sd[:, :-1, :])
                        total_loss += distance_coef * loss_dist
                        loss_dict['distance_loss'] = loss_dist.item()
                        epoch_distance_loss += loss_dist.item()

                    # 4. Euclidean Loss (PRESTO Aux)
                    if euclidean_coef > 0 and fk_fn is not None:
                         x_pred_sd = fk_fn(u_pred_sd) # FK on predicted joints [B, Seq, Task]
                         with th.no_grad():
                             # Unnormalize ground truth action (already swapped)
                             u_true_sd = normalizer.unnormalize(batch['trajectory'].swapaxes(-1, -2))
                             x_true_sd = fk_fn(u_true_sd) # FK on true joints
                         loss_eucd = F.mse_loss(x_pred_sd, x_true_sd)
                         total_loss += euclidean_coef * loss_eucd
                         loss_dict['euclidean_loss'] = loss_eucd.item()
                         epoch_euclidean_loss += loss_eucd.item()


                # Apply Diffusion Reweighting if enabled
                if reweight_loss_by_coll:
                     if cost is not None and u_pred_sd is not None:
                         with th.no_grad():
                             col_label = batch.get('col_label', None)
                             if col_label is not None:
                                 # Calculate collision cost per trajectory point/batch element
                                 coll_cost_per_element = cost(u_pred_sd, col_label) # Shape [B] or [B, Seq]? Assume [B] for now
                                 # Simple weighting: increase weight for higher cost (more collision)
                                 # Normalize cost, add 1, potentially clamp? Cost function details matter here.
                                 # Example: weight = 1.0 + torch.clamp(coll_cost_per_element / cost_margin, 0, 5) # Clamp max weight
                                 # This needs careful tuning based on cost function output range
                                 weight = 1.0 + coll_cost_per_element # Simplified weighting
                                 weight = weight / weight.mean() # Normalize weights per batch
                                 # Reshape weight to match loss_ddpm if it's per-element [B, Obs, Seq]
                                 if len(loss_ddpm.shape) > 1:
                                     weight = weight.view(-1, *([1]*(len(loss_ddpm.shape)-1)))

                                 loss_ddpm = (loss_ddpm * weight).mean() # Apply weight and take mean
                             else:
                                 print("Warning: Reweighting enabled but no 'col_label' for cost function.")
                                 loss_ddpm = loss_ddpm.mean() # Fallback to simple mean
                     else:
                         print("Warning: Reweighting enabled but cost function or predicted trajectory unavailable.")
                         loss_ddpm = loss_ddpm.mean() # Fallback to simple mean
                else:
                     loss_ddpm = loss_ddpm.mean() # Simple mean if not reweighting

                total_loss += diffusion_coef * loss_ddpm
                loss_dict['diffusion_loss'] = loss_ddpm.item()
                epoch_diffusion_loss += loss_ddpm.item()


                # 5. Grasp Quality Loss (GIGA) - BCE loss
                loss_qual = _qual_loss_fn(pred_grasp_qual, gt_grasp_label.squeeze(-1)).mean() # Mean over points and batch
                total_loss += grasp_coef * loss_qual
                loss_dict['grasp_qual_loss'] = loss_qual.item()

                # 6. Grasp Rotation Loss (GIGA) - Min quaternion distance loss
                loss_rot = _rot_loss_fn(pred_grasp_rot, gt_grasp_rot).mean() # Mean over points and batch
                total_loss += grasp_coef * loss_rot # Use same grasp_coef
                loss_dict['grasp_rot_loss'] = loss_rot.item()

                # 7. Grasp Width Loss (GIGA) - Scaled MSE loss
                loss_width = _width_loss_fn(pred_grasp_width, gt_grasp_width.squeeze(-1)).mean() # Mean over points and batch
                total_loss += grasp_coef * 0.01 * loss_width # Use GIGA's 0.01 scaling relative to grasp_coef
                loss_dict['grasp_width_loss'] = loss_width.item()

                # Aggregate grasp losses for logging
                current_grasp_loss = loss_qual + loss_rot + 0.01 * loss_width
                epoch_grasp_loss += current_grasp_loss.item() # Log combined grasp loss metric

                # 8. TSDF Prediction Loss (GIGA/Combined) - BCE loss
                if tsdf_coef > 0:
                    loss_tsdf = _tsdf_loss_fn(pred_tsdf, gt_tsdf_value.squeeze(-1)).mean() # Mean over points and batch
                    total_loss += tsdf_coef * loss_tsdf
                    loss_dict['tsdf_loss'] = loss_tsdf.item()
                    epoch_tsdf_loss += loss_tsdf.item()


            # --- Backward Pass & Optimization Step ---
            scaler.scale(total_loss).backward()
            # Optional: Gradient clipping
            # scaler.unscale_(optimizer)
            # th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_total_loss += total_loss.item()
            global_step += 1

            # --- Logging ---
            log_data = {
                'train/total_loss': total_loss.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch,
                'train/global_step': global_step,
            }
            # Add individual losses to log
            log_data.update({f'train/{k}': v for k, v in loss_dict.items()})

            if writer is not None and log_by == 'step':
                 writer.log(log_data, step=global_step)

            progress_bar.set_postfix(loss=total_loss.item())


        # --- End of Epoch ---
        avg_epoch_loss = epoch_total_loss / num_batches
        ic(f"Epoch {epoch+1} finished. Avg Total Loss: {avg_epoch_loss:.4f}")

        # Log epoch averages
        if writer is not None:
             epoch_log_data = {
                 'train/epoch_avg_total_loss': avg_epoch_loss,
                 'train/epoch_avg_diffusion_loss': epoch_diffusion_loss / num_batches,
                 'train/epoch_avg_grasp_loss': epoch_grasp_loss / num_batches, # Combined grasp loss
                 'train/epoch_avg_tsdf_loss': epoch_tsdf_loss / num_batches if tsdf_coef > 0 else 0,
                 'train/epoch_avg_collision_loss': epoch_collision_loss / num_batches if collision_coef > 0 else 0,
                 'train/epoch_avg_distance_loss': epoch_distance_loss / num_batches if distance_coef > 0 else 0,
                 'train/epoch_avg_euclidean_loss': epoch_euclidean_loss / num_batches if euclidean_coef > 0 else 0,
                 'epoch': epoch + 1 # Log epoch number itself
             }
             # Log at the end of the epoch, using global_step as the step counter
             writer.log(epoch_log_data, step=global_step)


        # --- Save Checkpoint ---
        if (epoch + 1) % save_epoch == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(log_dir, f"ckpt_epoch_{epoch+1}.pth")
            save_data = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                # Save normalizer state if it's not identity/None
                'normalizer': normalizer.state_dict() if hasattr(normalizer, 'state_dict') else \
                              ({'mean': normalizer.mean, 'std': normalizer.std} if normalizer else None),
            }
            if lr_scheduler is not None:
                save_data['lr_scheduler'] = lr_scheduler.state_dict()

            save_ckpt(save_data, ckpt_path)
            ic(f"Checkpoint saved to {ckpt_path}")

    # --- End of Training ---
    end_time = time.time()
    ic(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
    if writer is not None:
        writer.finish()

# --- Main Training Function (No Hydra/Config) ---
def train(
    log_dir_base: str = "runs/combined_train_standalone",
    use_wandb: bool = False,
    wandb_project: str = "presto_giga_combined",
    wandb_entity: Optional[str] = None, # Set your wandb entity here
    # --- Key Hyperparameters ---
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    lr_warmup_steps: int = 100, # Reduced for shorter dummy training
    lr_schedule: str = 'cos', # 'cos' or None
    save_epoch: int = 10,
    log_by: str = 'epoch', # <-- ADD THIS PARAMETER (default to 'epoch')
    # --- Loss Coefficients ---
    diffusion_coef: float = 1.0,
    grasp_coef: float = 1.0, # GIGA combined loss coef
    tsdf_coef: float = 0.5,  # TSDF prediction loss coef
    collision_coef: float = 0.5, # PRESTO aux collision loss coef
    distance_coef: float = 0.5,  # PRESTO aux distance loss coef
    euclidean_coef: float = 0.5, # PRESTO aux euclidean loss coef
    reweight_loss_by_coll: bool = True, # PRESTO reweighting flag
    # --- Diffusion Settings ---
    beta_schedule='squaredcos_cap_v2',
    beta_start=0.0001,
    beta_end=0.02,
    num_train_timesteps=1000,
    prediction_type='epsilon', # 'epsilon' or 'sample'
    x0_type: str = 'step', # 'step' or 'iter'
    x0_iter: int = 1,
    step_max: Optional[int] = None, # Max diffusion timestep to use
    # --- Model Dimensions (from dummy data) ---
    seq_len: int = 50,
    obs_dim: int = 7,      # Franka joints
    cond_dim: int = 104,   # Example condition dim
    tsdf_dim: int = 32,    # TSDF grid size
    num_grasp_points: int = 64, # Num grasp queries
    num_tsdf_points: int = 32,  # Num TSDF queries
    # --- Model Architecture Params (Example for PrestoGIGA) ---
    model_depth: int = 12,
    model_num_heads: int = 6,
    model_hidden_size: int = 384,
    model_patch_size: int = 1, # For 1D DiT
    giga_encoder: str = 'voxel_simple_local', # Example GIGA encoder
    giga_decoder: str = 'simple_local', # Example GIGA decoder
    giga_c_dim: int = 32, # GIGA feature dimension
    # --- Hardware ---
    device_str: str = "auto", # "auto", "cuda", "cpu"
    use_amp: bool = True, # Automatic Mixed Precision
    # --- Checkpointing ---
    load_checkpoint_path: Optional[str] = None, # Path to load checkpoint from
    # --- Cost Function Params (if CuroboCost is used) ---
    cost_margin: float = 0.1, # Example margin for collision cost
    # Add other cost-related params if needed
):
    """
    Sets up and runs the training loop with hardcoded parameters.
    """
    # --- Device Setup ---
    if device_str == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        device = device_str
    ic(f"Using device: {device}")

    # --- Path Setup ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_dir_base, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    ic(f"Logging to: {log_dir}")

    # --- Logger ---
    writer = None
    if use_wandb and WandbLogger is not None:
        try:
            writer = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                # group=None, # Optional: group runs
                run_id=None, # Optional: resume run
                mode=None,
                # Pass key hyperparameters to wandb config
                cfg={
                    'learning_rate': learning_rate, 'batch_size': batch_size,
                    'num_epochs': num_epochs, 'diffusion_coef': diffusion_coef,
                    'grasp_coef': grasp_coef, 'tsdf_coef': tsdf_coef,
                    'collision_coef': collision_coef, 'distance_coef': distance_coef,
                    'euclidean_coef': euclidean_coef, 'obs_dim': obs_dim,
                    'cond_dim': cond_dim, 'tsdf_dim': tsdf_dim,
                    'model_depth': model_depth, 'model_hidden_size': model_hidden_size,
                    'beta_schedule': beta_schedule, 'prediction_type': prediction_type,
                 }
            )
            ic("Using WandbLogger.")
        except Exception as e:
            print(f"Failed to initialize WandbLogger: {e}. Logging disabled.")
            writer = None
    else:
        ic("Wandb logging disabled.")


    # --- Normalizer ---
    # Using identity normalizer for dummy data, replace if using real data stats
    normalizer = Normalize.identity(dim=obs_dim, device=device)
    ic("Using identity normalizer.")

    # --- Cost Function ---
    cost = None
    from presto.cost.curobo_cost import CuroboCost
    from presto.cost.cached_curobo_cost import CachedCuroboCost, cached_curobo_cost_with_ng
    if CuroboCost is not None and (collision_coef > 0 or distance_coef > 0 or reweight_loss_by_coll):
         ic("Instantiating CuroboCost...")
         # Use default CuroboCost config for simplicity
         from presto.cost.curobo_cost import CuroboCost # Ensure import
         try:
             cost_config = CuroboCost.Config() # Use default config
             cost = CuroboCost(cfg=cost_config, batch_size=batch_size, device=device)
             ic(f"CuroboCost instantiated with margin: {cost_margin}")
         except Exception as e:
             print(f"Failed to instantiate CuroboCost: {e}. Disabling related losses.")
             cost = None
             collision_coef = 0.0
             distance_coef = 0.0
             reweight_loss_by_coll = False
    else:
        ic("CuroboCost not needed or not available.")


    # --- Forward Kinematics ---
    current_fk_fn = None
    if euclidean_coef > 0:
        current_fk_fn = franka_fk # Use imported fk function
        ic("Using franka_fk function for Euclidean loss.")
    else:
        ic("Euclidean loss disabled, FK function not needed.")


    # --- Model ---
    ic("Instantiating PrestoGIGA model...")
    # Define model config explicitly (can be a dict or ModelConfig dataclass)
    # Example using a dictionary:
    model_config = {
         'input_size': seq_len, # Good practice to set this too
         'patch_size': 10,      # <<< ADD THIS LINE (e.g., 10 divides 50)
         'in_channels': obs_dim,
         'hidden_size': model_hidden_size,
         'num_layer': model_depth,
         'num_heads': model_num_heads,
         'cond_dim': cond_dim,
         'adaln_scale': 1.0,
         'learn_sigma': False,
         'use_cond': True,
         'use_flash_attn': False, # Set based on availability/preference
         # GIGA/ConvONet specific parts
         'grid_size': tsdf_dim,
         'encoder': giga_encoder,
         'encoder_kwargs': {'plane_type': ['grid', 'xz', 'xy', 'yz'],'grid_resolution': 32, 'plane_resolution': 32, 'unet3d': False, 'unet3d_kwargs': {'num_levels': 3, 'f_maps': 32, 'groups': 1, 'unet_feat_dim': giga_c_dim}},
         'decoder': giga_decoder,
         'decoder_kwargs': {'sample_mode': 'bilinear', 'hidden_size': 32},
         'c_dim': giga_c_dim, # Feature dim from GIGA encoder
         'padding': 0.1,
         'n_classes': 1, # For grasp quality
         'mlp_ratio': 4.0, # Add default or pass as arg if needed
         'class_dropout_prob': 0.0, # Add default
         'use_pos_emb': False, # Add default
         'dim_pos_emb': 3 * 2 * 32, # Add default or calculate
         'sin_emb_x': 0, # Add default
         'cat_emb_x': False, # Add default
         'use_cond_token': False, # Add default
         'use_cloud': False, # Add default
         'use_grasp': grasp_coef > 0 or tsdf_coef > 0,
         'use_tsdf': tsdf_coef > 0,
         'decoder_type': giga_decoder, # Already present via giga_decoder
         'use_joint_embeddings': True, # Add default
         'encoder_type': giga_encoder, # Already present via giga_encoder
    }
    model_config_ns = dict_to_namespace(model_config) # Convert dict to namespace
    model = PrestoGIGA(model_config_ns).to(device) # Pass namespace object
    ic(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")


    # --- Scheduler ---
    ic("Instantiating Diffusion Scheduler...")
    # Define diffusion config explicitly (can be a dict or DiffusionConfig dataclass)
    diffusion_config = {
        'beta_schedule': beta_schedule,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'num_train_timesteps': num_train_timesteps,
        'prediction_type': prediction_type,
        # Add other relevant scheduler params if needed, e.g., for DDIM
        'clip_sample': False, # Common DDIM parameter
        'set_alpha_to_one': False, # Common DDIM parameter
    }
    # Use get_scheduler factory or instantiate directly, e.g., DDPMScheduler
    # sched = get_scheduler(diffusion_config)
    if prediction_type == 'epsilon':
        sched = DDPMScheduler(
            num_train_timesteps=diffusion_config['num_train_timesteps'],
            beta_schedule=diffusion_config['beta_schedule'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            prediction_type=diffusion_config['prediction_type'],
        )
    else: # Example for 'sample' prediction type
         sched = DDPMScheduler( # Adjust scheduler type if needed for 'sample'
            num_train_timesteps=diffusion_config['num_train_timesteps'],
            beta_schedule=diffusion_config['beta_schedule'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end'],
            prediction_type=diffusion_config['prediction_type'],
        )
    ic(f"Scheduler: {type(sched).__name__}")


    # --- Optimizer ---
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    ic(f"Optimizer: AdamW (LR={learning_rate}, WD={weight_decay})")


    # --- LR Scheduler ---
    lr_scheduler = None
    if lr_schedule == 'cos':
        # Calculate total steps based on dummy data setup (1 batch per epoch)
        num_training_steps = num_epochs # Since we do 1 step per epoch
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=min(lr_warmup_steps, num_training_steps // 10), # Adjust warmup
            num_training_steps=num_training_steps)
        ic(f"LR Scheduler: Cosine with {lr_warmup_steps} warmup steps.")
    else:
        ic("LR Scheduler: None")


    # --- AMP Grad Scaler ---
    amp_enabled = use_amp and (device == 'cuda')
    scaler = th.cuda.amp.GradScaler(enabled=amp_enabled)
    ic(f"AMP Enabled: {amp_enabled}")


    # --- Load Checkpoint (Optional) ---
    start_epoch = 0
    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        ic(f"Loading checkpoint from: {load_checkpoint_path}")
        try:
            checkpoint = th.load(load_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint.get('epoch', -1) + 1 # Resume from next epoch
            global_step_loaded = checkpoint.get('global_step', 0)
            # Load normalizer state if saved and normalizer exists
            if 'normalizer' in checkpoint and normalizer is not None:
                 if hasattr(normalizer, 'load_state_dict'):
                     normalizer.load_state_dict(checkpoint['normalizer'])
                 elif isinstance(checkpoint['normalizer'], dict) and 'mean' in checkpoint['normalizer']:
                     normalizer.mean = checkpoint['normalizer']['mean'].to(device)
                     normalizer.std = checkpoint['normalizer']['std'].to(device)

            ic(f"Resuming training from epoch {start_epoch}, global step {global_step_loaded}")
            # Note: global_step in train_loop will restart from 0 unless passed in
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0


    # --- Run Training Loop ---
    train_loop(
        log_dir=log_dir,
        model=model,
        sched=sched,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        writer=writer,
        cost=cost,
        normalizer=normalizer,
        fk_fn=current_fk_fn,
        device=device,
        # Pass hyperparameters
        num_epochs=num_epochs - start_epoch, # Adjust epochs remaining
        batch_size=batch_size,
        use_amp=amp_enabled,
        diffusion_coef=diffusion_coef,
        grasp_coef=grasp_coef,
        tsdf_coef=tsdf_coef,
        collision_coef=collision_coef,
        distance_coef=distance_coef,
        euclidean_coef=euclidean_coef,
        reweight_loss_by_coll=reweight_loss_by_coll,
        x0_type=x0_type,
        x0_iter=x0_iter,
        cost_margin=cost_margin,
        log_by=log_by,
        step_max=step_max,
        save_epoch=save_epoch,
        # Pass dummy data params
        seq_len=seq_len,
        obs_dim=obs_dim,
        cond_dim=cond_dim,
        tsdf_dim=tsdf_dim,
        num_grasp_points=num_grasp_points,
        num_tsdf_points=num_tsdf_points,
    )


# --- Main Execution ---
if __name__ == '__main__':
    # --- Set parameters directly here ---
    train(
        log_dir_base="runs/combined_train_standalone_test",
        use_wandb=False, # Set to True to enable WandB logging
        # wandb_project="your_project_name",
        # wandb_entity="your_entity_name",

        num_epochs=20,   # Short run for testing
        batch_size=8,    # Small batch size
        learning_rate=1e-4,
        save_epoch=5,

        # Keep losses simple initially
        diffusion_coef=1.0,
        grasp_coef=0.5, # Example: Lower grasp weight
        tsdf_coef=0.0,
        collision_coef=0.0, # Disable aux losses for initial test
        distance_coef=0.0,
        euclidean_coef=0.0,
        reweight_loss_by_coll=False,

        # Model dimensions (match dummy data)
        seq_len=50,
        obs_dim=7,
        cond_dim=104,
        tsdf_dim=32,
        num_grasp_points=64,
        num_tsdf_points=32,

        # Simplified model arch for faster testing
        model_depth=6,
        model_num_heads=4,
        model_hidden_size=256,

        device_str="auto", # Use GPU if available
        use_amp=True,
        # load_checkpoint_path=None # Set path to resume
    )