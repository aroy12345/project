#!/usr/bin/env python3

from typing import Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass, replace, field
from tqdm.auto import tqdm
import pickle
import numpy as np
from omegaconf import OmegaConf, MISSING
from icecream import ic
from contextlib import nullcontext
import time
import math

import warp as wp
import einops
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader # Explicit import

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
# NOTE: Import UNet1DOutput based on diffusers version
try:
    from diffusers.models.unet_1d import UNet1DOutput
except ImportError:
    from diffusers.models.unets.unet_1d import UNet1DOutput

# Assuming a combined dataset factory exists or is created
# from presto.data.factory import (DataConfig, get_combined_dataset)
from presto.data.factory import DataConfig # Placeholder
print(DataConfig)
# Import PrestoGIGA model and its config
from presto.network.combined import PrestoGIGA # Remove PrestoGIGAConfig from here
# Import diffusion utilities and scheduler factory
from presto.diffusion.util import pred_x0, diffusion_loss
from presto.network.factory import DiffusionConfig, get_scheduler
# Import utilities
from presto.util.ckpt import (load_ckpt, save_ckpt, last_ckpt)
from presto.util.path import RunPath, get_path
from presto.util.hydra_cli import hydra_cli
from presto.util.wandb_logger import WandbLogger

# --- Import CuroboCost ---
from presto.cost.curobo_cost import CuroboCost # Import the actual cost function
from presto.data.franka_util import franka_fk # <<< ADD THIS IMPORT

# --- Define GIGA-style Loss Functions ---
# Adapted from GIGA/scripts/train_giga.py (lines 158-177)
def _qual_loss_fn(pred, target):
    # pred is already sigmoid-ed in PrestoGIGA.forward_grasp
    # Use BCE loss directly
    return F.binary_cross_entropy(pred, target, reduction="none")

def _rot_loss_fn(pred, target):
    # pred is already normalized in PrestoGIGA.forward_grasp
    # Assumes target[:, 0] and target[:, 1] are the two symmetric quaternions
    # If target is just [B, M, 4], adjust accordingly
    if target.shape[-1] == 4 and len(target.shape) == 3: # Shape [B, M, 4]
         # Simple dot product loss if only one target rotation provided
         return 1.0 - torch.abs(torch.sum(pred * target, dim=-1))
    elif target.shape[-1] == 4 and len(target.shape) == 4 and target.shape[-2] == 2: # Shape [B, M, 2, 4]
        loss0 = _quat_loss_fn(pred, target[..., 0, :])
        loss1 = _quat_loss_fn(pred, target[..., 1, :])
        return torch.min(loss0, loss1)
    else:
        raise ValueError(f"Unsupported target rotation shape: {target.shape}")


def _quat_loss_fn(pred, target):
    # Ensure inputs are normalized (prediction should be already)
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 1.0 - torch.abs(torch.sum(pred_norm * target_norm, dim=-1))

def _width_loss_fn(pred, target, scale=40.0):
    # Simple MSE loss, potentially scaled
    return F.mse_loss(scale * pred, scale * target, reduction="none")

# --- Configuration Dataclasses ---

@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 500
    lr_schedule: str = 'cos'
    num_epochs: int = 512
    batch_size: int = 64 # Adjust as needed
    save_epoch: int = 10
    use_amp: Optional[bool] = None

    # Loss coefficients
    diffusion_coef: float = 1.0
    grasp_coef: float = 1.0 # Overall coefficient for the combined GIGA loss
    tsdf_coef: float = 0.0 # Set > 0 if training TSDF prediction
    # --- Added PRESTO Aux Loss Coefs ---
    collision_coef: float = 0.0 # Set > 0 to enable
    distance_coef: float = 0.0  # Set > 0 to enable
    euclidean_coef: float = 0.0 # Set > 0 to enable

    # --- Added PRESTO Reweighting/x0 Config ---
    reweight_loss_by_coll: bool = False # Set True to enable
    x0_type: str = 'step' # 'step' or 'iter'
    x0_iter: int = 1 # Number of iterations for x0_type='iter'

    log_by: str = 'epoch'
    step_max: Optional[int] = None # Max diffusion timestep

# --- ADD PrestoGIGAConfig definition here ---
@dataclass
class PrestoGIGAConfig:
    """Configuration for the combined PrestoGIGA model."""
    # Fields copied from PrestoGIGA.Config in presto/network/combined.py
    input_size: int = 1000  # Note: PRESTO's DiT usually uses obs_dim here, adjust if needed
    patch_size: int = 20
    in_channels: int = 7  # Will be overridden by dummy data dim in test
    hidden_size: int = 256
    num_layer: int = 4
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.0
    cond_dim: int = 104 # Will be overridden by dummy data dim in test
    learn_sigma: bool = True
    use_cond: bool = True
    use_pos_emb: bool = False
    dim_pos_emb: int = 3 * 2 * 32
    sin_emb_x: int = 0
    cat_emb_x: bool = False
    use_cond_token: bool = False
    use_cloud: bool = False # GIGA specific, might not be used in combined

    grid_size: int = 40 # Will be overridden by dummy data dim in test
    c_dim: int = 32 # Feature dimension from TSDF encoder/feature extractor
    use_grasp: bool = True # Enable grasp prediction heads
    use_tsdf: bool = True # Enable TSDF prediction head

    decoder_type: str = 'simple_fc' # GIGA decoder type
    decoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'hidden_size': 32,
        'n_blocks': 5,
        'leaky': False,
    })
    padding: float = 0.1 # GIGA decoder padding

    use_amp: Optional[bool] = None # Inherited, might not be needed directly here
    use_joint_embeddings: bool = True # Combine diffusion + TSDF features

    encoder_type: Optional[str] = 'voxel_simple_local' # GIGA TSDF encoder type
    encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'plane_type': ['xz', 'xy', 'yz'],
        'plane_resolution': 40, # Should match grid_size?
        'unet': True,
        'unet_kwargs': {
            'depth': 3,
            'merge_mode': 'concat',
            'start_filts': 32
        }
    })
    # Add any other fields specific to PrestoGIGA if needed


@dataclass
class MetaConfig:
    project: str = 'prestogiga'
    task: str = 'joint_train'
    group: Optional[str] = None
    run_id: Optional[str] = None
    resume: bool = False

@dataclass
class Config:
    meta: MetaConfig = MetaConfig()
    path: RunPath.Config = RunPath.Config(root='/tmp/prestogiga')
    train: TrainConfig = field(default_factory=TrainConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    # Embed PrestoGIGA's config directly
    model: PrestoGIGAConfig = field(default_factory=PrestoGIGAConfig)
    data: DataConfig = field(default_factory=DataConfig) # Assumes DataConfig is suitable for combined data
    device: str = 'cuda:0'
    cfg_file: Optional[str] = None
    load_ckpt: Optional[str] = None
    use_wandb: bool = True
    cost: CuroboCost.Config = field(default_factory=CuroboCost.Config) # Add CuroboCost config field
    seed: int = 0
    resume: Optional[str] = None
    # Add any other top-level configs needed (e.g., dataset config)
    dataset: Dict[str, Any] = field(default_factory=dict) # Example dataset config field

# --- Utility Functions ---

def _map_device(x, device):
    """ Recursively map tensors in a dict/list/tuple to a device. """
    if isinstance(x, dict):
        return {k: _map_device(v, device) for (k, v) in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(_map_device(item, device) for item in x)
    elif isinstance(x, th.Tensor):
        return x.to(device=device)
    return x

# Assume a collate function exists that handles the combined batch structure
# It should pad tensors appropriately if needed.
def combined_collate_fn(batch_list):
    # This is a placeholder - implementation depends heavily on the Dataset structure
    # It needs to handle all keys: 'trajectory', 'env-label', 'tsdf',
    # 'grasp_query_points', 'grasp_qual_labels', etc.
    # Use torch.utils.data.default_collate where possible, handle others manually.
    elem = batch_list[0]
    return {key: th.utils.data.default_collate([d[key] for d in batch_list])
            if isinstance(elem[key], th.Tensor) else
            [d[key] for d in batch_list] # Or custom stacking/padding
            for key in elem}


# --- GIGA-style Loss Functions ---
# (Keep the existing _qual_loss_fn, _rot_loss_fn, _quat_loss_fn, _width_loss_fn)

# --- Helper for x0 Prediction (Adapted from PRESTO/presto/scripts/train.py logic) ---
def predict_x0_from_model_out(model: PrestoGIGA, noisy_trajs, steps, label, diffusion_preds, sched: SchedulerMixin, cfg: Config):
    """ Predicts x0 based on model output and scheduler type. """
    if cfg.train.x0_type == 'step':
        # Assumes `diffusion_preds` is epsilon if scheduler predicts noise, or x0 if scheduler predicts x0
        if sched.config.prediction_type == "epsilon":
            # Use the utility function from presto.diffusion.util
            pred_traj_ds = pred_x0(sched, steps, diffusion_preds, noisy_trajs)
        elif sched.config.prediction_type == "sample":
             pred_traj_ds = diffusion_preds # Model directly predicts x0
        else:
             raise ValueError(f"Unsupported prediction type: {sched.config.prediction_type}")
    elif cfg.train.x0_type == 'iter':
        # Implementation based on PRESTO/presto/scripts/train.py lines 271-287
        pred_traj_ds = noisy_trajs
        if isinstance(sched, DDIMScheduler):
            # Calculate intermediate timesteps for iterative refinement
            # Ensure steps is a tensor on the correct device
            steps_tensor = steps if isinstance(steps, th.Tensor) else th.tensor([steps] * noisy_trajs.shape[0], device=noisy_trajs.device)

            # Create linspace for each item in the batch
            t_i_list = [th.linspace(0, s.item(), cfg.train.x0_iter + 1).long().flip(dims=(0,))[:-1].to(noisy_trajs.device)
                        for s in steps_tensor]

            # Iterate cfg.train.x0_iter times
            for i in range(cfg.train.x0_iter):
                intermediate_t = th.stack([t[i] for t in t_i_list]) # Get the i-th timestep for each batch item

                # Re-predict noise/x0 at the intermediate step using only the diffusion part
                with th.no_grad(): # No need for gradients during iterative refinement
                    intermediate_results = model.forward(
                        sample=pred_traj_ds,
                        timestep=intermediate_t,
                        class_labels=label,
                        mode="diffusion", # Only need diffusion output here
                        return_dict=True
                    )
                intermediate_preds = intermediate_results["sample"] # Noise or x0

                # Perform DDIM step to refine the trajectory prediction
                # Note: DDIMScheduler.step expects model_output (noise or x0), timestep, sample
                pred_traj_ds = sched.step(intermediate_preds, intermediate_t, pred_traj_ds).prev_sample
        else:
             raise ValueError(f'Iterative x0 prediction requires DDIMScheduler, got {type(sched)}')
    else:
        raise ValueError(f"Unsupported x0_type: {cfg.train.x0_type}")
    return pred_traj_ds

# --- Training Loop ---
def train_loop(cfg: Config,
               path: RunPath,
               dataset, # Assume this is your combined dataset instance
               model: PrestoGIGA,
               sched: SchedulerMixin,
               optimizer: th.optim.Optimizer,
               lr_scheduler: Optional[th.optim.lr_scheduler._LRScheduler],
               scaler: th.cuda.amp.GradScaler,
               writer: Optional[WandbLogger], # Use WandbLogger or Any
               # --- Added arguments ---
               cost: Optional[Any] = None, # Placeholder for CuroboCost or similar
               normalizer: Optional[Any] = None, # Placeholder for Normalize object
               fk_fn: Optional[callable] = None): # Placeholder for forward kinematics function

    device = cfg.device
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8, # Adjust as needed
        pin_memory=True,
        # collate_fn=combined_collate_fn # Use your custom collate if needed
    )

    # Check if auxiliary PRESTO costs or reweighting are needed
    # This determines if we need the unnormalized trajectory and 'none' reduction
    need_traj_for_aux_cost = (
        cfg.train.collision_coef > 0
        or cfg.train.distance_coef > 0
        or cfg.train.reweight_loss_by_coll # Check the reweight flag too
        or cfg.train.euclidean_coef > 0
    )
    # Ensure necessary components are provided if aux costs/reweighting are enabled
    if need_traj_for_aux_cost:
        if cost is None:
            raise ValueError("`cost` object must be provided when auxiliary costs or reweighting are enabled.")
        if normalizer is None:
             raise ValueError("`normalizer` object must be provided when auxiliary costs or reweighting are enabled.")
        if cfg.train.euclidean_coef > 0 and fk_fn is None:
             raise ValueError("`fk_fn` must be provided when euclidean_coef > 0.")
        if cfg.train.reweight_loss_by_coll and not hasattr(cost, 'cfg') or not hasattr(cost.cfg, 'margin'):
             print("Warning: `cost.cfg.margin` not found, needed for reweighting. Using default margin 0.0.")
             # Or raise ValueError("`cost.cfg.margin` must be defined for reweighting.")


    ic("Starting training loop...")
    global_step = 0
    try:
        diff_step_max: int = sched.config.num_train_timesteps
        if cfg.train.step_max is not None:
            diff_step_max = min(diff_step_max, cfg.train.step_max)

        for epoch in tqdm(range(cfg.train.num_epochs), desc=f"Epochs ({path.dir.name})"):
            # Running averages for logging
            loss_total_ra = []
            loss_ddpm_ra = []
            loss_coll_ra = []
            loss_dist_ra = []
            loss_eucd_ra = []
            loss_grasp_qual_ra = []
            loss_grasp_rot_ra = []
            loss_grasp_width_ra = []
            loss_grasp_comb_ra = [] # GIGA-style combined grasp loss
            loss_tsdf_ra = []

            model.train() # Ensure model is in training mode

            for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
                optimizer.zero_grad()

                # --- 1. Prepare Batch Data ---
                batch = _map_device(batch, device=device)
                # PRESTO expects trajectory as [B, S, C], then swaps internally if needed
                # Let's assume dataset provides [B, S, C] or adjust here
                # If dataset provides [B, C, S] like DummyDataset:
                # true_traj = batch['trajectory'].swapaxes(-1, -2) # [B, S, C]
                # If dataset provides [B, S, C] directly:
                true_traj = batch['trajectory'] # [B, S, C] Ground truth trajectory (unnormalized)

                # Normalize trajectory for diffusion input/target
                # Ensure normalizer is on the correct device
                if normalizer is None:
                     raise ValueError("Normalizer must be provided")
                current_normalizer = normalizer.to(device)
                true_traj_ds = current_normalizer(true_traj) # [B, S, C] Normalized

                # Swap axes for DiT input: [B, C, S]
                true_traj_ds_dit = true_traj_ds.swapaxes(-1, -2)

                env_label = batch['env-label']        # [B, D_cond] Diffusion condition
                tsdf_input = batch['tsdf']            # [B, 1, G, G, G] Grasp condition
                grasp_query_points = batch['grasp_query_points'] # [B, M, 3] Grasp query
                grasp_qual_labels = batch['grasp_qual_labels']   # [B, M, 1] Grasp targets
                grasp_rot_labels = batch['grasp_rot_labels']     # [B, M, 4] or [B, M, 2, 4]
                grasp_width_labels = batch['grasp_width_labels'] # [B, M, 1]
                # Optional data
                col_label = batch.get('col-label') or batch.get('prim-label') # For PRESTO aux costs
                tsdf_query_points = batch.get('tsdf_query_points') # For TSDF prediction
                tsdf_labels = batch.get('tsdf_labels') # For TSDF prediction

                # --- 2. Diffusion Noise Setup (PRESTO Style) ---
                steps = th.randint(
                    0, diff_step_max, (true_traj_ds_dit.shape[0],), device=device
                ).long()
                noise = th.randn(true_traj_ds_dit.shape, device=device) # Noise shape matches DiT input [B, C, S]
                noisy_trajs = sched.add_noise(true_traj_ds_dit, noise, steps)

                # --- 3. Model Forward Pass (Joint) ---
                # Use AMP context manager for forward pass
                with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
                    results = model.forward(
                        sample=noisy_trajs,            # Diffusion input [B, C, S]
                        timestep=steps,                # Diffusion timestep
                        class_labels=env_label,        # Diffusion condition
                        tsdf=tsdf_input,               # Grasp condition
                        p=grasp_query_points,          # Grasp query points
                        p_tsdf=tsdf_query_points,      # Optional: TSDF query points
                        mode="joint",                  # Ensure joint mode
                        return_dict=True
                    )
                    diffusion_preds = results.get("sample") # Noise or x0 prediction [B, C, S]
                    grasp_qual_preds = results.get("qual")  # [B, M, 1]
                    grasp_rot_preds = results.get("rot")    # [B, M, 4]
                    grasp_width_preds = results.get("width")# [B, M, 1]
                    tsdf_preds = results.get("tsdf")        # Optional [B, N_tsdf, 1]

                    # --- 4. Calculate Diffusion Loss (including PRESTO aux structure) ---
                    loss_ddpm_raw = th.tensor(0.0, device=device)
                    loss_coll = th.tensor(0.0, device=device) # PRESTO Aux
                    loss_dist = th.tensor(0.0, device=device) # PRESTO Aux
                    loss_eucd = th.tensor(0.0, device=device) # PRESTO Aux
                    final_loss_ddpm = th.tensor(0.0, device=device) # After reweighting/aux

                    if cfg.train.diffusion_coef > 0 and diffusion_preds is not None:
                        # Determine reduction based on PRESTO logic
                        reduction = 'none' if need_traj_for_aux_cost else 'mean'

                        # Calculate base diffusion loss (expects target [B, C, S])
                        loss_ddpm_raw = diffusion_loss(
                            sched=sched,
                            noise=noise,
                            model_output=diffusion_preds,
                            target=true_traj_ds_dit, # Use normalized target for diffusion loss
                            steps=steps,
                            reduction=reduction # Use conditional reduction
                        ) # Shape [B] if reduction='none', scalar if 'mean'

                        # Predict x0 (trajectory) from model output
                        # Output shape: [B, C, S]
                        pred_traj_ds = predict_x0_from_model_out(
                            model, noisy_trajs, steps, env_label, diffusion_preds, sched, cfg
                        )

                        # --- PRESTO Auxiliary Costs & Reweighting ---
                        if need_traj_for_aux_cost:
                            # Unnormalize predicted trajectory for physical costs
                            # Swap axes back to [B, S, C] for cost functions
                            u_pred_sd = current_normalizer.unnormalize(pred_traj_ds.swapaxes(-1, -2))

                            # Calculate SDF using CuroboCost
                            # cost() likely calls forward_collision_cost which handles reset internally if c is passed
                            # It expects q shape [B, S, C] and c shape [B, D_cond]
                            sdf = cost(u_pred_sd, col_label) # Shape [B, S] or similar depending on cost impl.

                            # Calculate auxiliary costs based on SDF (Signed Distance Field)
                            # Mimicking PRESTO train.py logic (lines 330-336)
                            cost_margin = cfg.cost.margin # Get margin from cost config

                            if cfg.train.collision_coef > 0 or cfg.train.reweight_loss_by_coll:
                                # Collision loss: Penalize negative SDF (penetration) up to margin
                                loss_coll = F.relu(cost_margin - sdf)
                                # Reduce loss_coll (e.g., mean over sequence and batch)
                                # PRESTO train.py seems to reduce later during reweighting or final aggregation
                                # Let's keep it per-batch for reweighting: mean over sequence length
                                if loss_coll.ndim > 1:
                                     loss_coll = loss_coll.mean(dim=1) # Shape [B]

                            if cfg.train.distance_coef > 0:
                                # Distance loss: Penalize positive SDF (distance)
                                loss_dist = F.relu(sdf)
                                # Reduce loss_dist (e.g., mean over sequence and batch)
                                # Keep per-batch: mean over sequence length
                                if loss_dist.ndim > 1:
                                     loss_dist = loss_dist.mean(dim=1) # Shape [B]

                            if cfg.train.euclidean_coef > 0:
                                # Calculate Euclidean loss in workspace (requires FK)
                                # Assumes fk_fn takes [B, S, C] -> [B, S, 3] (e.g., EEF position)
                                pred_xyz = fk_fn(u_pred_sd)
                                true_xyz = fk_fn(true_traj) # Use original unnormalized trajectory
                                loss_eucd = th.linalg.norm(pred_xyz - true_xyz, dim=-1).mean() # Mean over points and batch

                            # Reweight diffusion loss by collision (PRESTO logic)
                            if cfg.train.reweight_loss_by_coll:
                                # Ensure loss_ddpm_raw is per-batch element [B]
                                if reduction != 'none':
                                     raise RuntimeError("Diffusion loss reduction must be 'none' for reweighting.")
                                # Ensure loss_coll is per-batch element [B]
                                if loss_coll.ndim == 1: # Check if loss_coll is per-batch [B]
                                     # Original PRESTO weight: 1.0 + cost(u_pred_sd, col) / cost.cfg.margin
                                     # cost() here returns SDF, not the raw cost value used in original weight formula.
                                     # Let's use an exponential weight based on collision loss (distance below margin)
                                     # weight = th.exp(-loss_coll / cost_margin).detach() + 1e-2 # Alternative weighting
                                     # OR try to replicate original logic more closely if cost returns appropriate value
                                     # For now, using the exponential weight based on derived collision loss:
                                     coll_weight = th.exp(-loss_coll / cost_margin).detach() + 1e-2 # Shape [B]
                                     final_loss_ddpm = (loss_ddpm_raw * coll_weight).mean() # Apply weight and reduce
                                else: # If coll_loss is not per-batch, cannot reweight per sample
                                     print(f"Warning: Collision loss has unexpected shape {loss_coll.shape}, cannot reweight per sample. Using raw diffusion loss mean.")
                                     final_loss_ddpm = loss_ddpm_raw.mean()
                            else:
                                # If no reweighting, just take the mean if reduction was 'none'
                                final_loss_ddpm = loss_ddpm_raw.mean() if reduction == 'none' else loss_ddpm_raw

                        else:
                            # If no aux costs/reweighting, final loss is just the raw (mean) loss
                            final_loss_ddpm = loss_ddpm_raw

                    # --- 5. Calculate Grasp Losses (Exact GIGA Style) ---
                    combined_grasp_loss = th.tensor(0.0, device=device)
                    # For logging individual components (mean over all points)
                    log_loss_qual = th.tensor(0.0, device=device)
                    log_loss_rot = th.tensor(0.0, device=device)
                    log_loss_width = th.tensor(0.0, device=device)

                    if cfg.train.grasp_coef > 0 and \
                       grasp_qual_preds is not None and grasp_rot_preds is not None and grasp_width_preds is not None:

                        # Calculate per-point losses (reduction='none')
                        # Shapes: [B, M, 1], [B, M], [B, M, 1]
                        loss_qual_raw = _qual_loss_fn(grasp_qual_preds, grasp_qual_labels)
                        loss_rot_raw = _rot_loss_fn(grasp_rot_preds, grasp_rot_labels)
                        loss_width_raw = _width_loss_fn(grasp_width_preds, grasp_width_labels)

                        # --- Exact GIGA Loss Combination (per point) ---
                        # Reference: GIGA/scripts/train_giga.py line 168
                        # Reference: GIGA/scripts/train_vgn.py line 165
                        # Ensure label is broadcastable: [B, M, 1]
                        label_mask = grasp_qual_labels.detach() # Use detached label for masking

                        # Combine per-point losses using label mask and hardcoded width coef (0.01)
                        # loss_rot_raw and loss_width_raw need squeezing if they have trailing dim 1
                        loss_per_point = loss_qual_raw + label_mask * (
                            loss_rot_raw.unsqueeze(-1) + 0.01 * loss_width_raw
                        ) # Shape [B, M, 1]

                        # Final GIGA loss for backprop: mean over all points (B * M)
                        combined_grasp_loss = loss_per_point.mean()

                        # Calculate individual means *only* for logging purposes
                        with th.no_grad():
                             log_loss_qual = loss_qual_raw.mean()
                             # Mask rotation/width loss before averaging for logging consistency
                             log_loss_rot = (label_mask * loss_rot_raw.unsqueeze(-1)).mean()
                             log_loss_width = (label_mask * loss_width_raw).mean()


                    # --- 6. Calculate Optional TSDF Loss ---
                    loss_tsdf = th.tensor(0.0, device=device)
                    if cfg.train.tsdf_coef > 0 and tsdf_preds is not None and tsdf_labels is not None:
                        # Assuming binary classification task for TSDF occupancy (like GIGA)
                        # Model output is logits, labels are target probabilities (0 or 1)
                        # GIGA's _occ_loss_fn uses BCEWithLogitsLoss
                        loss_tsdf = F.binary_cross_entropy_with_logits(tsdf_preds, tsdf_labels, reduction='mean')

                    # --- 7. Combine Losses ---
                    # Ensure aux losses are reduced before combining if they weren't already
                    final_loss_coll = loss_coll.mean() if isinstance(loss_coll, th.Tensor) and loss_coll.ndim > 0 else loss_coll
                    final_loss_dist = loss_dist.mean() if isinstance(loss_dist, th.Tensor) and loss_dist.ndim > 0 else loss_dist
                    final_loss_eucd = loss_eucd # Already reduced

                    total_loss = (cfg.train.diffusion_coef * final_loss_ddpm +
                                  cfg.train.collision_coef * final_loss_coll + # Use reduced aux loss
                                  cfg.train.distance_coef * final_loss_dist + # Use reduced aux loss
                                  cfg.train.euclidean_coef * final_loss_eucd + # Use reduced aux loss
                                  cfg.train.grasp_coef * combined_grasp_loss + # Use combined GIGA loss
                                  cfg.train.tsdf_coef * loss_tsdf) # TSDF loss already reduced

                # --- 8. Gradient Step & LR Schedule ---
                # Use standard AMP gradient scaling
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if lr_scheduler is not None:
                    lr_scheduler.step()
                global_step += 1


                # --- 9. Log Losses ---
                loss_total_ra.append(total_loss.item())
                loss_ddpm_ra.append(final_loss_ddpm.item()) # Log the final (potentially reweighted) diffusion loss
                loss_coll_ra.append(loss_coll.item())
                loss_dist_ra.append(loss_dist.item())
                loss_eucd_ra.append(loss_eucd.item())
                # Log individual grasp components (averaged over all points)
                loss_grasp_qual_ra.append(log_loss_qual.item())
                loss_grasp_rot_ra.append(log_loss_rot.item())
                loss_grasp_width_ra.append(log_loss_width.item())
                loss_grasp_comb_ra.append(combined_grasp_loss.item()) # Log the combined GIGA loss used for backprop
                loss_tsdf_ra.append(loss_tsdf.item())

            # --- End of Epoch Logging & Checkpointing ---
            if writer is not None:
                 log_data = {
                     'epoch': epoch,
                     'lr': optimizer.param_groups[0]['lr'],
                     'loss/total': np.mean(loss_total_ra),
                     'loss/diffusion': np.mean(loss_ddpm_ra),
                     'loss/collision': np.mean(loss_coll_ra),
                     'loss/distance': np.mean(loss_dist_ra),
                     'loss/euclidean': np.mean(loss_eucd_ra),
                     'loss/grasp_quality': np.mean(loss_grasp_qual_ra), # Logged individual mean
                     'loss/grasp_rotation': np.mean(loss_grasp_rot_ra), # Logged individual mean (masked)
                     'loss/grasp_width': np.mean(loss_grasp_width_ra),  # Logged individual mean (masked)
                     'loss/grasp_combined': np.mean(loss_grasp_comb_ra),# Logged combined GIGA loss
                     'loss/tsdf': np.mean(loss_tsdf_ra),
                 }
                 writer.log(log_data, step=global_step if cfg.train.log_by == 'step' else epoch)

            # Save checkpoint
            if (epoch + 1) % cfg.train.save_epoch == 0:
                # Include normalizer state if it exists and has state_dict or similar
                normalizer_state = None
                if normalizer is not None:
                     if hasattr(normalizer, 'state_dict'):
                         normalizer_state = normalizer.state_dict()
                     elif hasattr(normalizer, 'mean') and hasattr(normalizer, 'std'):
                         # Basic saving for mean/std normalizer
                         normalizer_state = {'mean': normalizer.mean, 'std': normalizer.std}

                save_ckpt(path=path,
                          model=model,
                          optim=optimizer,
                          sched=lr_scheduler,
                          scaler=scaler,
                          cfg=cfg,
                          step=epoch, # Save epoch number
                          global_step=global_step, # Save global step too
                          normalizer=normalizer_state) # Save normalizer state

    except KeyboardInterrupt:
        print('Graceful exit')
    finally:
        # Save final checkpoint
        normalizer_state = None
        if normalizer is not None:
             if hasattr(normalizer, 'state_dict'):
                 normalizer_state = normalizer.state_dict()
             elif hasattr(normalizer, 'mean') and hasattr(normalizer, 'std'):
                 normalizer_state = {'mean': normalizer.mean, 'std': normalizer.std}

        save_ckpt(path=path,
                  model=model,
                  optim=optimizer,
                  sched=lr_scheduler,
                  scaler=scaler,
                  cfg=cfg,
                  step=epoch if 'epoch' in locals() else -1, # Handle early exit
                  global_step=global_step,
                  normalizer=normalizer_state,
                  final=True)
        if writer is not None:
            writer.finish()

    return model # Return trained model


# --- Placeholder Imports (Ensure these are available in your environment) ---
# from curobo.wrap.reacher.cost import CuroboCost # Example Cost function
# from presto.data.normalize import Normalize # Example Normalizer
# from presto.data.franka_util import franka_fk # Example FK function

# --- Dummy Implementations (Replace with actual imports/logic) ---
# class DummyCost: # Replace with actual CuroboCost or similar
#     def __init__(self): self.cfg = OmegaConf.create({'margin': 0.05}) # Example margin
#     def coll_loss(self, traj, label): return th.rand(traj.shape[0], device=traj.device) * 0.1 # Return per-batch loss
#     def dist_loss(self, traj, label): return th.rand(traj.shape[0], device=traj.device) * 0.01 # Return per-batch loss

# --- (Keep Normalize if still needed as placeholder, otherwise delete) ---
class Normalize: # Replace with actual import
    def __init__(self, mean, std): self.mean=mean; self.std=std
    def to(self, device): self.mean=self.mean.to(device); self.std=self.std.to(device); return self
    def __call__(self, x): return (x - self.mean) / self.std
    def unnormalize(self, x): return x * self.std + self.mean
    @staticmethod
    def from_minmax(min_val, max_val):
        mean = (min_val + max_val) / 2.0
        std = (max_val - min_val) / 2.0
        std[std == 0] = 1.0 # Avoid division by zero
        return Normalize(mean, std)
    @staticmethod
    def identity(dim, device='cpu'): return Normalize(th.zeros(dim, device=device), th.ones(dim, device=device))

def train(cfg: Config):
    device: str = cfg.device
    path = RunPath(cfg.path)
    path.dump_cfg(cfg) # Save config

    # --- Logger ---
    writer = None
    if cfg.use_wandb:
        # Make sure WandbLogger is correctly imported
        # from presto.util.wandb_logger import WandbLogger
        writer = WandbLogger(
            project=cfg.meta.project,
            entity=None, # Set your wandb entity
            group=cfg.meta.group,
            run_id=cfg.meta.run_id,
            mode=None,
            cfg=OmegaConf.to_container(cfg, resolve=True)
        )
    ic("Logging to", path.dir)

    # --- !!! TESTING WITH DUMMY DATA !!! ---
    print("--- GENERATING DUMMY DATA FOR TESTING ---")
    # Define dimensions consistent with your config or reasonable defaults
    test_seq_len = 50
    test_obs_dim = 7 # From Franka
    test_cond_dim = 104 # Example
    test_tsdf_dim = 32 # Example
    test_num_grasp_points = 64 # Example
    test_num_tsdf_points = 32 # Example

    # Create a single data point
    single_dummy_dp = generate_dummy_datapoint(
        seq_len=test_seq_len,
        obs_dim=test_obs_dim,
        cond_dim=test_cond_dim,
        tsdf_dim=test_tsdf_dim,
        num_grasp_points=test_num_grasp_points,
        num_tsdf_points=test_num_tsdf_points,
        device=device # Create directly on target device
    )

    # Create a small batch (e.g., batch size 4)
    batch_size_for_test = 4
    dummy_batch_list = [generate_dummy_datapoint(
                            seq_len=test_seq_len, obs_dim=test_obs_dim, cond_dim=test_cond_dim,
                            tsdf_dim=test_tsdf_dim, num_grasp_points=test_num_grasp_points,
                            num_tsdf_points=test_num_tsdf_points, device='cpu' # Generate on CPU first
                        ) for _ in range(batch_size_for_test)]
    dummy_batch = create_batch_from_datapoints(dummy_batch_list, device=device)
    print(f"--- Generated dummy batch with size {batch_size_for_test} ---")
    # You can inspect the shapes here:
    # for key, value in dummy_batch.items():
    #     print(f"Batch key '{key}' shape: {value.shape}")
    # --- END TESTING WITH DUMMY DATA ---


    # --- Dataset & Normalizer ---
    # Replace DummyDataset with your actual implementation when ready
    # train_dataset = YourActualCombinedDataset(...)
    # print("WARNING: Using DummyDataset. Replace with actual combined dataset loader.")
    # Use dummy dimensions for placeholder normalizer if needed
    # obs_dim_from_dummy = test_obs_dim

    # --- Normalizer ---
    # Initialize normalizer based on dummy data dimensions for testing
    normalizer = None
    # ... (rest of normalizer logic, potentially using test_obs_dim) ...
    if normalizer is None:
        print("Warning: No normalizer found or loaded. Using identity normalizer for testing.")
        normalizer = Normalize.identity(dim=test_obs_dim, device=device) # Use dummy dim
    else:
        normalizer = normalizer.to(device)


    # --- Cost Function ---
    # Instantiate CuroboCost using the config
    cost = None
    if cfg.train.collision_coef > 0 or cfg.train.distance_coef > 0 or cfg.train.reweight_loss_by_coll:
         ic("Instantiating CuroboCost...")
         cost = CuroboCost(cfg=cfg.cost,
                           batch_size=batch_size_for_test, # Use test batch size
                           device=device)
         ic("CuroboCost instantiated.")


    # --- Forward Kinematics ---
    fk_fn = franka_fk
    ic("Using franka_fk function.")


    # --- Model & Scheduler ---
    # Update model config based on dummy data specifics for testing
    cfg.model.in_channels = test_obs_dim # Use dummy dim
    # cfg.model.input_size = test_seq_len # DiT input_size is usually obs_dim (channels)
    cfg.model.cond_dim = test_cond_dim # Use dummy dim
    cfg.model.grid_size = test_tsdf_dim # Use dummy dim

    model = PrestoGIGA(cfg.model).to(device)
    sched = get_scheduler(cfg.diffusion)
    ic(model)
    ic(sched)
    ic(f"Model Parameters: {count_parameters(model)}")

    # --- Optimizer & LR Scheduler ---
    optimizer = th.optim.AdamW(model.parameters(),
                               lr=cfg.train.learning_rate,
                               weight_decay=cfg.train.weight_decay)
    # No LR scheduler needed for single step test usually

    # --- AMP Grad Scaler ---
    scaler = th.cuda.amp.GradScaler(enabled=cfg.train.use_amp)

    # --- !!! TEST SINGLE FORWARD/BACKWARD PASS !!! ---
    print("--- TESTING SINGLE FORWARD/BACKWARD PASS ---")
    model.train()
    optimizer.zero_grad()

    # Use the generated dummy batch
    batch_for_test = dummy_batch

    # --- Mimic start of train_loop ---
    current_normalizer = normalizer # Use the initialized normalizer
    true_traj = batch_for_test['trajectory'] # Shape [B, S, C]
    # Normalize trajectory [B, S, C] -> [B, C, S] for diffusion model
    norm_traj = current_normalizer(true_traj).swapaxes(-1, -2) # Shape [B, C, S]
    env_label = batch_for_test['env-label']
    col_label = batch_for_test['col-label']
    tsdf_volume = batch_for_test.get('tsdf') # Use .get for optional keys
    grasp_query = batch_for_test['grasp_query_points']
    tsdf_query = batch_for_test.get('tsdf_query_points')

    # Sample noise and timesteps
    noise = th.randn_like(norm_traj)
    steps = th.randint(0, sched.config.num_train_timesteps, (batch_for_test['trajectory'].shape[0],), device=device).long()
    noisy_trajs = sched.add_noise(norm_traj, noise, steps)

    # Forward pass
    with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
        model_out = model(
            sample=noisy_trajs,                 # [B, C, S]
            timestep=steps,
            cond_embed=env_label,               # [B, D_cond]
            tsdf_volume=tsdf_volume,            # [B, 1, G, G, G]
            grasp_query_points=grasp_query,     # [B, M, 3]
            tsdf_query_points=tsdf_query,       # [B, N_tsdf, 3]
            return_dict=True
        )
        # Extract outputs
        diffusion_preds = model_out["sample"] # Noise or x0 prediction [B, C, S]
        grasp_qual_preds = model_out.get("grasp_quality") # [B, M, 1] (sigmoid-ed)
        grasp_rot_preds = model_out.get("grasp_rotation") # [B, M, 4] (normalized)
        grasp_width_preds = model_out.get("grasp_width") # [B, M, 1]
        tsdf_preds = model_out.get("tsdf_occupancy") # [B, N_tsdf, 1] (logits)

        # --- Calculate Losses (Mimic train_loop logic) ---
        # (Copy relevant loss calculation sections from train_loop here,
        #  using batch_for_test['grasp_qual_labels'], etc. as targets)
        # ... (diffusion_loss, aux costs, grasp loss, tsdf loss) ...
        # Example placeholder:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True) # Replace with actual loss calc

    # Backward pass
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"--- Single pass completed. Loss: {total_loss.item()} ---")
    # --- END SINGLE PASS TEST ---


    # --- Load Checkpoint (if resuming - skip for initial test) ---
    # ...

    # --- Run Training Loop (Comment out for initial test) ---
    # print("--- SKIPPING FULL TRAINING LOOP FOR INITIAL TEST ---")
    # train_loop(cfg=cfg, path=path, dataset=train_dataset, ...)

    ic('Done with test setup.')

# --- (Hydra entry point remains the same) ---
@hydra_cli(config_path='../config', config_name='combined_train')
def main(cfg: Config):
    ic(OmegaConf.to_yaml(cfg))
    train(cfg)

if __name__ == '__main__':
    main()