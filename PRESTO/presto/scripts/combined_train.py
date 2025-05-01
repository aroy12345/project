#!/usr/bin/env python3
import types
import logging
import warnings # Import warnings

# Set up logging to file
def setup_logging():
    logging.basicConfig(
        filename='log.txt',
        filemode='w',  # 'w' to overwrite, 'a' to append
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("--- Logging Initialized ---\n")

# Call this at the start of your script
setup_logging()

# --- Specific logger setup for the temp function ---
def setup_temp_logging():
    """Sets up a dedicated logger for the temp function writing to log4.txt."""
    temp_logger = logging.getLogger('temp_logger')
    temp_logger.setLevel(logging.INFO)
    # Prevent propagation to the root logger (which logs to log.txt)
    temp_logger.propagate = False 
    
    # Check if handlers are already added to avoid duplicates if called multiple times
    if not temp_logger.handlers:
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('log4.txt', mode='w')
        fh.setLevel(logging.INFO)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        # Add the handlers to the logger
        temp_logger.addHandler(fh)
        temp_logger.info("--- temp_logger Initialized (log4.txt) ---")


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



from data_temp import generate_dummy_datapoint, create_batch_from_datapoints



from presto.data.franka_util import franka_fk

from presto.network.combined import (
    GIGA, GIGAConfig,
    Presto, PrestoConfig,
    # Keep other necessary imports like TSDFEmbedder, encoder/decoder dicts if needed directly
)
from presto.network.factory import get_scheduler, DiffusionConfig, ModelConfig


from presto.diffusion.util import pred_x0, diffusion_loss



from presto.cost.curobo_cost import CuroboCost
from presto.cost.cached_curobo_cost import CachedCuroboCost, cached_curobo_cost_with_ng
from presto.data.normalize import Normalize

from presto.util.ckpt import (load_ckpt, save_ckpt, last_ckpt)

import datetime

from presto.util.wandb_logger import WandbLogger
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, replace
from tqdm.auto import tqdm

import pickle
import numpy as np
import logging
import os
from omegaconf import OmegaConf
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

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from presto.data.factory import (DataConfig, get_dataset)
from presto.data.franka_util import franka_fk
from presto.network.common import grad_step, MLP
from presto.network.factory import (ModelConfig, get_model,
                                    DiffusionConfig, get_scheduler)
from presto.network.dit_cloud import DiTCloud
from presto.diffusion.util import pred_x0, diffusion_loss
from presto.cost.curobo_cost import CuroboCost
from presto.cost.cached_curobo_cost import CachedCuroboCost, cached_curobo_cost_with_ng

from presto.util.ckpt import (load_ckpt,
                              save_ckpt,
                              last_ckpt)
from presto.util.path import RunPath, get_path
from presto.util.hydra_cli import hydra_cli
from presto.util.wandb_logger import WandbLogger


@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2

    lr_warmup_steps: int = 500
    lr_schedule: str = 'cos'

    num_epochs: int = 512
    batch_size: int = 256
    num_configs: int = 1000
    save_epoch: int = 1
    use_amp: Optional[bool] = None

    collision_coef: float = 0.0
    distance_coef: float = 0.0
    diffusion_coef: float = 1.0
    euclidean_coef: float = 0.0

    mp_cost_type: str = 'ddpm'

    cost: CuroboCost.Config = CuroboCost.Config()
    cached_cost: CachedCuroboCost.Config = CachedCuroboCost.Config()
    use_cached: bool = True
    use_ng: bool = False

    log_by: str = 'epoch'
    step_max: Optional[int] = None
    check_cost: bool = False

    # FOR flow-matching scheduling
    weighting_scheme: str = 'logit_normal'
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29


@dataclass
class MetaConfig:
    project: str = 'trajopt'
    task: str = 'test'
    group: Optional[str] = None
    # NOTE(ycho): if `run_id` is None,
    # then we'll ask for the run_id via `input()` during runtime
    # if `wandb` is enabled.
    run_id: Optional[str] = None
    resume: bool = False


@dataclass
class Config:
    # Global config
    meta: MetaConfig = MetaConfig()
    path: RunPath.Config = RunPath.Config(root='/tmp/presto')
    train: TrainConfig = TrainConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    device: str = 'cuda:0'
    cfg_file: Optional[str] = None
    x0_type: str = 'step'
    x0_iter: int = 4  # iterate 4-steps
    load_ckpt: Optional[str] = None
    reweight_loss_by_coll: bool = False
    use_wandb: bool = True

    @property
    def need_traj(self):
        # We need (unnormalized) joint trajectories only
        # when evaluating auxiliary collision/distance costs.
        return (
            self.train.collision_coef > 0
            or self.train.distance_coef > 0
            or self.reweight_loss_by_coll
            or self.train.euclidean_coef > 0
        )


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

def collate_fn(xlist):
    cols = [x.pop('col-label') for x in xlist]
    out = th.utils.data.default_collate(xlist)
    out['col-label'] = np.stack(cols, axis=0)
    return out

def _map_device(x, device):
    """ Recursively map tensors in a dict to a device.  """
    if isinstance(x, dict):
        return {k: _map_device(v, device) for (k, v) in x.items()}
    if isinstance(x, th.Tensor):
        return x.to(device=device)
    return x

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
    return 1.0 - th.abs(th.sum(pred_norm * target_norm, dim=-1))

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
    # --- Stage Control ---
    training_mode: str, # 'giga' or 'presto'
    # --- Models ---
    model: nn.Module, # Either GIGA or Presto instance
    giga_model_eval: Optional[nn.Module], # Pre-trained GIGA model for Presto's aux loss
    # --- Standard Components ---
    log_dir: str,
    sched: Optional[SchedulerMixin], # Optional for GIGA stage
    optimizer: th.optim.Optimizer,
    lr_scheduler,
    scaler: th.cuda.amp.GradScaler,
    writer,
    cost, # For Presto aux losses
    normalizer, # For Presto data normalization
    fk_fn, # For Presto aux losses and affordance query
    device: str,
    # --- Hyperparameters ---
    num_epochs: int,
    batch_size: int,
    use_amp: bool,
    # Loss Coefficients (relevant ones passed based on mode)
    diffusion_coef: float,
    grasp_coef: float,
    tsdf_coef: float,
    collision_coef: float,
    distance_coef: float,
    euclidean_coef: float,
    affordance_loss_coef: float, # NEW: Weight for Presto's affordance loss
    # Other params
    reweight_loss_by_coll: bool,
    x0_type: Optional[str], # Optional for GIGA stage
    x0_iter: Optional[int], # Optional for GIGA stage
    cost_margin: float,
    log_by: str,
    step_max: Optional[int],
    save_epoch: int,
    # --- Dummy Data Params ---
    seq_len: int,
    obs_dim: int,
    cond_dim: int,
    tsdf_dim: int,
    num_grasp_points: int,
    num_tsdf_points: int,
    trajectory_data = None,
    cost_v = None
):
    """
    Main training loop for a single stage (GIGA or Presto).
    Returns the path to the last saved checkpoint for this stage.
    """
    start_time = time.time()
    global_step = 0 # Or load from checkpoint if resuming steps
    last_ckpt_path = None # Variable to store the last saved checkpoint path

    # --- Diffusion Setup (Only for Presto stage) ---
    diff_step_max = 0
    substepss = None
    if training_mode == 'presto':
        if sched is None:
            raise ValueError("Scheduler must be provided for Presto training mode.")
        diff_step_max = sched.config.num_train_timesteps
        if step_max is not None:
            diff_step_max = min(step_max, diff_step_max) # Use smaller if step_max is set

        if x0_type == 'iter':
            # (Keep the substepss calculation logic as before)
            # ... (copy logic from original train_loop lines 151-161)
            substepss_list = [
                th.round(
                    th.arange(i, 0, -i / x0_iter, device=device)).long()
                for i in range(1, sched.config.num_train_timesteps + 1)]
            max_len = x0_iter
            padded_substepss = [F.pad(s, (0, max_len - len(s))) for s in substepss_list]
            substepss = th.stack(padded_substepss, dim=0).to(
                device=device,
                dtype=th.long).sub_(1).clamp_(min=0)
            print("Precomputed iterative diffusion substeps.")

    logging.info(f"--- Starting Training Loop --- MODE: {training_mode.upper()} ---")
    logging.info(f"Epochs: {num_epochs}, Batch Size: {batch_size}")
    # Log relevant loss coefficients based on mode
    if training_mode == 'giga':
        logging.info(f"Loss coefficients: grasp={grasp_coef}, tsdf={tsdf_coef}")
    elif training_mode == 'presto':
        logging.info(f"Loss coefficients: diffusion={diffusion_coef}, collision={collision_coef}, "
                     f"distance={distance_coef}, euclidean={euclidean_coef}, affordance={affordance_loss_coef}")
        logging.info(f"Reweight by collision: {reweight_loss_by_coll}")



    ic(f"Starting training loop for mode: {training_mode}...")
 
    loader = th.utils.data.DataLoader(trajectory_data,
                                      batch_size=16,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=collate_fn)

    for epoch in range(num_epochs):
        logging.info(f"\n=== MODE: {training_mode.upper()} | EPOCH {epoch+1}/{num_epochs} ===")
        model.train() # Set current model (GIGA or Presto) to train mode
        if training_mode == 'presto' and giga_model_eval is not None:
            giga_model_eval.eval() # Ensure loaded GIGA model is in eval mode
            logging.info("GIGA model loaded and set to eval mode")

        # Log memory
        if device.startswith('cuda'):
            mem_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)
            mem_reserved = th.cuda.memory_reserved(device) / (1024 ** 3)
            logging.info(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        epoch_total_loss = 0.0
        # Track specific losses per epoch
        epoch_losses = {} # Dictionary to store different loss components

        # --- Dummy Data Generation for the Epoch ---
        # Generate a fixed set of dummy data for the epoch for simplicity
        # In a real scenario, this would be a DataLoader
        num_batches = 5 # Example: simulate 5 batches per epoch
        epoch_data = []
        for _ in range(num_batches * batch_size):
             epoch_data.append(generate_dummy_datapoint(
                 seq_len, obs_dim, cond_dim, tsdf_dim,
                 num_grasp_points, num_tsdf_points, device='cpu' # Generate on CPU first
             ))

        first_five_batches = []
        for i, batch_p in enumerate(tqdm(loader, leave=False)):
            first_five_batches.append(batch_p)
            if i >= 4: 
                break
            
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        #logging.info(f"epoch {epoch_data} - epoch {len(epoch_data)}")
        logging.info(f"pbars: {pbar}")
        logging.info(f"num_batches: {num_batches}")
        for batch_idx, batch_temp in enumerate(first_five_batches):
            # Create batch
            batch_start = batch_idx * batch_size
            batch_end = batch_start + batch_size
            batch_list = epoch_data[batch_start:batch_end]
            batch = create_batch_from_datapoints(batch_list, device=device)
            logging.info(f"Batch sizes: {batch_size}")
            logging.info(f"Batch keys: {batch.keys()}")
            logging.info(f"Batch trajectory data: {batch['trajectory'].shape}")
            optimizer.zero_grad()

            # Use AMP context
            amp_context = th.cuda.amp.autocast(enabled=use_amp) if device.startswith('cuda') else nullcontext()

            with amp_context:
                total_loss = 0.0
                loss_dict = {} # Losses for this step

                # --- GIGA Forward and Loss (Stage 1) ---
                if training_mode == 'giga':
                    tsdf_input = batch['tsdf']
                    p_grasp = batch['grasp_query_points']
                    p_tsdf = batch['tsdf_query_points'] # For TSDF prediction loss

                    # GIGA forward pass
                    # GIGA model now directly takes tsdf and points
                    giga_output = model(tsdf=tsdf_input, p_grasp=p_grasp, p_tsdf=p_tsdf)

                    # Calculate GIGA losses
                    giga_loss = 0.0
                    # Quality Loss
                    if 'qual' in giga_output and grasp_coef > 0:
                        qual_loss = _qual_loss_fn(giga_output['qual'].squeeze(-1), batch['grasp_qual_labels'].squeeze(-1))
                        loss_dict['giga_qual_loss'] = qual_loss.mean()
                        giga_loss += grasp_coef * loss_dict['giga_qual_loss']
                        logging.info(f"GIGA Qual loss: {giga_output['qual'].squeeze(-1).shape, batch['grasp_qual_labels'].squeeze(-1).shape}")
                    # Rotation Loss
                    if 'rot' in giga_output and grasp_coef > 0:
                        rot_loss = _rot_loss_fn(giga_output['rot'], batch['grasp_rot_labels'])
                        loss_dict['giga_rot_loss'] = rot_loss.mean()
                        giga_loss += grasp_coef * loss_dict['giga_rot_loss']
                        logging.info(f"GIGA Rot loss: {giga_output['rot'].shape, batch['grasp_rot_labels'].shape}")
                    # Width Loss
                    if 'width' in giga_output and grasp_coef > 0:
                        width_loss = _width_loss_fn(giga_output['width'].squeeze(-1), batch['grasp_width_labels'].squeeze(-1))
                        loss_dict['giga_width_loss'] = width_loss.mean()
                        giga_loss += grasp_coef * loss_dict['giga_width_loss']
                        logging.info(f"GIGA Width loss: {giga_output['width'].squeeze(-1).shape, batch['grasp_width_labels'].squeeze(-1).shape}")
                    # TSDF Prediction Loss
                    if 'tsdf_pred' in giga_output and tsdf_coef > 0:
                        tsdf_pred_loss = _tsdf_loss_fn(giga_output['tsdf_pred'].squeeze(-1), batch['tsdf_labels'].squeeze(-1))
                        loss_dict['giga_tsdf_loss'] = tsdf_pred_loss.mean()
                        giga_loss += tsdf_coef * loss_dict['giga_tsdf_loss']
                        logging.info(f"GIGA TSDF loss: {giga_output['tsdf_pred'].squeeze(-1).shape, batch['tsdf_labels'].squeeze(-1).shape}")
                    total_loss = giga_loss
                    loss_dict['total_loss'] = total_loss


                # --- Presto Forward and Loss (Stage 2) ---
                elif training_mode == 'presto':
                    # Prepare Presto inputs
                    traj_data = batch_temp['trajectory'] # Shape [B, S, C]
                    start_data = batch_temp['start'] # Shape [B, C]
                    end_data = batch_temp['goal'] # Shape [B, C]
                    # Presto expects [B, C, S]
                    sample_input = traj_data.permute(0,1,2) # [B, C_in, T]
                    print(f"Sample input shape: {sample_input.shape}")
                    cond_input = batch_temp['env-label'] # Shape [B, cond_dim]
                    tsdf_input = batch['tsdf'] # Shape [B, 1, D, D, D]
                    col_label = batch_temp.get('col-label') # For aux losses

                    # Normalize trajectory data if normalizer is provided
                    if normalizer:
                        # Ensure normalizer stats are on the correct device
                        normalizer.to(device)
                        sample_norm = normalizer.normalize(sample_input)
                        print(f"Normalized sample range: [{sample_norm.min():.4f}, {sample_norm.max():.4f}]")
                    else:
                        sample_norm = sample_input
                        print("Skipping normalization.")


                    # Sample timesteps
                    timesteps = th.randint(0, diff_step_max, (batch_size,), device=device).long()
                    print(f"Timesteps shape: {timesteps.shape}")

                    # 1. Sample noise
                    noise = th.randn_like(sample_norm)
                    print(f"Noise shape: {noise.shape}")

                    # 2. Create noisy samples (x_t)
                    noisy_samples = sched.add_noise(sample_norm, noise, timesteps)
                    print(f"Noisy samples shape: {noisy_samples.shape}")

                    # 3. Get model prediction (preds)
                    # The model predicts noise (epsilon) or x0 based on its config/scheduler
                    model_output = model(
                        sample=noisy_samples,
                        timestep=timesteps,
                        class_labels=cond_input,
                        tsdf=tsdf_input,
                        start=start_data,
                        end=end_data # Pass TSDF conditioning
                    )
                    print(f"Model output shape: {model_output.shape}")

                    # 4. Calculate diffusion loss using the utility function
                    diff_loss = diffusion_loss(
                        preds=model_output,
                        trajs=sample_norm, # Original clean data (x_start)
                        noise=noise,
                        noisy_trajs=noisy_samples, # Noisy data (x_t)
                        steps=timesteps,
                        sched=sched,
                        reduction='mean' # Explicitly use mean reduction
                    )

                    # 5. Calculate predicted x0 if needed for visualization/aux losses
                    pred_x0_vis = None
                    # We need pred_x0 for aux losses, so calculate it
                    pred_x0_vis = pred_x0(
                        sched=sched,
                        t=timesteps,
                        v=model_output, # Model output (epsilon, sample, or v)
                        x=noisy_samples # Noisy input (x_t)
                    )
                    print(f"Predicted x0 shape: {pred_x0_vis.shape}")

                    loss_dict['diffusion_loss'] = diff_loss
                    total_loss += diffusion_coef * diff_loss
                    print(f"Diffusion loss shape: {diff_loss.shape}")
                    print(f"Diffusion loss: diffusion_coef: {diffusion_coef}, diff_loss: {diff_loss}")
                    print(f"Euclidean loss: euclidean_coef: {euclidean_coef}, collision_coef: {collision_coef}, distance_coef: {distance_coef}")

                    # --- Auxiliary Presto Losses (Collision, Distance, Euclidean) ---
                    # Denormalize predicted x0 for physical losses
                    if normalizer:
                        pred_x0_denorm = normalizer.unnormalize(pred_x0_vis)
                    else:
                        pred_x0_denorm = pred_x0_vis # Use directly if no normalization

                   
                    pred_x0_traj = pred_x0_denorm.permute(0, 1, 2)
                    print(f"pred_x0_traj: {pred_x0_traj.shape}")
                    if cost_v and collision_coef > 0:
                        print('check')
                        print(f"cost_v: {cost_v}")
                        loss_coll = cost_v(pred_x0_traj, col_label).mean()
                        print(f"loss_coll: {loss_coll}")
                        loss_dict['collision_loss'] = loss_coll
                        total_loss += collision_coef * loss_coll
                    
                    if distance_coef > 0:
                        loss_dist = F.mse_loss(pred_x0_traj[..., 1:, :],
                                                pred_x0_traj[..., :-1, :])
                        print(f"loss_dist: {loss_dist}")
                        loss_dict['distance_loss'] = loss_dist
                        total_loss += distance_coef * loss_dist

                    if fk_fn is not None and euclidean_coef > 0:
                        x_pred_sd = franka_fk(pred_x0_traj)
                        with th.no_grad():
                                x_true_sd = franka_fk(
                                    dataset.normalizer.unnormalize(batch_temp['trajectory'])
                                )
                        loss_eucd = F.mse_loss(x_pred_sd, x_true_sd)
                        print(f"loss_euclidean: {loss_eucd}")
                        loss_dict['euclidean_loss'] = loss_eucd
                        total_loss += euclidean_coef * loss_eucd
                        
                    print(f"Affordance loss coef: {affordance_loss_coef}")
                    if giga_model_eval is not None and affordance_loss_coef > 0:
                        if fk_fn is None:
                             print("Warning: Affordance loss requires fk_fn to get final EE position. Skipping affordance loss.")
                        else:
                             with th.no_grad():
                                  print(f"Pred x0 traj shape: {pred_x0_traj.shape}")
                                  final_joint_state = pred_x0_traj[:, -1, :] # [B, C_in]
                                  print(f"Final joint state shape: {final_joint_state.shape}")

                                  # Call FK to get the full 4x4 transformation matrix
                                  final_ee_transform = fk_fn(final_joint_state) # Shape: [B, 4, 4]

                                  # Extract the position (first 3 elements of the 4th column)
                                  final_ee_pos = final_ee_transform[:, :3, 3] # Shape: [B, 3]

                                  print(f"Final EE pos shape: {final_ee_pos.shape}") # Should now be [B, 3]
                                  giga_query_points = final_ee_pos.unsqueeze(1) # [B, 1, 3]
                                  print(f"GIGA query points shape: {giga_query_points.shape}")
                                  
                                  giga_model_eval.to(device)
                                  tsdf_features_for_giga = giga_model_eval.encode_tsdf(tsdf_input)
                                  logging.info(f"GIGA model: {giga_model_eval.cfg}")
                                  logging.info("Successfully ran tsdf encoding.")
                                  if not tsdf_features_for_giga:
                                       print("Warning: Failed to get TSDF features from loaded GIGA model. Skipping affordance loss.")
                                  else:
                                       # Get affordance predictions (only need quality for this example)
                                       affordance_preds = giga_model_eval.forward_affordance(
                                           tsdf_features=tsdf_features_for_giga,
                                           p_grasp=giga_query_points
                                       )
                                       logging.info("Successfully ran affordance prediction.")
                                       if 'qual' in affordance_preds:
                                            # Penalize low predicted quality (logits)
                                            # Higher logit = better quality -> minimize negative logit
                                            predicted_qual_logits = affordance_preds['qual'] # [B, 1, 1]
                                            # Affordance loss: encourage high quality logits
                                            print(f"Predicted qual logits shape: {predicted_qual_logits.shape}")
                                            print(f"Predicted qual logits: {predicted_qual_logits}")
                                            afford_loss = -predicted_qual_logits.sum() # Minimize negative quality
                                            loss_dict['affordance_loss'] = afford_loss
                                            total_loss += affordance_loss_coef * afford_loss*3
                                            print(f"Calculated affordance loss: {afford_loss.item():.4f}")
                                       else:
                                            print("Warning: 'qual' not found in loaded GIGA model output. Skipping affordance loss.")

                    print(f"total loss.shape: {total_loss.shape}")
                    print(f"Total loss: {total_loss.item():.4f}")
                    loss_dict['total_loss'] = total_loss
                    print(f"loss_dict: {loss_dict}")

            # --- Backpropagation ---
            if total_loss > 0 : # Check if loss is valid before backward pass
                 if scaler is not None: # Check if scaler is initialized
                     scaler.scale(total_loss).backward()
                     scaler.step(optimizer)
                     scaler.update()
                 else:
                     total_loss.backward()
                     optimizer.step()


            # --- Learning Rate Step ---
            if lr_scheduler is not None:
                lr_scheduler.step()

            # --- Logging ---
            step_loss = total_loss.item() if isinstance(total_loss, th.Tensor) else total_loss
            epoch_total_loss += step_loss

            # Update epoch loss tracking
            for k, v in loss_dict.items():
                 if isinstance(v, th.Tensor):
                     epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            pbar.set_postfix({
                'loss': f"{step_loss:.4f}",
                'lr': optimizer.param_groups[0]['lr']
            })

            # Log step metrics if log_by == 'step'
            if log_by == 'step' and writer is not None:
                 log_data = {f'loss/{k}': v.item() if isinstance(v, th.Tensor) else v for k, v in loss_dict.items()}
                 log_data['lr'] = optimizer.param_groups[0]['lr']
                 writer.log(log_data, step=global_step)

            global_step += 1
            if step_max is not None and global_step >= step_max:
                 logging.info(f"Reached max steps ({step_max}). Stopping training.")
                 print(f"Reached max steps ({step_max}). Stopping training.")
                 # --- Save Final Checkpoint Before Exiting ---
                 ckpt_path = os.path.join(log_dir, f"{training_mode}_step_{global_step}.pth")
                 save_dict = {
                     'epoch': epoch,
                     'global_step': global_step,
                     'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict() if scaler is not None else None,
                     'config': model.cfg.__dict__ if model.cfg is not None else None,
                 }
                 if lr_scheduler is not None:
                     save_dict['lr_scheduler'] = lr_scheduler.state_dict()
                 if normalizer is not None and hasattr(normalizer, 'state_dict'):
                     save_dict['normalizer'] = normalizer.state_dict()
                 logging.info(f"GIGA Stage: Preparing to save checkpoint. Keys: {list(save_dict.keys())}")
                 logging.info(f"GIGA Stage: Type of model.cfg: {type(model.cfg)}") # Check if config exists
                 if 'model' not in save_dict or 'config' not in save_dict:
                      logging.error("GIGA Stage: CRITICAL - 'model' or 'config' key missing BEFORE saving!")
                 elif save_dict['config'] is None:
                      logging.warning("GIGA Stage: 'config' key is None BEFORE saving!")
                 save_ckpt(save_dict, ckpt_path)
                 logging.info(f"Saved final checkpoint to {ckpt_path}")
                 print(f"Saved final checkpoint to {ckpt_path}")
                 last_ckpt_path = ckpt_path # Store path
                 return last_ckpt_path # Return immediately after saving final step checkpoint

        # --- Epoch End ---
        avg_epoch_loss = epoch_total_loss / num_batches
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

        # Log epoch metrics if log_by == 'epoch'
        if log_by == 'epoch' and writer is not None:
             log_data = {f'loss_epoch/{k}': v / num_batches for k, v in epoch_losses.items()}
             log_data['epoch'] = epoch + 1
             writer.log(log_data, step=global_step) # Log against global step

        # --- Save Checkpoint (End of Epoch) ---
        if (epoch + 1) % save_epoch == 0:
            ckpt_path = os.path.join(log_dir, f"{training_mode}_epoch_{epoch+1:04d}.pth")
            save_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'config': model.cfg.__dict__ if model.cfg is not None else None,
            }
            if lr_scheduler is not None:
                save_dict['lr_scheduler'] = lr_scheduler.state_dict()
            if normalizer is not None and hasattr(normalizer, 'state_dict'):
                 save_dict['normalizer'] = normalizer.state_dict() # Save normalizer state

            # --- Add Debug Logging Here (GIGA Stage Epoch Save) ---
            if training_mode == 'giga':
                logging.info(f"GIGA Stage (Epoch Save): Preparing save_dict. Keys: {list(save_dict.keys())}")
                logging.info(f"GIGA Stage (Epoch Save): Type of config: {type(save_dict.get('config'))}")
                if 'config' not in save_dict:
                    logging.error("GIGA Stage (Epoch Save): 'config' key MISSING before save_ckpt!")
                elif save_dict.get('config') is None:
                     logging.warning("GIGA Stage (Epoch Save): 'config' key is None before save_ckpt!")
            # --- End Debug Logging ---

            save_ckpt(save_dict, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")
            print(f"Saved checkpoint to {ckpt_path}")
            last_ckpt_path = ckpt_path # Update last saved path

    # --- End Epoch Loop ---
    end_time = time.time()
    logging.info(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # Return the path of the last checkpoint saved (could be None if save_epoch > num_epochs and step_max not reached)
    return last_ckpt_path


# === MODIFIED Main `train` Function ===
def train(
    # --- Stage Control ---
    training_mode: str = 'joint', # 'giga', 'presto', 'joint' (original - now invalid)
    giga_checkpoint_path: Optional[str] = None, # Path to load pre-trained GIGA for Presto stage
    # --- Logging/Saving ---
    log_dir_base: str = "runs/combined_train_stages",
    use_wandb: bool = False,
    wandb_project: str = "presto_giga_stages",
    wandb_entity: Optional[str] = None,
    save_epoch: int = 10,
    log_by: str = 'epoch',
    # --- Key Hyperparameters ---
    num_epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    lr_warmup_steps: int = 100,
    lr_schedule: str = 'cos',
    # --- Loss Coefficients ---
    diffusion_coef: float = 1.0,
    grasp_coef: float = 1.0,
    tsdf_coef: float = 0.5,
    collision_coef: float = 0.5,
    distance_coef: float = 0.5,
    euclidean_coef: float = 0.5,
    affordance_loss_coef: float = 0.2, # NEW coefficient for Presto's affordance loss
    reweight_loss_by_coll: bool = True,
    # --- Diffusion Settings (for Presto) ---
    beta_schedule='squaredcos_cap_v2',
    beta_start=0.0001,
    beta_end=0.02,
    num_train_timesteps=1000,
    prediction_type='epsilon',
    x0_type: str = 'step',
    x0_iter: int = 1,
    step_max: Optional[int] = None,
    # --- Model Dimensions (from dummy data) ---
    seq_len: int = 256,
    obs_dim: int = 7,
    cond_dim: int = 1294,
    tsdf_dim: int = 32,
    num_grasp_points: int = 64,
    num_tsdf_points: int = 32,
    # --- Model Architecture Params ---
    # Shared / GIGA
    giga_encoder_type: str = 'voxel_simple_local',
    giga_decoder_type: str = 'simple_local',
    giga_c_dim: int = 32,
    giga_plane_resolution: int = 32, # Resolution for encoder's output planes
    giga_rot_dim: int = 4,
    # Presto specific
    presto_depth: int = 12,
    presto_num_heads: int = 6,
    presto_hidden_size: int = 384,
    presto_patch_size: int = 1,
    # --- Hardware ---
    device_str: str = "auto",
    use_amp: bool = True,
    # --- Checkpointing ---
    load_checkpoint_path: Optional[str] = None, # Checkpoint for the *current* stage model
    # --- Cost Function Params ---
    cost_margin: float = 0.1,
    trajectory_data = None,
    cost_v = None
):
    """
    Main training function supporting two stages: 'giga' and 'presto'.
    Returns the path to the last saved checkpoint for the completed stage.
    """
    if training_mode == 'joint':
        raise ValueError("Training mode 'joint' is deprecated. Use 'giga' or 'presto'.")
    if training_mode == 'presto' and giga_checkpoint_path is None:
        print("Warning: Running Presto training without a pre-trained GIGA model specified via 'giga_checkpoint_path'. Affordance loss will be disabled.")
        affordance_loss_coef = 0.0 # Disable loss if no GIGA model provided

    # --- Setup (Logging, Device, Folders, WandB) ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir_base, f"{training_mode}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    # Setup file logging within the specific run directory
    
    log_file_path = os.path.join(log_dir, 'train.log')
    # Remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Also print to console
        ]
    )
    logging.info(f"--- Starting Training Run ---")
    logging.info(f"Mode: {training_mode.upper()}")
    logging.info(f"Log directory: {log_dir}")


    # Device
    if device_str == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        device = device_str
    logging.info(f"Using device: {device}")

    # WandB
    writer = None
    if use_wandb:
        # Make sure wandb is installed: pip install wandb
        try:
            # Pass all hyperparameters to wandb config
            config_dict = locals()
            # Remove non-serializable items if necessary
            config_dict.pop('writer', None)
            config_dict.pop('device', None) # Log device_str instead

            writer = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                output_dir=log_dir,
                config=config_dict,
                name=f"{training_mode}_{timestamp}" # Unique run name
            )
            logging.info(f"WandB logging enabled. Project: {wandb_project}, Run: {writer.run_id}")
        except ImportError:
            logging.warning("WandB not installed. Skipping WandB logging.")
        except Exception as e:
            logging.warning(f"Failed to initialize WandB: {e}. Skipping WandB logging.")


    # --- Normalizer (Identity for now) ---
    # Use 'center' and 'radius' as expected by Normalize.__init__
    normalizer = Normalize(center=th.zeros(obs_dim), radius=th.ones(obs_dim))
    logging.info("Initialized identity normalizer.")
   
    

    # --- Cost Function (Primarily for Presto) ---
    cost = None
    if training_mode == 'presto' and (collision_coef > 0 or distance_coef > 0 or reweight_loss_by_coll):
        # Setup the dedicated logger for temp function just before it might be needed (though it's called earlier)
        # This ensures it's set up before any potential Curobo cost setup errors
  
        # (Keep CuroboCost instantiation logic as before)
        # ... (copy logic from original train function lines 762-780)
        try:
            from presto.cost.curobo_cost import CuroboCost # Ensure import
            ic("Instantiating CuroboCost...")
            cost_config = CuroboCost.Config() # Use default config
            # Adjust batch size dynamically in loop if needed, or ensure it matches
            cost = CuroboCost(cfg=cost_config, batch_size=batch_size, device=device)
            ic(f"CuroboCost instantiated with margin: {cost_margin}") # Assuming margin is handled internally
            logging.info("CuroboCost instantiated.")
        except ImportError:
            print("CuroboCost not found. Install curobo_python to use cost functions.")
            logging.warning("CuroboCost not found. Disabling related losses.")
            cost = None
            collision_coef = 0.0
            distance_coef = 0.0
            reweight_loss_by_coll = False
        except Exception as e:
            print(f"Failed to instantiate CuroboCost: {e}. Disabling related losses.")
            logging.warning(f"Failed to instantiate CuroboCost: {e}. Disabling related losses.")
            cost = None
            collision_coef = 0.0
            distance_coef = 0.0
            reweight_loss_by_coll = False
    else:
        logging.info("CuroboCost not needed for this stage or disabled.")


    # --- Forward Kinematics (Primarily for Presto) ---
    current_fk_fn = None
    if training_mode == 'presto' and (euclidean_coef > 0 or affordance_loss_coef > 0):
        try:
            from presto.data.franka_util import franka_fk # Ensure import
            current_fk_fn = franka_fk
            logging.info("Using franka_fk function.")
        except ImportError:
             logging.error("franka_fk not found. Cannot compute Euclidean or Affordance loss.")
             current_fk_fn = None
             euclidean_coef = 0.0
             affordance_loss_coef = 0.0 # Disable if FK is missing
    else:
        logging.info("FK function not needed for this stage or disabled.")


    # --- Model Instantiation ---
    model = None
    model_config_ns = None # To store the config used

    # GIGA Stage
    if training_mode == 'giga':
        logging.info("Instantiating GIGA model...")
        giga_config = GIGAConfig(
            encoder_type=giga_encoder_type,
            encoder_kwargs={ # Explicitly define encoder kwargs for GIGA
                'plane_type': ['xy', 'xz', 'yz'],
                'plane_resolution': giga_plane_resolution,
                'grid_resolution': tsdf_dim, # Base grid res
                'c_dim': giga_c_dim,
                # Add other necessary encoder params like unet flags/kwargs if needed
                'unet': False, # Example
                'unet3d': False, # Example
                'unet_kwargs': { 'depth': 3, 'merge_mode': 'concat', 'start_filts': 32 }, # Example
            },
            decoder_type=giga_decoder_type,
            decoder_kwargs={ # Explicitly define decoder kwargs for GIGA
                'dim': 3,
                'sample_mode': 'bilinear',
                'hidden_size': 32, # Example decoder hidden size
                'concat_feat': True,
                'c_dim': giga_c_dim # Ensure decoder knows input feature dim
            },
            c_dim=giga_c_dim,
            rot_dim=giga_rot_dim,
            use_tsdf_pred=(tsdf_coef > 0), # Enable TSDF head if loss coeff > 0
            padding=0.1, # Example padding
            hidden_size=presto_hidden_size, # Use a hidden size, e.g., from Presto config for consistency if needed
            use_amp=use_amp,
        )
        model = GIGA(giga_config).to(device)
        model_config_ns = giga_config # Store config

    # Presto Stage
    elif training_mode == 'presto':
        logging.info("Instantiating Presto model...")
        presto_config = PrestoConfig(
            # Diffusion params
            input_size=seq_len,
            patch_size=presto_patch_size,
            in_channels=obs_dim,
            hidden_size=presto_hidden_size,
            num_layer=presto_depth,
            num_heads=presto_num_heads,
            mlp_ratio=4.0,
            class_dropout_prob=0.0,
            cond_dim=cond_dim,
            learn_sigma=False, # Example: predict epsilon only
            use_cond=True,
            use_pos_emb=False,
            dim_pos_emb=3 * 2 * 32, # Example
            sin_emb_x=0,
            cat_emb_x=False,
            use_cond_token=False,
            use_cloud=False,
            use_joint_embeddings=True, # Condition on TSDF embedding
            # Encoder params (for TSDF conditioning)
            encoder_type=giga_encoder_type, # Use the same encoder type as GIGA
             encoder_kwargs={ # Ensure kwargs match GIGA's for consistency
                'plane_type': ['xy', 'xz', 'yz'],
                'plane_resolution': giga_plane_resolution,
                'grid_resolution': tsdf_dim,
                'c_dim': giga_c_dim,
                'unet': False, # Example
                'unet3d': False, # Example
                'unet_kwargs': { 'depth': 3, 'merge_mode': 'concat', 'start_filts': 32 }, # Example
            },
            c_dim=giga_c_dim, # Feature dim from encoder
            padding=0.1, # Example padding
            use_amp=use_amp,
            # Decoder params are not directly used by Presto model itself
            decoder_type=giga_decoder_type, # Keep for consistency if needed elsewhere
            decoder_kwargs={}, # Not directly used by Presto init
        )
        model = Presto(presto_config).to(device)
        model_config_ns = presto_config # Store config

    logging.info(f"Model instantiated: {type(model).__name__}")
    logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Log the model config
    logging.info("\n=== Model Configuration ===")
    for key, value in model_config_ns.__dict__.items():
         # Avoid logging large nested dicts directly if too verbose
         if isinstance(value, dict) and len(str(value)) > 100:
              logging.info(f"  {key}: {{...}}") # Log placeholder for large dicts
         else:
              logging.info(f"  {key}: {value}")


    # --- Load Pre-trained GIGA Model (for Presto Stage 2) ---
    giga_model_eval = None
    if training_mode == 'presto' and giga_checkpoint_path is not None:
        logging.info(f"Loading pre-trained GIGA model from: {giga_checkpoint_path}")
        if not os.path.exists(giga_checkpoint_path):
            logging.error(f"GIGA checkpoint not found at {giga_checkpoint_path}. Cannot use affordance loss.")
            affordance_loss_coef = 0.0
        else:
        
            giga_ckpt = th.load(giga_checkpoint_path, map_location='cpu')
            logging.info(f"Presto Stage: Successfully loaded GIGA ckpt. Keys found: {list(giga_ckpt.keys())}")

            logging.info("Presto Stage: Found 'model' and 'config' keys in GIGA checkpoint.")
                     # Reconstruct GIGAConfig from the saved dictionary
            saved_config_dict = giga_ckpt['config']
            giga_config = GIGAConfig(**saved_config_dict)
            logging.info("Presto Stage: Reconstructed GIGAConfig from saved dict.")
            giga_model_eval = GIGA(giga_config).to(device) # Move to target device
            giga_model_eval.load_state_dict(giga_ckpt['model'])
            giga_model_eval.eval() # Set to evaluation mode
            logging.info(f"GIGA model eval config: {giga_model_eval.cfg}")
            logging.info("Successfully loaded pre-trained GIGA model for evaluation.")

    


    # --- Scheduler (Only for Presto stage) ---
    sched = None
    if training_mode == 'presto':
        logging.info("Instantiating Diffusion Scheduler...")
        # (Keep scheduler instantiation logic as before, using DDPMScheduler)
        # ... (copy logic from original train function lines 874-904)
        if prediction_type == 'epsilon':
            sched = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                prediction_type=prediction_type,
            )
        else: # Example for 'sample' prediction type
             sched = DDPMScheduler( # Adjust scheduler type if needed for 'sample'
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
                prediction_type=prediction_type,
            )
        sched.set_timesteps(num_train_timesteps) # Set inference steps = train steps
        logging.info(f"Initialized DDIMScheduler with {num_train_timesteps} steps.")


    # --- Optimizer ---
    optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logging.info(f"Optimizer: AdamW (LR={learning_rate}, WD={weight_decay})")


    # --- LR Scheduler ---
    lr_scheduler = None
    if lr_schedule == 'cos':
        # Calculate total steps based on dummy data setup (num_batches per epoch)
        num_batches_per_epoch = 5 # Match the value used in train_loop
        num_training_steps = num_epochs * num_batches_per_epoch
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps, # Use the direct value
            num_training_steps=num_training_steps)
        logging.info(f"LR Scheduler: Cosine with {lr_warmup_steps} warmup steps over {num_training_steps} total steps.")
    else:
        logging.info("LR Scheduler: None")


    # --- AMP Grad Scaler ---
    amp_enabled = use_amp and (device == 'cuda')
    # Ensure scaler is only enabled if AMP is truly active
    scaler = th.cuda.amp.GradScaler(enabled=amp_enabled)
    logging.info(f"AMP Enabled: {amp_enabled}")


    # --- Load Checkpoint for Current Stage (Optional) ---
    start_epoch = 0
    global_step_start = 0
    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        logging.info(f"Loading checkpoint for {training_mode.upper()} stage from: {load_checkpoint_path}")
        try:
            # Load checkpoint for the model being trained in this stage
            checkpoint = th.load(load_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if amp_enabled and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                 scaler.load_state_dict(checkpoint['scaler'])
            if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint.get('epoch', -1) + 1
            global_step_start = checkpoint.get('global_step', 0) # Load global step

            # Load normalizer state if saved
            if 'normalizer' in checkpoint and normalizer is not None:
                 if hasattr(normalizer, 'load_state_dict') and isinstance(checkpoint['normalizer'], dict):
                      normalizer.load_state_dict(checkpoint['normalizer'])
                      logging.info("Loaded normalizer state from checkpoint.")
                 elif isinstance(checkpoint['normalizer'], dict) and 'mean' in checkpoint['normalizer']: # Legacy format?
                      normalizer.mean = checkpoint['normalizer']['mean'].to(device)
                      normalizer.std = checkpoint['normalizer']['std'].to(device)
                      logging.info("Loaded normalizer state (mean/std) from checkpoint.")


            logging.info(f"Resuming {training_mode.upper()} training from epoch {start_epoch}, global step {global_step_start}")
        except Exception as e:
            logging.error(f"Error loading checkpoint for {training_mode.upper()} stage: {e}. Starting training from scratch.")
            start_epoch = 0
            global_step_start = 0 # Reset step count
    else:
         logging.info(f"No checkpoint specified or found for {training_mode.upper()} stage. Starting training from scratch.")


    # --- Run Training Loop ---
    epochs_to_run = num_epochs - start_epoch
    final_ckpt_path = None # Initialize
    if epochs_to_run <= 0:
         logging.info("Loaded checkpoint is already at or beyond the target number of epochs. Exiting.")
         # If resuming from a completed checkpoint, maybe return its path?
         final_ckpt_path = load_checkpoint_path # Return the path of the loaded checkpoint
         # return final_ckpt_path # Or handle differently if needed
    else:
        # Call setup for the temp logger here as well, ensuring it's set before train_loop
        # if train() wasn't called first (though it is in the current structure)

        final_ckpt_path = train_loop( # Capture the returned path
            # Stage control
            training_mode=training_mode,
            # Models
            model=model,
            giga_model_eval=giga_model_eval, # Pass loaded GIGA model
            # Standard components
            log_dir=log_dir,
            sched=sched, # Pass scheduler (None for GIGA stage)
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            writer=writer,
            cost=cost, # Pass cost object
            normalizer=normalizer, # Pass normalizer
            fk_fn=current_fk_fn, # Pass FK function
            device=device,
            # Hyperparameters
            num_epochs=epochs_to_run, # Run remaining epochs
            batch_size=batch_size,
            use_amp=amp_enabled, # Pass the actual AMP status
            # Loss Coefficients
            diffusion_coef=diffusion_coef,
            grasp_coef=grasp_coef,
            tsdf_coef=tsdf_coef,
            collision_coef=collision_coef,
            distance_coef=distance_coef,
            euclidean_coef=euclidean_coef,
            affordance_loss_coef=affordance_loss_coef, # Pass new coeff
            # Other params
            reweight_loss_by_coll=reweight_loss_by_coll,
            x0_type=x0_type if training_mode == 'presto' else None,
            x0_iter=x0_iter if training_mode == 'presto' else None,
            cost_margin=cost_margin,
            log_by=log_by,
            step_max=step_max,
            save_epoch=save_epoch,
            # Dummy Data Params
            seq_len=seq_len,
            obs_dim=obs_dim,
            cond_dim=cond_dim,
            tsdf_dim=tsdf_dim,
            num_grasp_points=num_grasp_points,
            num_tsdf_points=num_tsdf_points,
            # Pass start global step? train_loop recalculates it currently.
            # global_step_start=global_step_start # If train_loop needs to resume step count
            trajectory_data=trajectory_data,
            cost_v = cost_v
        )

    logging.info(f"--- Training Run Completed (Mode: {training_mode.upper()}) ---")
    return final_ckpt_path # Return the path


# --- Main Execution ---    

@hydra_cli(
    # NOTE(ycho): you may need to configure `config_path`
    # depending on where you run this script.
    # config_path='presto/src/presto/cfg/',
    config_name='train_v2')
def main(cfg: Config):
    print("--- Entering temp function ---")
    print(f"Initial cfg object type: {type(cfg)}")
    
    cfg0 = OmegaConf.structured(Config())
    print("Created structured Config() as cfg0.")
    cfg0.merge_with(cfg)
    print("Merged input cfg into cfg0.")
    cfg = cfg0
    if cfg.cfg_file is not None:
        print(f"cfg.cfg_file found: {cfg.cfg_file}. Merging...")
        cfg.merge_with(OmegaConf.load(cfg.cfg_file))
        cfg.merge_with_cli()
        print("Merged cfg_file and CLI args.")
    else:
        print("No cfg.cfg_file found.")
    cfg = OmegaConf.to_object(cfg)
    print(f"Converted cfg to object type: {cfg}")
    trajectory_data = get_dataset(cfg.data, 'train', device='cuda')
    cost_v = CachedCuroboCost(cfg.train.cached_cost,
                                batch_size=cfg.train.batch_size,
                                device=cfg.device,
                                n_prim={'sphere': 12, 'cuboid': 19, 'cylinder': 14},
                                )
    
    import argparse
    import logging # Ensure logging is imported here too
    setup_temp_logging()



    parser = argparse.ArgumentParser(description="Train GIGA then Presto sequentially.")
    # Remove mode, giga_ckpt, load_ckpt arguments
    # parser.add_argument('--mode', type=str, required=True, choices=['giga', 'presto'],
    #                     help="Training stage ('giga' or 'presto').")
    # parser.add_argument('--giga_ckpt', type=str, default=None,
    #                     help="Path to pre-trained GIGA checkpoint (required for 'presto' mode with affordance loss).")
    # parser.add_argument('--load_ckpt', type=str, default=None,
    #                     help="Path to checkpoint to resume training for the current stage.")

    # Keep general arguments
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs to train *each* stage.") # Clarify epoch usage
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--aff_coef', type=float, default=0.2, help="Affordance loss coefficient for Presto stage.")
    parser.add_argument('--device', type=str, default='auto', help="Device ('auto', 'cuda', 'cpu').")
    parser.add_argument('--no_amp', action='store_true', help="Disable Automatic Mixed Precision.")
    parser.add_argument('--wandb', action='store_true', help="Enable WandB logging.")
    parser.add_argument('--log_base', type=str, default="runs/combined_train_sequential", help="Base directory for log/checkpoint runs.")
    parser.add_argument('--save_freq', type=int, default=10, help="Checkpoint save frequency (in epochs).")
    # Add other relevant arguments if needed (e.g., separate epochs/lr for each stage if desired)

    args = parser.parse_args()

    # --- Stage 1: GIGA Training ---
    print("\n" + "="*40)
    print("=== STARTING GIGA TRAINING STAGE ===")
    print("="*40 + "\n")
    # Note: Using basicConfig here might conflict if train() also calls it.
    # The train() function now handles its own logging setup per run.
    # Consider setting up a root logger once if needed outside train().

    final_giga_ckpt_path = train(
        training_mode='giga',
        giga_checkpoint_path=None, # No pre-trained GIGA needed for GIGA stage
        load_checkpoint_path=None, # Not supporting resume for combined run yet

        log_dir_base=args.log_base, # Use the base log directory
        use_wandb=args.wandb,
        # wandb_project="your_project_name_giga", # Consider separate projects/tags
        # wandb_entity="your_entity_name",

        num_epochs=args.epochs, # Use shared epoch count
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_epoch=args.save_freq,

        # Loss coefficients for GIGA stage
        diffusion_coef=0.0,     # Not used in GIGA
        grasp_coef=1.0,         # GIGA loss
        tsdf_coef=0.5,          # GIGA loss
        collision_coef=0.0,     # Not used in GIGA
        distance_coef=0.0,      # Not used in GIGA
        euclidean_coef=0.0,     # Not used in GIGA
        affordance_loss_coef=0.0, # Not used in GIGA
        reweight_loss_by_coll=False, # Not used in GIGA

        # Model dimensions (match dummy data - keep consistent)
        seq_len=256, obs_dim=7, cond_dim=1294, tsdf_dim=32,
        num_grasp_points=64, num_tsdf_points=32,

        # Model arch params (keep consistent)
        presto_depth=12, presto_num_heads=3, presto_hidden_size=50, presto_patch_size=2,
        giga_encoder_type='voxel_simple_local', giga_decoder_type='simple_local',
        giga_c_dim=32, giga_plane_resolution=32, giga_rot_dim=4,

        device_str=args.device,
        use_amp=(not args.no_amp),
        trajectory_data=trajectory_data
    )

    if final_giga_ckpt_path is None:
        print("\n" + "="*40)
        print("!!! ERROR: GIGA training stage did not produce a checkpoint path. Cannot proceed to Presto stage. !!!")
        print("="*40 + "\n")
        exit(1) # Exit if GIGA failed to save

    print("\n" + "="*40)
    print(f"=== GIGA TRAINING STAGE COMPLETE ===")
    print(f"Final GIGA Checkpoint: {final_giga_ckpt_path}")
    print("=== STARTING PRESTO TRAINING STAGE ===")
    print("="*40 + "\n")

    # --- Stage 2: Presto Training ---
    final_presto_ckpt_path = train(
        training_mode='presto',
        giga_checkpoint_path=final_giga_ckpt_path, # Use the checkpoint from Stage 1
        load_checkpoint_path=None, # Not supporting resume for combined run yet

        log_dir_base=args.log_base, # Use the same base log directory
        use_wandb=args.wandb,
        # wandb_project="your_project_name_presto", # Consider separate projects/tags
        # wandb_entity="your_entity_name",

        num_epochs=args.epochs, # Use shared epoch count
        batch_size=args.batch_size,
        learning_rate=args.lr, # Consider if Presto needs a different LR
        save_epoch=args.save_freq,

        # Loss coefficients for Presto stage
        diffusion_coef=1.0,         # Presto loss
        grasp_coef=0.0,             # Not used in Presto (GIGA is eval only)
        tsdf_coef=0.0,              # Not used in Presto (GIGA is eval only)
        collision_coef=0.1,         # Presto aux loss (example: disabled)
        distance_coef=0.1,          # Presto aux loss (example: disabled)
        euclidean_coef=0.1,         # Presto aux loss (example: disabled)
        affordance_loss_coef=args.aff_coef, # Presto aux loss using GIGA
        reweight_loss_by_coll=False,    # Presto aux loss feature

        # Model dimensions (match dummy data - keep consistent)
        seq_len=256, obs_dim=7, cond_dim=1294, tsdf_dim=32,
        num_grasp_points=64, num_tsdf_points=32,

        # Model arch params (keep consistent)
        presto_depth=12, presto_num_heads=5, presto_hidden_size=300, presto_patch_size=2,
        giga_encoder_type='voxel_simple_local', giga_decoder_type='simple_local',
        giga_c_dim=32, giga_plane_resolution=32, giga_rot_dim=4,

        device_str=args.device,
        use_amp=(not args.no_amp),
        trajectory_data=trajectory_data,
        cost_v = cost_v
    )

    print("\n" + "="*40)
    if final_presto_ckpt_path:
        print(f"Final Presto Checkpoint: {final_presto_ckpt_path}")
    print("=== PRESTO TRAINING STAGE COMPLETE ===")
    print("=== COMBINED RUN FINISHED ===")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()