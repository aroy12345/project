#!/usr/bin/env python3

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import argparse
from dataclasses import dataclass
from functools import partial
import time

# Import PRESTO classes
from presto.data.presto_shelf import PrestoDatasetShelf
from presto.data.factory import DataConfig, get_dataset
from presto.diffusion.presto_pipeline import PrestoPipeline, PrestoGenerator
from presto.network.factory import get_scheduler, get_model
from presto.network.dit import DiT


def visualize_diffusion_steps(steps, title="Diffusion Steps"):
    """Visualize different steps of the diffusion process for a single joint."""
    num_steps = len(steps)
    num_joints = steps[0].shape[-1]
    
    # Plot first joint for each diffusion step
    fig, axes = plt.subplots(num_steps, 1, figsize=(10, 2*num_steps))
    
    for i, step_data in enumerate(steps):
        if num_steps == 1:
            ax = axes
        else:
            ax = axes[i]
        
        # Plot each joint with a different color
        for j in range(num_joints):
            ax.plot(step_data[:, j].cpu().numpy(), label=f"Joint {j}")
        
        ax.set_ylabel(f"Step {i}")
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate PRESTO Trajectories")
    parser.add_argument("--dataset_dir", type=str, default="data/presto_shelf/rename",
                        help="Directory containing the dataset files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of diffusion steps for inference")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--init_type", type=str, default="random", 
                        choices=["random", "linear", "true"],
                        help="Type of trajectory initialization")
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load dataset
    print(f"Loading dataset from {args.dataset_dir}")
    cfg = PrestoDatasetShelf.Config(
        dataset_dir=args.dataset_dir,
        device=device,
        normalize=True
    )
    dataset = PrestoDatasetShelf(cfg, split='train')
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # 2. Load model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    checkpoint = th.load(args.checkpoint, map_location=device)
    
    # Extract model configuration from checkpoint or use default
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Create a default model config based on dataset dimensions
        print("Model config not found in checkpoint, using default config")
        model_config = DiT.Config(
            input_size=dataset.seq_len,
            patch_size=20,
            in_channels=dataset.obs_dim,
            num_layer=4,
            num_heads=8,
            cond_dim=dataset.cond_dim,
            learn_sigma=True,
            use_cond=True
        )
    
    # Create model and load weights
    model = get_model(model_config).to(device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully")
    
    # 3. Setup scheduler
    print("Creating scheduler")
    scheduler_config = {
        'num_train_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'beta_schedule': 'linear',
        'clip_sample': False,
        'prediction_type': 'v_prediction'
    }
    scheduler = get_scheduler(scheduler_config, scheduler_type='ddpm')
    
    # 4. Create PRESTO pipeline
    print("Setting up PRESTO pipeline")
    pipeline_config = PrestoPipeline.Config(
        init_type=args.init_type,
        apply_constraint=True,
        n_denoise_step=1
    )
    pipeline = PrestoPipeline(
        pipeline_config,
        unet=model,
        scheduler=scheduler,
        batch_size=args.batch_size
    )
    
    # 5. Setup data generator
    print("Setting up data generator")
    generator = PrestoGenerator(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        init_type=args.init_type
    )
    
    # 6. Generate trajectories
    print(f"Generating trajectories with {args.num_inference_steps} diffusion steps")
    with th.no_grad():
        output = pipeline(
            data_fn=generator,
            num_inference_steps=args.num_inference_steps,
            return_dict=True
        )
    
    # 7. Process and visualize results
    print("Processing results")
    
    # Get the generated trajectories
    trajs = output['trajs']  # Shape should be [num_steps, batch_size, obs_dim, seq_len]
    print(f"Generated trajectories shape: {trajs.shape}")
    
    # Unnormalize if needed
    final_trajs = dataset.normalizer.unnormalize(
        trajs[-1].swapaxes(-1, -2)
    ).swapaxes(-1, -2)
    
    # Swap axes for visualization (to get batch, seq_len, obs_dim)
    visualization_trajs = final_trajs.swapaxes(-1, -2)
    
    # Collect intermediate steps for visualization
    diffusion_steps = [trajs[i].swapaxes(-1, -2)[0] for i in 
                       range(0, len(trajs), max(1, len(trajs)//5))]
    
    # Visualize different steps of the diffusion process
    visualize_diffusion_steps(diffusion_steps, "Diffusion_Process")
    
    # Print final trajectory statistics
    print("\nGenerated Trajectory Statistics:")
    print(f"Shape: {visualization_trajs.shape}")
    print(f"Min/Max: {visualization_trajs.min().item():.4f}/{visualization_trajs.max().item():.4f}")
    
    # Compare with original trajectory if available
    if 'true_traj' in output:
        true_traj = output['true_traj'][0]  # First batch item
        if dataset.cfg.normalize:
            true_traj = dataset.normalizer.unnormalize(true_traj)
        
        # Calculate MSE between generated and true trajectory
        mse = th.mean((visualization_trajs[0] - true_traj) ** 2).item()
        print(f"MSE with true trajectory: {mse:.6f}")
    
    print("\nTrajectory generation complete. Visualizations saved as PNG files.")


if __name__ == '__main__':
    main() 