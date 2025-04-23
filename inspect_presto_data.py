#!/usr/bin/env python3

import torch as th
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from dataclasses import dataclass, replace
import argparse

# Import PRESTO dataset class
from presto.data.presto_shelf import PrestoDatasetShelf
from presto.data.factory import DataConfig, get_dataset


def visualize_trajectory(trajectory, title="Robot Trajectory"):
    """Visualize a trajectory for each joint."""
    num_joints = trajectory.shape[-1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints))
    
    for i in range(num_joints):
        if num_joints == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(trajectory[:, i].cpu().numpy())
        ax.set_ylabel(f"Joint {i}")
        ax.grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect PRESTO dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/presto_shelf/rename",
                        help="Directory containing the dataset files")
    parser.add_argument("--pattern", type=str, default="*.pkl", 
                        help="File pattern to match dataset files")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of sample trajectories to display")
    args = parser.parse_args()

    # Configure dataset
    print(f"Loading dataset from {args.dataset_dir} with pattern {args.pattern}")
    device = "cuda" if th.cuda.is_available() else "cpu"
    
    # Option 1: Direct initialization
    cfg = PrestoDatasetShelf.Config(
        dataset_dir=args.dataset_dir,
        pattern=args.pattern,
        device=device
    )
    dataset = PrestoDatasetShelf(cfg, split='train')
    
    # Option 2: Using factory method (alternative)
    # data_cfg = DataConfig(
    #     dataset_type='shelf',
    #     dataset_dir=args.dataset_dir,
    #     normalize=True
    # )
    # dataset = get_dataset(data_cfg, split='train', device=device)
    
    print(f"\nDataset loaded successfully!")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Trajectory shape: {dataset.data['trajectory'].shape}")
    print(f"Sequence length: {dataset.seq_len}")
    print(f"Observation dimension: {dataset.obs_dim}")
    print(f"Condition dimension: {dataset.cond_dim}")
    
    # Print normalization parameters
    print("\nNormalization parameters:")
    print(f"Center: {dataset.normalizer.center}")
    print(f"Radius: {dataset.normalizer.radius}")
    
    # Display some sample trajectories
    print("\nDisplaying sample trajectories:")
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        
        # Print trajectory stats
        traj = sample['trajectory']
        print(f"  Trajectory shape: {traj.shape}")
        print(f"  Trajectory min/max: {traj.min().item():.4f}/{traj.max().item():.4f}")
        
        # Print start/goal
        print(f"  Start: {sample['start']}")
        print(f"  Goal: {sample['goal']}")
        
        # Print environment label stats
        env_label = sample['env-label']
        print(f"  Environment label shape: {env_label.shape}")
        print(f"  Environment label preview: {env_label[:5]}...")
        
        # Check if we have collision labels and print their type
        if 'col-label' in sample:
            col_label = sample['col-label']
            print(f"  Collision label type: {type(col_label)}")
            if isinstance(col_label, dict):
                print(f"  Collision label keys: {col_label.keys()}")
            else:
                print(f"  Collision label shape/length: {len(col_label) if hasattr(col_label, '__len__') else 'scalar'}")
        
        # Check for primitive labels
        if 'prim-label' in sample:
            prim_label = sample['prim-label']
            print(f"  Primitive label keys: {prim_label.keys() if isinstance(prim_label, dict) else 'not a dict'}")
        
        # Visualize the trajectory
        visualize_trajectory(traj, f"Sample_{i}_Trajectory")
    
    print("\nData inspection complete. Trajectory visualizations saved as PNG files.")


if __name__ == '__main__':
    main() 