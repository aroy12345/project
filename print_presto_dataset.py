#!/usr/bin/env python3

import torch as th
import numpy as np
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, replace

# Import PRESTO dataset class
from presto.data.presto_shelf import PrestoDatasetShelf

# Optional visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_trajectory(traj, title="Sample Trajectory"):
    """Visualize a trajectory if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for visualization")
        return
        
    if isinstance(traj, th.Tensor):
        traj = traj.detach().cpu().numpy()
    
    # Ensure traj has shape [timesteps, joints]
    if len(traj.shape) > 2:
        print(f"Warning: Trajectory has shape {traj.shape}, reshaping for visualization")
        traj = traj.reshape(-1, traj.shape[-1])
    
    num_joints = traj.shape[1]
    fig, axes = plt.subplots(num_joints, 1, figsize=(10, 2*num_joints))
    
    if num_joints == 1:
        axes = [axes]
        
    for i in range(num_joints):
        axes[i].plot(traj[:, i])
        axes[i].set_ylabel(f"Joint {i}")
        axes[i].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saved visualization to {title.replace(' ', '_')}.png")
    plt.close()


def print_tensor_info(name, tensor, max_elements=5):
    """Print information about a tensor."""
    if isinstance(tensor, th.Tensor):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Type: {tensor.dtype}")
        if tensor.numel() > 0:
            print(f"  Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            flat = tensor.flatten().detach().cpu()
            print(f"  Values: {flat[:min(max_elements, len(flat))].tolist()}")
    elif hasattr(tensor, 'shape'):  # NumPy array
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        if tensor.size > 0:
            print(f"  Range: [{tensor.min():.6f}, {tensor.max():.6f}]")
            flat = tensor.flatten()
            print(f"  Values: {flat[:min(max_elements, len(flat))].tolist()}")
    else:
        print(f"{name}: {type(tensor)}")
        try:
            print(f"  Value: {tensor}")
        except:
            print("  Could not print value")


def analyze_presto_dataset(dataset_dir, pattern="*.pkl", device="cpu", num_samples=3):
    """Load and analyze a PRESTO dataset."""
    print(f"\n=== Loading PRESTO dataset from {dataset_dir} with pattern {pattern} ===\n")
    
    # Configure and load the dataset
    cfg = PrestoDatasetShelf.Config(
        dataset_dir=dataset_dir,
        pattern=pattern,
        device=device
    )
    
    try:
        dataset = PrestoDatasetShelf(cfg, split='train')
        print(f"Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Sequence length: {dataset.seq_len}")
        print(f"Observation dimension: {dataset.obs_dim}")
        print(f"Condition dimension: {dataset.cond_dim}")
        
        # Print normalizer information
        print("\nNormalizer information:")
        if hasattr(dataset, 'normalizer'):
            print(f"Center: {dataset.normalizer.center}")
            print(f"Radius: {dataset.normalizer.radius}")
        
        # Print dataset stats
        print("\nDataset statistics:")
        for key in dataset.data.keys():
            if isinstance(dataset.data[key], th.Tensor):
                tensor = dataset.data[key]
                print(f"  {key}: shape={tensor.shape}, range=[{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        
        # Analyze individual samples
        print(f"\n=== Analyzing {min(num_samples, len(dataset))} individual samples ===")
        
        for i in range(min(num_samples, len(dataset))):
            print(f"\n--- Sample {i} ---")
            sample = dataset[i]
            
            # Print each key in the sample
            for key, value in sample.items():
                if isinstance(value, dict):
                    print(f"{key}: dictionary with keys {list(value.keys())}")
                    # Print first-level nested items
                    for sub_key, sub_value in value.items():
                        print_tensor_info(f"  {sub_key}", sub_value)
                else:
                    print_tensor_info(key, value)
            
            # Visualize trajectory
            if 'trajectory' in sample:
                visualize_trajectory(sample['trajectory'], f"Sample_{i}_Trajectory")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze PRESTO dataset with PrestoDatasetShelf")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing the dataset files")
    parser.add_argument("--pattern", type=str, default="*.pkl", 
                        help="File pattern to match dataset files")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load tensors on (cpu or cuda)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to analyze")
    args = parser.parse_args()
    
    dataset = analyze_presto_dataset(
        args.dataset_dir, 
        args.pattern, 
        args.device, 
        args.num_samples
    )


if __name__ == "__main__":
    main() 