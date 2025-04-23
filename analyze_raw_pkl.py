#!/usr/bin/env python3

import pickle
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Try to import torch but don't require it
try:
    import torch as th
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def visualize_trajectory(traj, title="Sample Trajectory", save_path=None):
    """
    Visualize a trajectory if matplotlib is available.
    
    Args:
        traj: NumPy array of shape [timesteps, joints]
        title: Plot title
        save_path: Path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for visualization")
        return
    
    # Convert torch tensor to numpy if needed
    if HAS_TORCH and isinstance(traj, th.Tensor):
        traj = traj.detach().cpu().numpy()
    
    # Ensure proper shape for visualization
    if len(traj.shape) > 2:
        print(f"Warning: Trajectory shape is {traj.shape}, reshaping for visualization")
        if traj.shape[0] == 1:  # If batch dimension
            traj = traj[0]
        else:
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
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        file_name = f"{title.replace(' ', '_')}.png"
        plt.savefig(file_name)
        print(f"Saved visualization to {file_name}")
    
    plt.close()


def describe_object(obj, name="Object", max_items=5, indent=""):
    """
    Recursively describe an object's type, shape, and content.
    
    Args:
        obj: The object to describe
        name: Name to display for the object
        max_items: Maximum number of array/list items to display
        indent: String for indentation
    """
    next_indent = indent + "  "
    
    # Type information
    type_name = type(obj).__name__
    print(f"{indent}{name} ({type_name}):", end="")
    
    # Handle different types
    if isinstance(obj, dict):
        print(f" dict with {len(obj)} keys")
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{next_indent}... ({len(obj) - max_items} more keys)")
                break
            describe_object(value, key, max_items, next_indent)
    
    elif isinstance(obj, (list, tuple)):
        print(f" {type_name} with {len(obj)} items")
        for i, item in enumerate(obj[:min(max_items, len(obj))]):
            if i >= max_items:
                print(f"{next_indent}... ({len(obj) - max_items} more items)")
                break
            describe_object(item, f"Item {i}", max_items, next_indent)
    
    elif isinstance(obj, np.ndarray):
        print(f" ndarray shape={obj.shape}, dtype={obj.dtype}")
        if obj.size > 0:
            flat = obj.flatten()
            sample = flat[:min(max_items, len(flat))]
            print(f"{next_indent}Range: [{np.min(obj)}, {np.max(obj)}]")
            print(f"{next_indent}Sample values: {sample}")
            
            # If this looks like a trajectory, visualize it
            if len(obj.shape) == 2 and obj.shape[1] <= 10:  # Reasonable number of joints
                filename = f"{name.replace(' ', '_')}_traj.png"
                visualize_trajectory(obj, f"Trajectory - {name}", filename)
    
    elif HAS_TORCH and isinstance(obj, th.Tensor):
        print(f" Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device}")
        if obj.numel() > 0:
            flat = obj.flatten().detach().cpu()
            sample = flat[:min(max_items, len(flat))]
            print(f"{next_indent}Range: [{obj.min().item()}, {obj.max().item()}]")
            print(f"{next_indent}Sample values: {sample.tolist()}")
            
            # If this looks like a trajectory, visualize it
            if len(obj.shape) == 2 and obj.shape[1] <= 10:  # Reasonable number of joints
                filename = f"{name.replace(' ', '_')}_traj.png"
                visualize_trajectory(obj, f"Trajectory - {name}", filename)
    
    elif isinstance(obj, (int, float, str, bool)):
        print(f" {obj}")
    
    else:
        print(f" {str(obj)[:100]}")


def analyze_pkl_file(file_path, sample_idx=None, max_items=5):
    """
    Load and analyze a pickle file.
    
    Args:
        file_path: Path to the pickle file
        sample_idx: Index of sample to analyze (if the file contains a list or dict of samples)
        max_items: Maximum number of items to display
    """
    print(f"Loading pickle file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("\n=== Data Structure Overview ===")
        describe_object(data, "Root", max_items)
        
        # Look for specific PRESTO data patterns
        if isinstance(data, dict):
            # Check for common PRESTO keys
            if 'qs' in data:
                print("\n=== Found PRESTO trajectory data ('qs') ===")
                qs = data['qs']
                describe_object(qs, "qs (trajectories)", max_items)
                
                # If sample_idx is provided, visualize a specific sample
                if sample_idx is not None and sample_idx < len(qs):
                    print(f"\n=== Visualizing Sample {sample_idx} ===")
                    visualize_trajectory(qs[sample_idx], f"Sample_{sample_idx}_Trajectory")
                # Otherwise visualize the first sample
                elif isinstance(qs, np.ndarray) and qs.ndim > 2:
                    visualize_trajectory(qs[0], "First_Trajectory")
            
            if 'ys' in data:
                print("\n=== Found PRESTO environment data ('ys') ===")
                describe_object(data['ys'], "ys (environment labels)", max_items)
            
            if 'ws' in data:
                print("\n=== Found PRESTO collision data ('ws') ===")
                describe_object(data['ws'], "ws (collision labels)", max_items)
        
        return data
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyze raw pickle files for PRESTO data")
    parser.add_argument("file_path", type=str, help="Path to the pickle file")
    parser.add_argument("--sample_idx", type=int, default=None, 
                        help="Index of sample to analyze (if dataset contains multiple samples)")
    parser.add_argument("--max_items", type=int, default=5,
                        help="Maximum number of items to display")
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} not found")
        sys.exit(1)
    
    analyze_pkl_file(args.file_path, args.sample_idx, args.max_items)


if __name__ == "__main__":
    main() 