#!/usr/bin/env python3

import torch as th
import numpy as np
import pickle
import argparse
import os
from pathlib import Path
from tqdm import tqdm

# Optional visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_dict_structure(d, prefix='', max_items=5, max_depth=3, current_depth=0):
    """Print the structure of a nested dictionary or list with sample values."""
    if current_depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
        
    if isinstance(d, dict):
        print(f"{prefix}Dict with {len(d)} keys:")
        for i, (k, v) in enumerate(d.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({len(d) - max_items} more keys)")
                break
            print(f"{prefix}  {k}:")
            print_dict_structure(v, prefix + "    ", max_items, max_depth, current_depth + 1)
    elif isinstance(d, list):
        print(f"{prefix}List with {len(d)} items:")
        for i, item in enumerate(d[:min(max_items, len(d))]):
            if i >= max_items:
                break
            print(f"{prefix}  Item {i}:")
            print_dict_structure(item, prefix + "    ", max_items, max_depth, current_depth + 1)
    elif isinstance(d, np.ndarray):
        print(f"{prefix}NumPy array of shape {d.shape}, dtype {d.dtype}")
        if d.size > 0:
            flat = d.flatten()
            print(f"{prefix}  Sample values: {flat[:min(5, len(flat))]} ... Range: [{np.min(d)}, {np.max(d)}]")
    elif isinstance(d, th.Tensor):
        print(f"{prefix}Torch tensor of shape {d.shape}, dtype {d.dtype}, device {d.device}")
        if d.numel() > 0:
            flat = d.detach().cpu().flatten()
            print(f"{prefix}  Sample values: {flat[:min(5, len(flat))]} ... Range: [{d.min().item()}, {d.max().item()}]")
    else:
        print(f"{prefix}{type(d).__name__}: {str(d)[:100]}")


def load_and_print_pkl(pkl_path, max_items=5):
    """Load a pickle file and print its structure."""
    print(f"Loading pickle file: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\n=== Data Structure ===")
    print_dict_structure(data, max_items=max_items)
    
    # Return the loaded data for further analysis
    return data


def visualize_trajectory(traj, title="Sample Trajectory"):
    """Visualize a trajectory if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available for visualization")
        return
        
    if isinstance(traj, th.Tensor):
        traj = traj.detach().cpu().numpy()
    
    fig, axes = plt.subplots(traj.shape[1], 1, figsize=(10, 2*traj.shape[1]))
    if traj.shape[1] == 1:
        axes = [axes]
        
    for i in range(traj.shape[1]):
        axes[i].plot(traj[:, i])
        axes[i].set_ylabel(f"Joint {i}")
        axes[i].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"Saved visualization to {title.replace(' ', '_')}.png")
    plt.close()


def analyze_single_sample(sample_data):
    """Analyze a single sample's structure and content."""
    if isinstance(sample_data, dict):
        print("\n=== Sample Content Analysis ===")
        for key, value in sample_data.items():
            print(f"\nKey: {key}")
            if isinstance(value, (np.ndarray, th.Tensor)):
                shape_str = value.shape if isinstance(value, np.ndarray) else tuple(value.shape)
                dtype_str = value.dtype
                print(f"  Type: {type(value).__name__}")
                print(f"  Shape: {shape_str}")
                print(f"  Dtype: {dtype_str}")
                
                # Print sample values for arrays/tensors
                if isinstance(value, np.ndarray):
                    flat = value.flatten()
                    if flat.size > 0:
                        print(f"  Sample values: {flat[:min(5, len(flat))]}")
                        print(f"  Range: [{np.min(value)}, {np.max(value)}]")
                else:  # Tensor
                    flat = value.detach().cpu().flatten()
                    if flat.numel() > 0:
                        print(f"  Sample values: {flat[:min(5, len(flat))].tolist()}")
                        print(f"  Range: [{value.min().item()}, {value.max().item()}]")
                
                # Visualize trajectory data if present
                if 'trajectory' in key.lower() or key == 'qs':
                    visualize_trajectory(value, f"Sample_{key}")
            elif isinstance(value, dict):
                print(f"  Type: dict with {len(value)} keys")
                print(f"  Keys: {list(value.keys())}")
            elif isinstance(value, list):
                print(f"  Type: list with {len(value)} items")
                if len(value) > 0:
                    print(f"  First item type: {type(value[0]).__name__}")
            else:
                print(f"  Type: {type(value).__name__}")
                print(f"  Value: {str(value)[:100]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PRESTO dataset")
    parser.add_argument("--pkl_path", type=str, help="Path to the pickle file")
    parser.add_argument("--sample_index", type=int, default=0, help="Index of sample to analyze")
    parser.add_argument("--max_items", type=int, default=5, help="Maximum number of items to print")
    args = parser.parse_args()
    
    # Load and print the structure of the pickle file
    try:
        data = load_and_print_pkl(args.pkl_path, args.max_items)
        
        # Check if it's a single sample or a dataset with multiple samples
        if isinstance(data, dict) and 'qs' in data:
            # This appears to be the standard PRESTO dataset format
            print("\n=== Dataset Overview ===")
            print(f"Number of trajectories: {len(data['qs'])}")
            print(f"Trajectory shape: {data['qs'].shape if hasattr(data['qs'], 'shape') else 'variable'}")
            
            # Get a single trajectory sample
            if args.sample_index < len(data['qs']):
                sample = {}
                for key in data:
                    try:
                        sample[key] = data[key][args.sample_index]
                    except (IndexError, TypeError):
                        sample[key] = data[key]  # If not indexable
                
                analyze_single_sample(sample)
            else:
                print(f"Sample index {args.sample_index} out of range (max: {len(data['qs'])-1})")
        elif isinstance(data, list) and len(data) > 0:
            # This might be a list of samples
            print(f"\n=== Dataset Overview ===")
            print(f"Number of samples: {len(data)}")
            
            if args.sample_index < len(data):
                analyze_single_sample(data[args.sample_index])
            else:
                print(f"Sample index {args.sample_index} out of range (max: {len(data)-1})")
        else:
            # Just analyze whatever we have
            analyze_single_sample(data)
            
    except Exception as e:
        print(f"Error analyzing data: {e}")


if __name__ == "__main__":
    main() 