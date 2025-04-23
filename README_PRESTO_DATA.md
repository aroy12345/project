# PRESTO Data Inspection and Generation

This guide explains how to load, inspect, and generate data with the PRESTO (Pose-Driven Robust rEpresentation for Scene-based Trajectory Optimization) system.

## Setup

First, ensure you have the PRESTO repository installed and all dependencies:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/UT-Austin-RPL/PRESTO.git
cd PRESTO

# Install dependencies
pip install -e .
```

## Inspecting PRESTO Data

The `inspect_presto_data.py` script allows you to load and visualize trajectories from the PRESTO dataset.

### Usage

```bash
python inspect_presto_data.py --dataset_dir path/to/your/dataset --pattern "*.pkl" --num_samples 3
```

### Arguments

- `--dataset_dir`: Path to the directory containing the dataset PKL files
- `--pattern`: File pattern to match dataset files (default: "*.pkl")
- `--num_samples`: Number of sample trajectories to display (default: 3)

### What the Script Shows

The script will load the dataset and print:
- Dataset size and dimensions
- Normalization parameters
- For each sample:
  - Trajectory statistics (shape, min/max values)
  - Start and goal positions
  - Environment and collision label information
  - Visualizations of each joint trajectory (saved as PNG files)

## Generating PRESTO Trajectories

The `generate_presto_data.py` script demonstrates how to generate new trajectories using a trained PRESTO model.

### Usage

```bash
python generate_presto_data.py --dataset_dir path/to/your/dataset --checkpoint path/to/model_checkpoint.pt --num_inference_steps 50 --init_type random
```

### Arguments

- `--dataset_dir`: Path to the directory containing the dataset PKL files
- `--checkpoint`: Path to a trained model checkpoint (.pt file) [REQUIRED]
- `--num_inference_steps`: Number of diffusion steps for inference (default: 50)
- `--batch_size`: Batch size for generation (default: 1)
- `--init_type`: Type of trajectory initialization:
  - `random`: Start with random noise
  - `linear`: Start with linear interpolation from start to goal
  - `true`: Start with the ground truth trajectory

### What the Script Shows

The script will:
1. Load the PRESTO dataset
2. Load a trained model from a checkpoint
3. Set up the diffusion scheduler and pipeline
4. Generate new trajectories
5. Visualize the diffusion process steps
6. Print statistics about the generated trajectories
7. Compare with ground truth if available

## Understanding PRESTO Data

The PRESTO dataset typically includes:

1. **Trajectories** (`qs`): Joint configurations over time, shape [num_samples, seq_len, obs_dim]
2. **Environmental Information** (`ys`): Scene descriptions, shape [num_samples, cond_dim]  
3. **Collision Labels** (`ws`): Information about collision objects
4. **Start/Goal Positions**: The initial and final joint configurations

## How PRESTO Generates Data

PRESTO uses a diffusion-based model to generate robot trajectories:

1. **Initialization**: Start with either random noise, linear interpolation, or ground truth
2. **Conditioning**: Use environmental information to condition the diffusion process
3. **Denoising**: Apply the learned model to progressively denoise the trajectory
4. **Constraints**: Optionally enforce start/goal constraints and collision avoidance

The diffusion process produces a sequence of progressively refined trajectories, eventually yielding a smooth, collision-free path that connects the start and goal positions while respecting environmental constraints.

## Troubleshooting

If you encounter issues:

- Ensure your dataset path is correct
- Check that the model checkpoint is compatible with your code
- Verify that the dataset format matches what PRESTO expects
- For GPU errors, try using `--device cpu` to run on CPU instead 