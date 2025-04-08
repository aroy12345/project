import torch
import torch.nn.functional as F

def generate_dummy_datapoint(
    seq_len=100,
    obs_dim=7,      # Franka Panda joints
    cond_dim=104,   # Example conditioning dimension
    tsdf_dim=32,    # Example grid size for TSDF
    num_grasp_points=64,
    num_tsdf_points=32,
    device='cpu'
):
    """
    Generates a single dummy data point dictionary with tensors conforming
    to the expected structure for combined_train.py.

    Args:
        seq_len (int): Length of the trajectory sequence.
        obs_dim (int): Dimension of the observation/joint space.
        cond_dim (int): Dimension of the conditioning vector ('env-label').
        tsdf_dim (int): Grid size for the TSDF volume (G).
        num_grasp_points (int): Number of grasp query points (M).
        num_tsdf_points (int): Number of TSDF query points (N_tsdf).
        device (str): Device to place tensors on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing dummy tensors for a single data sample.
    """

    # Generate normalized quaternion(s) for grasp labels
    grasp_rots = F.normalize(torch.randn(num_grasp_points, 4, device=device), dim=-1)

    # Generate dummy collision label (adjust shape/content based on CuroboCost needs)
    # Example: Flattened parameters or an embedding
    col_label_data = torch.randn(cond_dim, device=device) # Match env-label dim for simplicity

    data_point = {
        # Trajectory: Shape [S, C] as expected by PRESTO dataset __getitem__
        'trajectory': torch.randn(seq_len, obs_dim, device=device, dtype=torch.float32),

        # Conditioning vector for diffusion
        'env-label': torch.randn(cond_dim, device=device, dtype=torch.float32),

        # Collision information for CuroboCost
        'col-label': col_label_data,

        # TSDF Volume (Optional, but often used for GIGA conditioning)
        # Shape: [1, G, G, G]
        'tsdf': torch.randn(1, tsdf_dim, tsdf_dim, tsdf_dim, device=device, dtype=torch.float32),

        # Grasp Query Points and Labels
        # Shape: [M, 3]
        'grasp_query_points': torch.rand(num_grasp_points, 3, device=device, dtype=torch.float32) * 0.6 - 0.3, # Example range [-0.3, 0.3]
        # Shape: [M, 1] - Binary labels (0.0 or 1.0)
        'grasp_qual_labels': torch.randint(0, 2, (num_grasp_points, 1), device=device).float(),
        # Shape: [M, 4] - Normalized quaternions
        'grasp_rot_labels': grasp_rots,
        # Shape: [M, 1] - Example width range [0.0, 0.1]
        'grasp_width_labels': torch.rand(num_grasp_points, 1, device=device, dtype=torch.float32) * 0.1,

        # TSDF Query Points and Labels (Optional, for TSDF prediction loss)
        # Shape: [N_tsdf, 3]
        'tsdf_query_points': torch.rand(num_tsdf_points, 3, device=device, dtype=torch.float32) * 0.6 - 0.3, # Example range
        # Shape: [N_tsdf, 1] - Binary labels (0.0 or 1.0)
        'tsdf_labels': torch.randint(0, 2, (num_tsdf_points, 1), device=device).float()
    }

    return data_point

def create_batch_from_datapoints(datapoints, device='cpu'):
    """
    Collates a list of single data point dictionaries into a batch dictionary.
    Uses torch.stack for simplicity. Assumes all datapoints have identical tensor shapes.

    Args:
        datapoints (list[dict]): A list of data point dictionaries generated by
                                 generate_dummy_datapoint.
        device (str): Device for the final batch tensors.

    Returns:
        dict: A dictionary where each key maps to a stacked batch tensor.
    """
    if not datapoints:
        return {}
    # Get keys from the first datapoint
    batch = {key: [] for key in datapoints[0]}
    # Collect tensors for each key
    for dp in datapoints:
        for key in batch:
            batch[key].append(dp[key])
    # Stack tensors along the batch dimension (dim=0)
    for key in batch:
        batch[key] = torch.stack(batch[key], dim=0).to(device)
    return batch
