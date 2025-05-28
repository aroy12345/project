import torch
import numpy as np
import random
import math

# --- Part 1: TSDF Generation for Primitive Objects ---

def create_3d_grid(grid_min, grid_max, resolution, device='cpu'):
    """
    Creates a 3D grid of query points.

    Args:
        grid_min (list or tuple of 3 floats): Minimum coordinates (x, y, z).
        grid_max (list or tuple of 3 floats): Maximum coordinates (x, y, z).
        resolution (int): Number of points along each dimension.
        device (str): Device for the output tensor ('cpu' or 'cuda').

    Returns:
        torch.Tensor: A tensor of shape (resolution*resolution*resolution, 3)
                      representing the grid points.
    """
    x_coords = torch.linspace(grid_min[0], grid_max[0], resolution, device=device)
    y_coords = torch.linspace(grid_min[1], grid_max[1], resolution, device=device)
    z_coords = torch.linspace(grid_min[2], grid_max[2], resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    return grid_points

def get_sphere_tsdf(points, center, radius):
    """
    Computes the TSDF for a sphere.
    TSDF value is distance to surface: (distance_from_center - radius).
    Negative inside, positive outside, zero on the surface.

    Args:
        points (torch.Tensor): Query points, shape (N, 3).
        center (torch.Tensor): Sphere center, shape (3,).
        radius (float): Sphere radius.

    Returns:
        torch.Tensor: TSDF values for each point, shape (N,).
    """
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=points.dtype, device=points.device)
    distances_from_center = torch.norm(points - center.unsqueeze(0), dim=1)
    return distances_from_center - radius

def get_axis_aligned_cube_tsdf(points, center, size):
    """
    Computes the TSDF for an axis-aligned cube.
    Args:
        points (torch.Tensor): Query points, shape (N, 3).
        center (torch.Tensor or list/tuple): Cube center.
        size (torch.Tensor or list/tuple or float): Cube full side lengths (dx, dy, dz).
                                                 If float, assumes a uniform cube.
    Returns:
        torch.Tensor: TSDF values for each point, shape (N,).
    """
    target_device = points.device
    target_dtype = points.dtype

    # Ensure center is a tensor on the correct device and dtype
    if not isinstance(center, torch.Tensor):
        center_t = torch.tensor(center, dtype=target_dtype, device=target_device)
    else:
        center_t = center.to(device=target_device, dtype=target_dtype)

    # Ensure half_size is a tensor on the correct device and dtype
    # 'size' parameter is interpreted as full side lengths
    if isinstance(size, torch.Tensor):
        half_size_t = size.to(device=target_device, dtype=target_dtype) / 2.0
    elif isinstance(size, list) or isinstance(size, tuple):
        half_size_t = torch.tensor(size, dtype=target_dtype, device=target_device) / 2.0
    elif isinstance(size, (int, float)): # Single number, assume uniform cube
        half_size_t = torch.tensor([size/2.0, size/2.0, size/2.0], dtype=target_dtype, device=target_device)
    else:
        raise TypeError(f"Unsupported type for cube size: {type(size)}")

    if half_size_t.numel() == 1: # Should not happen if size is properly [sx,sy,sz] or becomes it
        half_size_t = half_size_t.repeat(3)
    elif half_size_t.numel() != 3:
        raise ValueError(f"Cube size must result in 3 dimensions for half_size, got {half_size_t.numel()}")


    # Transform points to cube's local frame (relative to center)
    local_points = points - center_t.unsqueeze(0)

    q = torch.abs(local_points) - half_size_t.unsqueeze(0)
    
    signed_distance_to_axis_planes = torch.max(q, torch.zeros_like(q))
    distance_outside = torch.norm(signed_distance_to_axis_planes, dim=1)
    
    distance_inside = torch.min(torch.max(q, dim=1)[0], torch.zeros_like(distance_outside))
    
    return distance_outside + distance_inside


def combine_tsdfs(tsdf_list):
    """
    Combines multiple TSDFs by taking the minimum value at each point.
    This corresponds to the union of the objects.

    Args:
        tsdf_list (list of torch.Tensor): List of TSDFs, each of shape (N,).

    Returns:
        torch.Tensor: Combined TSDF, shape (N,).
    """
    if not tsdf_list:
        return torch.empty(0)
    stacked_tsdfs = torch.stack(tsdf_list, dim=0)
    return torch.min(stacked_tsdfs, dim=0)[0]

# --- Part 2: Trajectory Generation (Skeleton and Placeholders) ---

class PrimitiveObstacle:
    def __init__(self, type, params, unique_id=None):
        self.type = type
        self.params = params
        self.device = 'cpu'
        self.unique_id = unique_id if unique_id is not None else random.randint(0, 1_000_000)

    def to(self, device):
        self.device = device
        center_val = self.params.get('center')
        if isinstance(center_val, list) or isinstance(center_val, tuple) or isinstance(center_val, np.ndarray):
            self.params['center'] = torch.tensor(center_val, dtype=torch.float32, device=self.device)
        elif isinstance(center_val, torch.Tensor):
            self.params['center'] = center_val.to(device=self.device, dtype=torch.float32)

        if self.type == 'cube':
            size_val = self.params.get('size')
            if isinstance(size_val, list) or isinstance(size_val, tuple) or isinstance(size_val, np.ndarray):
                self.params['size'] = torch.tensor(size_val, dtype=torch.float32, device=self.device)
            elif isinstance(size_val, (int, float)):
                self.params['size'] = torch.tensor([size_val, size_val, size_val], dtype=torch.float32, device=self.device)
            elif isinstance(size_val, torch.Tensor):
                self.params['size'] = size_val.to(device=self.device, dtype=torch.float32)
        return self

    def get_tsdf(self, points):
        center_param = self.params['center']
        if self.type == 'sphere':
            return get_sphere_tsdf(points, center_param, self.params['radius'])
        elif self.type == 'cube':
            return get_axis_aligned_cube_tsdf(points, center_param, self.params['size'])
        else:
            raise ValueError(f"Unknown obstacle type: {self.type}")

    def get_parameters_for_col_label(self):
        """Returns a flat list of parameters for the col-label."""
        # Type: 0 for sphere, 1 for cube
        type_code = 0 if self.type == 'sphere' else 1
        center = self.params['center'].tolist() # Should be a tensor
        if self.type == 'sphere':
            radius = [self.params['radius'], 0, 0] # Pad radius to 3 elements like size
            return [type_code] + center + radius
        else: # cube
            size = self.params['size'].tolist() # Should be a tensor
            return [type_code] + center + size

class Environment:
    def __init__(self, obstacles, workspace_min, workspace_max):
        self.obstacles = obstacles
        self.workspace_min = torch.tensor(workspace_min, dtype=torch.float32)
        self.workspace_max = torch.tensor(workspace_max, dtype=torch.float32)

    def to(self, device):
        self.workspace_min = self.workspace_min.to(device)
        self.workspace_max = self.workspace_max.to(device)
        for obs in self.obstacles:
            obs.to(device)
        return self

    def get_combined_tsdf(self, points):
        if not self.obstacles:
            return torch.full((points.shape[0],), float('inf'), device=points.device, dtype=points.dtype)
        all_tsdfs = [obs.get_tsdf(points) for obs in self.obstacles]
        return combine_tsdfs(all_tsdfs)

    def get_obstacle_params_for_col_label(self, max_obs=8):
        """Generates a flattened list of parameters for all obstacles for 'col-label'."""
        col_label_params = []
        num_obs = 0
        for obs in self.obstacles:
            if num_obs < max_obs:
                col_label_params.extend(obs.get_parameters_for_col_label())
                num_obs +=1
            else:
                break
        # Pad if fewer than max_obs obstacles (each obs has 1 for type + 3 for center + 3 for size/radius = 7 params)
        padding_per_obs = 7
        while num_obs < max_obs:
            col_label_params.extend([0.0] * padding_per_obs) # Pad with "empty" obstacle data
            num_obs +=1
        return col_label_params

# --- Part 2: Collision Checking and RRT Planner ---

class SphericalRobot:
    def __init__(self, radius):
        self.radius = radius

def is_collision(robot_config_point, robot_model: SphericalRobot, environment: Environment):
    """
    Checks collision for a spherical robot at a given point configuration.
    Args:
        robot_config_point (torch.Tensor): Current 3D position of the robot center.
        robot_model (SphericalRobot): The robot model.
        environment (Environment): The environment.
    Returns:
        bool: True if in collision, False otherwise.
    """
    if robot_config_point.ndim == 1:
        robot_config_point = robot_config_point.unsqueeze(0) # Ensure (1,3) for TSDF query

    # Check workspace boundaries
    if torch.any(robot_config_point < environment.workspace_min + robot_model.radius) or \
       torch.any(robot_config_point > environment.workspace_max - robot_model.radius):
        return True # Collision with workspace boundary

    tsdf_value_at_robot_center = environment.get_combined_tsdf(robot_config_point)
    return tsdf_value_at_robot_center.item() < robot_model.radius

class RRTNode:
    def __init__(self, config):
        self.config = config # 3D point for this simple RRT
        self.parent = None

def rrt_planner(start_config, goal_config, robot_model: SphericalRobot, environment: Environment,
                max_iter=1000, step_size=0.1, goal_bias=0.1, device='cpu'):
    """
    Basic RRT planner for a point/spherical robot in 3D.
    """
    start_node = RRTNode(start_config.to(device))
    goal_node = RRTNode(goal_config.to(device))

    if is_collision(start_node.config, robot_model, environment) or \
       is_collision(goal_node.config, robot_model, environment):
        # print("RRT Error: Start or Goal in collision.")
        return None, False

    node_list = [start_node]

    for i in range(max_iter):
        # Sample random configuration or goal
        if random.random() < goal_bias:
            rand_config = goal_node.config
        else:
            rand_config = torch.rand(3, device=device) * \
                          (environment.workspace_max - environment.workspace_min) + \
                          environment.workspace_min
        
        # Find nearest node
        nearest_node = node_list[0]
        min_dist = torch.norm(rand_config - nearest_node.config)
        for node in node_list[1:]:
            dist = torch.norm(rand_config - node.config)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        # Steer towards random config
        direction = rand_config - nearest_node.config
        dist_to_rand = torch.norm(direction)
        if dist_to_rand < 1e-6: continue # Avoid division by zero if rand_config is too close

        unit_direction = direction / dist_to_rand
        
        new_config = nearest_node.config + unit_direction * min(step_size, dist_to_rand)
        
        # Check collision for the new segment (simplified: just check the new_config)
        # A more robust check would discretize and check along the segment
        if not is_collision(new_config, robot_model, environment):
            new_node = RRTNode(new_config)
            new_node.parent = nearest_node
            node_list.append(new_node)

            # Check if goal reached
            if torch.norm(new_node.config - goal_node.config) < step_size:
                goal_node.parent = new_node # Connect goal
                path = []
                curr = goal_node
                while curr is not None:
                    path.append(curr.config)
                    curr = curr.parent
                # print(f"RRT: Path found in {i+1} iterations.")
                return torch.stack(path[::-1]), True # Reverse path

    # print(f"RRT: Failed to find path after {max_iter} iterations.")
    return None, False

# --- Part 3: Dynamic Environment and Dataset Generation ---

def generate_random_environment(
    num_obstacles_range=(2, 5), # Min/max number of obstacles
    workspace_min=(-1.0, -1.0, -0.5),
    workspace_max=(1.0, 1.0, 1.0),
    obstacle_size_range=(0.1, 0.4), # Radius for spheres, side length for cubes
    device='cpu'
):
    """Generates a random environment with spheres and cubes."""
    num_obstacles = random.randint(*num_obstacles_range)
    obstacles = []
    env_min = torch.tensor(workspace_min, device=device)
    env_max = torch.tensor(workspace_max, device=device)

    for i in range(num_obstacles):
        obj_type = random.choice(['sphere', 'cube'])
        
        # Ensure obstacle center is within a slightly smaller workspace to avoid edge cases
        center_margin = 0.2 
        center_x = random.uniform(workspace_min[0] + center_margin, workspace_max[0] - center_margin)
        center_y = random.uniform(workspace_min[1] + center_margin, workspace_max[1] - center_margin)
        center_z = random.uniform(workspace_min[2] + center_margin, workspace_max[2] - center_margin)
        center = torch.tensor([center_x, center_y, center_z], device=device)

        if obj_type == 'sphere':
            radius = random.uniform(obstacle_size_range[0], obstacle_size_range[1])
            obstacles.append(PrimitiveObstacle(type='sphere', params={'center': center, 'radius': radius}, unique_id=i).to(device))
        else: # cube
            size_val = random.uniform(obstacle_size_range[0], obstacle_size_range[1])
            size = torch.tensor([size_val, size_val, size_val], device=device) # Uniform cubes for simplicity
            obstacles.append(PrimitiveObstacle(type='cube', params={'center': center, 'size': size}, unique_id=i).to(device))
            
    return Environment(obstacles, workspace_min, workspace_max).to(device)

def generate_valid_start_goal(environment: Environment, robot_model: SphericalRobot, max_tries=50):
    """Generates collision-free start and goal configurations within the workspace."""
    for _ in range(max_tries):
        start_config = torch.rand(3, device=environment.workspace_min.device) * \
                       (environment.workspace_max - environment.workspace_min) + \
                       environment.workspace_min
        goal_config = torch.rand(3, device=environment.workspace_min.device) * \
                      (environment.workspace_max - environment.workspace_min) + \
                      environment.workspace_min
        
        # Ensure start and goal are not too close
        if torch.norm(start_config - goal_config) < 0.5: # Minimum distance
            continue

        if not is_collision(start_config, robot_model, environment) and \
           not is_collision(goal_config, robot_model, environment):
            return start_config, goal_config
    return None, None # Failed to find valid start/goal

def generate_single_datapoint_rrt(
    tsdf_grid_min, tsdf_grid_max, tsdf_resolution,
    robot_model: SphericalRobot,
    seq_len=50, # Desired length for the trajectory (robot configs are 3D points)
    obs_dim=3,  # For a point robot, obs_dim is 3 (x,y,z)
    max_obs_for_label=5, # Max obstacles to include in col-label
    rrt_max_iter=2000,
    rrt_step_size=0.1,
    rrt_goal_bias=0.1,
    device='cpu'
):
    path_found_for_env = False
    generated_environment = None
    start_config_robot = None
    goal_config_robot = None
    raw_trajectory = None
    
    max_env_gen_tries = 10
    for _ in range(max_env_gen_tries):
        generated_environment = generate_random_environment(
            num_obstacles_range=(2,max_obs_for_label), # Control density
            workspace_min=tsdf_grid_min, # Align TSDF and workspace
            workspace_max=tsdf_grid_max,
            device=device
        )
        start_config_robot, goal_config_robot = generate_valid_start_goal(generated_environment, robot_model)

        if start_config_robot is None or goal_config_robot is None:
            # print("Could not generate valid start/goal for this environment, retrying env.")
            continue

        raw_trajectory, success = rrt_planner(
            start_config_robot, goal_config_robot,
            robot_model, generated_environment,
            max_iter=rrt_max_iter, step_size=rrt_step_size, goal_bias=rrt_goal_bias, device=device
        )
        if success and raw_trajectory is not None:
            path_found_for_env = True
            break
        # else:
            # print(f"RRT failed for current env/start/goal. Retrying env or start/goal.")

    if not path_found_for_env or raw_trajectory is None:
        # print("Failed to generate a valid trajectory after multiple environment/start/goal attempts.")
        return None

    # 1. Generate TSDF for the successful environment
    grid_points = create_3d_grid(tsdf_grid_min, tsdf_grid_max, tsdf_resolution, device=device)
    tsdf_values = generated_environment.get_combined_tsdf(grid_points)
    tsdf_volume = tsdf_values.reshape(1, tsdf_resolution, tsdf_resolution, tsdf_resolution)
        
    # 2. Post-process trajectory
    if raw_trajectory.shape[0] > seq_len:
        # Uniformly sample seq_len points
        indices = torch.linspace(0, raw_trajectory.shape[0] - 1, seq_len, device=device).long()
        processed_trajectory = raw_trajectory[indices]
    elif raw_trajectory.shape[0] < seq_len:
        padding_needed = seq_len - raw_trajectory.shape[0]
        padding = raw_trajectory[-1, :].unsqueeze(0).repeat(padding_needed, 1)
        processed_trajectory = torch.cat([raw_trajectory, padding], dim=0)
    else:
        processed_trajectory = raw_trajectory
    
    if processed_trajectory.shape[1] != obs_dim: # Should be 3 for point robot
        # This indicates a mismatch, should not happen if robot_model is consistent
        print(f"ERROR: Trajectory dim {processed_trajectory.shape[1]} != obs_dim {obs_dim}")
        return None # Critical error

    # 3. Generate Labels
    # env-label: a simple representation, e.g., number of obstacles (could be more complex)
    env_label_scalar = float(len(generated_environment.obstacles))
    # For PRESTO, cond_dim might be large, so this needs to be a vector.
    # Let's use a one-hot encoding of num_obstacles if cond_dim allows, or just repeat.
    # This is still a placeholder, real env_label needs careful design.
    # For now, let's just use the obstacle parameters as the env_label as well.
    col_label_params = generated_environment.get_obstacle_params_for_col_label(max_obs=max_obs_for_label)
    env_label_tensor = torch.tensor(col_label_params, dtype=torch.float32, device=device)


    data_point = {
        'trajectory': processed_trajectory.to(torch.float32),
        'start': processed_trajectory[0, :].to(torch.float32),
        'goal': processed_trajectory[-1, :].to(torch.float32),
        'tsdf': tsdf_volume.to(torch.float32),
        'env-label': env_label_tensor, # Using obstacle params
        'col-label': env_label_tensor, # Using the same for now
    }
    return data_point

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Test Basic TSDF and Collision (Optional, can comment out) ---
    # ... (previous TSDF tests can remain here or be commented)

    # --- Test Full Data Generation Loop ---
    print("\n--- Generating Diverse Dataset (RRT based) ---")
    
    tsdf_resolution_main = 32
    # Define workspace, also used for TSDF grid
    main_grid_min = [-1.0, -1.0, -0.5]
    main_grid_max = [1.0, 1.0, 1.0]
    
    robot = SphericalRobot(radius=0.05) # Small spherical robot
    
    num_datapoints_to_generate = 10 # Generate a small dataset for testing
    dataset = []
    
    generated_count = 0
    attempts = 0
    max_attempts = num_datapoints_to_generate * 5 # Try more times than needed

    pbar = range(num_datapoints_to_generate)
    if 'tqdm' in globals(): # Check if tqdm is available
        try:
            from tqdm import tqdm
            pbar = tqdm(range(num_datapoints_to_generate), desc="Generating Datapoints")
        except ImportError:
            pass


    for i in pbar: # Use pbar here
        attempts = 0
        while attempts < 10: # Max 10 tries per datapoint (env/pathgen)
            datapoint = generate_single_datapoint_rrt(
                tsdf_grid_min=main_grid_min,
                tsdf_grid_max=main_grid_max,
                tsdf_resolution=tsdf_resolution_main,
                robot_model=robot,
                seq_len=50,
                obs_dim=3, # Point robot in 3D
                max_obs_for_label=5,
                rrt_max_iter=3000, # Increased iterations for RRT
                rrt_step_size=0.15,
                rrt_goal_bias=0.2,
                device=device
            )
            if datapoint:
                dataset.append(datapoint)
                if not isinstance(pbar, range): # if tqdm is used
                    pbar.set_postfix({"Generated": len(dataset)})
                break
            attempts += 1
        if not datapoint and isinstance(pbar, range): # if not tqdm
             print(f"Failed to generate datapoint {i+1} after {attempts} tries for it.")


    print(f"\nSuccessfully generated {len(dataset)} datapoints.")

    if dataset:
        print("\nExample of first generated datapoint:")
        for key, value in dataset[0].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}, device {value.device}")
            else:
                print(f"  {key}: {value}")
        
        # Example: Save the dataset (list of dictionaries with tensors)
        # You might want to convert tensors to numpy for wider compatibility if pickling
        # or save in a more structured format like .pt files or HDF5.
        # For now, just a print.
        # import pickle
        # with open("generated_rrt_dataset.pkl", "wb") as f:
        #    pickle.dump(dataset, f)
        # print("Saved dataset to generated_rrt_dataset.pkl")

        # --- Basic Validation (Visual or Statistical) ---
        # 1. Check TSDF values at trajectory points:
        #    For each trajectory in the dataset, sample points along it.
        #    Query the TSDF of the corresponding environment at these points.
        #    Values should be > robot_radius for collision-free paths.
        print("\n--- Basic Validation (example on first datapoint) ---")
        dp0 = dataset[0]
        traj0 = dp0['trajectory'] # Shape [seq_len, 3]
        env_for_dp0 = generate_random_environment( # Need to regenerate or store env with dp
             num_obstacles_range=(2,5), workspace_min=main_grid_min, workspace_max=main_grid_max, device=device
        )
        # This is tricky: need to reconstruct the exact environment used for that trajectory.
        # Better: Store environment object or its parameters with the datapoint, or regenerate with a seed.
        # For now, this validation part is just illustrative of the concept.
        
        # A better way: when generating, also store the list of obstacle parameters,
        # then rebuild the environment for validation.
        # The 'col-label' and 'env-label' now store obstacle parameters, so we can use that.

        params_from_label = dp0['env-label'].cpu().tolist()
        reconstructed_obstacles = []
        num_params_per_obs = 7 # type + center(3) + size(3)
        for i in range(0, len(params_from_label), num_params_per_obs):
            obs_chunk = params_from_label[i:i+num_params_per_obs]
            if sum(abs(p) for p in obs_chunk) < 1e-5 : # Heuristic for empty/padded obstacle
                continue 
            obs_type_code = int(obs_chunk[0])
            center = obs_chunk[1:4]
            dims = obs_chunk[4:7]
            if obs_type_code == 0: # Sphere
                reconstructed_obstacles.append(
                    PrimitiveObstacle(type='sphere', params={'center': center, 'radius': dims[0]}, unique_id=i//num_params_per_obs).to(device)
                )
            elif obs_type_code == 1: # Cube
                 reconstructed_obstacles.append(
                    PrimitiveObstacle(type='cube', params={'center': center, 'size': dims}, unique_id=i//num_params_per_obs).to(device)
                )
        
        if reconstructed_obstacles:
            reconstructed_env = Environment(reconstructed_obstacles, main_grid_min, main_grid_max).to(device)
            
            collision_found_in_validation = False
            for point_idx, point_on_traj in enumerate(traj0):
                if is_collision(point_on_traj, robot, reconstructed_env):
                    print(f"  VALIDATION FAIL: Collision detected for robot at trajectory point {point_idx}: {point_on_traj.cpu().numpy()}")
                    collision_found_in_validation = True
                    break
            if not collision_found_in_validation:
                print("  VALIDATION PASS (basic): No collisions detected for robot along the first trajectory.")
        else:
            print("  VALIDATION SKIP: Could not reconstruct environment from label for first datapoint.")
