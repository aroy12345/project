import argparse
from pathlib import Path
import resource

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world


OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6


def main(args, rank):
    print(f"Process {rank} started!")
    print(f"Debug: Worker {rank} settings - grasps_per_scene={args.grasps_per_scene}, seed={np.random.randint(0, 1000) + rank}")
    GRASPS_PER_SCENE = args.grasps_per_scene
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)
    print(f"Debug: Worker {rank} initialized with seed {seed}")
    sim = ClutterRemovalSim(args.scene, args.object_set, gui=args.sim_gui)
    finger_depth = sim.gripper.finger_depth
    print(f"Debug: Worker {rank} finger_depth={finger_depth}")
    grasps_per_worker = args.num_grasps // args.num_proc
    pbar = tqdm(total=grasps_per_worker, disable=rank != 0)

    if rank == 0:
        print(f"Process {rank} started!")
        (args.root / "scenes").mkdir(parents=True)
        print(f"Debug: Created scenes directory at {args.root / 'scenes'}")
        write_setup(
            args.root,
            sim.size,
            sim.camera.intrinsic,
            sim.gripper.max_opening_width,
            sim.gripper.finger_depth,
        )
        print(f"Debug: Wrote setup data with sim.size={sim.size}, max_opening_width={sim.gripper.max_opening_width}")
        if args.save_scene:
            (args.root / "mesh_pose_list").mkdir(parents=True)
            print(f"Debug: Created mesh_pose_list directory at {args.root / 'mesh_pose_list'}")

    for scene_idx in range(grasps_per_worker // GRASPS_PER_SCENE):
        # generate heap
        print(f"Debug: Worker {rank} generating heap #{scene_idx}!")
        object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
        print(f"Debug: Worker {rank} generating {object_count} objects")
        sim.reset(object_count)
        sim.save_state()
        print(f"Debug: Worker {rank} saved simulation state")

        # render synthetic depth images
        n = MAX_VIEWPOINT_COUNT
        print(f"Debug: Worker {rank} rendering {n} depth images")
        depth_imgs, extrinsics = render_images(sim, n)
        print(f"Debug: Worker {rank} rendered {len(depth_imgs)} top-down images")
        depth_imgs_side, extrinsics_side = render_side_images(sim, 1, args.random)
        print(f"Debug: Worker {rank} rendered {len(depth_imgs_side)} side images with random={args.random}")

        # reconstrct point cloud using a subset of the images
        print(f"Debug: Worker {rank} creating TSDF with size={sim.size}")
        tsdf = create_tsdf(sim.size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        print(f"Debug: Worker {rank} TSDF creation complete, extracting point cloud")
        pc = tsdf.get_cloud()
        print(f"Debug: Worker {rank} extracted point cloud with {len(pc.points)} points")
        print(f"Debug: Worker {rank} point cloud extraction complete with memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024} MB")

        # crop surface and borders from point cloud
        bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
        print(f"Debug: Worker {rank} cropping with bounds {sim.lower} to {sim.upper}")
        pc = pc.crop(bounding_box)
        print(f"Debug: Worker {rank} cropped point cloud has {len(pc.points)} points")
        # o3d.visualization.draw_geometries([pc])

        if pc.is_empty():
            print(f"Debug: Worker {rank} point cloud empty, skipping scene")
            continue

        # store the raw data
        print(f"Debug: Worker {rank} starting to write sensor data to disk")
        scene_id = write_sensor_data(args.root, depth_imgs_side, extrinsics_side)
        print(f"Debug: Worker {rank} created scene with ID {scene_id}")
        if args.save_scene:
            print(f"Debug: Worker {rank} saving mesh pose list for scene {scene_id}")
            mesh_pose_list = get_mesh_pose_list_from_world(sim.world, args.object_set)
            write_point_cloud(args.root, scene_id, mesh_pose_list, name="mesh_pose_list")
            print(f"Debug: Worker {rank} saved mesh pose list with {len(mesh_pose_list)} items")

        for grasp_idx in range(GRASPS_PER_SCENE):
            print(f"Debug: Worker {rank} sampling grasp point {grasp_idx+1}/{GRASPS_PER_SCENE} for scene {scene_id}")
            # sample and evaluate a grasp point
            point, normal = sample_grasp_point(pc, finger_depth)
            print(f"Debug: Worker {rank} sampled point {point} with normal {normal}")
            grasp, label = evaluate_grasp_point(sim, point, normal)
            print(f"Debug: Worker {rank} evaluated grasp with width={grasp.width}, label={label}")

            # store the sample
            write_grasp(args.root, scene_id, grasp, label)
            pbar.update()

    pbar.close()
    print(f'Debug: Process {rank} finished all {grasps_per_worker} grasps!')


def render_images(sim, n):
    print(f"Debug: render_images called with n={n}")
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    print(f"Debug: camera dimensions {width}x{height}")
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])
    print(f"Debug: camera origin at {origin.translation}")

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)
        print(f"Debug: camera {i} params: r={r}, theta={theta}, phi={phi}")

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]
        print(f"Debug: rendered image {i} with shape {depth_img.shape}")

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics

def render_side_images(sim, n=1, random=False):
    print(f"Debug: render_side_images called with n={n}, random={random}")
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, sim.size / 3])
    print(f"Debug: side camera origin at {origin.translation}")

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)

    for i in range(n):
        if random:
            r = np.random.uniform(1.6, 2.4) * sim.size
            theta = np.random.uniform(np.pi / 4.0, 5.0 * np.pi / 12.0)
            phi = np.random.uniform(- 5.0 * np.pi / 5, - 3.0 * np.pi / 8.0)
            print(f"Debug: random side camera {i} params: r={r}, theta={theta}, phi={phi}")
        else:
            r = 2 * sim.size
            theta = np.pi / 3.0
            phi = - np.pi / 2.0
            print(f"Debug: fixed side camera {i} params: r={r}, theta={theta}, phi={phi}")

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        depth_img = sim.camera.render(extrinsic)[1]
        print(f"Debug: rendered side image {i} with shape {depth_img.shape}")

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img

    return depth_imgs, extrinsics


def sample_grasp_point(point_cloud, finger_depth, eps=0.1):
    print(f"Debug: sample_grasp_point called with {len(point_cloud.points)} points, finger_depth={finger_depth}, eps={eps}")
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)
    ok = False
    attempts = 0
    while not ok:
        # TODO this could result in an infinite loop, though very unlikely
        attempts += 1
        idx = np.random.randint(len(points))
        point, normal = points[idx], normals[idx]
        ok = normal[2] > -0.1  # make sure the normal is poitning upwards
        if attempts % 10 == 0:
            print(f"Debug: Made {attempts} attempts to find valid grasp point, normal[2]={normal[2]}")
    grasp_depth = np.random.uniform(-eps * finger_depth, (1.0 + eps) * finger_depth)
    print(f"Debug: Selected point index {idx} after {attempts} attempts, grasp_depth={grasp_depth}")
    point = point + normal * grasp_depth
    return point, normal


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    print(f"Debug: evaluate_grasp_point called with pos={pos}, normal={normal}, num_rotations={num_rotations}")
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
        print(f"Debug: x_axis aligned with z_axis, using alternative x_axis")
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)
    print(f"Debug: Initial grasp frame computed with z_axis={z_axis}")

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    print(f"Debug: Testing {len(yaws)} yaw angles: {yaws}")
    outcomes, widths = [], []
    for i, yaw in enumerate(yaws):
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        print(f"Debug: Yaw {i}: {yaw:.2f} rad -> outcome={outcome}, width={width}")
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    print(f"Debug: Success pattern: {successes}")
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        print(f"Debug: Found {len(peaks)} success peaks with widths {properties['widths']}")
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]
        print(f"Debug: Selected peak at index {idx_of_widest_peak} with yaw={yaws[idx_of_widest_peak]:.2f}, width={width}")
    else:
        print(f"Debug: No successful grasps found")

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="blocks")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--grasps-per-scene", type=int, default=120)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--random", action="store_true", help="Add distrubation to camera pose")
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    args.save_scene = True
    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)
