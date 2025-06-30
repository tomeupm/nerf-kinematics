#!/usr/bin/env python3
"""
Convert a poses.txt file with bracket-and-semicolon matrices into an Instant-NGP (NeRF) compatible transforms.json,
including per-frame sharpness computed via variance of Laplacian (PIL-only).

Supports image filenames with zero-based indices (e.g., "TestNERF 0.jpg").
If an image for index X is missing, it tries X+1, X+2, ... for the same matrix,
and ensures each image is used only once for subsequent frames.
Also recenters and optionally scales camera translations to the scene origin.
"""
import os
import re
import json
import math
import argparse
import numpy as np
from PIL import Image, ImageFilter


def parse_poses_file(poses_file_path):
    """
    Parse poses.txt containing matrices in the form:
    [r1c1, r1c2, ..., r1c4 ; ... ; r4c1, ..., r4c4]
    Returns a list of 4x4 matrices as Python lists.
    """
    with open(poses_file_path, 'r') as f:
        content = f.read()
    matrix_pattern = r"\[\s*(.*?)\s*\]"
    raw_mats = re.findall(matrix_pattern, content, re.DOTALL)
    mats = []
    for raw in raw_mats:
        rows = raw.split(';')
        mat = []
        for row in rows:
            row_clean = row.strip().replace(',', ' ')
            if not row_clean:
                continue
            nums = re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', row_clean)
            if len(nums) != 4:
                continue
            mat.append([float(x) for x in nums])
        if len(mat) == 4:
            mats.append(np.array(mat).tolist())
    if not mats:
        raise RuntimeError(f"No matrices parsed from {poses_file_path}")
    return mats


def compute_sharpness(image_path):
    """Compute image sharpness as variance of Laplacian using PIL."""
    img = Image.open(image_path).convert('L')
    kernel = ImageFilter.Kernel(
        size=(3, 3),
        kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0],
        scale=1,
        offset=0
    )
    lap = img.filter(kernel)
    lap_arr = np.array(lap, dtype=np.float64)
    return float(lap_arr.var())


def generate_test_poses(center_pos, radius=50, n_poses=8):
    """Generate circular test poses around a center position."""
    poses = []
    for i in range(n_poses):
        angle = 2 * math.pi * i / n_poses
        x = center_pos[0] + radius * math.cos(angle)
        y = center_pos[1] + radius * math.sin(angle)
        z = center_pos[2]
        
        # Look towards center
        forward = np.array([center_pos[0] - x, center_pos[1] - y, center_pos[2] - z])
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build transform matrix
        transform = [
            [right[0], up[0], -forward[0], x],
            [right[1], up[1], -forward[1], y],
            [right[2], up[2], -forward[2], z],
            [0.0, 0.0, 0.0, 1.0]
        ]
        poses.append(transform)
    
    return poses


def generate_video_poses(center_pos, radius=40, n_poses=60):
    """Generate smooth circular video poses."""
    poses = []
    for i in range(n_poses):
        angle = 2 * math.pi * i / n_poses
        # Slight vertical variation for more dynamic movement
        height_offset = 5 * math.sin(4 * angle)
        
        x = center_pos[0] + radius * math.cos(angle)
        y = center_pos[1] + radius * math.sin(angle)
        z = center_pos[2] + height_offset
        
        # Look towards center
        forward = np.array([center_pos[0] - x, center_pos[1] - y, center_pos[2] - z])
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build transform matrix
        transform = [
            [right[0], up[0], -forward[0], x],
            [right[1], up[1], -forward[1], y],
            [right[2], up[2], -forward[2], z],
            [0.0, 0.0, 0.0, 1.0]
        ]
        poses.append(transform)
    
    return poses


def create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, k1, k2, p1, p2, aabb_scale):
    """Create base JSON structure with camera parameters."""
    return {
        "camera_angle_x": cax,
        "camera_angle_y": cay,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": aabb_scale,
        "frames": []
    }


def calculate_aabb_scale(matrices, scale_multiplier=1.0):
    """
    Calculate appropriate aabb_scale based on camera positions.
    Returns a power-of-2 value that encompasses the scene bounding box with some margin.
    """
    # Extract camera positions
    positions = np.array([[m[0][3], m[1][3], m[2][3]] for m in matrices])
    
    # Calculate scene center and maximum distance
    center = positions.mean(axis=0)
    max_distance = np.linalg.norm(positions - center, axis=1).max()
    
    print(f"Scene center: {center}")
    print(f"Max distance from center: {max_distance}")
    
    # Calculate target size with margin
    target_size = max_distance * 2 * scale_multiplier  # Diameter with margin
    
    # Round up to nearest power of 2 (minimum 1.0, maximum 128.0)
    aabb_scale = 1.0
    while aabb_scale < target_size and aabb_scale < 128.0:
        aabb_scale *= 2.0
    
    return aabb_scale


def main():
    parser = argparse.ArgumentParser(
        description='Convert poses.txt to Instant-NGP transforms.json with sharpness')
    parser.add_argument('--poses', default='poses.txt', help='Path to poses.txt')
    parser.add_argument('--image_folder', default='images_robot', help='Folder with images')
    parser.add_argument('--image_prefix', default='TestNERF ', help='Image name prefix')
    parser.add_argument('--image_ext', default='jpg', help='Image file extension')
    parser.add_argument('--camera_angle_x', type=float, default=87.0, help='Horizontal FOV (deg)')
    parser.add_argument('--camera_angle_y', type=float, default=58.0, help='Vertical FOV (deg)')
    parser.add_argument('--cx', type=float, default=None, help='Principal point x (defaults to w/2)')
    parser.add_argument('--cy', type=float, default=None, help='Principal point y (defaults to h/2)')
    parser.add_argument('--k1', type=float, default=0.0, help='Radial distortion k1')
    parser.add_argument('--k2', type=float, default=0.0, help='Radial distortion k2')
    parser.add_argument('--p1', type=float, default=0.0, help='Tangential distortion p1')
    parser.add_argument('--p2', type=float, default=0.0, help='Tangential distortion p2')
    parser.add_argument('--recenter', action='store_true', help='Recenter cameras to scene origin')
    parser.add_argument('--scale_trans', type=float, default=1.0, help='Uniform scale for camera translations')
    parser.add_argument('--output', default='transforms.json', help='Output JSON name')
    args = parser.parse_args()

    # Parse transform matrices
    mats = parse_poses_file(args.poses)
    n_mats = len(mats)

    # Separate validation data (first matrix/image)
    if n_mats > 1:
        val_mat = mats[0]  # First matrix for validation
        train_mats = mats[1:]  # All except first for training
        n_train_mats = len(train_mats)
    else:
        raise RuntimeError("Need at least 2 matrices to separate training and validation")

    # Calculate initial aabb_scale using training matrices only
    initial_aabb_scale = calculate_aabb_scale(train_mats)
    print(f"Initial aabb_scale (before transformations): {initial_aabb_scale}")

    # Optional: recenter and scale translations
    if args.recenter:
        # Calculate centroid from training matrices only
        centers = np.array([[m[0][3], m[1][3], m[2][3]] for m in train_mats])
        center = centers.mean(axis=0) if args.recenter else np.zeros(3)
        

        # Use max distance approach for normalization
        max_distance = np.linalg.norm(centers - center, axis=1).max()
        scale_factor = args.scale_trans / max_distance
        
        # Apply transformations to training matrices
        for m in train_mats:
            m[0][3] = (m[0][3] - center[0]) * scale_factor
            m[1][3] = (m[1][3] - center[1]) * scale_factor
            m[2][3] = (m[2][3] - center[2]) * scale_factor
        
        # Apply same transformations to validation matrix
        val_mat[0][3] = (val_mat[0][3] - center[0]) * scale_factor
        val_mat[1][3] = (val_mat[1][3] - center[1]) * scale_factor
        val_mat[2][3] = (val_mat[2][3] - center[2]) * scale_factor
        
        print(f"Applied recentering: {args.recenter}, scale factor: {scale_factor}")

    # Calculate final aabb_scale after all transformations
    final_aabb_scale = calculate_aabb_scale(train_mats)
    print(f"Final aabb_scale (after transformations): {final_aabb_scale}")

    # Calculate scene center for test poses (after transformations, using training data)
    centers = np.array([[m[0][3], m[1][3], m[2][3]] for m in train_mats])
    scene_center = centers.mean(axis=0)

    # Determine image resolution using first available image
    w = h = None
    for idx in range(n_train_mats * 2):
        fname = f"{args.image_prefix}{idx}.{args.image_ext}"
        fpath = os.path.join(args.image_folder, fname)
        if os.path.isfile(fpath):
            with Image.open(fpath) as im:
                w, h = im.size
            break
    if w is None or h is None:
        raise FileNotFoundError("No images found matching prefix and extension in image_folder")

    # Compute intrinsics
    cax = math.radians(args.camera_angle_x)
    cay = math.radians(args.camera_angle_y)
    fl_x = 0.5 * w / math.tan(cax / 2)
    fl_y = 0.5 * h / math.tan(cay / 2)
    cx = args.cx if args.cx is not None else w / 2.0
    cy = args.cy if args.cy is not None else h / 2.0

    # Build main JSON structure (training)
    out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, final_aabb_scale)

    # Populate training frames: sequentially assign next available image (excluding first)
    next_idx = 1  # Start from index 1 to skip first image
    for mat in train_mats:
        found = False
        while next_idx < n_mats * 2:
            fname = f"{args.image_prefix}{next_idx}.{args.image_ext}"
            fpath = os.path.join(args.image_folder, fname)
            next_idx += 1
            if os.path.isfile(fpath):
                sharp = compute_sharpness(fpath)
                out["frames"].append({
                    "file_path": fpath,
                    "sharpness": sharp,
                    "transform_matrix": mat
                })
                found = True
                break
        if not found:
            raise FileNotFoundError(f"No image found for pose, checked up to index {next_idx-1}.")

    # Write main training JSON
    with open(args.output, 'w') as of:
        json.dump(out, of, indent=2)
    print(f"Wrote {args.output} with {n_train_mats} training frames (size {w}x{h}).")

    # Generate validation JSON with the first image/matrix
    val_out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, final_aabb_scale)
    
    # Find the first available image for validation
    val_found = False
    for idx in range(n_mats * 2):  # Search from beginning
        fname = f"{args.image_prefix}{idx}.{args.image_ext}"
        fpath = os.path.join(args.image_folder, fname)
        if os.path.isfile(fpath):
            sharp = compute_sharpness(fpath)
            val_out["frames"].append({
                "file_path": fpath,
                "sharpness": sharp,
                "transform_matrix": val_mat
            })
            val_found = True
            break
    
    if not val_found:
        raise FileNotFoundError("No image found for validation.")
    
    val_filename = args.output.replace('.json', '_val.json')
    with open(val_filename, 'w') as of:
        json.dump(val_out, of, indent=2)
    print(f"Wrote {val_filename} with 1 validation frame.")

    # Generate test JSON
    test_out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, final_aabb_scale)
    test_poses = generate_test_poses(scene_center, radius=50, n_poses=8)
    
    for i, pose in enumerate(test_poses):
        test_out["frames"].append({
            "file_path": f"./Test{i}.jpg",
            "transform_matrix_start": pose
        })
    
    test_filename = args.output.replace('.json', '_test.json')
    with open(test_filename, 'w') as of:
        json.dump(test_out, of, indent=2)
    print(f"Wrote {test_filename} with {len(test_poses)} test frames.")

    # Generate video test JSON
    video_out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, final_aabb_scale)
    video_poses = generate_video_poses(scene_center, radius=40, n_poses=60)
    
    for i, pose in enumerate(video_poses):
        video_out["frames"].append({
            "transform_matrix": pose
        })
    
    video_filename = args.output.replace('.json', '_test_video.json')
    with open(video_filename, 'w') as of:
        json.dump(video_out, of, indent=2)
    print(f"Wrote {video_filename} with {len(video_poses)} video frames.")


if __name__ == '__main__':
    main()
