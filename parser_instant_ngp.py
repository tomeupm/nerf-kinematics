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
    parser.add_argument('--aabb_scale', type=float, default=4.0, help='AABB scale for scene bounds')
    parser.add_argument('--recenter', action='store_true', help='Recenter cameras to scene origin')
    parser.add_argument('--scale_trans', type=float, default=1.0, help='Uniform scale for camera translations')
    parser.add_argument('--output', default='transforms_converted.json', help='Output JSON name')
    args = parser.parse_args()

    # Parse transform matrices
    mats = parse_poses_file(args.poses)
    n_mats = len(mats)

    # Optional: recenter and scale translations
    if args.recenter or args.scale_trans != 1.0:
        centers = np.array([[m[0][3], m[1][3], m[2][3]] for m in mats])
        centroid = centers.mean(axis=0) if args.recenter else np.zeros(3)
        for m in mats:
            m[0][3] = (m[0][3] - centroid[0]) / args.scale_trans
            m[1][3] = (m[1][3] - centroid[1]) / args.scale_trans
            m[2][3] = (m[2][3] - centroid[2]) / args.scale_trans

    # Calculate scene center for test poses
    centers = np.array([[m[0][3], m[1][3], m[2][3]] for m in mats])
    scene_center = centers.mean(axis=0)

    # Determine image resolution using first available image
    w = h = None
    for idx in range(n_mats * 2):
        fname = f"{args.image_prefix}{idx}.{args.image_ext}"
        fpath = os.path.join(args.image_folder, fname)
        if os.path.isfile(fpath):
            with Image.open(fpath) as im:
                w, h = im.size
            break
    if w is None:
        raise FileNotFoundError("No images found matching prefix and extension in image_folder")

    # Compute intrinsics
    cax = math.radians(args.camera_angle_x)
    cay = math.radians(args.camera_angle_y)
    fl_x = 0.5 * w / math.tan(cax / 2)
    fl_y = 0.5 * h / math.tan(cay / 2)
    cx = args.cx if args.cx is not None else w / 2.0
    cy = args.cy if args.cy is not None else h / 2.0

    # Build main JSON structure
    out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, args.aabb_scale)

    # Populate frames: sequentially assign next available image
    next_idx = 0
    for mat in mats:
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

    # Write main JSON
    with open(args.output, 'w') as of:
        json.dump(out, of, indent=2)
    print(f"Wrote {args.output} with {n_mats} frames (size {w}x{h}).")

    # Generate test JSON
    test_out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, args.aabb_scale)
    test_poses = generate_test_poses(scene_center, radius=50, n_poses=8)
    
    for i, pose in enumerate(test_poses):
        test_out["frames"].append({
            "file_path": f"./Test{i}.jpg",
            "sharpness": 170.0,
            "transform_matrix": pose
        })
    
    test_filename = args.output.replace('.json', '_test.json')
    with open(test_filename, 'w') as of:
        json.dump(test_out, of, indent=2)
    print(f"Wrote {test_filename} with {len(test_poses)} test frames.")

    # Generate video test JSON
    video_out = create_base_json(w, h, cax, cay, fl_x, fl_y, cx, cy, args.k1, args.k2, args.p1, args.p2, args.aabb_scale)
    video_poses = generate_video_poses(scene_center, radius=40, n_poses=60)
    
    for i, pose in enumerate(video_poses):
        video_out["frames"].append({
            "file_path": f"./Video{i:03d}.jpg",
            "sharpness": 170.0,
            "transform_matrix": pose
        })
    
    video_filename = args.output.replace('.json', '_test_video.json')
    with open(video_filename, 'w') as of:
        json.dump(video_out, of, indent=2)
    print(f"Wrote {video_filename} with {len(video_poses)} video frames.")


if __name__ == '__main__':
    main()
