"""
Convert 3D cup metadata to YOLO-format bounding box labels.

For each JSON metadata file, projects the 3D bounding box corners of each cup
into 2D image coordinates using the camera intrinsics and extrinsics, then
computes the tightest axis-aligned 2D bounding box.

Output: YOLO format txt files (class x_center y_center width height), normalized.

Usage:
    python make_yolo_labels.py --dataset-dir /workspace/beer_pong_dataset
    python make_yolo_labels.py --dataset-dir /workspace/beer_pong_dataset --class-name cup_top --margin-top 0.6

Options:
    --margin-top: Fraction of bbox height to crop from bottom (0.6 = keep top 40%
                  of the cup bounding box, roughly the rim). Default 0 (full cup).
"""

import argparse
import json
import os
import glob
import numpy as np


def project_3d_to_2d(points_3d, K, cam_extrinsics):
    """
    Project 3D world points to 2D image coordinates.

    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        K: 3x3 camera intrinsic matrix
        cam_extrinsics: 4x4 camera-to-world matrix (camera pose in world)

    Returns:
        Nx2 array of 2D pixel coordinates
    """
    K = np.array(K)
    cam_ext = np.array(cam_extrinsics)

    # Invert to get world-to-camera transform
    world_to_cam = np.linalg.inv(cam_ext)

    # Convert points to homogeneous
    pts = np.array(points_3d)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # Nx4

    # Transform to camera space
    pts_cam = (world_to_cam @ pts_h.T).T[:, :3]  # Nx3

    # BlenderProc/Blender camera: -Z is forward, X is right, Y is up
    # OpenCV convention: Z is forward, X is right, Y is down
    # Convert: x_cv = x_bl, y_cv = -y_bl, z_cv = -z_bl
    pts_cv = pts_cam.copy()
    pts_cv[:, 1] = -pts_cam[:, 1]
    pts_cv[:, 2] = -pts_cam[:, 2]

    # Project
    pts_2d = (K @ pts_cv.T).T  # Nx3
    depths = pts_2d[:, 2]

    # Avoid division by zero
    valid = depths > 0.01
    pts_2d[valid, 0] /= depths[valid]
    pts_2d[valid, 1] /= depths[valid]
    pts_2d[~valid] = -1  # invalid points

    return pts_2d[:, :2], valid


def main():
    parser = argparse.ArgumentParser(description='Generate YOLO labels from 3D metadata')
    parser.add_argument('--dataset-dir', type=str, default='/workspace/beer_pong_dataset')
    parser.add_argument('--margin-top', type=float, default=0.0,
                        help='Fraction of bbox height to crop from bottom (for cup-top only)')
    parser.add_argument('--class-id', type=int, default=0,
                        help='YOLO class ID for cups')
    parser.add_argument('--min-visibility', type=float, default=0.3,
                        help='Min fraction of bbox corners that must be visible')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir for YOLO labels (default: dataset_dir/yolo_labels)')
    args = parser.parse_args()

    labels_dir = os.path.join(args.dataset_dir, 'labels')
    yolo_dir = args.output_dir or os.path.join(args.dataset_dir, 'yolo_labels')
    os.makedirs(yolo_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(labels_dir, '*.json')))
    print(f"Processing {len(json_files)} label files...")

    total_cups = 0
    total_visible = 0

    for json_path in json_files:
        with open(json_path) as f:
            meta = json.load(f)

        img_w = meta['image_width']
        img_h = meta['image_height']
        K = meta['camera_intrinsics']
        cam_ext = meta['camera_extrinsics_4x4']

        yolo_lines = []

        for cup in meta['cups']:
            total_cups += 1
            corners_3d = np.array(cup['bbox_3d_corners'])  # 8x3

            pts_2d, valid = project_3d_to_2d(corners_3d, K, cam_ext)

            # Check visibility
            if valid.sum() < len(valid) * args.min_visibility:
                continue

            # Get 2D bounding box from visible projected corners
            visible_pts = pts_2d[valid]
            x_min = max(0, np.min(visible_pts[:, 0]))
            x_max = min(img_w, np.max(visible_pts[:, 0]))
            y_min = max(0, np.min(visible_pts[:, 1]))
            y_max = min(img_h, np.max(visible_pts[:, 1]))

            if x_max <= x_min or y_max <= y_min:
                continue

            # Apply margin-top (crop from bottom to get cup rim only)
            if args.margin_top > 0:
                bbox_h = y_max - y_min
                y_max = y_min + bbox_h * (1.0 - args.margin_top)

            # Convert to YOLO format (normalized x_center, y_center, width, height)
            bw = (x_max - x_min) / img_w
            bh = (y_max - y_min) / img_h
            cx = (x_min + x_max) / 2.0 / img_w
            cy = (y_min + y_max) / 2.0 / img_h

            # Skip tiny boxes
            if bw < 0.005 or bh < 0.005:
                continue

            yolo_lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_visible += 1

        # Write YOLO label file
        base = os.path.splitext(os.path.basename(json_path))[0]
        with open(os.path.join(yolo_dir, f"{base}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))
            if yolo_lines:
                f.write('\n')

    print(f"Done! {total_visible}/{total_cups} cups visible across {len(json_files)} images")
    print(f"YOLO labels written to: {yolo_dir}")


if __name__ == '__main__':
    main()
