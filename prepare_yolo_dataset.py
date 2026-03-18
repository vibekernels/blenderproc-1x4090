"""
Prepare the beer pong dataset for YOLOv8 training.

1. Generate YOLO labels from 3D metadata (margin-top for cup rim detection)
2. Split into train/val/test sets (80/10/10)
3. Create YOLO dataset.yaml config
"""

import os
import sys
import glob
import shutil
import random
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='/workspace/beer_pong_dataset')
    parser.add_argument('--yolo-dir', default='/workspace/beer_pong_yolo')
    parser.add_argument('--margin-top', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=0.80)
    parser.add_argument('--val-ratio', type=float, default=0.10)
    args = parser.parse_args()

    # Step 1: Generate YOLO labels
    print("Generating YOLO labels...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    label_script = os.path.join(script_dir, 'make_yolo_labels.py')
    yolo_labels_dir = os.path.join(args.dataset_dir, 'yolo_labels_cup_top')
    ret = os.system(
        f'{sys.executable} {label_script} '
        f'--dataset-dir {args.dataset_dir} '
        f'--margin-top {args.margin_top} '
        f'--output-dir {yolo_labels_dir}'
    )
    if ret != 0:
        print("Failed to generate YOLO labels")
        sys.exit(1)

    # Step 2: Gather all image/label pairs
    images_dir = os.path.join(args.dataset_dir, 'images')
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    print(f"Found {len(image_files)} images")

    # Match labels
    pairs = []
    for img_path in image_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(yolo_labels_dir, f"{base}.txt")
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))

    print(f"Matched {len(pairs)} image/label pairs")

    # Step 3: Split
    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    # Step 4: Create YOLO directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.yolo_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.yolo_dir, split, 'labels'), exist_ok=True)

    def copy_pairs(pair_list, split_name):
        for img_path, lbl_path in pair_list:
            fname = os.path.basename(img_path)
            base = os.path.splitext(fname)[0]
            # Symlink instead of copy to save disk space
            img_dst = os.path.join(args.yolo_dir, split_name, 'images', fname)
            lbl_dst = os.path.join(args.yolo_dir, split_name, 'labels', f"{base}.txt")
            if not os.path.exists(img_dst):
                os.symlink(os.path.abspath(img_path), img_dst)
            if not os.path.exists(lbl_dst):
                os.symlink(os.path.abspath(lbl_path), lbl_dst)

    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs, 'val')
    copy_pairs(test_pairs, 'test')

    # Step 5: Write dataset.yaml
    yaml_path = os.path.join(args.yolo_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(args.yolo_dir)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: cup_top\n")

    print(f"\nDataset ready at: {args.yolo_dir}")
    print(f"Config: {yaml_path}")


if __name__ == '__main__':
    main()
