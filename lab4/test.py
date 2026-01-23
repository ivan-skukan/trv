import os
import sys
import argparse
from math import floor
import numpy as np
import cv2

#!/usr/bin/env python3
# Visualize a few stereo pairs from KITTI odometry sequences
# Save as test.py and run from the repository root that contains ./dataset/sequences/00

import matplotlib.pyplot as plt

def find_stereo_pair_dirs(seq_path):
    # prefer color pair image_2/image_3, else image_0/image_1
    prefer = [("image_2", "image_3"), ("image_0", "image_1")]
    for a, b in prefer:
        a_p = os.path.join(seq_path, a)
        b_p = os.path.join(seq_path, b)
        if os.path.isdir(a_p) and os.path.isdir(b_p):
            return a_p, b_p
    raise FileNotFoundError(f"No stereo image pairs found in {seq_path}")

def list_images(dirpath):
    imgs = sorted([f for f in os.listdir(dirpath) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    return [os.path.join(dirpath, f) for f in imgs]

def read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if len(im.shape) == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def sample_indices(n_imgs, n_samples):
    if n_samples <= 0:
        return []
    if n_samples >= n_imgs:
        return list(range(n_imgs))
    return [int(floor(i)) for i in np.linspace(0, n_imgs - 1, n_samples)]

def visualize_pairs(left_list, right_list, indices, figsize=(12, 3)):
    n = len(indices)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(figsize[0], figsize[1]*n))
    if n == 1:
        axes = np.array([axes])
    for row, idx in enumerate(indices):
        li = left_list[idx]
        ri = right_list[idx]
        L = read_rgb(li)
        R = read_rgb(ri)
        axl = axes[row, 0]
        axr = axes[row, 1]
        axl.imshow(L); axl.axis("off"); axl.set_title(os.path.basename(li))
        axr.imshow(R); axr.axis("off"); axr.set_title(os.path.basename(ri))
    plt.tight_layout()
    plt.show()

def main():
    p = argparse.ArgumentParser(description="Visualize KITTI stereo pairs")
    p.add_argument("--seq", default="lab4/dataset/sequences/00", help="sequence folder (default ./dataset/sequences/00)")
    p.add_argument("--n", type=int, default=5, help="number of pairs to display")
    p.add_argument("--indices", nargs="*", type=int, default=None, help="explicit indices to show (overrides --n)")
    args = p.parse_args()

    if not os.path.isdir(args.seq):
        print("Sequence path not found:", args.seq, file=sys.stderr)
        sys.exit(1)

    left_dir, right_dir = find_stereo_pair_dirs(args.seq)
    left_list = list_images(left_dir)
    right_list = list_images(right_dir)
    if len(left_list) == 0 or len(right_list) == 0:
        print("No images found in:", left_dir, right_dir, file=sys.stderr)
        sys.exit(1)
    if len(left_list) != len(right_list):
        m = min(len(left_list), len(right_list))
        left_list = left_list[:m]; right_list = right_list[:m]

    if args.indices:
        indices = [i for i in args.indices if 0 <= i < len(left_list)]
        if not indices:
            print("No valid indices provided.", file=sys.stderr)
            sys.exit(1)
    else:
        indices = sample_indices(len(left_list), args.n)

    visualize_pairs(left_list, right_list, indices)

if __name__ == "__main__":
    main()