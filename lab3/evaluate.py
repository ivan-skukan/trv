import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Evaluate KITTI poses with evo.")
    parser.add_argument(
        "--gt",
        default=os.path.join("dataset", "sequences", "gt", "00", "00.txt"),
        help="Path to ground truth KITTI poses file.",
    )
    parser.add_argument(
        "--est",
        default=os.path.join("results", "estimated.txt"),
        help="Path to estimated KITTI poses file.",
    )
    parser.add_argument(
        "--velodyne-dir",
        default=os.path.join("dataset", "sequences", "00", "velodyne"),
        help="Velodyne directory used to align GT frames by filename indices.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show evo plots (requires GUI).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        print(f"Ground truth file not found: {args.gt}")
        return 1
    if not os.path.exists(args.est):
        print(f"Estimated file not found: {args.est}")
        return 1

    gt_path = args.gt
    est_path = args.est

    # If velodyne frames are missing, align GT by available frame indices.
    if os.path.isdir(args.velodyne_dir):
        frame_files = sorted(
            f for f in os.listdir(args.velodyne_dir) if f.endswith(".bin")
        )
        if frame_files:
            indices = [int(os.path.splitext(f)[0]) for f in frame_files]
            with open(args.gt, "r") as f:
                gt_lines = f.readlines()
            filtered_gt = [gt_lines[i] for i in indices if i < len(gt_lines)]
            results_dir = os.path.dirname(os.path.abspath(args.est))
            gt_path = os.path.join(results_dir, "gt_aligned.txt")
            with open(gt_path, "w") as f:
                f.writelines(filtered_gt)

    evo_ape = shutil.which("evo_ape")
    if evo_ape is None:
        print("evo_ape not found. Install evo or run evaluation manually.")
        print("Example: evo_ape kitti <gt> <est> -a -r full")
        return 1

    cmd = [evo_ape, "kitti", gt_path, est_path, "-a", "-r", "full"]
    if args.plot:
        cmd.append("--plot")

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
