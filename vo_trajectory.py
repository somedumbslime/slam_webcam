from pathlib import Path
import argparse

import cv2
import numpy as np

import match_and_pose as mp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mini monocular VO: estimate and accumulate camera trajectory from image sequence."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory with sequential images (default: images).",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="outputs/camera_calibration.npz",
        help="Calibration file path (default: outputs/camera_calibration.npz).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory (default: outputs).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in sorted image list (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive). -1 means full sequence.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame step in sequence (default: 1).",
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.75,
        help="Lowe ratio for ORB matching (default: 0.75).",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold for Essential matrix in pixels (default: 1.0).",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=20,
        help="Minimum pose inliers required to accept step (default: 20).",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Scale factor for each accepted translation step (default: 1.0).",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=900,
        help="Trajectory image size in pixels (default: 900).",
    )
    return parser.parse_args()


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def select_sequence(args):
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

    paths = list_images(images_dir)
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 images for VO.")

    start = max(0, args.start)
    end = len(paths) if args.end == -1 else min(len(paths), args.end)
    step = max(1, args.step)

    seq = paths[start:end:step]
    if len(seq) < 2:
        raise RuntimeError(
            f"Selected sequence too short: {len(seq)} image(s). "
            "Adjust --start/--end/--step."
        )
    return seq


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return img


def draw_trajectory(entries, out_path: Path, canvas_size: int):
    canvas_size = max(300, int(canvas_size))
    margin = int(canvas_size * 0.08)
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    xyz = np.array([e["pos"] for e in entries], dtype=np.float64)
    x = xyz[:, 0]
    z = xyz[:, 2]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    if abs(x_max - x_min) < 1e-9:
        x_min -= 1.0
        x_max += 1.0
    if abs(z_max - z_min) < 1e-9:
        z_min -= 1.0
        z_max += 1.0

    draw_w = canvas_size - 2 * margin
    draw_h = canvas_size - 2 * margin

    def to_px(xx, zz):
        u = int(margin + (xx - x_min) / (x_max - x_min) * draw_w)
        # Flip Z so "forward" tends to go up on the image.
        v = int(canvas_size - margin - (zz - z_min) / (z_max - z_min) * draw_h)
        return u, v

    pts = [to_px(e["pos"][0], e["pos"][2]) for e in entries]

    for i in range(1, len(pts)):
        prev_ok = entries[i]["accepted"]
        color = (0, 170, 0) if prev_ok else (180, 180, 180)
        cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    cv2.circle(canvas, pts[0], 6, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "START",
        (pts[0][0] + 8, pts[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.circle(canvas, pts[-1], 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "END",
        (pts[-1][0] + 8, pts[-1][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Mini VO trajectory (X-Z plane, arbitrary scale)",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), canvas)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = select_sequence(args)
    k, dist = mp.load_calibration(Path(args.calib))

    # Pose as camera orientation in world (R_wc) and camera center in world (C_w).
    r_wc = np.eye(3, dtype=np.float64)
    c_w = np.zeros(3, dtype=np.float64)

    entries = [
        {
            "idx": 0,
            "image": str(seq[0]),
            "matches": 0,
            "inliers": 0,
            "accepted": True,
            "pos": c_w.copy(),
            "note": "start",
        }
    ]

    accepted_steps = 0
    skipped_steps = 0

    prev_gray = load_gray(seq[0])

    for i in range(len(seq) - 1):
        curr_path = seq[i + 1]
        curr_gray = load_gray(curr_path)

        accepted = False
        matches = 0
        inliers = 0
        note = "ok"

        try:
            kp1, kp2, good = mp.detect_and_match(prev_gray, curr_gray, args.ratio_test)
            matches = len(good)
            pose = mp.estimate_pose(
                kp1, kp2, good, k, dist, args.ransac_threshold
            )
            inliers = int(pose["inliers_pose"])

            if inliers >= args.min_inliers:
                r_rel = pose["R"].astype(np.float64)
                t_rel = pose["t"].reshape(3).astype(np.float64)
                t_norm = float(np.linalg.norm(t_rel))
                if t_norm < 1e-12:
                    note = "tiny translation"
                else:
                    t_rel = t_rel / t_norm
                    # Camera center of frame (i+1) in frame-i coordinates.
                    c_next_in_prev = -r_rel.T @ t_rel * float(args.step_scale)
                    # Convert motion to world and accumulate.
                    c_w = c_w + r_wc @ c_next_in_prev
                    r_wc = r_wc @ r_rel.T
                    accepted = True
                    accepted_steps += 1
            else:
                note = f"low inliers ({inliers})"
                skipped_steps += 1
        except Exception as exc:
            note = f"failed: {exc}"
            skipped_steps += 1

        entries.append(
            {
                "idx": i + 1,
                "image": str(curr_path),
                "matches": matches,
                "inliers": inliers,
                "accepted": accepted,
                "pos": c_w.copy(),
                "note": note,
            }
        )
        prev_gray = curr_gray

    traj_img_path = out_dir / "vo_trajectory.png"
    draw_trajectory(entries, traj_img_path, args.canvas_size)

    csv_path = out_dir / "vo_trajectory.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("idx,image,matches,inliers,accepted,x,y,z,note\n")
        for e in entries:
            x, y, z = e["pos"]
            f.write(
                f"{e['idx']},{e['image']},{e['matches']},{e['inliers']},"
                f"{int(e['accepted'])},{x:.6f},{y:.6f},{z:.6f},{e['note']}\n"
            )

    pts = np.array([e["pos"] for e in entries], dtype=np.float64)
    np.savez(
        out_dir / "vo_trajectory.npz",
        trajectory_xyz=pts,
        accepted=np.array([e["accepted"] for e in entries], dtype=bool),
        images=np.array([e["image"] for e in entries], dtype=object),
    )

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else np.array([])
    total_path = float(np.sum(dists)) if len(dists) > 0 else 0.0

    report_path = out_dir / "vo_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Mini VO report\n")
        f.write(f"images_used: {len(seq)}\n")
        f.write(f"accepted_steps: {accepted_steps}\n")
        f.write(f"skipped_steps: {skipped_steps}\n")
        f.write(f"min_inliers: {args.min_inliers}\n")
        f.write(f"step_scale: {args.step_scale}\n")
        f.write(f"total_path_length_arbitrary: {total_path:.6f}\n")
        f.write(f"final_position: {pts[-1].tolist()}\n")

    print("=== MINI VO RESULT ===")
    print(f"images used: {len(seq)}")
    print(f"accepted steps: {accepted_steps}")
    print(f"skipped steps: {skipped_steps}")
    print(f"final position (arbitrary scale): {pts[-1]}")
    print(f"total path length (arbitrary): {total_path:.6f}")
    print(f"Saved: {traj_img_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'vo_trajectory.npz'}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
