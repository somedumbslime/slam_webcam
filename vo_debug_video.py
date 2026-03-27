from pathlib import Path
import argparse

import cv2
import numpy as np

import match_and_pose as mp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a debug video for monocular VO with matches/inliers and trajectory."
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
        "--fps",
        type=float,
        default=5.0,
        help="Output video FPS (default: 5.0).",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=120,
        help="Maximum number of matches drawn per step (default: 120).",
    )
    parser.add_argument(
        "--traj-panel-size",
        type=int,
        default=540,
        help="Trajectory panel size in pixels (default: 540).",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="vo_debug.mp4",
        help="Output video filename inside out-dir (default: vo_debug.mp4).",
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


def render_trajectory_panel(entries, panel_size):
    panel_size = max(320, int(panel_size))
    panel = np.full((panel_size, panel_size, 3), 245, dtype=np.uint8)
    margin = int(panel_size * 0.10)

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

    draw_w = panel_size - 2 * margin
    draw_h = panel_size - 2 * margin

    def to_px(xx, zz):
        u = int(margin + (xx - x_min) / (x_max - x_min) * draw_w)
        v = int(panel_size - margin - (zz - z_min) / (z_max - z_min) * draw_h)
        return u, v

    pts = [to_px(e["pos"][0], e["pos"][2]) for e in entries]
    for i in range(1, len(pts)):
        color = (0, 150, 0) if entries[i]["accepted"] else (150, 150, 150)
        cv2.line(panel, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    cv2.circle(panel, pts[0], 5, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(panel, pts[-1], 6, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(
        panel,
        "Trajectory X-Z (arb. scale)",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    # Axes hints
    cv2.line(
        panel,
        (margin, panel_size - margin),
        (margin + 80, panel_size - margin),
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        panel,
        (margin, panel_size - margin),
        (margin, panel_size - margin - 80),
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "+X",
        (margin + 84, panel_size - margin + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "+Z",
        (margin - 6, panel_size - margin - 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    return panel


def stack_match_and_panel(match_img, panel_img):
    h = match_img.shape[0]
    scale = h / panel_img.shape[0]
    panel_w = int(panel_img.shape[1] * scale)
    panel_rs = cv2.resize(panel_img, (panel_w, h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([match_img, panel_rs])


def draw_status_text(img, lines, color=(20, 220, 20)):
    y = 28
    for line in lines:
        cv2.putText(
            img,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 28


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

    prev_color = cv2.imread(str(seq[0]), cv2.IMREAD_COLOR)
    if prev_color is None:
        raise RuntimeError(f"Cannot read image: {seq[0]}")
    prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)

    video_path = out_dir / args.video_name
    writer = None

    for i in range(len(seq) - 1):
        curr_path = seq[i + 1]
        curr_color = cv2.imread(str(curr_path), cv2.IMREAD_COLOR)
        if curr_color is None:
            skipped_steps += 1
            entries.append(
                {
                    "idx": i + 1,
                    "image": str(curr_path),
                    "matches": 0,
                    "inliers": 0,
                    "accepted": False,
                    "pos": c_w.copy(),
                    "note": "read failed",
                }
            )
            continue
        curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)

        accepted = False
        matches = 0
        inliers = 0
        note = "ok"
        draw_matches = []
        kp1 = []
        kp2 = []

        try:
            kp1, kp2, good = mp.detect_and_match(prev_gray, curr_gray, args.ratio_test)
            matches = len(good)
            pose = mp.estimate_pose(kp1, kp2, good, k, dist, args.ransac_threshold)
            inliers = int(pose["inliers_pose"])

            mask = pose["mask_pose"]
            inlier_matches = [m for idx, m in enumerate(good) if mask[idx]]

            if len(inlier_matches) > 0:
                draw_matches = sorted(inlier_matches, key=lambda m: m.distance)[: args.max_draw]
            else:
                draw_matches = sorted(good, key=lambda m: m.distance)[: args.max_draw]

            if inliers >= args.min_inliers:
                r_rel = pose["R"].astype(np.float64)
                t_rel = pose["t"].reshape(3).astype(np.float64)
                t_norm = float(np.linalg.norm(t_rel))
                if t_norm < 1e-12:
                    note = "tiny translation"
                else:
                    t_rel = t_rel / t_norm
                    c_next_in_prev = -r_rel.T @ t_rel * float(args.step_scale)
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

        if len(draw_matches) > 0:
            match_vis = cv2.drawMatches(
                prev_color,
                kp1,
                curr_color,
                kp2,
                draw_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
        else:
            match_vis = np.hstack([prev_color, curr_color])

        traj_panel = render_trajectory_panel(entries, args.traj_panel_size)
        frame = stack_match_and_panel(match_vis, traj_panel)

        status = "ACCEPTED" if accepted else "SKIPPED"
        color = (30, 220, 30) if accepted else (30, 140, 255)
        lines = [
            f"step {i} -> {i+1} | {status}",
            f"matches={matches}  inliers={inliers}  min_inliers={args.min_inliers}",
            f"note: {note}",
            f"pos=[{c_w[0]:.2f}, {c_w[1]:.2f}, {c_w[2]:.2f}] (arb)",
        ]
        draw_status_text(frame, lines, color=color)

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, float(args.fps), (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {video_path}")
        writer.write(frame)

        prev_color = curr_color
        prev_gray = curr_gray

    if writer is not None:
        writer.release()

    pts = np.array([e["pos"] for e in entries], dtype=np.float64)
    total_path = (
        float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if len(pts) > 1
        else 0.0
    )

    final_panel = render_trajectory_panel(entries, 900)
    final_traj_path = out_dir / "vo_trajectory_from_video.png"
    cv2.imwrite(str(final_traj_path), final_panel)

    csv_path = out_dir / "vo_debug_steps.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("idx,image,matches,inliers,accepted,x,y,z,note\n")
        for e in entries:
            x, y, z = e["pos"]
            f.write(
                f"{e['idx']},{e['image']},{e['matches']},{e['inliers']},"
                f"{int(e['accepted'])},{x:.6f},{y:.6f},{z:.6f},{e['note']}\n"
            )

    report_path = out_dir / "vo_debug_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("VO debug video report\n")
        f.write(f"images_used: {len(seq)}\n")
        f.write(f"accepted_steps: {accepted_steps}\n")
        f.write(f"skipped_steps: {skipped_steps}\n")
        f.write(f"total_path_length_arbitrary: {total_path:.6f}\n")
        f.write(f"final_position: {pts[-1].tolist()}\n")
        f.write(f"video: {video_path}\n")

    print("=== VO DEBUG VIDEO RESULT ===")
    print(f"images used: {len(seq)}")
    print(f"accepted steps: {accepted_steps}")
    print(f"skipped steps: {skipped_steps}")
    print(f"final position (arbitrary scale): {pts[-1]}")
    print(f"total path length (arbitrary): {total_path:.6f}")
    print(f"Saved: {video_path}")
    print(f"Saved: {final_traj_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
