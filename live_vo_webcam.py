from pathlib import Path
from datetime import datetime
import argparse
import time

import cv2
import numpy as np

import match_and_pose as mp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live monocular VO from webcam with trajectory visualization."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
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
        help="Output directory for logs/results (default: outputs).",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested width (default: 640).")
    parser.add_argument(
        "--height", type=int, default=480, help="Requested height (default: 480)."
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.75,
        help="Lowe ratio threshold for ORB matching (default: 0.75).",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold in pixels for Essential matrix (default: 1.0).",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=20,
        help="Minimum inliers to accept VO step (default: 20).",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Translation scale for each accepted step (default: 1.0).",
    )
    parser.add_argument(
        "--traj-panel-size",
        type=int,
        default=460,
        help="Trajectory panel size in px (default: 460).",
    )
    parser.add_argument(
        "--traj-ppu",
        type=float,
        default=65.0,
        help="Initial trajectory scale in pixels-per-unit (default: 65.0).",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="Draw inlier matches between previous/current frame.",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=120,
        help="Max matches to draw when --show-matches is enabled (default: 120).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save displayed stream to outputs/live_vo_*.mp4.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=15.0,
        help="Output video fps when --save-video is set (default: 15.0).",
    )
    return parser.parse_args()


def choose_grid_step_units(ppu):
    target_px = 55.0
    raw = target_px / max(ppu, 1e-6)
    steps = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for s in steps:
        if s >= raw:
            return s
    return steps[-1]


def update_view_state(entries, size, view_state):
    size = max(320, int(size))
    margin = int(size * 0.12)
    radius_px = max(10, size // 2 - margin)

    xyz = np.array([e["pos"] for e in entries], dtype=np.float64)
    max_abs = max(
        1.0,
        float(np.max(np.abs(xyz[:, 0]))),
        float(np.max(np.abs(xyz[:, 2]))),
    )
    needed_ppu = radius_px / (max_abs * 1.15)
    needed_ppu = max(6.0, needed_ppu)

    # Auto-zoom out only. This keeps the panel stable and avoids jitter.
    if needed_ppu < view_state["ppu"]:
        view_state["ppu"] = needed_ppu


def render_traj_panel(entries, size, view_state):
    size = max(320, int(size))
    panel = np.full((size, size, 3), 245, dtype=np.uint8)
    margin = int(size * 0.08)
    c = size // 2
    ppu = float(view_state["ppu"])

    def to_px(xx, zz):
        u = int(c + xx * ppu)
        v = int(c - zz * ppu)
        return u, v

    # Grid
    grid_step = choose_grid_step_units(ppu)
    half_units = (size // 2 - margin) / max(ppu, 1e-6)
    tick_vals = np.arange(-half_units, half_units + grid_step, grid_step)
    for t in tick_vals:
        u, _ = to_px(t, 0.0)
        _, v = to_px(0.0, t)
        if margin <= u < size - margin:
            color = (220, 220, 220) if abs(t) > 1e-9 else (120, 120, 120)
            cv2.line(panel, (u, margin), (u, size - margin), color, 1, cv2.LINE_AA)
        if margin <= v < size - margin:
            color = (220, 220, 220) if abs(t) > 1e-9 else (120, 120, 120)
            cv2.line(panel, (margin, v), (size - margin, v), color, 1, cv2.LINE_AA)

    pts = [to_px(e["pos"][0], e["pos"][2]) for e in entries]
    for i in range(1, len(pts)):
        color = (0, 150, 0) if entries[i]["accepted"] else (160, 160, 160)
        cv2.line(panel, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

    cv2.circle(panel, pts[0], 5, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(panel, pts[-1], 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "Live VO trajectory (X-Z)",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"scale: {ppu:.1f} px/u",
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "+X",
        (size - margin - 28, c - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "+Z",
        (c + 8, margin + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    return panel


def stack_with_panel(left_img, panel):
    h = left_img.shape[0]
    scale = h / panel.shape[0]
    panel_w = int(panel.shape[1] * scale)
    panel_rs = cv2.resize(panel, (panel_w, h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([left_img, panel_rs])


def draw_lines(img, lines, color=(20, 220, 20), start_y=26):
    y = start_y
    for line in lines:
        cv2.putText(
            img,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 26


def save_results(out_dir: Path, run_name: str, entries):
    xyz = np.array([e["pos"] for e in entries], dtype=np.float64)
    accepted = np.array([e["accepted"] for e in entries], dtype=bool)
    timestamps = np.array([e["timestamp"] for e in entries], dtype=np.float64)

    csv_path = out_dir / f"{run_name}_trajectory.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("idx,timestamp,matches,inliers,accepted,x,y,z,note\n")
        for e in entries:
            x, y, z = e["pos"]
            f.write(
                f"{e['idx']},{e['timestamp']:.6f},{e['matches']},{e['inliers']},"
                f"{int(e['accepted'])},{x:.6f},{y:.6f},{z:.6f},{e['note']}\n"
            )

    npz_path = out_dir / f"{run_name}_trajectory.npz"
    np.savez(
        npz_path,
        trajectory_xyz=xyz,
        accepted=accepted,
        timestamps=timestamps,
    )

    total_path = (
        float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
        if len(xyz) > 1
        else 0.0
    )
    accepted_steps = int(np.sum(accepted[1:])) if len(accepted) > 1 else 0
    skipped_steps = max(0, len(entries) - 1 - accepted_steps)

    report_path = out_dir / f"{run_name}_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Live VO webcam report\n")
        f.write(f"frames_processed: {len(entries)}\n")
        f.write(f"accepted_steps: {accepted_steps}\n")
        f.write(f"skipped_steps: {skipped_steps}\n")
        f.write(f"total_path_length_arbitrary: {total_path:.6f}\n")
        f.write(f"final_position: {xyz[-1].tolist()}\n")

    return csv_path, npz_path, report_path, accepted_steps, skipped_steps, total_path, xyz[-1]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k, dist = mp.load_calibration(Path(args.calib))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Live VO started.")
    print("Controls: q=quit, r=reset trajectory")
    print(f"Camera resolution: {actual_w}x{actual_h}")

    run_name = "live_vo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    video_writer = None
    video_path = out_dir / f"{run_name}.mp4"

    r_wc = np.eye(3, dtype=np.float64)
    c_w = np.zeros(3, dtype=np.float64)
    view_state = {"ppu": max(6.0, float(args.traj_ppu))}
    t0 = time.perf_counter()
    idx = 0

    entries = [
        {
            "idx": 0,
            "timestamp": 0.0,
            "matches": 0,
            "inliers": 0,
            "accepted": True,
            "pos": c_w.copy(),
            "note": "start",
        }
    ]

    prev_gray = None
    prev_color = None
    status = "waiting first frame"
    matches = 0
    inliers = 0
    draw_matches = []
    kp1 = []
    kp2 = []

    try:
        while True:
            ok, curr_color = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)
            ts = time.perf_counter() - t0

            accepted = False
            note = "ok"
            draw_matches = []
            kp1 = []
            kp2 = []

            if prev_gray is not None:
                try:
                    kp1, kp2, good = mp.detect_and_match(prev_gray, curr_gray, args.ratio_test)
                    matches = len(good)
                    pose = mp.estimate_pose(
                        kp1, kp2, good, k, dist, args.ransac_threshold
                    )
                    inliers = int(pose["inliers_pose"])
                    mask = pose["mask_pose"]
                    inlier_matches = [m for i, m in enumerate(good) if mask[i]]
                    draw_matches = sorted(inlier_matches, key=lambda m: m.distance)[: args.max_draw]

                    if inliers >= args.min_inliers:
                        r_rel = pose["R"].astype(np.float64)
                        t_rel = pose["t"].reshape(3).astype(np.float64)
                        t_norm = float(np.linalg.norm(t_rel))
                        if t_norm < 1e-12:
                            note = "tiny translation"
                            status = "SKIP tiny translation"
                        else:
                            t_rel = t_rel / t_norm
                            c_next_in_prev = -r_rel.T @ t_rel * float(args.step_scale)
                            c_w = c_w + r_wc @ c_next_in_prev
                            r_wc = r_wc @ r_rel.T
                            accepted = True
                            status = "ACCEPTED"
                    else:
                        note = f"low inliers ({inliers})"
                        status = "SKIP low inliers"
                except Exception as exc:
                    matches = 0
                    inliers = 0
                    note = f"failed: {exc}"
                    status = "SKIP failed"

                idx += 1
                entries.append(
                    {
                        "idx": idx,
                        "timestamp": ts,
                        "matches": matches,
                        "inliers": inliers,
                        "accepted": accepted,
                        "pos": c_w.copy(),
                        "note": note,
                    }
                )

            update_view_state(entries, args.traj_panel_size, view_state)
            panel = render_traj_panel(entries, args.traj_panel_size, view_state)

            if args.show_matches and prev_color is not None and len(draw_matches) > 0:
                left = cv2.drawMatches(
                    prev_color,
                    kp1,
                    curr_color,
                    kp2,
                    draw_matches,
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
            else:
                left = curr_color.copy()

            frame = stack_with_panel(left, panel)
            color = (30, 220, 30) if "ACCEPTED" in status else (30, 140, 255)
            info = [
                f"{status}  matches={matches} inliers={inliers} min={args.min_inliers}",
                f"pos=[{c_w[0]:.2f}, {c_w[1]:.2f}, {c_w[2]:.2f}]  scale=arb",
                f"frames={len(entries)}  q=quit r=reset",
            ]
            draw_lines(frame, info, color=color)
            cv2.imshow("Live VO Webcam", frame)

            if args.save_video:
                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, float(args.video_fps), (w, h)
                    )
                video_writer.write(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                r_wc = np.eye(3, dtype=np.float64)
                c_w = np.zeros(3, dtype=np.float64)
                idx = 0
                entries = [
                    {
                        "idx": 0,
                        "timestamp": ts,
                        "matches": 0,
                        "inliers": 0,
                        "accepted": True,
                        "pos": c_w.copy(),
                        "note": "reset",
                    }
                ]
                status = "RESET"
                matches = 0
                inliers = 0
                view_state = {"ppu": max(6.0, float(args.traj_ppu))}

            prev_gray = curr_gray
            prev_color = curr_color
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()

    (
        csv_path,
        npz_path,
        report_path,
        accepted_steps,
        skipped_steps,
        total_path,
        final_pos,
    ) = save_results(out_dir, run_name, entries)

    print("=== LIVE VO RESULT ===")
    print(f"frames processed: {len(entries)}")
    print(f"accepted steps: {accepted_steps}")
    print(f"skipped steps: {skipped_steps}")
    print(f"final position (arbitrary scale): {final_pos}")
    print(f"total path length (arbitrary): {total_path:.6f}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {report_path}")
    if args.save_video:
        print(f"Saved: {video_path}")


if __name__ == "__main__":
    main()
