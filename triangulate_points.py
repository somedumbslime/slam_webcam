from pathlib import Path
import argparse

import cv2
import numpy as np

import match_and_pose as mp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triangulate 3D points from two images using calibrated camera."
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path to first image. If omitted, selected from images dir.",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Path to second image. If omitted, selected from images dir.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Images directory for auto-selection (default: images).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=8,
        help="When image paths are omitted, pick pair [-offset, -1] (default: 8).",
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
        "--max-reproj-error",
        type=float,
        default=3.0,
        help="Max reprojection error in pixels for keeping triangulated points (default: 3.0).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=8,
        help="Minimum final 3D points required (default: 8).",
    )
    return parser.parse_args()


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def select_image_pair(args):
    if args.image1 and args.image2:
        return Path(args.image1), Path(args.image2)

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir.resolve()}")

    paths = list_images(images_dir)
    if len(paths) < 2:
        raise RuntimeError("Need at least two images.")

    offset = max(1, int(args.offset))
    if len(paths) <= offset:
        return paths[-2], paths[-1]
    return paths[-1 - offset], paths[-1]


def project_points(p, points_3d):
    ones = np.ones((points_3d.shape[0], 1), dtype=np.float64)
    x_h = (p @ np.hstack([points_3d, ones]).T).T
    x = x_h[:, :2] / x_h[:, 2:3]
    return x


def write_ply(points_3d, colors_bgr, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, bgr in zip(points_3d, colors_bgr):
            b, g, r = [int(v) for v in bgr]
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {r} {g} {b}\n")


def sample_colors(img_bgr, pts_2d):
    h, w = img_bgr.shape[:2]
    pts_int = np.round(pts_2d).astype(int)
    pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w - 1)
    pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h - 1)
    return img_bgr[pts_int[:, 1], pts_int[:, 0]]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img1_path, img2_path = select_image_pair(args)
    k, dist = mp.load_calibration(Path(args.calib))
    img1, gray1 = mp.load_image(img1_path)
    _, gray2 = mp.load_image(img2_path)

    kp1, kp2, good_matches = mp.detect_and_match(gray1, gray2, args.ratio_test)
    pose = mp.estimate_pose(kp1, kp2, good_matches, k, dist, args.ransac_threshold)

    mask_pose = pose["mask_pose"]
    pts1 = pose["pts1"][mask_pose]
    pts2 = pose["pts2"][mask_pose]
    if len(pts1) < 8:
        raise RuntimeError(f"Too few pose inliers for triangulation: {len(pts1)}")

    pts1_und = mp.undistort_points(pts1, k, dist)
    pts2_und = mp.undistort_points(pts2, k, dist)

    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([pose["R"], pose["t"]])

    x_h = cv2.triangulatePoints(
        p1,
        p2,
        pts1_und.T.astype(np.float64),
        pts2_und.T.astype(np.float64),
    )
    points_3d = (x_h[:3] / x_h[3]).T

    # Keep only physically valid points in front of both cameras.
    z1 = points_3d[:, 2]
    points_cam2 = (pose["R"] @ points_3d.T + pose["t"]).T
    z2 = points_cam2[:, 2]
    mask_positive_depth = (z1 > 0) & (z2 > 0) & np.isfinite(points_3d).all(axis=1)

    # Reprojection filter in undistorted pixel domain.
    pts1_proj = project_points(p1, points_3d)
    pts2_proj = project_points(p2, points_3d)
    err1 = np.linalg.norm(pts1_proj - pts1_und, axis=1)
    err2 = np.linalg.norm(pts2_proj - pts2_und, axis=1)
    mask_reproj = (err1 < args.max_reproj_error) & (err2 < args.max_reproj_error)

    final_mask = mask_positive_depth & mask_reproj
    points_3d_final = points_3d[final_mask]
    pts1_final = pts1[final_mask]
    err1_final = err1[final_mask]
    err2_final = err2[final_mask]

    if len(points_3d_final) < args.min_points:
        raise RuntimeError(
            f"Too few triangulated points after filtering: {len(points_3d_final)} "
            f"(required >= {args.min_points}). "
            "Try a larger frame offset or lower --max-reproj-error."
        )

    colors = sample_colors(img1, pts1_final)
    ply_path = out_dir / "triangulated_points.ply"
    write_ply(points_3d_final, colors, ply_path)

    np.savez(
        out_dir / "triangulated_points.npz",
        points_3d=points_3d_final,
        points2d_img1=pts1_final,
        reproj_err_img1=err1_final,
        reproj_err_img2=err2_final,
        image1=str(img1_path),
        image2=str(img2_path),
        K=k,
        R=pose["R"],
        t=pose["t"],
        good_matches=len(good_matches),
        pose_inliers=pose["inliers_pose"],
        triangulated_before_filter=len(points_3d),
        triangulated_final=len(points_3d_final),
    )

    report_path = out_dir / "triangulation_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Triangulation report\n")
        f.write(f"image1: {img1_path}\n")
        f.write(f"image2: {img2_path}\n")
        f.write(f"good_matches: {len(good_matches)}\n")
        f.write(f"pose_inliers: {pose['inliers_pose']}\n")
        f.write(f"triangulated_before_filter: {len(points_3d)}\n")
        f.write(f"triangulated_final: {len(points_3d_final)}\n")
        f.write(
            f"mean_reproj_error_img1: {float(np.mean(err1_final)):.4f} px\n"
        )
        f.write(
            f"mean_reproj_error_img2: {float(np.mean(err2_final)):.4f} px\n"
        )

    print("=== TRIANGULATION RESULT ===")
    print(f"image1: {img1_path}")
    print(f"image2: {img2_path}")
    print(f"good matches: {len(good_matches)}")
    print(f"pose inliers: {pose['inliers_pose']}")
    print(f"triangulated before filter: {len(points_3d)}")
    print(f"triangulated final: {len(points_3d_final)}")
    print(f"mean reproj err img1: {float(np.mean(err1_final)):.4f}px")
    print(f"mean reproj err img2: {float(np.mean(err2_final)):.4f}px")
    print(f"Saved: {out_dir / 'triangulated_points.npz'}")
    print(f"Saved: {out_dir / 'triangulated_points.ply'}")
    print(f"Saved: {out_dir / 'triangulation_report.txt'}")


if __name__ == "__main__":
    main()
