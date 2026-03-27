from pathlib import Path
import argparse

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate relative camera pose from two images using ORB + Essential matrix."
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path to first image. If omitted, uses one of the latest images.",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Path to second image. If omitted, uses the latest image.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Images directory for auto-selection (default: images).",
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
        help="Lowe ratio threshold (default: 0.75).",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold in pixels for findEssentialMat (default: 1.0).",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=120,
        help="Max number of matches to draw (default: 120).",
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
        raise RuntimeError("Need at least two images for pose estimation.")

    # Use the latest two images when paths are not explicitly provided.
    return paths[-2], paths[-1]


def load_calibration(calib_path: Path):
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path.resolve()}")

    data = np.load(calib_path, allow_pickle=True)
    if "camera_matrix" not in data:
        raise KeyError("camera_matrix not found in calibration file.")

    k = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64) if "dist_coeffs" in data else None
    return k, dist


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def detect_and_match(gray1, gray2, ratio_test):
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        raise RuntimeError("Not enough ORB features found in one of the images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError(f"Too few good matches: {len(good)} (need >= 8).")

    return kp1, kp2, good


def undistort_points(pts, k, dist):
    if dist is None:
        return pts
    und = cv2.undistortPoints(pts.reshape(-1, 1, 2), k, dist, P=k)
    return und.reshape(-1, 2)


def estimate_pose(kp1, kp2, good_matches, k, dist, ransac_threshold):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    pts1_und = undistort_points(pts1, k, dist)
    pts2_und = undistort_points(pts2, k, dist)

    e, mask_e = cv2.findEssentialMat(
        pts1_und,
        pts2_und,
        k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )

    if e is None or mask_e is None:
        raise RuntimeError("findEssentialMat failed.")

    if e.shape != (3, 3):
        # When multiple solutions are returned, use the first 3x3 block.
        e = e[:3, :3]

    inliers_e = int(mask_e.ravel().sum())
    if inliers_e < 8:
        raise RuntimeError(f"Too few E inliers: {inliers_e} (need >= 8).")

    _, r, t, mask_pose = cv2.recoverPose(e, pts1_und, pts2_und, k, mask=mask_e)
    inliers_pose = int(mask_pose.ravel().sum())

    return {
        "E": e,
        "R": r,
        "t": t,
        "mask_e": mask_e.ravel().astype(bool),
        "mask_pose": mask_pose.ravel().astype(bool),
        "inliers_e": inliers_e,
        "inliers_pose": inliers_pose,
        "pts1": pts1,
        "pts2": pts2,
    }


def save_visualizations(img1, img2, kp1, kp2, good_matches, pose_result, out_dir, max_draw):
    out_dir.mkdir(parents=True, exist_ok=True)

    good_sorted = sorted(good_matches, key=lambda m: m.distance)
    draw_all = good_sorted[:max_draw]

    vis_all = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        draw_all,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_dir / "matches_all.jpg"), vis_all)

    inlier_matches = [m for i, m in enumerate(good_matches) if pose_result["mask_pose"][i]]
    inlier_matches = sorted(inlier_matches, key=lambda m: m.distance)[:max_draw]
    vis_inliers = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(out_dir / "matches_inliers.jpg"), vis_inliers)

    return len(inlier_matches)


def save_pose_report(out_dir, img1_path, img2_path, k, pose_result):
    t = pose_result["t"].reshape(-1)
    t_norm = float(np.linalg.norm(t))
    if t_norm > 0:
        t_dir = t / t_norm
    else:
        t_dir = t

    report_path = out_dir / "pose_result.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Relative pose estimation result\n")
        f.write(f"image1: {img1_path}\n")
        f.write(f"image2: {img2_path}\n")
        f.write(f"good_matches: {len(pose_result['pts1'])}\n")
        f.write(f"E_inliers: {pose_result['inliers_e']}\n")
        f.write(f"pose_inliers: {pose_result['inliers_pose']}\n\n")
        f.write("Camera matrix K:\n")
        f.write(np.array2string(k, precision=6))
        f.write("\n\nEssential matrix E:\n")
        f.write(np.array2string(pose_result["E"], precision=6))
        f.write("\n\nRotation R:\n")
        f.write(np.array2string(pose_result["R"], precision=6))
        f.write("\n\nTranslation t (unit direction, scale unknown):\n")
        f.write(np.array2string(t_dir.reshape(3, 1), precision=6))
        f.write("\n")

    np.savez(
        out_dir / "pose_result.npz",
        E=pose_result["E"],
        R=pose_result["R"],
        t=pose_result["t"],
        t_unit=t_dir.reshape(3, 1),
        inliers_e=pose_result["inliers_e"],
        inliers_pose=pose_result["inliers_pose"],
    )


def main():
    args = parse_args()

    img1_path, img2_path = select_image_pair(args)
    calib_path = Path(args.calib)
    out_dir = Path(args.out_dir)

    k, dist = load_calibration(calib_path)
    img1, gray1 = load_image(img1_path)
    img2, gray2 = load_image(img2_path)

    kp1, kp2, good_matches = detect_and_match(gray1, gray2, args.ratio_test)
    pose_result = estimate_pose(
        kp1, kp2, good_matches, k, dist, args.ransac_threshold
    )

    drawn_inliers = save_visualizations(
        img1, img2, kp1, kp2, good_matches, pose_result, out_dir, args.max_draw
    )
    save_pose_report(out_dir, img1_path, img2_path, k, pose_result)

    print("=== POSE ESTIMATION RESULT ===")
    print(f"image1: {img1_path}")
    print(f"image2: {img2_path}")
    print(f"good matches: {len(good_matches)}")
    print(f"E inliers: {pose_result['inliers_e']}")
    print(f"pose inliers: {pose_result['inliers_pose']}")
    print("Rotation R:")
    print(pose_result["R"])
    print("Translation direction t (scale unknown):")
    t = pose_result["t"].reshape(-1)
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        print((t / t_norm).reshape(3, 1))
    else:
        print(pose_result["t"])
    print(f"Saved: {out_dir / 'matches_all.jpg'}")
    print(f"Saved: {out_dir / 'matches_inliers.jpg'} (drawn {drawn_inliers} inliers)")
    print(f"Saved: {out_dir / 'pose_result.txt'}")
    print(f"Saved: {out_dir / 'pose_result.npz'}")


if __name__ == "__main__":
    main()
