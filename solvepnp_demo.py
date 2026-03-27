from pathlib import Path
import argparse

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate camera pose with solvePnP using a ChArUco board."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image. If omitted, best image from --images-dir is selected.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory with images (default: images).",
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
        "--aruco-dict",
        type=str,
        default="DICT_6X6_250",
        help="ArUco dictionary name (default: DICT_6X6_250).",
    )
    parser.add_argument(
        "--charuco-squares-x",
        type=int,
        default=5,
        help="Number of ChArUco squares along X (default: 5).",
    )
    parser.add_argument(
        "--charuco-squares-y",
        type=int,
        default=7,
        help="Number of ChArUco squares along Y (default: 7).",
    )
    parser.add_argument(
        "--charuco-square-length",
        type=float,
        default=12.0,
        help="ChArUco square length in mm (default: 12.0).",
    )
    parser.add_argument(
        "--charuco-marker-length",
        type=float,
        default=10.0,
        help="ChArUco marker length in mm (default: 10.0).",
    )
    parser.add_argument(
        "--min-corners",
        type=int,
        default=6,
        help="Minimum ChArUco corners required for solvePnP (default: 6).",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=20.0,
        help="Axis length for drawing pose (same units as board, default: 20.0).",
    )
    return parser.parse_args()


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def load_calibration(calib_path: Path):
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path.resolve()}")
    data = np.load(calib_path, allow_pickle=True)
    if "camera_matrix" not in data:
        raise KeyError("camera_matrix not found in calibration file.")
    k = data["camera_matrix"].astype(np.float64)
    dist = data["dist_coeffs"].astype(np.float64) if "dist_coeffs" in data else None
    return k, dist


def build_board(args):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco module is missing. Install opencv-contrib-python."
        )
    if not hasattr(cv2.aruco, args.aruco_dict):
        raise ValueError(f"Unknown ArUco dictionary: {args.aruco_dict}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    board = cv2.aruco.CharucoBoard(
        (args.charuco_squares_x, args.charuco_squares_y),
        args.charuco_square_length,
        args.charuco_marker_length,
        aruco_dict,
    )
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return board, detector


def detect_charuco(gray, board, detector):
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if charuco_ids is None or num_corners is None:
        return None

    return {
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "num_corners": int(num_corners),
    }


def find_best_image(images_dir: Path, board, detector):
    paths = list_images(images_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir.resolve()}")

    best = None
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det = detect_charuco(gray, board, detector)
        if det is None:
            continue
        if best is None or det["num_corners"] > best[1]["num_corners"]:
            best = (path, det)

    if best is None:
        raise RuntimeError("No image with detectable ChArUco corners found.")
    return best


def charuco_correspondences(board, charuco_corners, charuco_ids):
    obj_all = board.getChessboardCorners()
    ids = charuco_ids.reshape(-1).astype(int)
    obj_pts = obj_all[ids].reshape(-1, 3).astype(np.float32)
    img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    return obj_pts, img_pts


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k, dist = load_calibration(Path(args.calib))
    board, detector = build_board(args)

    if args.image:
        image_path = Path(args.image)
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        det = detect_charuco(gray, board, detector)
        if det is None:
            raise RuntimeError("No ChArUco corners found on selected image.")
    else:
        image_path, det = find_best_image(Path(args.images_dir), board, detector)
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if det["num_corners"] < args.min_corners:
        raise RuntimeError(
            f"Too few ChArUco corners: {det['num_corners']} (need >= {args.min_corners})."
        )

    obj_pts, img_pts = charuco_correspondences(
        board, det["charuco_corners"], det["charuco_ids"]
    )

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=k,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        confidence=0.999,
        iterationsCount=200,
    )

    if not ok:
        raise RuntimeError("solvePnPRansac failed.")

    inliers_count = 0 if inliers is None else int(len(inliers))

    # Optional refinement for slightly better estimate.
    rvec, tvec = cv2.solvePnPRefineLM(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=k,
        distCoeffs=dist,
        rvec=rvec,
        tvec=tvec,
    )

    # Compute reprojection RMSE.
    reproj, _ = cv2.projectPoints(obj_pts, rvec, tvec, k, dist)
    reproj = reproj.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((img_pts - reproj) ** 2, axis=1))))

    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, det["marker_corners"], det["marker_ids"])
    cv2.aruco.drawDetectedCornersCharuco(
        vis, det["charuco_corners"], det["charuco_ids"]
    )
    cv2.drawFrameAxes(vis, k, dist, rvec, tvec, args.axis_length, 2)
    vis_path = out_dir / "solvepnp_axes.jpg"
    cv2.imwrite(str(vis_path), vis)

    pose_path = out_dir / "solvepnp_result.npz"
    np.savez(
        pose_path,
        image=str(image_path),
        rvec=rvec,
        tvec=tvec,
        R=cv2.Rodrigues(rvec)[0],
        rmse=rmse,
        corners_used=len(obj_pts),
        inliers=inliers_count,
        K=k,
        dist=dist,
    )

    report_path = out_dir / "solvepnp_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("solvePnP report\n")
        f.write(f"image: {image_path}\n")
        f.write(f"charuco_corners_detected: {det['num_corners']}\n")
        f.write(f"points_used: {len(obj_pts)}\n")
        f.write(f"inliers: {inliers_count}\n")
        f.write(f"reprojection_rmse_px: {rmse:.6f}\n\n")
        f.write("rvec:\n")
        f.write(np.array2string(rvec.reshape(3, 1), precision=6))
        f.write("\n\ntvec:\n")
        f.write(np.array2string(tvec.reshape(3, 1), precision=6))
        f.write("\n\nR:\n")
        f.write(np.array2string(cv2.Rodrigues(rvec)[0], precision=6))
        f.write("\n")

    print("=== SOLVEPNP RESULT ===")
    print(f"image: {image_path}")
    print(f"charuco corners detected: {det['num_corners']}")
    print(f"points used: {len(obj_pts)}")
    print(f"inliers: {inliers_count}")
    print(f"reprojection RMSE: {rmse:.6f}px")
    print("rvec:")
    print(rvec.reshape(3, 1))
    print("tvec (same units as board lengths):")
    print(tvec.reshape(3, 1))
    print(f"Saved: {vis_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {pose_path}")


if __name__ == "__main__":
    main()
