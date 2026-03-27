from pathlib import Path

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Choose calibration target:
# "chessboard" or "charuco"
CALIBRATION_METHOD = "charuco"

# Chessboard config (number of INNER corners)
CHESSBOARD_SIZE = (5, 7)  # (cols, rows)
CHESSBOARD_SQUARE_SIZE_MM = 12.0

# ChArUco config
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LENGTH_MM = 12.0
CHARUCO_MARKER_LENGTH_MM = 7.0
ARUCO_DICT_NAME = "DICT_6X6_250"

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)


def list_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def build_chessboard_object_points(board_size, square_size):
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def compute_reprojection_rmse(
    objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
):
    total_sq_error = 0.0
    total_points = 0

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        total_sq_error += cv2.norm(imgp, projected, cv2.NORM_L2SQR)
        total_points += len(projected)

    return float(np.sqrt(total_sq_error / total_points))


def get_aruco_dict(dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco module is missing. Install opencv-contrib-python for ChArUco support."
        )

    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

    dict_id = getattr(cv2.aruco, dict_name)
    return cv2.aruco.getPredefinedDictionary(dict_id)


def detect_markers(gray, aruco_dict):
    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector.detectMarkers(gray)

    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)


def calibrate_from_chessboard(image_paths):
    objp = build_chessboard_object_points(CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE_MM)
    objpoints = []
    imgpoints = []
    used_images = []
    image_size = None

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot open image: {img_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if not found:
            print(f"[SKIP] Pattern not found: {img_path.name}")
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)

        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used_images.append(img_path)

        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners_refined, found)
        cv2.imwrite(str(OUTPUT_DIR / f"corners_{img_path.name}"), vis)
        print(f"[OK] Used image: {img_path.name}")

    if len(used_images) < 10:
        raise RuntimeError(
            f"Too few valid images: {len(used_images)}. Collect at least 10 good shots."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    rmse = compute_reprojection_rmse(
        objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    metadata = {
        "method": "chessboard",
        "pattern_size": CHESSBOARD_SIZE,
        "square_size_mm": CHESSBOARD_SQUARE_SIZE_MM,
    }
    return ret, rmse, camera_matrix, dist_coeffs, rvecs, tvecs, used_images, metadata


def calibrate_from_charuco(image_paths):
    aruco_dict = get_aruco_dict(ARUCO_DICT_NAME)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_MM,
        CHARUCO_MARKER_LENGTH_MM,
        aruco_dict,
    )

    all_charuco_corners = []
    all_charuco_ids = []
    used_images = []
    image_size = None

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot open image: {img_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        marker_corners, marker_ids, _ = detect_markers(gray, aruco_dict)
        if marker_ids is None or len(marker_ids) == 0:
            print(f"[SKIP] No ArUco markers: {img_path.name}")
            continue

        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

        if charuco_ids is None or num_corners < 6:
            print(f"[SKIP] Too few ChArUco corners: {img_path.name}")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        used_images.append(img_path)

        vis = img.copy()
        cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
        cv2.imwrite(str(OUTPUT_DIR / f"corners_{img_path.name}"), vis)
        print(f"[OK] Used image: {img_path.name} (corners={int(num_corners)})")

    if len(used_images) < 10:
        raise RuntimeError(
            f"Too few valid images: {len(used_images)}. Collect at least 10 good shots."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )

    # For ChArUco, OpenCV returns RMS reprojection error as 'ret'.
    rmse = float(ret)
    metadata = {
        "method": "charuco",
        "squares_x": CHARUCO_SQUARES_X,
        "squares_y": CHARUCO_SQUARES_Y,
        "square_length_mm": CHARUCO_SQUARE_LENGTH_MM,
        "marker_length_mm": CHARUCO_MARKER_LENGTH_MM,
        "aruco_dict": ARUCO_DICT_NAME,
    }
    return ret, rmse, camera_matrix, dist_coeffs, rvecs, tvecs, used_images, metadata


def save_outputs(
    ret,
    rmse,
    camera_matrix,
    dist_coeffs,
    rvecs,
    tvecs,
    used_images,
    metadata,
):
    print("\n=== CALIBRATION RESULT ===")
    print(f"Method: {metadata['method']}")
    print(f"OpenCV RMS: {ret:.6f}")
    print(f"Reprojection RMSE: {rmse:.6f}")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coeffs:\n", dist_coeffs.ravel())

    save_payload = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": np.array(rvecs, dtype=object),
        "tvecs": np.array(tvecs, dtype=object),
        "rmse": rmse,
        "opencv_rms": float(ret),
        "method": metadata["method"],
    }
    for key, value in metadata.items():
        if key != "method":
            save_payload[key] = value

    np.savez(OUTPUT_DIR / "camera_calibration.npz", **save_payload)

    sample_img = cv2.imread(str(used_images[0]))
    if sample_img is not None:
        h, w = sample_img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(
            sample_img, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y : y + rh, x : x + rw]

        cv2.imwrite(str(OUTPUT_DIR / "undistorted_sample.jpg"), undistorted)
        print("Undistorted image saved to outputs/undistorted_sample.jpg")


def main():
    image_paths = list_images(IMAGES_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {IMAGES_DIR.resolve()}")

    method = CALIBRATION_METHOD.strip().lower()
    if method == "chessboard":
        result = calibrate_from_chessboard(image_paths)
    elif method == "charuco":
        result = calibrate_from_charuco(image_paths)
    else:
        raise ValueError(
            f"Unsupported CALIBRATION_METHOD='{CALIBRATION_METHOD}'. "
            "Use 'chessboard' or 'charuco'."
        )

    save_outputs(*result)


if __name__ == "__main__":
    main()
