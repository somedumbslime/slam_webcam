from pathlib import Path
from datetime import datetime
import argparse
import time

import cv2


DEFAULT_FILTER = "charuco"
DEFAULT_ARUCO_DICT = "DICT_6X6_250"
DEFAULT_CHARUCO_SQUARES_X = 5
DEFAULT_CHARUCO_SQUARES_Y = 7
DEFAULT_MIN_CHARUCO_CORNERS = 6
DEFAULT_CHESSBOARD_COLS = 7
DEFAULT_CHESSBOARD_ROWS = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture frames from camera and save one image every N seconds."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between saved frames (default: 1.0)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="images",
        help="Directory to save images (default: images)",
    )
    parser.add_argument(
        "--filter",
        choices=["none", "charuco", "chessboard"],
        default=DEFAULT_FILTER,
        help=f"Save only frames with detected pattern (default: {DEFAULT_FILTER})",
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default=DEFAULT_ARUCO_DICT,
        help=f"ArUco dictionary for charuco filter (default: {DEFAULT_ARUCO_DICT})",
    )
    parser.add_argument(
        "--charuco-squares-x",
        type=int,
        default=DEFAULT_CHARUCO_SQUARES_X,
        help=f"ChArUco squares along X (default: {DEFAULT_CHARUCO_SQUARES_X})",
    )
    parser.add_argument(
        "--charuco-squares-y",
        type=int,
        default=DEFAULT_CHARUCO_SQUARES_Y,
        help=f"ChArUco squares along Y (default: {DEFAULT_CHARUCO_SQUARES_Y})",
    )
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=DEFAULT_MIN_CHARUCO_CORNERS,
        help=f"Minimum interpolated ChArUco corners to accept frame (default: {DEFAULT_MIN_CHARUCO_CORNERS})",
    )
    parser.add_argument(
        "--chessboard-cols",
        type=int,
        default=DEFAULT_CHESSBOARD_COLS,
        help=f"Chessboard inner corners in X (default: {DEFAULT_CHESSBOARD_COLS})",
    )
    parser.add_argument(
        "--chessboard-rows",
        type=int,
        default=DEFAULT_CHESSBOARD_ROWS,
        help=f"Chessboard inner corners in Y (default: {DEFAULT_CHESSBOARD_ROWS})",
    )
    return parser.parse_args()


def build_charuco_detector(args):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco module is missing. Install opencv-contrib-python for ChArUco filter."
        )
    if not hasattr(cv2.aruco, args.aruco_dict):
        raise ValueError(f"Unknown ArUco dictionary: {args.aruco_dict}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.aruco_dict))
    board = cv2.aruco.CharucoBoard(
        (args.charuco_squares_x, args.charuco_squares_y),
        1.0,
        0.7,
        aruco_dict,
    )
    params = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    else:
        detector = None

    return aruco_dict, board, params, detector


def charuco_ok(frame, args, detector_ctx):
    aruco_dict, board, params, detector = detector_ctx
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if detector is not None:
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )

    if marker_ids is None or len(marker_ids) == 0:
        return False, "no markers"

    num_corners, _, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    corners_count = int(num_corners) if num_corners is not None else 0

    if charuco_ids is None or corners_count < args.min_charuco_corners:
        return False, f"corners={corners_count}"

    return True, f"corners={corners_count}"


def chessboard_ok(frame, args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    board_size = (args.chessboard_cols, args.chessboard_rows)
    found, _ = cv2.findChessboardCorners(gray, board_size, None)
    return found, "found" if found else "not found"


def frame_passes_filter(frame, args, detector_ctx):
    if args.filter == "none":
        return True, "no filter"
    if args.filter == "charuco":
        return charuco_ok(frame, args, detector_ctx)
    return chessboard_ok(frame, args)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_ctx = None
    if args.filter == "charuco":
        detector_ctx = build_charuco_detector(args)

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Capturing started. Press 'q' to stop.")
    print(f"Saving one frame every {args.interval:.2f}s to: {out_dir.resolve()}")
    print(f"Filter mode: {args.filter}")

    next_save_at = time.time()
    saved = 0
    skipped = 0
    last_status = "waiting"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            now = time.time()
            if now >= next_save_at:
                is_good, reason = frame_passes_filter(frame, args, detector_ctx)
                if is_good:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    file_path = out_dir / f"img_{ts}.jpg"
                    cv2.imwrite(str(file_path), frame)
                    saved += 1
                    last_status = f"saved ({reason})"
                    print(f"[{saved}] saved: {file_path.name} [{reason}]")
                else:
                    skipped += 1
                    last_status = f"skipped ({reason})"
                    print(f"[skip {skipped}] rejected frame [{reason}]")
                next_save_at = now + args.interval

            cv2.putText(
                frame,
                f"saved={saved} skipped={skipped} | {last_status}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Preview (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Total saved images: {saved}, skipped: {skipped}")


if __name__ == "__main__":
    main()
