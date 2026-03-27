from pathlib import Path
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ORB-SLAM3 monocular settings YAML from camera_calibration.npz."
    )
    parser.add_argument(
        "--calib",
        type=str,
        default="outputs/camera_calibration.npz",
        help="Calibration npz path (default: outputs/camera_calibration.npz).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/orbslam3_webcam.yaml",
        help="Output YAML path (default: outputs/orbslam3_webcam.yaml).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera image width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera image height (default: 480).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Camera fps used during sequence capture (default: 15.0).",
    )
    parser.add_argument(
        "--rgb-order",
        type=int,
        default=1,
        choices=[0, 1],
        help="Set 1 for RGB input, 0 for BGR (default: 1).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=1200,
        help="ORBextractor.nFeatures (default: 1200).",
    )
    return parser.parse_args()


def load_calibration(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path.resolve()}")

    data = np.load(path, allow_pickle=True)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise KeyError("camera_matrix or dist_coeffs missing in calibration file.")

    k = data["camera_matrix"].astype(float)
    dist = data["dist_coeffs"].reshape(-1).astype(float)

    if k.shape != (3, 3):
        raise ValueError(f"camera_matrix has unexpected shape: {k.shape}")
    if dist.size < 5:
        raise ValueError(f"dist_coeffs must have at least 5 values, got {dist.size}")

    return k, dist


def yaml_text(k, d, width, height, fps, rgb_order, n_features):
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    k1, k2, p1, p2, k3 = [float(v) for v in d[:5]]

    # OpenCV FileStorage style YAML that ORB-SLAM3 expects.
    return f"""%YAML:1.0
File.version: "1.0"

Camera.type: "PinHole"
Camera.fx: {fx:.8f}
Camera.fy: {fy:.8f}
Camera.cx: {cx:.8f}
Camera.cy: {cy:.8f}
Camera.k1: {k1:.10f}
Camera.k2: {k2:.10f}
Camera.p1: {p1:.10f}
Camera.p2: {p2:.10f}
Camera.k3: {k3:.10f}
Camera.width: {int(width)}
Camera.height: {int(height)}
Camera.fps: {float(fps):.6f}
Camera.RGB: {int(rgb_order)}

ORBextractor.nFeatures: {int(n_features)}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""


def main():
    args = parse_args()

    calib_path = Path(args.calib)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k, d = load_calibration(calib_path)
    text = yaml_text(
        k,
        d,
        args.width,
        args.height,
        args.fps,
        args.rgb_order,
        args.n_features,
    )
    out_path.write_text(text, encoding="utf-8")

    print("ORB-SLAM3 YAML exported.")
    print(f"Input calib: {calib_path.resolve()}")
    print(f"Output YAML: {out_path.resolve()}")
    print(
        f"fx={k[0,0]:.3f}, fy={k[1,1]:.3f}, cx={k[0,2]:.3f}, cy={k[1,2]:.3f}, "
        f"dist=[{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}, {d[3]:.4f}, {d[4]:.4f}]"
    )


if __name__ == "__main__":
    main()
