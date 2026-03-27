from pathlib import Path
import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ORB-SLAM3 mono_tum on a recorded TUM-like sequence."
    )
    parser.add_argument(
        "--orbslam-root",
        type=str,
        required=True,
        help="Path to ORB_SLAM3 repository root.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=str,
        required=True,
        help="Path to sequence directory containing rgb.txt and images/.",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="outputs/orbslam3_webcam.yaml",
        help="Path to ORB-SLAM3 settings YAML (default: outputs/orbslam3_webcam.yaml).",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help="Path to mono_tum executable. If omitted, script tries common locations.",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="Path to ORBvoc.txt. If omitted, script tries common locations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved command without running.",
    )
    return parser.parse_args()


def pick_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_binary(root: Path, explicit: str | None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"mono_tum binary not found: {p.resolve()}")
        return p

    candidates = [
        root / "Examples" / "Monocular" / "mono_tum",
        root / "Examples" / "Monocular" / "mono_tum.exe",
        root / "build" / "Examples" / "Monocular" / "mono_tum",
        root / "build" / "Examples" / "Monocular" / "mono_tum.exe",
    ]
    chosen = pick_existing(candidates)
    if chosen is None:
        raise FileNotFoundError(
            "Cannot find mono_tum executable. "
            "Pass --binary explicitly or build ORB-SLAM3."
        )
    return chosen


def resolve_vocab(root: Path, explicit: str | None):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {p.resolve()}")
        return p

    candidates = [
        root / "Vocabulary" / "ORBvoc.txt",
        root / "ORBvoc.txt",
    ]
    chosen = pick_existing(candidates)
    if chosen is None:
        raise FileNotFoundError(
            "Cannot find ORBvoc.txt. Pass --vocab explicitly."
        )
    return chosen


def validate_sequence(seq_dir: Path):
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence dir not found: {seq_dir.resolve()}")
    rgb = seq_dir / "rgb.txt"
    img_dir = seq_dir / "images"
    if not rgb.exists():
        raise FileNotFoundError(f"Missing rgb.txt in sequence: {rgb.resolve()}")
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images/ in sequence: {img_dir.resolve()}")
    return rgb, img_dir


def main():
    args = parse_args()

    orb_root = Path(args.orbslam_root)
    if not orb_root.exists():
        raise FileNotFoundError(f"ORB-SLAM3 root not found: {orb_root.resolve()}")

    seq_dir = Path(args.sequence_dir)
    settings = Path(args.settings)
    if not settings.exists():
        raise FileNotFoundError(f"Settings YAML not found: {settings.resolve()}")

    validate_sequence(seq_dir)
    binary = resolve_binary(orb_root, args.binary)
    vocab = resolve_vocab(orb_root, args.vocab)

    cmd = [
        str(binary),
        str(vocab),
        str(settings),
        str(seq_dir),
    ]

    print("Resolved command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("Note: ORB-SLAM3 viewer requires a GUI session (X/Pangolin).")

    if args.dry_run:
        return

    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"mono_tum exited with code: {proc.returncode}")
            sys.exit(proc.returncode)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to execute mono_tum. Check binary path and permissions. {exc}"
        ) from exc


if __name__ == "__main__":
    main()
