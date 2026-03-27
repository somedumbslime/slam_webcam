# ORB-SLAM3 Webcam Flow (Monocular)

This project now has scripts to run a full webcam -> dataset -> ORB-SLAM3 pipeline.

## 1) What you need to install

You need ORB-SLAM3 built on Linux (recommended: WSL2 Ubuntu on Windows 11).

Use:

```bash
bash setup_orbslam3_ubuntu.sh ~/slam
```

This installs dependencies and builds ORB-SLAM3 in `~/slam/ORB_SLAM3`.

Notes:
- Use a project path without spaces/cyrillic if possible.
- ORB-SLAM3 viewer (Pangolin) needs GUI support (WSLg on Windows 11 is usually enough).

## 2) Record a TUM-like sequence from webcam

Run from this repo:

```bash
python capture_tum_dataset.py --fps 15 --width 640 --height 480
```

It creates:
- `datasets/<sequence_name>/images/*.png`
- `datasets/<sequence_name>/rgb.txt`

## 3) Export ORB-SLAM3 settings from your calibration

```bash
python export_orbslam3_yaml.py --width 640 --height 480 --fps 15
```

Output:
- `outputs/orbslam3_webcam.yaml`

## 4) Run ORB-SLAM3 mono_tum

From the same repo (inside the Linux/WSL environment where ORB-SLAM3 is built):

```bash
python run_orbslam3_mono.py \
  --orbslam-root ~/slam/ORB_SLAM3 \
  --sequence-dir datasets/<sequence_name> \
  --settings outputs/orbslam3_webcam.yaml
```

If needed, pass explicit binary/vocab:

```bash
python run_orbslam3_mono.py \
  --orbslam-root ~/slam/ORB_SLAM3 \
  --sequence-dir datasets/<sequence_name> \
  --settings outputs/orbslam3_webcam.yaml \
  --binary ~/slam/ORB_SLAM3/Examples/Monocular/mono_tum \
  --vocab ~/slam/ORB_SLAM3/Vocabulary/ORBvoc.txt
```

## 5) Typical failure reasons

- Wrong camera calibration resolution (YAML width/height must match captured sequence).
- Too fast camera motion / motion blur.
- Low-texture scene (plain wall).
- Auto-exposure/auto-focus changes too aggressively.
- ORB-SLAM3 binary path or vocabulary path is wrong.

## 6) Minimum "good demo" checklist

- Sequence length: 30-90 seconds.
- Slow camera motion, frequent overlaps, some loops in trajectory.
- Good lighting, textured scene.
- Run returns stable tracking and visible map points/keyframes.
