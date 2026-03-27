# SLAM Practice Project (Русский гайд)

Этот репозиторий — практический мини-пайплайн для изучения геометрического CV и базовых шагов к SLAM:

- калибровка камеры (Chessboard/ChArUco),
- matching и оценка относительной позы (`Essential matrix`, `recoverPose`),
- триангуляция 3D-точек,
- `solvePnP`,
- мини-VO (траектория по последовательности кадров),
- debug-видео VO,
- опционально: запуск ORB-SLAM3 на своей вебкамерной последовательности.

---

## 1. Что нужно установить

### 1.1 Python-зависимости

Рекомендуется Python 3.10+.

```bash
pip install numpy opencv-contrib-python
```

Почему `opencv-contrib-python`: нужен модуль `cv2.aruco` для ChArUco.

### 1.2 Структура папок

Скрипты работают с папками:

- `images/` — входные изображения,
- `outputs/` — результаты,
- `datasets/` — последовательности в TUM-подобном формате для ORB-SLAM3.

Если папка отсутствует, скрипт обычно создаст ее сам.

---

## 2. Быстрый маршрут (пошагово)

Ниже рекомендованный порядок выполнения.

### Шаг 1: собрать кадры для калибровки

Запуск автосбора:

```bash
python capture_images.py
```

По умолчанию включен фильтр `charuco` (кадр сохраняется только если паттерн найден).

Полезные параметры:

```bash
python capture_images.py --filter charuco --aruco-dict DICT_6X6_250 --charuco-squares-x 5 --charuco-squares-y 7 --min-charuco-corners 6
python capture_images.py --filter none
```

Остановка: клавиша `q`.

### Шаг 2: калибровка камеры

```bash
python -m calibrate_camera
```

Выходные файлы:

- `outputs/camera_calibration.npz` — матрица камеры и дисторсия,
- `outputs/corners_*.jpg` — визуализация найденных углов,
- `outputs/undistorted_sample.jpg` — пример undistort.

Если меняется паттерн, поправь константы в `calibrate_camera.py`:

- `CALIBRATION_METHOD` (`charuco` / `chessboard`),
- `ARUCO_DICT_NAME`,
- `CHARUCO_SQUARES_X`, `CHARUCO_SQUARES_Y`,
- размеры `CHARUCO_SQUARE_LENGTH_MM`, `CHARUCO_MARKER_LENGTH_MM`.

### Шаг 3: matching + relative pose (Day 2)

```bash
python match_and_pose.py
```

Выход:

- `outputs/matches_all.jpg`,
- `outputs/matches_inliers.jpg`,
- `outputs/pose_result.txt`,
- `outputs/pose_result.npz`.

### Шаг 4: triangulation (3D точки)

```bash
python triangulate_points.py
```

Выход:

- `outputs/triangulated_points.npz`,
- `outputs/triangulated_points.ply`,
- `outputs/triangulation_report.txt`.

### Шаг 5: solvePnP demo (Day 3)

```bash
python solvepnp_demo.py
```

Выход:

- `outputs/solvepnp_axes.jpg`,
- `outputs/solvepnp_report.txt`,
- `outputs/solvepnp_result.npz`.

### Шаг 6: мини-VO траектория

```bash
python vo_trajectory.py
```

Выход:

- `outputs/vo_trajectory.png`,
- `outputs/vo_trajectory.csv`,
- `outputs/vo_trajectory.npz`,
- `outputs/vo_report.txt`.

### Шаг 7: debug-видео VO

```bash
python vo_debug_video.py
```

Выход:

- `outputs/vo_debug.mp4`,
- `outputs/vo_trajectory_from_video.png`,
- `outputs/vo_debug_steps.csv`,
- `outputs/vo_debug_report.txt`.

### Шаг 8: realtime VO с вебкамеры

```bash
python live_vo_webcam.py
```

Полезные опции:

```bash
python live_vo_webcam.py --show-matches --save-video
python live_vo_webcam.py --min-inliers 15 --ratio-test 0.8
```

Управление в окне:

- `q` — выход,
- `r` — сброс траектории.

---

## 3. ORB-SLAM3 на своей вебкамере (опционально)

Есть готовые скрипты:

- `capture_tum_dataset.py` — запись последовательности (`images/*.png` + `rgb.txt`),
- `export_orbslam3_yaml.py` — генерация настроек ORB-SLAM3 из калибровки,
- `run_orbslam3_mono.py` — запуск `mono_tum`.

Подробности: `ORB_SLAM3_WEBCAM.md`.

Короткий сценарий:

1. Сборка ORB-SLAM3 в Ubuntu/WSL:

```bash
bash setup_orbslam3_ubuntu.sh ~/slam
```

2. Запись датасета:

```bash
python capture_tum_dataset.py --fps 15 --width 640 --height 480
```

3. Экспорт yaml:

```bash
python export_orbslam3_yaml.py --width 640 --height 480 --fps 15
```

4. Запуск ORB-SLAM3:

```bash
python run_orbslam3_mono.py --orbslam-root ~/slam/ORB_SLAM3 --sequence-dir datasets/<sequence_name> --settings outputs/orbslam3_webcam.yaml
```

---

## 4. Советы по съемке (очень важно)

- Калибруй в том же разрешении, в котором потом работаешь.
- Для ChArUco: разные дистанции, разные углы, разные зоны кадра (включая края).
- Избегай смаза и слишком темных кадров.
- Для VO/SLAM: медленное движение, текстурная сцена, хороший свет, перекрытие кадров.

---

## 5. Частые проблемы

### `No ArUco markers` / `Too few ChArUco corners`

- неверный словарь (`ARUCO_DICT_NAME`),
- не те размеры доски (`CHARUCO_SQUARES_X/Y`),
- сильный блюр/пересвет/малый размер паттерна в кадре.

### Плохая калибровка (большой RMS)

- мало валидных кадров,
- паттерн почти всегда в центре и на одной дистанции,
- плохой свет.

### VO часто “SKIPPED”

- низкая текстура,
- слишком быстрые движения,
- высокие пороги `min_inliers`,
- смаз.

Пробуй:

```bash
python vo_debug_video.py --min-inliers 15 --ratio-test 0.8
```

---

## 6. Что уже покрыто этим проектом

Ты получаешь практику по:

- calibration / undistortion,
- feature matching + RANSAC,
- essential matrix + relative pose,
- triangulation,
- solvePnP,
- базовому monocular VO.

Это хорошая “собесная база” для входа в тему SLAM/VIO.
