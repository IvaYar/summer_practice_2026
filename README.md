# Raspberry Pi 5 Realtime Car Detector

Базовый проект для детекта машин на Raspberry Pi 5 с CSI-камерой через Picamera2. Runtime использует OpenCV DNN + YOLO ONNX: это проще всего запустить на Pi 5 без отдельного NPU.

## Что важно по FPS

Raspberry Pi 5 16 GB построен на Broadcom BCM2712: 4 ядра Arm Cortex-A76 @ 2.4 GHz и VideoCore VII GPU. Для камеры и UI это бодро, но нейросетевой детект машин в честные 30 FPS на CPU зависит от размера модели, `input_size`, разрешения и охлаждения. Этот проект показывает камеру около 30 FPS и запускает детектор в отдельном потоке на последних кадрах. Реальный `detect_fps` надо измерить на твоем Pi.

Если нужна стабильная детекция 30 FPS, почти наверняка понадобится AI-ускоритель: Raspberry Pi AI HAT+/AI Kit на Hailo или другой NPU/TPU. Важно: если M.2-разъем уже занят SSD, Hailo M.2-ускоритель обычно некуда поставить без другой схемы подключения.

Официальные ссылки:

- Raspberry Pi 5: https://www.raspberrypi.com/products/raspberry-pi-5/
- Camera software / Picamera2: https://www.raspberrypi.com/documentation/computers/camera_software.html
- Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- Raspberry Pi AI Kit: https://www.raspberrypi.com/products/ai-kit/

## Установка на Raspberry Pi OS

Лучше использовать Raspberry Pi OS Bookworm 64-bit на SSD.

```bash
sudo apt update
sudo apt full-upgrade -y
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
source .venv/bin/activate
```

Проверь камеру:

```bash
rpicam-hello -t 5000
```

## Экспорт модели

Экспорт можно сделать прямо на Pi или на ПК, а потом перенести файл `models/*.onnx` на Pi.

```bash
source .venv/bin/activate
python -m pip install -r requirements-export.txt
python tools/export_yolo_onnx.py --weights yolo11n.pt --imgsz 320 --output models/yolo11n_320.onnx
```

Более быстрый, но менее точный вариант:

```bash
python tools/export_yolo_onnx.py --weights yolo11n.pt --imgsz 256 --output models/yolo11n_256.onnx
```

## Запуск

Стандартный запуск с авто-выбором Picamera2:

```bash
python -m car_detector.app --config configs/pi5_cpu.yaml
```

Быстрый профиль:

```bash
python -m car_detector.app --config configs/pi5_fast.yaml
```

Headless-режим без окна, удобен по SSH:

```bash
python -m car_detector.app --config configs/pi5_cpu.yaml --headless
```

Отладка на видеофайле:

```bash
python -m car_detector.app --config configs/video_debug.yaml --video path/to/road.mp4
```

В репозитории есть короткий тестовый sample:

```bash
python -m car_detector.app --config configs/video_debug.yaml --video test_videos/own_test_sample.mp4
```

Быстрый ROI-режим для дорожного видео: модель смотрит только на область дороги и может работать с меньшим `input_size`:

```bash
python -m car_detector.app --config configs/video_roi_fast.yaml --video test_videos/own_test.mp4
```

Границы ROI задаются долями кадра. Например, чтобы чуть выше захватить дальние машины:

```bash
python -m car_detector.app --config configs/video_roi_fast.yaml --video test_videos/own_test.mp4 --roi-y1-ratio 0.22
```

Выход из окна: `q` или `Esc`.

## Бенчмарк настоящего FPS детектора

```bash
python tools/benchmark.py --model models/yolo11n_320.onnx --source picamera2 --seconds 30
```

Если `detector_fps` сильно ниже 30:

- попробуй `configs/pi5_fast.yaml`;
- уменьши `--input-size 256`;
- поставь `--width 640 --height 360`;
- подними `--conf 0.4`;
- проверь охлаждение и питание Pi 5;
- для стабильных 30 FPS переходи на Hailo/TPU backend.

## Структура

- `car_detector/app.py` - realtime-приложение;
- `car_detector/detector.py` - YOLO ONNX postprocess под COCO-классы `car,bus,truck`;
- `car_detector/camera.py` - Picamera2 и OpenCV input;
- `tools/export_yolo_onnx.py` - экспорт Ultralytics YOLO в ONNX;
- `tools/benchmark.py` - измерение реального detector FPS;
- `configs/*.yaml` - профили запуска.
