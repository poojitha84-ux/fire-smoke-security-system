# Detection Models & Provenance

This directory stores deep learning weights for real-time fire/smoke surveillance and mobile phone false-positive rejection.

## 1. Fire and Smoke Detector Model
- **File Name:** `fire_smoke_yolov8n.pt`
- **Architecture:** YOLOv8n (Nano - optimized for edge real-time camera inference)
- **Provenance / Repository:** [`rabahdev/fire-smoke-yolov8n`](https://huggingface.co/rabahdev/fire-smoke-yolov8n)
- **Dataset:** D-Fire Dataset (Fine-tuned on 4,306 test split images for early flame and smoke dispersion)
- **mAP50:** 0.754
- **Classes:**
  - `0`: `smoke`
  - `1`: `fire`
- **Licence:** AGPL-3.0
- **Direct Download URL:**
  `https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt`

## 2. Screen & Mobile Phone Detector Model
- **File Name:** `yolov8n.pt`
- **Architecture:** YOLOv8n (COCO pre-trained)
- **Provenance:** Ultralytics official model releases
- **Classes Utilized for Screen Filtering:**
  - `67`: `cell phone`
  - `62`: `tv`
  - `63`: `laptop`
- **Licence:** AGPL-3.0
- **Direct Download URL:**
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt`

## Automatic Downloading
The application automatically downloads and verifies these weights on first run if an internet connection is present.
For offline or air-gapped security operations, download the `.pt` files and save them directly into this `models/` directory.
