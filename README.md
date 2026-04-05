# Bird Detection and Classification Pipeline

A Python system for detecting and identifying bird species in video files using deep learning. Detected species are recorded in a SQLite database keyed to the source video; a web UI lets you browse results and play back videos in-browser.

> **Disclaimer**: Prototyped by hand but "productized" with AI.

## How it works

Two models run in sequence on each video frame:

1. **Detection** — YOLO (YOLO backend) or Hailo-8 NPU via GStreamer (Hailo backend) finds bounding boxes labelled `bird`.
2. **Classification** — A TFLite bird-ID model identifies the species from the cropped region.

When a video finishes processing, any species that crossed the confidence threshold are written to `birds.db` alongside the path to the source video file. No images are extracted; the original video is the record.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# installs: opencv-python-headless, ultralytics, tensorflow, flask
```

### Required assets

Place these in `assets/` before running:

| File | Used by | Purpose |
|------|---------|---------|
| `yolo26n.pt` | YOLO backend | Bird detection model |
| `model.tflite` | Both backends | Species classification model |
| `labels.txt` | Both backends | One species label per line, matching model output indices |

An optional species ignore list (one name per line) can be passed via `--bird-ignore-list`.

## Usage

```bash
source venv/bin/activate
```

### YOLO backend

```bash
# Single video file
python aiBirdDetector.py --backend yolo --video birds.mp4

# Batch-process a directory of MP4s, copying positives to ./videos/
python aiBirdDetector.py --backend yolo --video-dir /path/to/videos/ --output-dir ./videos/

# RTSP stream (detections logged to console; no DB entry for live streams)
python aiBirdDetector.py --backend yolo --rtsp rtsp://127.0.0.1:8554/stream1 --display

# Tune thresholds
python aiBirdDetector.py --backend yolo --video birds.mp4 \
  --confidence 0.7 --classification-confidence 0.85 --frame-skip 15
```

### Hailo backend

Uses a Hailo-8 NPU via GStreamer. Hailo-specific flags (`-i`, `--height`, `--width`, `--frame-rate`) pass through directly to `GStreamerDetectionApp`.

```bash
# Via convenience script (watches for new files with inotifywait)
bash run_pipeline.sh

# Directly — single video file at 4K, 1 fps, copying positives to ./videos/
python aiBirdDetector.py --backend hailo -i /path/to/birds.mp4 \
  --height 2160 --width 3840 --frame-rate 1 --output-dir ./videos/
```

### Key shared options

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `birds.db` | SQLite database path |
| `--classification-confidence` | 0.9 (YOLO) / 0.5 (Hailo) | Species confidence threshold |
| `--bird-model` | `assets/model.tflite` | TFLite classification model |
| `--bird-labels` | `assets/labels.txt` | Species labels file |
| `--bird-ignore-list` | — | Species to skip (YOLO only) |
| `--output-dir` | — | Directory to copy bird-positive videos into |

### Automated file watcher

`run_pipeline.sh` watches a directory for newly written MP4 files using `inotifywait` and runs the Hailo pipeline on each one automatically. Bird-positive videos are copied to `./videos/` and the originals are deleted. Edit `VIDEOS_DIR` at the top of the script to point at your incoming video folder.

```bash
# Requires: sudo apt install inotify-tools
bash run_pipeline.sh
```

## Web viewer

A Flask app for browsing results. Each row in the table represents a video file; species detected in it are shown as chips. Click **Play** to stream the video inline.

```bash
python viewer/birdViewer.py --db birds.db --port 5000
# then open http://localhost:5000
```

API endpoints (JSON):

- `GET /api/summary` — species counts across all videos
- `GET /api/videos?species=&date=&page=` — paginated video list with species
- `GET /videos/<id>` — streams the source video file

## Database schema

`birds.db` (SQLite) has two tables:

**`videos`** — one row per processed video file

| column | type | notes |
|--------|------|-------|
| `id` | INTEGER | primary key |
| `filepath` | TEXT | absolute path to source video |
| `recorded_at` | TEXT | ISO 8601, when processing started |
| `backend` | TEXT | `'yolo'` or `'hailo'` |

**`detections`** — one row per species found in a video

| column | type | notes |
|--------|------|-------|
| `id` | INTEGER | primary key |
| `video_id` | INTEGER | FK → `videos.id` |
| `timestamp` | TEXT | ISO 8601, time of first detection |
| `species` | TEXT | |
| `species_confidence` | REAL | classification model confidence (0–1) |
| `detection_confidence` | REAL | YOLO box confidence; NULL for Hailo |

## RTSP stream setup (Raspberry Pi camera)

**On the Raspberry Pi** (camera source):
```bash
rpicam-vid -t 0 -n --codec libav --libav-format mpegts -o tcp://192.168.0.23:4444
```

**On the Linux server** (relay to RTSP):
```bash
socat TCP4-LISTEN:4444,fork STDOUT | cvlc stream:///dev/stdin \
  --sout '#rtp{sdp=rtsp://:8554/stream1}' --rtsp-host=127.0.0.1
```

## Testing individual components

```bash
# Test species classification on a single image
python birdIdModel.py <image_path>

# Test raw Hailo hardware inference
python hailoSimple.py

# Smoke-test the database layer
python birdDatabase.py
```

## License

- YOLOv8 (Ultralytics) — AGPL-3.0
- TensorFlow — Apache 2.0
