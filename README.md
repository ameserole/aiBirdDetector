# Bird Detection and Classification Pipeline

A Python system for detecting and identifying bird species in video files and RTSP streams using deep learning models.

## Overview

**Disclaimer**: Prototyped by hand but "productized" with AI.

This project combines two models:
- **YOLO (You Only Look Once)**: For generic bird detection in video frames
- **Your Choice of TFLite Bird ID Model**: For bird species classification

The pipeline processes video frames, detects birds, crops the detected regions, and classifies them into specific species using the provided bird ID model.

## Features



## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.x (optional, for GPU acceleration)

### Setup

```bash
# Navigate to project directory
cd /home/messy/myBirdCamera

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install opencv-python tensorflow ultralytics pillow numpy
```

## Usage

```bash
python3 ./aiBirdDetector.py --help
usage: aiBirdDetector.py [-h] [--video VIDEO] [--rtsp RTSP] [--display] [--output OUTPUT] [--confidence CONFIDENCE]
                         [--frame-skip FRAME_SKIP] [--yolo-model YOLO_MODEL] [--bird-model BIRD_MODEL]
                         [--bird-labels BIRD_LABELS] [--bird-ignore-list BIRD_IGNORE_LIST] [--output-dir OUTPUT_DIR]

Bird Detection and Classification Pipeline

options:
  -h, --help            show this help message and exit
  --video VIDEO         Path to video file
  --rtsp RTSP           RTSP stream URL
  --display             Display video while processing
  --output OUTPUT       Path to save output video
  --confidence CONFIDENCE
                        Detection confidence threshold (0-1)
  --frame-skip FRAME_SKIP
                        Process every Nth frame
  --yolo-model YOLO_MODEL
                        YOLO model path
  --bird-model BIRD_MODEL
                        Bird ID model path
  --bird-labels BIRD_LABELS
                        Bird ID labels path
  --bird-ignore-list BIRD_IGNORE_LIST
                        Path to bird species ignore list
  --output-dir OUTPUT_DIR
                        Directory to save detected birds

Examples:
  # Process local video file
  python myBirdCameraDetect.py --video bird-detect-1/source/birds.mp4
  
  # Process RTSP stream with display
  python myBirdCameraDetect.py --rtsp rtsp://127.0.0.1:8554/stream1 --display
  
  # Save output video
  python myBirdCameraDetect.py --video input.mp4 --output output.mp4 --frame-skip 0
        
```

### Basic Examples

#### Process a local video file
```bash
python myBirdCameraDetect.py --video ./birds.mp4
```

#### Process RTSP stream with display
```bash
python myBirdCameraDetect.py --rtsp rtsp://127.0.0.1:8554/stream1 --display
```

#### Save output video with annotations
```bash
python myBirdCameraDetect.py --video input.mp4 --output output_annotated.mp4 --frame-skip 0
```

#### Adjust detection sensitivity
```bash
python myBirdCameraDetect.py --video input.mp4 --confidence 0.7 --display
```

#### Process every 15th frame for faster processing
```bash
python myBirdCameraDetect.py --rtsp rtsp://192.168.0.23:8554/stream --frame-skip 15
```

## Output

### Detected Bird Images
Saved to `images/` directory with timestamp:
```
images/
├── bird_2026-01-28_14-32-15-123456.jpg
├── bird_2026-01-28_14-32-45-234567.jpg
└── ...
```

## RTSP Stream Setup

### Raspberry Pi Camera to RTSP

**On Raspberry Pi** (camera source):
```bash
rpicam-vid -t 0 -n --codec libav --libav-format mpegts -o tcp://192.168.0.23:4444
```

**On Linux Server** (stream relay):
```bash
socat TCP4-LISTEN:4444,fork STDOUT | cvlc stream:///dev/stdin \
  --sout '#rtp{sdp=rtsp://:8554/stream1}' --rtsp-host=127.0.0.1
```

**Process stream**:
```bash
python myBirdCameraDetect.py --rtsp rtsp://127.0.0.1:8554/stream1 --display
```

## License

This project uses:
- YOLOv8 (Ultralytics) - AGPL-3.0
- TensorFlow - Apache 2.0
