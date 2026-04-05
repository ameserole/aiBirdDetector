import argparse
import os
import sys

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# ---------------------------------------------------------------------------
# Shared args parsed by this entry point (stripped from sys.argv before
# GStreamerDetectionApp initialises its own argparse in the hailo backend).
# ---------------------------------------------------------------------------
_OUR_FLAGS = {
    "--backend",
    "--bird-model",
    "--bird-labels",
    "--bird-ignore-list",
    "--output-dir",
    "--db",
    "--classification-confidence",
    "--video-dir",
}


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Bird Detection and Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # YOLO backend — local video file
  python aiBirdDetector.py --backend yolo --video birds.mp4

  # YOLO backend — RTSP stream with live display
  python aiBirdDetector.py --backend yolo --rtsp rtsp://127.0.0.1:8554/stream1 --display

  # Hailo backend — video file at 4K, 1 fps
  python aiBirdDetector.py --backend hailo -i /data/birds.mp4 --height 2160 --width 3840 --frame-rate 1
        """,
    )

    # ---- shared args -------------------------------------------------------
    parser.add_argument(
        "--backend", choices=["yolo", "hailo"], default="yolo",
        help="Detection backend to use (default: yolo)",
    )
    parser.add_argument(
        "--bird-model", default="assets/model.tflite",
        help="Path to Bird ID TFLite model",
    )
    parser.add_argument(
        "--bird-labels", default="assets/labels.txt",
        help="Path to Bird ID labels file",
    )
    parser.add_argument(
        "--bird-ignore-list", default=None,
        help="Path to species ignore list (one name per line)",
    )
    parser.add_argument(
        "--db", default="birds.db",
        help="SQLite database file path (default: birds.db)",
    )
    parser.add_argument(
        "--classification-confidence", type=float, default=None,
        help="Species classification confidence threshold (default: 0.9 for yolo, 0.5 for hailo)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to copy videos containing bird detections into",
    )

    # ---- YOLO-only args ----------------------------------------------------
    yolo_group = parser.add_argument_group("YOLO backend options")
    yolo_group.add_argument("--video", help="Path to a single video file")
    yolo_group.add_argument("--video-dir", help="Directory of MP4 files to process in sequence")
    yolo_group.add_argument("--rtsp", help="RTSP stream URL")
    yolo_group.add_argument(
        "--display", action="store_true",
        help="Display annotated video while processing",
    )
    yolo_group.add_argument("--output", help="Path to save annotated output video")
    yolo_group.add_argument(
        "--confidence", type=float, default=0.8,
        help="YOLO detection confidence threshold (default: 0.8)",
    )
    yolo_group.add_argument(
        "--frame-skip", type=int, default=30,
        help="Process every Nth frame (default: 30)",
    )
    yolo_group.add_argument(
        "--yolo-model", default="assets/yolo26n.pt",
        help="Path to YOLO model",
    )

    return parser


def _strip_our_args_from_argv():
    """
    Remove our custom flags and their values from sys.argv so that
    GStreamerDetectionApp only sees its own expected arguments.
    """
    filtered = [sys.argv[0]]
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        # Match --flag or --flag=value
        flag = arg.split("=")[0]
        if flag in _OUR_FLAGS:
            # Skip the flag; if it's --flag value (not --flag=value) skip next too
            if "=" not in arg and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
        else:
            filtered.append(arg)
            i += 1
    sys.argv = filtered


def _extract_hailo_input() -> str:
    """Return the value of the -i flag from sys.argv, or '' if not present."""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "-i" and i + 1 < len(argv):
            return argv[i + 1]
    return ""


def main():
    parser = _build_parser()
    # parse_known_args so unrecognised Hailo flags don't cause an error
    args, _ = parser.parse_known_args()

    from birdDatabase import BirdDatabase
    db = BirdDatabase(args.db)

    try:
        if args.backend == "yolo":
            _run_yolo(args, db)
        else:
            _run_hailo(args, db)
    finally:
        db.close()


def _run_yolo(args, db):
    if not args.video and not args.rtsp and not args.video_dir:
        print("ERROR: --backend yolo requires --video, --video-dir, or --rtsp")
        sys.exit(1)

    if args.video and args.video_dir:
        print("ERROR: --video and --video-dir are mutually exclusive")
        sys.exit(1)

    classification_confidence = args.classification_confidence if args.classification_confidence is not None else 0.9

    from pipeline_yolo import BirdDetectionPipeline
    pipeline = BirdDetectionPipeline(
        yolo_model_path=args.yolo_model,
        bird_id_model_path=args.bird_model,
        bird_id_labels_path=args.bird_labels,
        bird_id_ignore_list_path=args.bird_ignore_list,
        detection_confidence=args.confidence,
        classification_confidence=classification_confidence,
        frame_skip=args.frame_skip,
        output_dir=args.output_dir,
        db=db,
    )

    if args.video_dir:
        import glob
        video_files = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
        if not video_files:
            print(f"ERROR: No MP4 files found in {args.video_dir}")
            sys.exit(1)
        print(f"Found {len(video_files)} video(s) in {args.video_dir}")
        for video_path in video_files:
            pipeline.process_video(
                source=video_path,
                display=args.display,
                output_video=args.output,
                seen_species=set(),
            )
        sys.exit(0)
    elif args.video:
        success = pipeline.process_video(
            source=args.video,
            display=args.display,
            output_video=args.output,
            seen_species=set(),
        )
        sys.exit(0 if success else 1)
    else:  # rtsp
        success = pipeline.process_video(
            source=args.rtsp,
            display=args.display,
            output_video=args.output,
        )
        sys.exit(0 if success else 1)


def _run_hailo(args, db):
    classification_confidence = args.classification_confidence if args.classification_confidence is not None else 0.5

    # Extract the -i (input) path before stripping our flags, then strip
    video_filepath = _extract_hailo_input()
    _strip_our_args_from_argv()

    from pipeline_hailo import run as hailo_run
    hailo_run(
        bird_id_model_path=args.bird_model,
        bird_id_labels_path=args.bird_labels,
        video_filepath=video_filepath,
        classification_confidence=classification_confidence,
        output_dir=args.output_dir,
        db=db,
    )


if __name__ == "__main__":
    main()
