import cv2
import datetime
from pathlib import Path
import shutil
import time
from ultralytics import YOLO
from birdIdModel import BirdIdTFLiteModel


def _copy_video(src: str, output_dir: str) -> str:
    """Copy src into output_dir, avoiding overwrites with a counter suffix.

    Returns the absolute path of the copy.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    dest = dest_dir / src_path.name
    if dest.exists():
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{src_path.stem}_{counter}{src_path.suffix}"
            counter += 1
    shutil.copy2(str(src_path), str(dest))
    return str(dest.resolve())


class BirdDetectionPipeline:
    """
    Detects and identifies birds in video/RTSP streams using:
    - YOLO for bird detection
    - Bird ID TFLite model for species classification

    Detections are accumulated in memory during process_video() and written
    to the database as a single video record at the end of each file.
    RTSP sources are not written to the database (no file to reference).
    """

    def __init__(self,
                 yolo_model_path="assets/yolo26n.pt",
                 bird_id_model_path="assets/model.tflite",
                 bird_id_labels_path="assets/labels.txt",
                 bird_id_ignore_list_path=None,
                 detection_confidence=0.8,
                 classification_confidence=0.9,
                 frame_skip=30,
                 output_dir=None,
                 db=None):
        self.yolo_model = YOLO(yolo_model_path)
        self.bird_id_model = BirdIdTFLiteModel(bird_id_model_path, bird_id_labels_path)
        self.detection_confidence = detection_confidence
        self.classification_confidence = classification_confidence
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        self.db = db
        self.bird_id_ignore_list = self._load_ignore_list(bird_id_ignore_list_path) if bird_id_ignore_list_path else set()

        self.frame_count = 0
        self.processed_frame_count = 0
        self.birds_detected = {}
        self.min_roi_size = (50, 50)
        self._seen_species = None
        self._detections: list[dict] = []

    def _load_ignore_list(self, ignore_list_path):
        ignore_set = set()
        try:
            with open(ignore_list_path, 'r') as f:
                for line in f:
                    species = line.strip()
                    if species:
                        ignore_set.add(species)
        except Exception as e:
            print(f"Error loading ignore list: {e}")
        return ignore_set

    def _classify_bird_from_roi(self, roi):
        try:
            results = self.bird_id_model.classify_from_array(roi)
            if results and float(results[0][1]) >= self.classification_confidence:
                return results[0][0], float(results[0][1])
            return None
        except Exception as e:
            print(f"Error classifying bird from ROI: {e}")
            return None

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return frame
        self.processed_frame_count += 1

        start = time.time()
        detections = self.yolo_model.predict(
            source=[frame],
            conf=self.detection_confidence,
            save=False,
            verbose=False
        )
        print(f"Frame {self.frame_count}: YOLO detection time: {time.time() - start:.2f}s")

        for detection in detections:
            for box in detection.boxes:
                class_name = self.yolo_model.names[int(box.cls)]
                if 'bird' not in class_name.lower():
                    continue

                detection_confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                if (x2 - x1) < self.min_roi_size[0] or (y2 - y1) < self.min_roi_size[1]:
                    continue

                roi = frame[y1:y2, x1:x2]
                current_time = datetime.datetime.now()

                start = time.time()
                classification = self._classify_bird_from_roi(roi)
                print(f"Frame {self.frame_count}: Bird classification time: {time.time() - start:.2f}s")

                if classification:
                    bird_species, species_confidence = classification

                    if bird_species in self.bird_id_ignore_list:
                        continue

                    if self._seen_species is not None and bird_species in self._seen_species:
                        continue

                    self._detections.append({
                        "species": bird_species,
                        "species_confidence": species_confidence,
                        "detection_confidence": detection_confidence,
                        "timestamp": current_time,
                    })

                    if self._seen_species is not None:
                        self._seen_species.add(bird_species)

                    self.birds_detected[bird_species] = self.birds_detected.get(bird_species, 0) + 1

                    bird_species_display = bird_species
                    species_confidence_display = species_confidence
                else:
                    bird_species_display = "Unknown"
                    species_confidence_display = 0.0

                print(f"[{current_time.strftime('%H:%M:%S')}] BIRD DETECTED")
                print(f"  Species: {bird_species_display} (confidence: {species_confidence_display:.1%})")
                print(f"  Detection confidence: {detection_confidence:.1%}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{bird_species_display} ({species_confidence_display:.0%})"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def process_video(self, source, display=False, output_video=None, seen_species=None):
        self.frame_count = 0
        self.processed_frame_count = 0
        self.birds_detected = {}
        self._seen_species = seen_species
        self._detections = []
        started_at = datetime.datetime.now()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video source: {source}")
            return False

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0

        print(f"\n{'='*60}")
        print(f"Processing: {source}")
        print(f"Resolution: {frame_width}x{frame_height}  FPS: {fps:.1f}")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = self.process_frame(frame)
                if writer:
                    writer.write(processed_frame)
                if display:
                    cv2.imshow("Bird Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break
                if self.processed_frame_count % 10 == 0 and total_frames > 0:
                    progress = self.frame_count / total_frames * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames} frames)")
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        # Write video + detections to DB (file sources only, not RTSP)
        is_rtsp = isinstance(source, str) and source.lower().startswith("rtsp://")
        if self._detections and not is_rtsp:
            abs_source = str(Path(source).resolve())
            saved_path = abs_source
            if self.output_dir:
                saved_path = _copy_video(abs_source, self.output_dir)
                print(f"Saved video: {saved_path}")
            if self.db is not None:
                video_id = self.db.create_video(saved_path, backend="yolo", recorded_at=started_at)
                for det in self._detections:
                    self.db.record_detection(video_id, **det)

        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Frames processed: {self.processed_frame_count}")
        print(f"Birds detected: {sum(self.birds_detected.values())}")
        if self.birds_detected:
            print("\nBird species detected:")
            for species, count in sorted(self.birds_detected.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {species}: {count}")
        print(f"{'='*60}\n")

        return True
