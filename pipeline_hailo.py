import cv2
import datetime
from pathlib import Path
import shutil
import time


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

import gi
gi.require_version("Gst", "1.0")

import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

from birdIdModel import BirdIdTFLiteModel

hailo_logger = get_logger(__name__)


def _convert_bbox_to_pixels(bbox, width, height):
    """Convert normalized Hailo bbox coordinates to pixel coordinates."""
    x1 = max(0, int(bbox.xmin() * width))
    y1 = max(0, int(bbox.ymin() * height))
    x2 = min(width, int((bbox.xmin() + bbox.width()) * width))
    y2 = min(height, int((bbox.ymin() + bbox.height()) * height))
    return x1, y1, x2, y2


def _process_bird_detection(bird_crop, bird_id_model, user_data,
                             classification_confidence=0.8):
    """Classify a bird crop and accumulate the result in user_data.

    Only the first detection of each species is recorded (dedup via seen_species).
    """
    start = time.time()
    results = bird_id_model.classify_from_array(bird_crop)
    elapsed = time.time() - start

    if not results:
        hailo_logger.info("No bird species identified.")
        return None

    bird_species = results[0][0]
    species_confidence = float(results[0][1])
    hailo_logger.info(
        f"Identified: {bird_species} ({species_confidence:.2f}) in {elapsed:.2f}s"
    )

    if species_confidence > classification_confidence:
        if bird_species in user_data.seen_species:
            hailo_logger.info(f"Skipping duplicate sighting: {bird_species}")
            return bird_species, species_confidence

        user_data.detections.append({
            "species": bird_species,
            "species_confidence": species_confidence,
            "detection_confidence": None,
            "timestamp": datetime.datetime.now(),
        })
        user_data.seen_species.add(bird_species)

    return bird_species, species_confidence


class HailoUserData(app_callback_class):
    """Carries shared state into the GStreamer callback."""

    def __init__(self, bird_id_model, db=None,
                 classification_confidence=0.5, video_filepath="", output_dir=None):
        super().__init__()
        self.bird_id_model = bird_id_model
        self.db = db
        self.classification_confidence = classification_confidence
        self.video_filepath: str = video_filepath
        self.output_dir = output_dir
        self.seen_species: set = set()
        self.detections: list[dict] = []


def app_callback(element, buffer, user_data):
    if buffer is None:
        hailo_logger.warning("Received None buffer.")
        return

    pad = element.get_static_pad("src")
    fmt, width, height = get_caps_from_pad(pad)
    frame = get_numpy_from_buffer(buffer, fmt, width, height)
    success, map_info = buffer.map(Gst.MapFlags.READ)

    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        bird_count = 0
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            hailo_logger.debug(f"Detection: {label}, confidence: {confidence:.2f}")

            if label != "bird":
                continue

            bird_count += 1
            bbox = detection.get_bbox()
            x1, y1, x2, y2 = _convert_bbox_to_pixels(bbox, width, height)
            bird_crop = cv2.flip(frame[y1:y2, x1:x2].copy(), 1)

            _process_bird_detection(
                bird_crop,
                user_data.bird_id_model,
                user_data,
                user_data.classification_confidence,
            )

        if bird_count > 0:
            hailo_logger.info(f"Total bird detections in frame: {bird_count}")

    except Exception as e:
        hailo_logger.error(f"Error processing buffer: {e}")
    finally:
        buffer.unmap(map_info)


class MyGStreamerDetectionApp(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data):
        super().__init__(app_callback, user_data)

    def on_eos(self):
        hailo_logger.info("End of stream reached.")
        ud = self.user_data
        if ud.detections and ud.video_filepath:
            abs_path = str(Path(ud.video_filepath).resolve())
            saved_path = abs_path
            if ud.output_dir:
                saved_path = _copy_video(abs_path, ud.output_dir)
                hailo_logger.info(f"Saved video: {saved_path}")
            if ud.db is not None:
                video_id = ud.db.create_video(saved_path, backend="hailo")
                for det in ud.detections:
                    ud.db.record_detection(video_id, **det)
                hailo_logger.info(
                    f"Wrote {len(ud.detections)} detection(s) for {saved_path}"
                )
        self.shutdown()


def run(bird_id_model_path="assets/model.tflite",
        bird_id_labels_path="assets/labels.txt",
        video_filepath="",
        classification_confidence=0.5,
        output_dir=None,
        db=None):
    """
    Start the Hailo GStreamer detection pipeline.
    Remaining CLI args (e.g. -i, --height, --width, --frame-rate) are consumed
    directly from sys.argv by GStreamerDetectionApp.
    """
    bird_id_model = BirdIdTFLiteModel(bird_id_model_path, bird_id_labels_path)
    user_data = HailoUserData(
        bird_id_model=bird_id_model,
        db=db,
        classification_confidence=classification_confidence,
        video_filepath=video_filepath,
        output_dir=output_dir,
    )

    hailo_logger.info("Starting Hailo Detection App.")
    app = MyGStreamerDetectionApp(app_callback, user_data)
    app.run()
