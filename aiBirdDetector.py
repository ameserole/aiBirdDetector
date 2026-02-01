import cv2
import datetime
import os
import sys
import argparse
from ultralytics import YOLO
import numpy as np
from birdIdModel import BirdIdTFLiteModel
from pathlib import Path

class BirdDetectionPipeline:
    """
    Detects and identifies birds in video/RTSP streams using:
    - YOLO for bird detection
    - Bird ID TFLite model for species classification
    """
    
    def __init__(self, 
                 yolo_model_path="assets/yolo26n.pt",
                 bird_id_model_path="assets/model.tflite",
                 bird_id_labels_path="assets/labels.txt",
                 bird_id_ignore_list_path=None,
                 output_dir="images",
                 detection_confidence=0.8,
                 frame_skip=30):
        """
        Initialize the bird detection pipeline.
        
        Args:
            yolo_model_path: Path to YOLO model
            bird_id_model_path: Path to Bird ID TFLite model
            bird_id_labels_path: Path to Bird ID labels
            bird_id_ignore_list_path: Path to ignore list
            output_dir: Directory to save detected bird images
            detection_confidence: Confidence threshold for YOLO detection (0-1)
            frame_skip: Skip frames for faster processing (every Nth frame)
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.bird_id_model = BirdIdTFLiteModel(bird_id_model_path, bird_id_labels_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detection_confidence = detection_confidence
        self.frame_skip = frame_skip
        self.bird_id_ignore_list = self._load_ignore_list(bird_id_ignore_list_path) if bird_id_ignore_list_path else set()

        # Statistics tracking
        self.frame_count = 0
        self.processed_frame_count = 0
        self.birds_detected = {}
        self.min_roi_size = (50, 50)  # Minimum ROI size to classify
    
    def _load_ignore_list(self, ignore_list_path):
        """Load bird species ignore list from file."""
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

    def _get_unique_filename(self, base_filename):
        """Generate unique filename to avoid overwrites."""
        if not self.output_dir.joinpath(base_filename).exists():
            return self.output_dir.joinpath(base_filename)
        
        path = Path(base_filename)
        counter = 1
        stem = path.stem
        suffix = path.suffix
        
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            full_path = self.output_dir.joinpath(new_name)
            if not full_path.exists():
                return full_path
            counter += 1
    
    def _crop_and_save_roi(self, frame, bbox, timestamp, name_prefix="bird"):
        """
        Crop region of interest (bird detection) and save to disk.
        
        Args:
            frame: Input frame
            bbox: Bounding box as [x1, y1, x2, y2]
            timestamp: datetime object for filename
        
        Returns:
            Path to saved image or None if failed
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Validate ROI dimensions
            width = x2 - x1
            height = y2 - y1
            if width < self.min_roi_size[0] or height < self.min_roi_size[1]:
                return None
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            # Save image
            filename = f"{name_prefix}_" + timestamp.strftime("%Y-%m-%d_%H-%M-%S-%f.jpg")
            filepath = self._get_unique_filename(filename)

            success = cv2.imwrite(str(filepath), roi)
            return filepath if success else None
        except Exception as e:
            print(f"Error saving ROI: {e}")
            return None
    
    def _classify_bird(self, image_path, confidence_threshold=0.9):
        """
        Classify bird species from image using Bird ID model.
        
        Args:
            image_path: Path to bird image
        
        Returns:
            Tuple of (species_name, confidence) or None if failed
        """
        try:
            results = self.bird_id_model.run_from_filepath(str(image_path))
            if results and float(results[0][1]) >= confidence_threshold:
                return results[0][0], float(results[0][1])
            return None
        except Exception as e:
            print(f"Error classifying bird: {e}")
            return None
    
    def _classify_bird_from_roi(self, roi, confidence_threshold=0.9):
        """
        Classify bird species from a ROI (region of interest) array without saving to disk.
        
        Args:
            roi: Region of interest as numpy array (BGR format from cv2)
        
        Returns:
            Tuple of (species_name, confidence) or None if failed
        """
        try:
            results = self.bird_id_model.classify_from_array(roi)
            if results and float(results[0][1]) >= confidence_threshold:
                return results[0][0], float(results[0][1])
            return None
        except Exception as e:
            print(f"Error classifying bird from ROI: {e}")
            return None
    
    def process_frame(self, frame):
        """
        Process single frame for bird detection and classification.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Annotated frame with detections
        """
        self.frame_count += 1
        
        # Skip frames for faster processing
        if self.frame_count % self.frame_skip != 0:
            return frame
        
        self.processed_frame_count += 1
        
        # Run YOLO detection
        detections = self.yolo_model.predict(
            source=[frame], 
            conf=self.detection_confidence, 
            save=False,
            verbose=False
        )
        
        # Process detections
        for detection in detections:
            for box in detection.boxes:
                class_name = self.yolo_model.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy[0].cpu().numpy()
                
                # Only process bird detections
                if 'bird' not in class_name.lower():
                    continue
                
                # Save and classify bird
                current_time = datetime.datetime.now()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
                # Validate ROI dimensions
                width = x2 - x1
                height = y2 - y1
                if width < self.min_roi_size[0] or height < self.min_roi_size[1]:
                    return None
                
                # Extract ROI
                roi = frame[y1:y2, x1:x2]
                classification = self._classify_bird_from_roi(roi)
                # Classify species
                bird_species = "Unknown"
                species_confidence = 0.0
                
                if classification:
                    bird_species, species_confidence = classification
                    
                    if bird_species in self.bird_id_ignore_list:
                        continue
                    
                    self._crop_and_save_roi(frame, bbox, current_time, bird_species)
                    # Track statistics
                    if bird_species not in self.birds_detected:
                        self.birds_detected[bird_species] = 0
                    self.birds_detected[bird_species] += 1
                
                # Log detection
                print(f"[{current_time.strftime('%H:%M:%S')}] BIRD DETECTED")
                print(f"  Species: {bird_species} (confidence: {species_confidence:.1%})")
                print(f"  Detection confidence: {confidence:.1%}")
                # print(f"  Saved to: {image_path.name}")
                
                # Draw bounding box
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label text
                label = f"{bird_species} ({species_confidence:.0%})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    frame, label,
                    (x1, y1 - 10),
                    font, 0.7,
                    (0, 255, 0), 2
                )
        
        return frame

    def process_video(self, source, display=False, output_video=None):
        """
        Process video file or RTSP stream.
        
        Args:
            source: Path to video file or RTSP URL
            display: Whether to display video while processing
            output_video: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open video source: {source}")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*60}")
        print(f"Processing: {source}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps:.1f}")
        print(f"Total frames: {total_frames}")
        print(f"{'='*60}\n")
        
        # Setup video writer if output requested
        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_video, fourcc, fps,
                (frame_width, frame_height)
            )
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write output video
                if writer:
                    writer.write(processed_frame)
                
                # Display if requested
                if display:
                    cv2.imshow("Bird Detection", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped by user")
                        break
                
                # Progress update
                if self.processed_frame_count % 10 == 0:
                    progress = (self.frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames} frames)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
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


def main():
    parser = argparse.ArgumentParser(
        description="Bird Detection and Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process local video file
  python myBirdCameraDetect.py --video bird-detect-1/source/birds.mp4
  
  # Process RTSP stream with display
  python myBirdCameraDetect.py --rtsp rtsp://127.0.0.1:8554/stream1 --display
  
  # Save output video
  python myBirdCameraDetect.py --video input.mp4 --output output.mp4 --frame-skip 0
        """
    )
    
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--rtsp", type=str, help="RTSP stream URL")
    parser.add_argument("--display", action="store_true", help="Display video while processing")
    parser.add_argument("--output", type=str, help="Path to save output video")
    parser.add_argument("--confidence", type=float, default=0.8, help="Detection confidence threshold (0-1)")
    parser.add_argument("--frame-skip", type=int, default=30, help="Process every Nth frame")
    parser.add_argument("--yolo-model", type=str, default="assets/yolo26n.pt", help="YOLO model path")
    parser.add_argument("--bird-model", type=str, default="assets/model.tflite", help="Bird ID model path")
    parser.add_argument("--bird-labels", type=str, default="assets/labels.txt", help="Bird ID labels path")
    parser.add_argument("--bird-ignore-list", type=str, default=None, help="Path to bird species ignore list")
    parser.add_argument("--output-dir", type=str, default="images", help="Directory to save detected birds")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not args.rtsp:
        parser.print_help()
        print("\nERROR: Must provide either --video or --rtsp")
        sys.exit(1)
    
    source = args.video or args.rtsp
    
    # Create pipeline
    pipeline = BirdDetectionPipeline(
        yolo_model_path=args.yolo_model,
        detection_confidence=args.confidence,
        frame_skip=args.frame_skip,
        bird_id_model_path=args.bird_model,
        bird_id_labels_path=args.bird_labels,
        bird_id_ignore_list_path=args.bird_ignore_list,
        output_dir=args.output_dir
    )
    
    # Process video
    success = pipeline.process_video(
        source=source,
        display=args.display,
        output_video=args.output
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
