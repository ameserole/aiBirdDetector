import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class BirdDatabase:
    """
    SQLite database for recording bird detection events.
    Thread-safe for use from GStreamer callback threads.

    Schema:
      videos     — one row per processed video file
      detections — one row per species identified within a video
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS videos (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        filepath    TEXT    NOT NULL,
        recorded_at TEXT    NOT NULL,
        backend     TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS detections (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id             INTEGER NOT NULL REFERENCES videos(id),
        timestamp            TEXT    NOT NULL,
        species              TEXT    NOT NULL,
        species_confidence   REAL    NOT NULL,
        detection_confidence REAL
    );
    """

    def __init__(self, db_path="birds.db"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        print(f"Database ready: {self.db_path}")

    def create_video(self, filepath: str, backend: str, recorded_at=None) -> int:
        """
        Insert a video record and return its id.

        Args:
            filepath:    Absolute path to the video file.
            backend:     'yolo' or 'hailo'
            recorded_at: datetime when processing started; defaults to now.
        """
        if recorded_at is None:
            recorded_at = datetime.now()
        ts = recorded_at.isoformat()

        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO videos (filepath, recorded_at, backend) VALUES (?, ?, ?)",
                (filepath, ts, backend),
            )
            self._conn.commit()
            return cur.lastrowid

    def record_detection(self, video_id: int, species: str,
                         species_confidence: float,
                         detection_confidence: float | None = None,
                         timestamp=None):
        """
        Insert one detection row linked to a video.

        Args:
            video_id:             Row id from create_video().
            species:              Classified species name.
            species_confidence:   Classification model confidence (0-1).
            detection_confidence: YOLO box confidence (0-1), or None for Hailo.
            timestamp:            datetime of the detection; defaults to now.
        """
        if timestamp is None:
            timestamp = datetime.now()
        ts = timestamp.isoformat()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO detections
                    (video_id, timestamp, species,
                     species_confidence, detection_confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (video_id, ts, species, species_confidence, detection_confidence),
            )
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    # Quick smoke-test
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp = f.name
    try:
        db = BirdDatabase(tmp)

        vid1 = db.create_video("/data/videos/morning_garden.mp4", "yolo")
        db.record_detection(vid1, "American Robin", 0.95, detection_confidence=0.87)
        db.record_detection(vid1, "House Sparrow", 0.72, detection_confidence=0.81)

        vid2 = db.create_video("/data/videos/afternoon.mp4", "hailo")
        db.record_detection(vid2, "Blue Jay", 0.88)

        videos = db._conn.execute("SELECT * FROM videos").fetchall()
        detections = db._conn.execute("SELECT * FROM detections").fetchall()
        print("Videos:")
        for row in videos:
            print(" ", row)
        print("Detections:")
        for row in detections:
            print(" ", row)

        db.close()
    finally:
        os.unlink(tmp)
