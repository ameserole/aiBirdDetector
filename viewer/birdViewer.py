import argparse
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

_config = {
    "db_path": "birds.db",
}

_PER_PAGE = 50

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_db():
    conn = sqlite3.connect(_config["db_path"])
    conn.row_factory = sqlite3.Row
    return conn


def _build_video_where(species, date):
    """Build WHERE + params to filter the videos list."""
    clauses, params = [], []
    if date:
        clauses.append("DATE(v.recorded_at) = ?")
        params.append(date)
    if species:
        clauses.append(
            "v.id IN (SELECT video_id FROM detections WHERE species = ?)"
        )
        params.append(species)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    conn = _get_db()
    try:
        rows = conn.execute("""
            SELECT species, COUNT(*) AS count, MAX(timestamp) AS last_seen
            FROM detections
            GROUP BY species
            ORDER BY count DESC
        """).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/videos")
def videos():
    species = request.args.get("species", "")
    date = request.args.get("date", "")
    page = max(1, int(request.args.get("page", 1)))

    where, params = _build_video_where(species, date)

    conn = _get_db()
    try:
        total = conn.execute(
            f"SELECT COUNT(DISTINCT v.id) FROM videos v {where}", params
        ).fetchone()[0]

        offset = (page - 1) * _PER_PAGE
        rows = conn.execute(
            f"""
            SELECT v.id, v.filepath, v.recorded_at, v.backend,
                   GROUP_CONCAT(DISTINCT d.species) AS species_list
            FROM videos v
            JOIN detections d ON d.video_id = v.id
            {where}
            GROUP BY v.id
            ORDER BY v.recorded_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [_PER_PAGE, offset],
        ).fetchall()
    except sqlite3.OperationalError:
        total, rows = 0, []
    finally:
        conn.close()

    result_rows = []
    for row in rows:
        d = dict(row)
        d["filename"] = Path(d["filepath"]).name
        d["species"] = d.pop("species_list", "").split(",") if d.get("species_list") else []
        d["video_url"] = f"/videos/{d['id']}"
        result_rows.append(d)

    return jsonify({
        "rows": result_rows,
        "total": total,
        "page": page,
        "pages": max(1, (total + _PER_PAGE - 1) // _PER_PAGE),
        "per_page": _PER_PAGE,
    })


@app.route("/videos/<int:video_id>")
def serve_video(video_id):
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT filepath FROM videos WHERE id = ?", (video_id,)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return "Not found", 404
    p = Path(row["filepath"])
    if not p.is_file():
        return "File not found on disk", 404
    return send_file(str(p), mimetype="video/mp4")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird sightings web viewer")
    parser.add_argument("--db",   default="birds.db", help="SQLite database path (default: birds.db)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    args = parser.parse_args()

    _config["db_path"] = args.db

    print(f"Starting Bird Sightings viewer at http://0.0.0.0:{args.port}")
    print(f"  Database : {args.db}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
