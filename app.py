r"""
Flask Webapplicatie voor het K-Means + Content-Based Movie Recommender System.
Start met: .\.venv\Scripts\python.exe app.py
Open daarna: http://127.0.0.1:5000
"""

import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = {
    "flask": "flask",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        __import__(module_name)
    except ImportError:
        print(f"Pakket '{pip_name}' ontbreekt. Bezig met automatische installatie...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

from flask import Flask, jsonify, render_template, request, send_file
from recommender_engine import MovieRecommenderEngine


BASE_DIR = Path(__file__).resolve().parent
PLOT_PATH = BASE_DIR / "kmeans_visualisatie.png"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True

print("Initialiseren van MovieRecommenderEngine...")
engine = MovieRecommenderEngine()
print(f"Recommender klaar met {len(engine.df):,} films!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/visualisatie.png")
def visualisatie_png():
    return send_file(PLOT_PATH, mimetype="image/png")


@app.route("/api/meta")
def api_meta():
    return jsonify(engine.get_metadata())


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "")
    limit = min(int(request.args.get("limit", 12)), 30)
    return jsonify({"results": engine.search(q, limit=limit)})


@app.route("/api/movie/<int:movie_id>")
def api_movie(movie_id: int):
    idx = engine.id_to_idx.get(movie_id)
    if idx is None:
        return jsonify({"error": "Film niet gevonden"}), 404
    return jsonify(engine.serialize_movie(idx))


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = request.get_json(silent=True) or {}
    seed_ids = [int(x) for x in payload.get("seed_ids", [])]
    content_weight = float(payload.get("content_weight", 0.65))
    cluster_mode = str(payload.get("cluster_mode", "hybrid"))
    genre = str(payload.get("genre", ""))
    min_rating = float(payload.get("min_rating", 0.0))
    year_min = int(payload.get("year_min", 1920))
    year_max = int(payload.get("year_max", 2026))
    max_runtime = int(payload.get("max_runtime", 300))
    limit = min(int(payload.get("limit", 12)), 24)

    result = engine.recommend(
        seed_ids=seed_ids,
        content_weight=content_weight,
        cluster_mode=cluster_mode,
        genre=genre,
        min_rating=min_rating,
        year_min=year_min,
        year_max=year_max,
        max_runtime=max_runtime,
        limit=limit,
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
