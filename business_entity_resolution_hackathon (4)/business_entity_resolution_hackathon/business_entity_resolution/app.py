from flask import Flask, render_template, jsonify, send_file, request
from pathlib import Path
import sys
import traceback
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.data_generator import generate_demo_dataset
from src.pipeline import (
    pipeline_status,
    preprocess_all,
    generate_candidates,
    make_features,
    train_model,
    evaluate_model,
    predict_test
)

app = Flask(__name__)


# =========================================================
# PAGES
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/model")
def model():
    return render_template("model.html")


@app.route("/testing")
def testing():
    return render_template("testing.html")


@app.route("/results")
def results():
    return render_template("results.html")


# =========================================================
# PIPELINE STATUS
# =========================================================

@app.get("/api/status")
def status():
    return jsonify(pipeline_status(BASE))


# =========================================================
# DATASET GENERATION
# =========================================================

@app.post("/api/generate")
def generate():
    try:
        info = generate_demo_dataset(BASE)
        return jsonify({
            "ok": True,
            **info
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# DATASET COUNTS
# =========================================================

@app.get("/api/dataset")
def dataset_info():

    train = BASE / "dataset" / "train"

    files = {
        "source1": train / "train_source1.tsv",
        "source2": train / "train_source2.tsv",
        "source3": train / "train_source3.tsv",
        "ground_truth": train / "train_ground_truth.tsv"
    }

    result = {}

    for name, path in files.items():

        if not path.exists():
            result[name] = {
                "exists": False,
                "total": 0
            }
            continue

        try:
            df = pd.read_csv(
                path,
                sep="\t",
                dtype=str
            )

            result[name] = {
                "exists": True,
                "total": len(df),
                "columns": list(df.columns),
                "file": str(path)
            }

        except Exception as e:

            result[name] = {
                "exists": False,
                "total": 0,
                "error": str(e)
            }

    return jsonify(result)


# =========================================================
# PREPROCESS
# =========================================================

@app.post("/api/preprocess")
def preprocess():

    try:
        info = preprocess_all(BASE)

        return jsonify({
            "ok": True,
            **info
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# CANDIDATES
# =========================================================

@app.post("/api/candidates")
def candidates():

    try:
        info = generate_candidates(BASE)

        return jsonify({
            "ok": True,
            **info
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# TRAIN
# =========================================================

@app.post("/api/train")
def train():

    try:
        info = train_model(BASE)

        return jsonify({
            "ok": True,
            **info
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# EVALUATE
# =========================================================

@app.post("/api/evaluate")
def evaluate():

    try:
        info = evaluate_model(BASE)

        return jsonify({
            "ok": True,
            **info
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# TEST
# =========================================================

@app.post("/api/test")
def test():

    try:

        data = request.get_json(silent=True) or {}

        threshold = float(
            data.get("threshold", 0.72)
        )

        info = predict_test(
            BASE,
            threshold
        )

        return jsonify({
            "ok": True,
            **info
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# =========================================================
# DATASET PREVIEW
# =========================================================

@app.get("/api/preview/<source>")
def preview(source):

    allowed = {
        "source1",
        "source2",
        "source3",
        "ground_truth"
    }

    if source not in allowed:

        return jsonify({
            "error": "Invalid source"
        }), 400

    train = BASE / "dataset" / "train"

    mapping = {

        "source1":
            train / "train_source1.tsv",

        "source2":
            train / "train_source2.tsv",

        "source3":
            train / "train_source3.tsv",

        "ground_truth":
            train / "train_ground_truth.tsv"
    }

    path = mapping[source]

    if not path.exists():

        return jsonify({
            "error": f"{path.name} not found in dataset/train"
        }), 404

    try:

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str
        ).fillna("")

        return jsonify({

            "columns": list(df.columns),

            "rows":
                df.head(20).to_dict(
                    "records"
                ),

            "total":
                len(df),

            "file":
                path.name

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# RESULTS
# =========================================================

@app.get("/api/results")
def api_results():

    p = BASE / "output" / "matching_results.tsv"

    if not p.exists():

        return jsonify({
            "error":
                "Run test prediction first."
        }), 404

    df = pd.read_csv(
        p,
        sep="\t",
        dtype=str
    ).fillna("")

    return jsonify({

        "columns":
            list(df.columns),

        "rows":
            df.head(100).to_dict(
                "records"
            ),

        "total":
            len(df)
    })


# =========================================================
# DOWNLOAD
# =========================================================

@app.get("/download/<name>")
def download(name):

    allowed = {
        "matching_results.tsv",
        "candidate_pairs.tsv"
    }

    if name not in allowed:

        return jsonify({
            "error": "Invalid file"
        }), 400

    p = BASE / "output" / name

    if not p.exists():

        return jsonify({
            "error":
                "File not available yet."
        }), 404

    return send_file(
        p,
        as_attachment=True
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )