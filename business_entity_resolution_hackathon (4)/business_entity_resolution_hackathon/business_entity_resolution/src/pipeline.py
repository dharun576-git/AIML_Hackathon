import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from rapidfuzz.fuzz import ratio, token_set_ratio

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "name_ratio",
    "name_token_ratio",
    "address_ratio",
    "address_token_ratio",
    "name_tfidf",
    "address_tfidf",
    "country_match",
    "name_len_diff",
    "address_len_diff"
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def norm(x):
    x = str(x).lower()

    x = x.replace("&", " and ")

    x = re.sub(r"[^a-z0-9\s]", " ", x)

    x = re.sub(r"\s+", " ", x).strip()

    replacements = {
        "private limited": "pvt ltd",
        "private ltd": "pvt ltd",
        "corporation": "corp",
        "limited": "ltd",
        "technologies": "tech",
        "technology": "tech",
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "parkway": "pkwy"
    }

    for a, b in replacements.items():
        x = x.replace(a, b)

    x = re.sub(r"\s+", " ", x).strip()

    return x


# ============================================================
# LOAD DATA
# ============================================================

def load_data(base):
    base = base.resolve()

    train_dir = base / "dataset" / "train"
    test_dir = base / "dataset" / "test"

    required_files = [
        train_dir / "train_source1.tsv",
        train_dir / "train_source2.tsv",
        train_dir / "train_source3.tsv",
        train_dir / "train_ground_truth.tsv",
        test_dir / "test_source1.tsv",
        test_dir / "test_source2.tsv",
        test_dir / "test_source3.tsv"
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Required dataset file not found:\n{file}"
            )

    return {
        "tr1": pd.read_csv(
            train_dir / "train_source1.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "tr2": pd.read_csv(
            train_dir / "train_source2.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "tr3": pd.read_csv(
            train_dir / "train_source3.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "truth": pd.read_csv(
            train_dir / "train_ground_truth.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "te1": pd.read_csv(
            test_dir / "test_source1.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "te2": pd.read_csv(
            test_dir / "test_source2.tsv",
            sep="\t",
            dtype=str
        ).fillna(""),

        "te3": pd.read_csv(
            test_dir / "test_source3.tsv",
            sep="\t",
            dtype=str
        ).fillna("")
    }


# ============================================================
# PREPROCESSING
# ============================================================

def prep(df):
    x = df.copy()

    x["business_name"] = x["business_name"].fillna("").astype(str)
    x["business_address"] = x["business_address"].fillna("").astype(str)
    x["country"] = x["country"].fillna("").astype(str)

    x["name_clean"] = x["business_name"].map(norm)
    x["address_clean"] = x["business_address"].map(norm)
    x["country_clean"] = x["country"].map(norm)

    return x


def preprocess_all(base):
    base = base.resolve()

    data = load_data(base)

    for key in data:
        if key != "truth":
            data[key] = prep(data[key])

    output_dir = base / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in [
        "tr1",
        "tr2",
        "tr3",
        "te1",
        "te2",
        "te3"
    ]:
        data[key].to_csv(
            output_dir / f"preprocessed_{key}.tsv",
            sep="\t",
            index=False
        )

    return {
        "message": "Preprocessing completed",
        "features_created": [
            "name_clean",
            "address_clean",
            "country_clean"
        ]
    }


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def candidate_pairs(s1, s2, s3, limit=8):
    """
    Generate candidate Source-2 / Source-3 records
    for every Source-1 record.
    """

    all_candidates = pd.concat(
        [
            s2.assign(_src="S2"),
            s3.assign(_src="S3")
        ],
        ignore_index=True
    )

    rows = []

    for _, r in s1.iterrows():

        scored = []

        for _, c in all_candidates.iterrows():

            # Country blocking
            if r["country_clean"] != c["country_clean"]:
                continue

            name_score = ratio(
                r["name_clean"],
                c["name_clean"]
            ) / 100.0

            address_score = ratio(
                r["address_clean"],
                c["address_clean"]
            ) / 100.0

            token_name_score = token_set_ratio(
                r["name_clean"],
                c["name_clean"]
            ) / 100.0

            # Combined blocking score
            score = (
                0.50 * name_score
                + 0.25 * token_name_score
                + 0.25 * address_score
            )

            # Keep plausible candidates
            if (
                name_score >= 0.30
                or token_name_score >= 0.35
                or address_score >= 0.50
            ):
                scored.append(
                    (
                        score,
                        c["entity_id"]
                    )
                )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        scored = scored[:limit]

        candidate_ids = [
            x[1]
            for x in scored
        ]

        rows.append(
            (
                r["entity_id"],
                ",".join(candidate_ids)
            )
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids"
        ]
    )


def generate_candidates(base):
    base = base.resolve()

    data = load_data(base)

    tr1 = prep(data["tr1"])
    tr2 = prep(data["tr2"])
    tr3 = prep(data["tr3"])

    te1 = prep(data["te1"])
    te2 = prep(data["te2"])
    te3 = prep(data["te3"])

    train_cp = candidate_pairs(
        tr1,
        tr2,
        tr3,
        limit=8
    )

    test_cp = candidate_pairs(
        te1,
        te2,
        te3,
        limit=8
    )

    output_dir = base / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_candidate_pairs.tsv"
    test_path = output_dir / "candidate_pairs.tsv"

    train_cp.to_csv(
        train_path,
        sep="\t",
        index=False
    )

    test_cp.to_csv(
        test_path,
        sep="\t",
        index=False
    )

    training_candidates = int(
        train_cp["candidate_entity_ids"]
        .map(
            lambda x: len(
                [
                    v
                    for v in str(x).split(",")
                    if v
                ]
            )
        )
        .sum()
    )

    test_candidates = int(
        test_cp["candidate_entity_ids"]
        .map(
            lambda x: len(
                [
                    v
                    for v in str(x).split(",")
                    if v
                ]
            )
        )
        .sum()
    )

    return {
        "message": "Candidate generation completed",
        "training_candidates": training_candidates,
        "test_candidates": test_candidates,
        "training_file": str(train_path),
        "test_file": str(test_path)
    }


# ============================================================
# TRUTH MAP
# ============================================================

def truth_map(truth):
    mapping = {}

    for _, row in truth.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        matched = str(
            row["matched_entity_ids"]
        ).strip()

        if not matched:
            mapping[source1_id] = set()
            continue

        mapping[source1_id] = set(
            x.strip()
            for x in matched.split(",")
            if x.strip()
        )

    return mapping


# ============================================================
# CANDIDATE FRAME
# ============================================================

def candidate_frame(s1, s2, s3, cp):

    all_candidates = pd.concat(
        [
            s2.assign(_src="S2"),
            s3.assign(_src="S3")
        ],
        ignore_index=True
    )

    lookup = {
        str(row["entity_id"]): row
        for _, row in all_candidates.iterrows()
    }

    cp_lookup = {}

    for _, row in cp.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        )

        candidate_ids = str(
            row["candidate_entity_ids"]
        )

        cp_lookup[source1_id] = [
            x.strip()
            for x in candidate_ids.split(",")
            if x.strip()
        ]

    rows = []

    for _, source1 in s1.iterrows():

        source1_id = str(
            source1["entity_id"]
        )

        ids = cp_lookup.get(
            source1_id,
            []
        )

        for candidate_id in ids:

            if candidate_id not in lookup:
                continue

            candidate = lookup[candidate_id].copy()

            candidate["_s1"] = source1_id

            rows.append(candidate)

    if not rows:

        return pd.DataFrame(
            columns=list(s2.columns) + [
                "_src",
                "_s1"
            ]
        )

    return pd.DataFrame(rows)


# ============================================================
# TF-IDF / FEATURE EXTRACTION
# ============================================================

def make_features(
    s1,
    candidates,
    truth_set=None,
    fit_vectorizers=False
):

    rows = []

    if candidates is None or len(candidates) == 0:
        return pd.DataFrame(
            columns=[
                "s1",
                "candidate"
            ] + FEATURES + ["label"]
        )

    s1 = s1.copy()
    candidates = candidates.copy()

    s1["name_clean"] = s1["name_clean"].fillna("")
    s1["address_clean"] = s1["address_clean"].fillna("")

    candidates["name_clean"] = (
        candidates["name_clean"]
        .fillna("")
    )

    candidates["address_clean"] = (
        candidates["address_clean"]
        .fillna("")
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    all_names = pd.concat(
        [
            s1["name_clean"],
            candidates["name_clean"]
        ],
        ignore_index=True
    ).astype(str)

    all_addresses = pd.concat(
        [
            s1["address_clean"],
            candidates["address_clean"]
        ],
        ignore_index=True
    ).astype(str)

    name_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1
    )

    address_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1
    )

    name_matrix = name_vectorizer.fit_transform(
        all_names
    )

    address_matrix = address_vectorizer.fit_transform(
        all_addresses
    )

    # Source-1 lookup
    s1_lookup = {
        str(row["entity_id"]): index
        for index, (_, row)
        in enumerate(s1.iterrows())
    }

    candidate_offset = len(s1)

    candidate_lookup = {
        str(row["entity_id"]): candidate_offset + index
        for index, (_, row)
        in enumerate(candidates.iterrows())
    }

    # --------------------------------------------------------
    # Pair features
    # --------------------------------------------------------

    for _, candidate in candidates.iterrows():

        source1_id = str(
            candidate["_s1"]
        )

        candidate_id = str(
            candidate["entity_id"]
        )

        if source1_id not in s1_lookup:
            continue

        if candidate_id not in candidate_lookup:
            continue

        source1_row = s1.iloc[
            s1_lookup[source1_id]
        ]

        i = s1_lookup[source1_id]
        j = candidate_lookup[candidate_id]

        name_cosine = float(
            name_matrix[i]
            .multiply(name_matrix[j])
            .sum()
        )

        address_cosine = float(
            address_matrix[i]
            .multiply(address_matrix[j])
            .sum()
        )

        name_value = str(
            source1_row["name_clean"]
        )

        candidate_name = str(
            candidate["name_clean"]
        )

        address_value = str(
            source1_row["address_clean"]
        )

        candidate_address = str(
            candidate["address_clean"]
        )

        name_ratio_value = (
            ratio(
                name_value,
                candidate_name
            ) / 100.0
        )

        name_token_value = (
            token_set_ratio(
                name_value,
                candidate_name
            ) / 100.0
        )

        address_ratio_value = (
            ratio(
                address_value,
                candidate_address
            ) / 100.0
        )

        address_token_value = (
            token_set_ratio(
                address_value,
                candidate_address
            ) / 100.0
        )

        name_length_diff = (
            abs(
                len(name_value)
                - len(candidate_name)
            )
            /
            max(
                1,
                len(name_value)
            )
        )

        address_length_diff = (
            abs(
                len(address_value)
                - len(candidate_address)
            )
            /
            max(
                1,
                len(address_value)
            )
        )

        label = None

        if truth_set is not None:

            true_matches = truth_set.get(
                source1_id,
                set()
            )

            label = int(
                candidate_id in true_matches
            )

        rows.append(
            {
                "s1": source1_id,
                "candidate": candidate_id,

                "name_ratio": name_ratio_value,

                "name_token_ratio": name_token_value,

                "address_ratio": address_ratio_value,

                "address_token_ratio": address_token_value,

                "name_tfidf": name_cosine,

                "address_tfidf": address_cosine,

                "country_match": int(
                    str(
                        source1_row["country_clean"]
                    )
                    ==
                    str(
                        candidate["country_clean"]
                    )
                ),

                "name_len_diff": name_length_diff,

                "address_len_diff": address_length_diff,

                "label": label
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_arrays(
    model,
    X,
    y
):

    if len(X) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f0_5": 0.0,
            "confusion_matrix": [
                [0.0, 0.0],
                [0.0, 0.0]
            ]
        }

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y,
        predictions,
        beta=0.5,
        zero_division=0
    )

    # Normalized confusion matrix
    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
        normalize="true"
    )

    cm = np.nan_to_num(
        cm,
        nan=0.0
    )

    return {
        "accuracy": round(
            float(accuracy),
            4
        ),

        "precision": round(
            float(precision),
            4
        ),

        "recall": round(
            float(recall),
            4
        ),

        "f0_5": round(
            float(f05),
            4
        ),

        "confusion_matrix": [
            [
                round(
                    float(value),
                    4
                )
                for value in row
            ]
            for row in cm
        ]
    }


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(base):

    base = base.resolve()

    data = load_data(base)

    tr1 = prep(data["tr1"])
    tr2 = prep(data["tr2"])
    tr3 = prep(data["tr3"])

    output_dir = base / "output"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_dir = base / "models"
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    candidate_file = (
        output_dir
        / "train_candidate_pairs.tsv"
    )

    # Automatically generate candidates
    # if they don't exist.
    if not candidate_file.exists():

        generate_candidates(base)

    cp = pd.read_csv(
        candidate_file,
        sep="\t",
        dtype=str
    ).fillna("")

    candidate_df = candidate_frame(
        tr1,
        tr2,
        tr3,
        cp
    )

    if len(candidate_df) == 0:

        raise ValueError(
            "No training candidate pairs were generated."
        )

    truth = truth_map(
        data["truth"]
    )

    features = make_features(
        tr1,
        candidate_df,
        truth
    )

    if len(features) == 0:

        raise ValueError(
            "Feature extraction returned zero rows."
        )

    X = features[
        FEATURES
    ].astype(float)

    y = features[
        "label"
    ].astype(int)

    groups = features[
        "s1"
    ].astype(str)

    # --------------------------------------------------------
    # Group split
    #
    # All candidate pairs belonging to the same Source-1
    # entity stay in either train or validation.
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_indices, val_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_train = X.iloc[
        train_indices
    ]

    X_val = X.iloc[
        val_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_val = y.iloc[
        val_indices
    ]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        model_dir
        / "entity_resolution_model.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES
        },
        model_path
    )

    # --------------------------------------------------------
    # Save validation dataset
    # --------------------------------------------------------

    validation_df = features.iloc[
        val_indices
    ].copy()

    validation_df.to_csv(
        output_dir
        / "validation_features.tsv",
        sep="\t",
        index=False
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    metrics = evaluate_arrays(
        model,
        X_val,
        y_val
    )

    with open(
        output_dir
        / "evaluation.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return {
        "message": "Model trained successfully",

        "training_pairs": int(
            len(X_train)
        ),

        "validation_pairs": int(
            len(X_val)
        ),

        "positive_training_pairs": int(
            y_train.sum()
        ),

        "positive_validation_pairs": int(
            y_val.sum()
        ),

        **metrics
    }


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(base, *args, **kwargs):

    base = base.resolve()

    model_path = (
        base
        / "models"
        / "entity_resolution_model.pkl"
    )

    validation_path = (
        base
        / "output"
        / "validation_features.tsv"
    )

    output_dir = (
        base
        / "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Please click "
            "'Train Model' before Evaluate."
        )

    # --------------------------------------------------------
    # Check validation data
    # --------------------------------------------------------

    if not validation_path.exists():

        raise FileNotFoundError(
            "Validation data not found. "
            "Please click 'Train Model' before Evaluate."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    package = joblib.load(
        model_path
    )

    model = package["model"]

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation_df = pd.read_csv(
        validation_path,
        sep="\t"
    ).fillna("")

    if len(validation_df) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    X_val = validation_df[
        FEATURES
    ].astype(float)

    y_val = validation_df[
        "label"
    ].astype(int)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_arrays(
        model,
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # Save evaluation
    # --------------------------------------------------------

    with open(
        output_dir
        / "evaluation.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    return metrics


# ============================================================
# TEST PREDICTION
# ============================================================

def predict_test(
    base,
    threshold=0.72
):

    base = base.resolve()

    data = load_data(base)

    te1 = prep(
        data["te1"]
    )

    te2 = prep(
        data["te2"]
    )

    te3 = prep(
        data["te3"]
    )

    output_dir = (
        base
        / "output"
    )

    model_path = (
        base
        / "models"
        / "entity_resolution_model.pkl"
    )

    candidate_path = (
        output_dir
        / "candidate_pairs.tsv"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Model not found. Please train the model first."
        )

    if not candidate_path.exists():

        generate_candidates(base)

    cp = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str
    ).fillna("")

    candidate_df = candidate_frame(
        te1,
        te2,
        te3,
        cp
    )

    package = joblib.load(
        model_path
    )

    model = package["model"]

    lookup = {}

    probabilities = []

    if len(candidate_df) > 0:

        features = make_features(
            te1,
            candidate_df
        )

        X = features[
            FEATURES
        ].astype(float)

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        features["probability"] = probabilities

        for _, row in features.iterrows():

            source1_id = row["s1"]

            candidate_id = row[
                "candidate"
            ]

            probability = float(
                row["probability"]
            )

            lookup.setdefault(
                source1_id,
                []
            ).append(
                (
                    candidate_id,
                    probability
                )
            )

    # --------------------------------------------------------
    # Build final output
    # --------------------------------------------------------

    rows = []

    for _, source1 in te1.iterrows():

        source1_id = str(
            source1["entity_id"]
        )

        candidates = lookup.get(
            source1_id,
            []
        )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        selected = []

        for candidate_id, probability in candidates:

            if probability >= threshold:

                selected.append(
                    candidate_id
                )

        # Remove duplicates
        selected = list(
            dict.fromkeys(
                selected
            )
        )

        rows.append(
            (
                source1_id,
                ",".join(selected)
            )
        )

    results = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    results_path = (
        output_dir
        / "matching_results.tsv"
    )

    results.to_csv(
        results_path,
        sep="\t",
        index=False
    )

    entities_with_match = sum(
        bool(value)
        for value
        in results["matched_entity_ids"]
    )

    total_matches = sum(
        len(
            [
                x
                for x in str(value).split(",")
                if x
            ]
        )
        for value
        in results["matched_entity_ids"]
    )

    average_confidence = (
        float(
            np.mean(
                probabilities
            )
        )
        if len(probabilities) > 0
        else 0.0
    )

    return {
        "message": "Test prediction completed",

        "source1_entities": int(
            len(results)
        ),

        "entities_with_match": int(
            entities_with_match
        ),

        "total_matches": int(
            total_matches
        ),

        "average_candidate_confidence": round(
            average_confidence,
            4
        ),

        "threshold": threshold,

        "results_file": str(
            results_path
        )
    }


# ============================================================
# PIPELINE STATUS
# ============================================================

def pipeline_status(base):

    base = base.resolve()

    output_dir = (
        base
        / "output"
    )

    train_dir = (
        base
        / "dataset"
        / "train"
    )

    model_dir = (
        base
        / "models"
    )

    def exists(path):
        return path.exists()

    return {
        "dataset": exists(
            train_dir
            / "train_source1.tsv"
        ),

        "preprocessed": exists(
            output_dir
            / "preprocessed_tr1.tsv"
        ),

        "candidates": exists(
            output_dir
            / "candidate_pairs.tsv"
        ),

        "model": exists(
            model_dir
            / "entity_resolution_model.pkl"
        ),

        "results": exists(
            output_dir
            / "matching_results.tsv"
        ),

        "validation": exists(
            output_dir
            / "validation_features.tsv"
        ),

        "evaluation": exists(
            output_dir
            / "evaluation.json"
        )
    }