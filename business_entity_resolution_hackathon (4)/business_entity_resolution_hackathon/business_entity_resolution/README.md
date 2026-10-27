
# Business Entity Resolution — AIML Hackathon

College-level end-to-end prototype based on the supplied Business Entity Resolution challenge.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## Workflow

Dashboard → Generate Demo Dataset → Preprocess → Candidate Generation → Train Model → Evaluate → Test Prediction → Results.

The model uses character TF-IDF cosine similarity, fuzzy name/address similarities, country match, and a Random Forest classifier.

The test set has no labels. `matching_results.tsv` is the final prediction and `candidate_pairs.tsv` is the candidate set passed to the matching stage.
