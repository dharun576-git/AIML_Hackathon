
# Business Entity Resolution — Methodology

## 1. Methodology Used
The pipeline normalizes business names and addresses, generates plausible candidate pairs, computes fuzzy and TF-IDF similarity features, and uses a Random Forest classifier to predict matches.

## 2. Candidate Generation / Blocking
Candidates are restricted using country agreement and fuzzy name/address similarity. The final candidate list is stored in `output/candidate_pairs.tsv`.

## 3. Model Architecture
A Random Forest classifier uses name similarity, address similarity, character TF-IDF cosine similarity, country match, and length-difference features.

## 4. Evaluation
Training candidates are split into training and validation subsets. Accuracy, precision, recall and F0.5 are reported.

## 5. Test Inference
The test set has no labels. The trained model scores candidate pairs and predictions above the selected confidence threshold are exported to `output/matching_results.tsv`.

## 6. Reproducibility
Run `pip install -r requirements.txt`, then `python app.py`. Use the dashboard workflow to regenerate data and outputs.
