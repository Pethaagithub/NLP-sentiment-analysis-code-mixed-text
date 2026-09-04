"""
TF-IDF + classical ML baselines (Logistic Regression, SVM, XGBoost).

Usage:
    python scripts/run_tfidf_baseline.py --lang tamil
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.datasets import fit_label_encoder, load_split
from src.embeddings import fit_tfidf, tfidf_transform
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics
from src.utils import load_config, make_run_dir, set_seed

CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": SVC(kernel="linear"),
    "xgboost": XGBClassifier(eval_metric="mlogloss"),
    "random_forest": RandomForestClassifier(n_estimators=200),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["tamil", "tulu"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    processed_dir = cfg["data"]["processed_dir"]
    train_df = load_split(processed_dir, args.lang, "train")
    val_df = load_split(processed_dir, args.lang, "validation")

    label_encoder = fit_label_encoder(train_df)
    y_train = label_encoder.transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])

    tfidf_cfg = cfg["embeddings"]["tfidf"]
    vectorizer = fit_tfidf(
        train_df["text"].tolist(),
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
    )
    X_train = tfidf_transform(vectorizer, train_df["text"].tolist())
    X_val = tfidf_transform(vectorizer, val_df["text"].tolist())

    for name, clf in CLASSIFIERS.items():
        print(f"\n=== {args.lang} | TF-IDF + {name} ===")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        metrics = compute_metrics(y_val, preds)
        print(metrics)

        run_dir = make_run_dir(cfg["output_dir"], f"{args.lang}_tfidf_{name}")
        save_metrics(metrics, run_dir)
        plot_confusion_matrix(y_val, preds, list(label_encoder.classes_), run_dir)


if __name__ == "__main__":
    main()
