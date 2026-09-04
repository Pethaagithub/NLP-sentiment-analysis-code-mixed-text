"""
Train a classical DL model (CNN/RNN/LSTM/BiLSTM/GRU) on top of a chosen
embedding strategy (TF-IDF or a frozen transformer).

Usage:
    python scripts/train_dl_model.py --lang tamil --embedding indicbert --model bilstm --epochs 10
    python scripts/train_dl_model.py --lang tulu --embedding tfidf --model cnn --epochs 10
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets import fit_label_encoder, load_split
from src.embeddings import TransformerEmbedder, fit_tfidf, tfidf_transform
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics
from src.models.dl_models import build_model
from src.train import evaluate_dl_model, train_dl_model
from src.utils import get_device, load_config, make_run_dir, set_seed
from torch.utils.data import DataLoader
from src.datasets import EmbeddingDataset


def get_embeddings(embedding_name: str, cfg: dict, train_texts, val_texts):
    if embedding_name == "tfidf":
        tfidf_cfg = cfg["embeddings"]["tfidf"]
        vectorizer = fit_tfidf(
            train_texts, max_features=tfidf_cfg["max_features"], ngram_range=tuple(tfidf_cfg["ngram_range"])
        )
        return tfidf_transform(vectorizer, train_texts), tfidf_transform(vectorizer, val_texts)

    embedder = TransformerEmbedder(embedding_name, max_length=cfg["embeddings"]["max_seq_length"])
    return embedder.embed(train_texts), embedder.embed(val_texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["tamil", "tulu"])
    parser.add_argument("--embedding", required=True,
                         choices=["tfidf", "mbert", "muril", "indicbert", "xlm_roberta"])
    parser.add_argument("--model", required=True, choices=["cnn", "rnn", "lstm", "bilstm", "gru"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    processed_dir = cfg["data"]["processed_dir"]
    train_df = load_split(processed_dir, args.lang, "train")
    val_df = load_split(processed_dir, args.lang, "validation")

    label_encoder = fit_label_encoder(train_df)
    y_train = label_encoder.transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    num_classes = len(label_encoder.classes_)

    print(f"Extracting '{args.embedding}' embeddings for {args.lang}...")
    X_train, X_val = get_embeddings(args.embedding, cfg, train_df["text"].tolist(), val_df["text"].tolist())

    dl_cfg = cfg["dl_models"]
    model = build_model(
        args.model,
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=dl_cfg["hidden_dim"],
        num_layers=dl_cfg["num_layers"],
        dropout=dl_cfg["dropout"],
    )

    epochs = args.epochs or dl_cfg["epochs"]
    model = train_dl_model(
        model, X_train, y_train, X_val, y_val, device,
        epochs=epochs, batch_size=dl_cfg["batch_size"], lr=dl_cfg["learning_rate"],
    )

    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=dl_cfg["batch_size"])
    metrics, y_true, y_pred = evaluate_dl_model(model, val_loader, device)
    print(f"\nFinal validation metrics: {metrics}")

    run_dir = make_run_dir(cfg["output_dir"], f"{args.lang}_{args.embedding}_{args.model}")
    save_metrics(metrics, run_dir)
    plot_confusion_matrix(y_true, y_pred, list(label_encoder.classes_), run_dir)
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
