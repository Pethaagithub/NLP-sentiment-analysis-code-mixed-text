"""
End-to-end fine-tuning of a multilingual transformer (mBERT / MuRIL / IndicBERT /
XLM-RoBERTa) for sentiment classification, using Hugging Face's Trainer API.

Usage:
    python scripts/finetune_transformer.py --lang tulu --model mbert --epochs 10
    python scripts/finetune_transformer.py --lang tamil --model xlm_roberta --epochs 10
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from transformers import Trainer, TrainingArguments

from src.datasets import TransformerTextDataset, fit_label_encoder, load_split
from src.evaluate import compute_metrics, plot_confusion_matrix, save_metrics
from src.models.transformer_models import load_tokenizer_and_model
from src.utils import load_config, make_run_dir, set_seed


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_metrics(labels, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, choices=["tamil", "tulu"])
    parser.add_argument("--model", required=True, choices=["mbert", "muril", "indicbert", "xlm_roberta"])
    parser.add_argument("--epochs", type=int, default=None)
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
    num_classes = len(label_encoder.classes_)

    tokenizer, model = load_tokenizer_and_model(args.model, num_labels=num_classes)

    max_len = cfg["embeddings"]["max_seq_length"]
    train_dataset = TransformerTextDataset(train_df["text"].tolist(), y_train, tokenizer, max_length=max_len)
    val_dataset = TransformerTextDataset(val_df["text"].tolist(), y_val, tokenizer, max_length=max_len)

    ft_cfg = cfg["transformer_finetuning"]
    epochs = args.epochs or ft_cfg["epochs"]
    run_dir = make_run_dir(cfg["output_dir"], f"{args.lang}_finetune_{args.model}")

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=ft_cfg["batch_size"],
        per_device_eval_batch_size=ft_cfg["batch_size"],
        learning_rate=ft_cfg["learning_rate"],
        weight_decay=ft_cfg["weight_decay"],
        warmup_ratio=ft_cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=os.path.join(run_dir, "logs"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=hf_compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"\nFinal validation metrics: {eval_result}")

    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)

    metrics = compute_metrics(y_val, y_pred)
    save_metrics(metrics, run_dir)
    plot_confusion_matrix(y_val, y_pred, list(label_encoder.classes_), run_dir)
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
