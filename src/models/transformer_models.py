"""
End-to-end transformer fine-tuning for sentiment classification.

Wraps AutoModelForSequenceClassification for mBERT / MuRIL / IndicBERT / XLM-RoBERTa,
so the encoder weights are updated jointly with a classification head, rather than
used as frozen feature extractors (contrast with src/embeddings.py).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINTS = {
    "mbert": "bert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlm_roberta": "xlm-roberta-large",
}


def load_tokenizer_and_model(model_key: str, num_labels: int):
    if model_key not in CHECKPOINTS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(CHECKPOINTS)}")

    checkpoint = CHECKPOINTS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    return tokenizer, model
