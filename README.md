# Multilingual Text Normalization & Sentiment Pipeline

A sentiment classification pipeline for **code-mixed Tamil-English** and **Tulu-English** social media text.
The pipeline normalizes noisy, multi-script, code-mixed input and benchmarks multiple embedding
strategies (TF-IDF, mBERT, MuRIL, IndicBERT, IndicFastText, XLM-RoBERTa) against classical deep
learning architectures (CNN, RNN, LSTM, BiLSTM, GRU) and fine-tuned transformer models.

> Based on data and methodology from the **DravidianLangTech@NAACL 2025** shared task on
> sentiment analysis in Tamil and Tulu code-mixed text.

## What this project does

1. **Normalizes** noisy, script-diverse, code-mixed text (Romanized Tamil, Kannada-script Tulu,
   embedded English) into a clean, structured input format — without transliterating or stripping
   script authenticity.
2. **Embeds** the normalized text using multiple strategies, from sparse (TF-IDF) to contextual
   multilingual transformer embeddings (mBERT, MuRIL, IndicBERT, IndicFastText, XLM-RoBERTa).
3. **Trains** both classical deep learning models (CNN / RNN / LSTM / BiLSTM / GRU) on top of
   embeddings, and fine-tunes transformer models end-to-end.
4. **Evaluates** every configuration on precision, recall, macro F1, and accuracy, and produces a
   comparison table + confusion matrices for both languages.

## Project structure

```
code-mixed-sentiment-pipeline/
├── configs/
│   └── config.yaml            # central config: paths, hyperparameters, model/embedding choices
├── data/
│   ├── README.md              # instructions for obtaining the dataset (not redistributed here)
│   └── (place raw/processed CSVs here)
├── src/
│   ├── preprocessing.py       # text normalization / cleaning pipeline
│   ├── datasets.py            # PyTorch Dataset classes + train/val/test loading
│   ├── embeddings.py          # TF-IDF + transformer embedding extraction
│   ├── models/
│   │   ├── dl_models.py       # CNN / RNN / LSTM / BiLSTM / GRU classifier heads
│   │   └── transformer_models.py  # fine-tuning wrapper (mBERT, MuRIL, IndicBERT, XLM-R)
│   ├── train.py                # generic training loop (DL models on frozen embeddings)
│   ├── evaluate.py             # metrics: precision/recall/F1/accuracy + confusion matrix
│   └── utils.py                 # seeding, logging, misc helpers
├── scripts/
│   ├── run_tfidf_baseline.py   # TF-IDF + classical ML baselines (LogReg, SVM, XGBoost)
│   ├── train_dl_model.py       # train a DL model (CNN/RNN/LSTM/BiLSTM/GRU) on chosen embedding
│   └── finetune_transformer.py # fine-tune a transformer end-to-end (mBERT/MuRIL/IndicBERT/XLM-R)
├── outputs/                     # metrics, logs, confusion matrices, saved checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

This repository does **not** include the DravidianLangTech dataset directly, since it is
distributed under the shared task's own terms. See `data/README.md` for instructions on
obtaining it and the expected file format.

Expected columns after preprocessing: `text`, `label`.

## Usage

**1. Preprocess raw data**
```bash
python -m src.preprocessing --input data/raw/tamil_train.csv --output data/processed/tamil_train.csv --lang tamil
```

**2. TF-IDF + classical ML baseline**
```bash
python scripts/run_tfidf_baseline.py --lang tamil
```

**3. Train a deep learning model on top of an embedding**
```bash
python scripts/train_dl_model.py --lang tamil --embedding indicbert --model bilstm --epochs 10
```

**4. Fine-tune a transformer end-to-end**
```bash
python scripts/finetune_transformer.py --lang tulu --model mbert --epochs 10
```

All runs write metrics (precision, recall, macro F1, accuracy) and a confusion matrix to
`outputs/<lang>_<embedding_or_model>_<timestamp>/`.

## Results (reference, from original study)

| Language | Best Config              | Macro F1 | Accuracy |
|----------|---------------------------|----------|----------|
| Tamil    | Fine-tuned XLM-RoBERTa     | 0.50     | 0.65     |
| Tulu     | Fine-tuned mBERT           | 0.57     | 0.69     |

## Notes

- Preprocessing intentionally preserves script (Romanized Tamil, Kannada-script Tulu) — no
  transliteration is performed, to keep the linguistic signal authentic.
- Tulu has no dedicated pretrained word embedding; multilingual transformers (mBERT, MuRIL) are
  used as the closest available approximation.
- This is a classification pipeline built on **encoder-based transformer models** (BERT-family),
  not a generative LLM / RAG pipeline.

## License

MIT — see `LICENSE`. Dataset usage is governed separately by the DravidianLangTech shared task terms.
