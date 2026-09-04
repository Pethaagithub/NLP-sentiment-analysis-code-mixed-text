"""
Text normalization pipeline for code-mixed Tamil-English and Tulu-English data.

Design goals (matching the project methodology):
  - Preserve original script (Romanized Tamil, Kannada-script Tulu) — no transliteration.
  - Lowercase only Latin-script tokens (Tamil/Tulu don't have a case distinction).
  - Strip noise that carries no sentiment signal: URLs, @mentions, '#' symbols, emojis.
  - Collapse repeated punctuation ("!!!" -> "!") while keeping the punctuation itself,
    since exclamation/question marks carry sentiment cues.
  - Drop rows with null/missing labels.
"""

import argparse
import re

import pandas as pd

try:
    import emoji
except ImportError:  # pragma: no cover
    emoji = None

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#(\w+)")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,])\1{1,}")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub(" ", text)


def strip_hashtag_symbol(text: str) -> str:
    """Keep the word inside a hashtag, drop the '#' symbol itself."""
    return HASHTAG_SYMBOL_PATTERN.sub(r"\1", text)


def remove_emojis(text: str) -> str:
    if emoji is None:
        return text
    return emoji.replace_emoji(text, replace=" ")


def collapse_repeated_punctuation(text: str) -> str:
    """'!!!' -> '!', '???' -> '?', but leaves single punctuation untouched."""
    return REPEATED_PUNCT_PATTERN.sub(r"\1", text)


def lowercase_latin_tokens(text: str) -> str:
    """
    Lowercase only tokens composed of Latin characters, leaving Tamil / Kannada
    script tokens untouched (they have no case distinction to normalize).
    """
    tokens = text.split()
    normalized = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z0-9.,!?'\"()\-]+", tok):
            normalized.append(tok.lower())
        else:
            normalized.append(tok)
    return " ".join(normalized)


def clean_text(text: str, cfg: dict) -> str:
    if not isinstance(text, str):
        return ""

    if cfg.get("remove_urls", True):
        text = remove_urls(text)
    if cfg.get("remove_mentions", True):
        text = remove_mentions(text)
    if cfg.get("remove_hashtags_symbol", True):
        text = strip_hashtag_symbol(text)
    if cfg.get("remove_emojis", True):
        text = remove_emojis(text)
    if cfg.get("lowercase_english_only", True):
        text = lowercase_latin_tokens(text)
    if cfg.get("collapse_repeated_punctuation", True):
        text = collapse_repeated_punctuation(text)

    text = MULTISPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame, cfg: dict, drop_null_labels: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].apply(lambda t: clean_text(t, cfg))

    if drop_null_labels:
        df = df.dropna(subset=["label"])

    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess code-mixed sentiment data.")
    parser.add_argument("--input", required=True, help="Path to raw CSV with 'text' and 'label' columns.")
    parser.add_argument("--output", required=True, help="Path to write the cleaned CSV.")
    parser.add_argument("--lang", required=True, choices=["tamil", "tulu"], help="Language of the input file.")
    args = parser.parse_args()

    cfg = {
        "remove_urls": True,
        "remove_mentions": True,
        "remove_hashtags_symbol": True,
        "remove_emojis": True,
        "lowercase_english_only": True,
        "collapse_repeated_punctuation": True,
    }

    df = pd.read_csv(args.input)
    cleaned = preprocess_dataframe(df, cfg)
    cleaned.to_csv(args.output, index=False)
    print(f"[{args.lang}] Preprocessed {len(df)} -> {len(cleaned)} rows. Saved to {args.output}")


if __name__ == "__main__":
    main()
