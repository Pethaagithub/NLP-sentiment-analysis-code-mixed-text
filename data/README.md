# Data

This project uses the code-mixed Tamil-English and Tulu-English sentiment datasets released for
the **DravidianLangTech@NAACL 2025** shared task on sentiment analysis.

The raw dataset is **not redistributed in this repository**. To run the pipeline:

1. Register for / download the shared task data from the official competition page:
   https://codalab.lisn.upsaclay.fr/competitions/20893
2. Place the raw files under `data/raw/` using this naming convention:
   ```
   data/raw/tamil_train.csv
   data/raw/tamil_val.csv
   data/raw/tamil_test.csv
   data/raw/tulu_train.csv
   data/raw/tulu_val.csv
   data/raw/tulu_test.csv
   ```
3. Each raw CSV should contain at minimum:
   - `text` — the raw code-mixed comment/post
   - `label` — the sentiment label

   **Tamil labels:** `Positive`, `Negative`, `Mixed feelings`, `Unknown state`
   **Tulu labels:** `Positive`, `Negative`, `Mixed`, `Neutral`, `Not Tulu`

4. Run preprocessing to generate the cleaned files under `data/processed/`:
   ```bash
   python -m src.preprocessing --input data/raw/tamil_train.csv --output data/processed/tamil_train.csv --lang tamil
   ```

## Notes on script handling

- **Tamil** text in this dataset is predominantly **Romanized** (Latin script) mixed with English.
- **Tulu** text is predominantly written in **Kannada script** with embedded English terms.
- Preprocessing does **not** transliterate between scripts — original script is preserved to keep
  the linguistic signal authentic to how users actually write online.
