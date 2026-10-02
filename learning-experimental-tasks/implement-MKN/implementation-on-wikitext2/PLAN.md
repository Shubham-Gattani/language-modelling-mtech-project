# Plan: MKN on WikiText-2

## Goal

Run the readable Modified Kneser-Ney implementation on a real word-level corpus
without hiding preprocessing, count construction, discount estimation, or
perplexity calculation behind a framework.

## Scope

- Download the standard WikiText-2 word-level configuration from Hugging Face
  once and save it locally.
- Load the saved dataset from disk during model training.
- Read one non-empty `text` row as one modeling sequence.
- Train an order-3 interpolated MKN model.
- Estimate `D1`, `D2`, and `D3+` separately for bigrams and trigrams.
- Evaluate validation perplexity with the training vocabulary.
- Start with bounded line prefixes before attempting the full split.

## Not in scope yet

- WikiText-103.
- Faster array or database-backed count storage.
- Sentence segmentation beyond the corpus's existing line boundaries.
- Hyperparameter search or comparison against neural language models.

## Verification

1. Run the toy-data unit tests in this folder.
2. Install the `datasets` dependency from `requirements.txt`.
3. Run `download_wikitext2.py` once to save `data/wikitext-2/`.
4. Run the default bounded experiment from the saved local dataset.
5. Inspect counts, discounts, OOV handling, perplexity, and normalization.
6. Only then try `--max-train-lines 0 --max-valid-lines 0`.
