# MKN on WikiText-2

This folder is the first real-corpus step after the 12-sentence MKN simulator.
The goal is to keep the implementation readable while exposing the practical
choices that do not appear in the tiny hand-worked corpus.

## Flow

```text
    download_wikitext2.py loads WikiText-2 from Hugging Face once
    -> save the DatasetDict under data/wikitext-2/
    -> mkn_wikitext2.py loads only from that local directory
    -> read non-empty text rows from the train/validation splits
    -> keep prefixes for the first experiment
    -> add <s> and </s> to each row
    -> count unigrams, bigrams, and trigrams
    -> estimate order-specific D1, D2, and D3+
    -> calculate recursive MKN probabilities
    -> replace validation OOV words with <unk>
    -> calculate validation perplexity
```

## Why WikiText-2 before WikiText-103?

The WikiText-2 training split has 36,718 text rows, while WikiText-103 has
1,801,350. The smaller split lets us inspect counts, discounts, memory use,
and perplexity before dealing with the much larger count tables.

The official dataset also provides raw and non-raw variants. This experiment
uses the `wikitext-2-v1` non-raw word-level configuration. Its rows already use
`<unk>` for words outside the word vocabulary.

## Install the dataset dependency

The implementation uses the Hugging Face `datasets` package:

```bash
python3 -m pip install -r requirements.txt
```

The download script downloads the dataset and saves it locally with
`save_to_disk()`. No dataset archive is stored in this repository.

Run the download step once:

```bash
python3 download_wikitext2.py
```

This creates:

```text
data/wikitext-2/
```

## Run a small real-corpus experiment

Start with 5,000 training lines and 500 validation lines:

```bash
python3 mkn_wikitext2.py \
  --dataset-path data/wikitext-2
```

The training script reads only from the saved local dataset. It does not call
Hugging Face or download data.

Run all training and validation rows only after the smaller run is understood:

```bash
python3 mkn_wikitext2.py \
  --dataset-path data/wikitext-2 \
  --max-train-lines 0 \
  --max-valid-lines 0
```

The default dataset path is `data/wikitext-2`. You can override it explicitly:

```bash
python3 mkn_wikitext2.py --dataset-path /path/to/saved/wikitext-2
```

## Preprocessing assumptions

ASSUMPTION: Each non-empty Hugging Face WikiText `text` row is treated as one
modeling sequence.
We do not join across blank lines, because that would create artificial
n-grams between separate article sections.

ASSUMPTION: Whitespace tokenization is sufficient because the selected
configuration is already word-level and space-separated.

ASSUMPTION: `<s>` is added at the start and `</s>` at the end of every row.
`<s>` is never predicted; `</s>` is predicted and therefore remains in the
model vocabulary.

ASSUMPTION: Validation words unseen during training are mapped to `<unk>`.
Validation must not expand the training vocabulary.

## What to inspect first

1. Compare the number of training rows with the number of distinct unigrams,
   bigrams, and trigrams.
2. Inspect the separate bigram and trigram frequency-of-frequencies values.
3. Confirm that order 2 and order 3 have different discount triples.
4. Check that selected recursive probabilities are positive.
5. Read the validation perplexity only after understanding what tokens were
   included in its denominator.

## Files

- `download_wikitext2.py`: one-time Hugging Face download and local save step.
- `mkn_wikitext2.py`: readable MKN model, local dataset loader, and perplexity CLI.
- `requirements.txt`: Python dependency needed for the Hugging Face steps.
- `test_mkn_wikitext2.py`: deterministic tests using the earlier toy corpus.
- `PLAN.md`: implementation scope and verification plan.

## Verification

From this directory, after installing `requirements.txt`:

```bash
python3 -m unittest -v
```

The tests use synthetic data so they do not require a network connection or
downloaded corpus.

## Reference

The dataset card reports the split sizes and explains the raw/non-raw files:
[Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext).
