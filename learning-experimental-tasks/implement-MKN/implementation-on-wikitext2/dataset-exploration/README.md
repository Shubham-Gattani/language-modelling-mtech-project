# WikiText-2 Dataset Exploration

These scripts inspect the locally saved WikiText-2 dataset before we use it for
MKN training.

## Run from this folder

```bash
cd learning-experimental-tasks/implement-MKN/implementation-on-wikitext2/dataset-exploration
source ../venv/bin/activate
```

The scripts expect the saved dataset at:

```text
../data/wikitext-2
```

## Save the first 5,000 training rows

```bash
python3 save_first_train_rows.py
```

This creates:

```text
first-5000-train-lines.txt
```

The file contains the first 5,000 non-empty `text` rows from the training
split. This matches the row-selection behavior used by the MKN experiment.

To choose a different prefix or output file:

```bash
python3 save_first_train_rows.py \
  --limit 1000 \
  --output first-1000-train-lines.txt
```

## Verify `<unk>`

Check the first 5,000 non-empty training rows:

```bash
python3 check_unk_token.py --split train --max-lines 5000
```

Check the complete training split:

```bash
python3 check_unk_token.py --split train --max-lines 0
```

Check all rows in the validation split:

```bash
python3 check_unk_token.py --split validation --max-lines 0
```

The script reports:

- how many rows were inspected;
- how many rows contain `<unk>`;
- the total number of `<unk>` tokens;
- the first inspected row containing `<unk>`.

## Explore all splits

```bash
python3 explore_dataset.py
```

This prints and saves `dataset-summary.json` with statistics for:

- train;
- validation;
- test;
- the first 5,000 non-empty training rows.

The summary includes raw and non-empty row counts, total token counts, number
of distinct tokens, row-length statistics, `<unk>` counts, and the 20 most
common tokens in each split.

## Why these checks matter

Before training MKN, we want to know:

```text
How many rows will be used?
How many tokens are present?
How many distinct token types exist?
How long are the rows?
Does <unk> exist in the actual training prefix?
Are train, validation, and test similar in basic size and vocabulary behavior?
```

The scripts do not modify the saved Hugging Face dataset. They only read it
and create exploration outputs in this folder.
