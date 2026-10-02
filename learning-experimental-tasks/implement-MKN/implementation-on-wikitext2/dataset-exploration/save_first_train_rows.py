"""Save the first non-empty WikiText-2 training rows as a text file.

Flow:
    locally saved DatasetDict
        -> load the train split
        -> keep non-empty text rows
        -> take the first requested number of rows
        -> write one dataset row per output line
"""

import argparse
from pathlib import Path


def get_first_non_empty_rows(dataset, limit):
    """Return the first `limit` non-empty text rows from a dataset split."""

    rows = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            rows.append(text)
        if len(rows) == limit:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Save a prefix of the WikiText-2 training split."
    )
    parser.add_argument(
        "--dataset-path",
        default="../data/wikitext-2",
        help="path created by download_wikitext2.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="number of non-empty training rows to save",
    )
    parser.add_argument(
        "--output",
        default="first-5000-train-lines.txt",
        help="output text file",
    )
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "install the datasets package with: "
            "python3 -m pip install -r ../requirements.txt"
        ) from error

    dataset_dict = load_from_disk(args.dataset_path)
    rows = get_first_non_empty_rows(dataset_dict["train"], args.limit)

    if len(rows) < args.limit:
        print(
            f"WARNING: requested {args.limit} rows, but only "
            f"{len(rows)} non-empty training rows were available."
        )

    output_path = Path(args.output)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"saved rows: {len(rows)}")
    print(f"output file: {output_path}")


if __name__ == "__main__":
    main()
