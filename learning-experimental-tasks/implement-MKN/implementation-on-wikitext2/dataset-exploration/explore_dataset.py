"""Explore the saved WikiText-2 train, validation, and test splits.

Flow:
    locally saved DatasetDict
        -> inspect train, validation, and test rows
        -> count empty rows and tokens
        -> count distinct tokens and <unk> occurrences
        -> measure row lengths
        -> inspect the first 5,000 non-empty train rows
        -> print and save a JSON summary
"""

import argparse
import json
from pathlib import Path


UNK_TOKEN = "<unk>"


def summarize_rows(rows):
    """Return useful size, vocabulary, length, and `<unk>` statistics."""

    raw_row_count = 0
    empty_row_count = 0
    non_empty_row_count = 0
    total_token_count = 0
    rows_with_unk = 0
    unk_token_count = 0
    token_counts = {}
    row_lengths = []

    for row in rows:
        raw_row_count += 1
        text = row["text"].strip()
        if not text:
            empty_row_count += 1
            continue

        non_empty_row_count += 1
        tokens = text.split()
        row_lengths.append(len(tokens))
        total_token_count += len(tokens)

        current_unk_count = tokens.count(UNK_TOKEN)
        if current_unk_count > 0:
            rows_with_unk += 1
            unk_token_count += current_unk_count

        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

    if row_lengths:
        average_row_length = total_token_count / len(row_lengths)
        shortest_row = min(row_lengths)
        longest_row = max(row_lengths)
    else:
        average_row_length = 0.0
        shortest_row = 0
        longest_row = 0

    most_common_tokens = sorted(
        token_counts.items(), key=lambda item: (-item[1], item[0])
    )[:20]

    return {
        "raw_row_count": raw_row_count,
        "empty_row_count": empty_row_count,
        "non_empty_row_count": non_empty_row_count,
        "total_token_count": total_token_count,
        "distinct_token_count": len(token_counts),
        "shortest_row_tokens": shortest_row,
        "longest_row_tokens": longest_row,
        "average_row_tokens": average_row_length,
        "rows_with_unk": rows_with_unk,
        "unk_token_count": unk_token_count,
        "most_common_tokens": [
            {"token": token, "count": count}
            for token, count in most_common_tokens
        ],
    }


def get_first_non_empty_rows(dataset, limit):
    """Return the first `limit` non-empty rows from one split."""

    rows = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            rows.append({"text": text})
        if len(rows) == limit:
            break
    return rows


def print_split_summary(split_name, summary):
    """Print the important statistics for one split."""

    print(f"[{split_name}]")
    print(f"raw rows: {summary['raw_row_count']}")
    print(f"empty rows: {summary['empty_row_count']}")
    print(f"non-empty rows: {summary['non_empty_row_count']}")
    print(f"total tokens: {summary['total_token_count']}")
    print(f"distinct tokens: {summary['distinct_token_count']}")
    print(f"average row length: {summary['average_row_tokens']:.2f} tokens")
    print(f"shortest row: {summary['shortest_row_tokens']} tokens")
    print(f"longest row: {summary['longest_row_tokens']} tokens")
    print(f"rows containing <unk>: {summary['rows_with_unk']}")
    print(f"total <unk> tokens: {summary['unk_token_count']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Explore all locally saved WikiText-2 splits."
    )
    parser.add_argument(
        "--dataset-path",
        default="../data/wikitext-2",
        help="path created by download_wikitext2.py",
    )
    parser.add_argument(
        "--output",
        default="dataset-summary.json",
        help="where to save the JSON summary",
    )
    parser.add_argument(
        "--train-prefix-size",
        type=int,
        default=5000,
        help="number of non-empty training rows to inspect separately",
    )
    args = parser.parse_args()

    if args.train_prefix_size <= 0:
        raise ValueError("--train-prefix-size must be greater than zero")

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "install the datasets package with: "
            "python3 -m pip install -r ../requirements.txt"
        ) from error

    dataset_dict = load_from_disk(args.dataset_path)
    summary = {}

    for split_name in ("train", "validation", "test"):
        summary[split_name] = summarize_rows(dataset_dict[split_name])

    train_prefix_rows = get_first_non_empty_rows(
        dataset_dict["train"], args.train_prefix_size
    )
    summary["train_prefix"] = {
        "requested_non_empty_rows": args.train_prefix_size,
        "statistics": summarize_rows(train_prefix_rows),
    }

    for split_name in ("train", "validation", "test"):
        print_split_summary(split_name, summary[split_name])

    print(f"[first {args.train_prefix_size} non-empty train rows]")
    print_split_summary("train_prefix", summary["train_prefix"]["statistics"])

    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"JSON summary: {output_path}")


if __name__ == "__main__":
    main()
