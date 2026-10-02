"""Check whether a saved WikiText-2 split contains the <unk> token.

Flow:
    saved DatasetDict
        -> load one split
        -> remove empty rows, matching the main program
        -> inspect the complete split
        -> inspect the selected prefix, such as the first 5,000 rows
        -> report row counts, token counts, and the first occurrence
"""

import argparse


UNK_TOKEN = "<unk>"


def inspect_rows(rows, max_lines):
    """Return simple <unk> statistics for a list of text rows.

    A limit of zero means that all non-empty rows are inspected.
    """

    non_empty_rows = []
    for row in rows:
        text = row["text"].strip()
        if text:
            non_empty_rows.append(text)

    if max_lines == 0:
        selected_rows = non_empty_rows
    else:
        selected_rows = non_empty_rows[:max_lines]

    rows_with_unk = 0
    unk_token_count = 0
    first_row_with_unk = None

    for row_number, text in enumerate(selected_rows):
        tokens = text.split()
        occurrences = tokens.count(UNK_TOKEN)
        if occurrences > 0:
            rows_with_unk += 1
            unk_token_count += occurrences
            if first_row_with_unk is None:
                first_row_with_unk = row_number

    return {
        "available_non_empty_rows": len(non_empty_rows),
        "inspected_rows": len(selected_rows),
        "rows_with_unk": rows_with_unk,
        "unk_token_count": unk_token_count,
        "first_inspected_row_with_unk": first_row_with_unk,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check <unk> in a locally saved WikiText-2 split."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/wikitext-2",
        help="path created by download_wikitext2.py",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
        help="dataset split to inspect",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="number of non-empty rows to inspect; use 0 for all rows",
    )
    args = parser.parse_args()

    if args.max_lines < 0:
        raise ValueError("--max-lines must be zero or positive")

    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError(
            "install the datasets package with: "
            "python3 -m pip install -r requirements.txt"
        ) from error

    dataset = load_from_disk(args.dataset_path)
    stats = inspect_rows(dataset[args.split], args.max_lines)

    print(f"split: {args.split}")
    print(f"non-empty rows available: {stats['available_non_empty_rows']}")
    print(f"rows inspected: {stats['inspected_rows']}")
    print(f"rows containing {UNK_TOKEN}: {stats['rows_with_unk']}")
    print(f"total {UNK_TOKEN} tokens: {stats['unk_token_count']}")
    print(
        "first inspected row containing <unk>: "
        f"{stats['first_inspected_row_with_unk']}"
    )

    if stats["unk_token_count"] == 0:
        print(
            "RESULT: <unk> was not found in the inspected rows. "
            "The current evaluation code cannot map unknown words safely "
            "unless <unk> is in the model vocabulary."
        )
    else:
        print(
            "RESULT: <unk> is present in the inspected rows, so unknown "
            "validation words can be mapped to this vocabulary token."
        )


if __name__ == "__main__":
    main()
