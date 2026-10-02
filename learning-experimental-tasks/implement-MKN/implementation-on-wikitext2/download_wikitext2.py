"""Download WikiText-2 from Hugging Face and save it locally.

Flow:
    load Salesforce/wikitext from Hugging Face
        -> save the complete DatasetDict under data/wikitext-2
        -> let mkn_wikitext2.py load only from that local directory

The separate download step keeps network access out of model training.
"""

import argparse
from pathlib import Path


DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-v1"


def download_and_save(output_directory, dataset_loader=None):
    """Download WikiText-2 once and save its train/validation/test splits."""

    output_directory = Path(output_directory)
    saved_dataset_marker = output_directory / "dataset_dict.json"
    if saved_dataset_marker.exists():
        print(f"Dataset directory already exists: {output_directory}")
        print("Delete it manually only if you want to download a fresh copy.")
        return

    if output_directory.exists() and any(output_directory.iterdir()):
        raise RuntimeError(
            f"dataset directory is not empty but is not a saved DatasetDict: "
            f"{output_directory}"
        )

    if dataset_loader is None:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "downloading WikiText-2 requires the 'datasets' package; "
                "install it with: python3 -m pip install -r requirements.txt"
            ) from error
        dataset_loader = load_dataset

    print(f"Loading {DATASET_NAME} ({DATASET_CONFIG}) from Hugging Face")
    dataset = dataset_loader(DATASET_NAME, DATASET_CONFIG)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_directory))
    print(f"Saved WikiText-2 under {output_directory}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and locally save the Hugging Face WikiText-2 dataset."
    )
    parser.add_argument(
        "--output",
        default="data/wikitext-2",
        help="directory where the saved DatasetDict will be written",
    )
    args = parser.parse_args()
    download_and_save(args.output)


if __name__ == "__main__":
    main()
