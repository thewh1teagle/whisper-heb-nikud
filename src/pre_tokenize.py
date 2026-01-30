"""
Prepare and cache the dataset with progress bars.
Example:
uv run src/pre_tokenize.py --data_dir dataset/ --model_name ivrit-ai/whisper-large-v3-turbo --dataset_cache_path ./dataset_cache
"""

from pathlib import Path

import datasets
from transformers import WhisperProcessor

from config import get_args
from data import load_dataset_from_csv, preprocess_dataset


def main():
    args = get_args()
    cache_path = Path(args.dataset_cache_path)

    if cache_path.exists():
        print(f"Cache already exists at {cache_path}. Delete it to rebuild.")
        return

    datasets.enable_progress_bar()

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language="Hebrew",
        task="transcribe",
    )

    dataset = load_dataset_from_csv(args.data_dir)
    dataset = preprocess_dataset(
        dataset,
        processor,
        batch_size=args.batch_size,
    )

    dataset.save_to_disk(cache_path)
    print(f"Saved pre-tokenized dataset to {cache_path}")


if __name__ == "__main__":
    main()
