"""Report the tokenized length distribution of the notes and how much text each max_length cuts.

Prints aggregate statistics only (percentiles and truncation rates), never note text.

    python scripts/token_lengths.py top50_base
    python scripts/token_lengths.py top50_base --max-lengths 2048 4096 8192 --sample 20000
"""

import argparse

import numpy as np
from transformers import AutoTokenizer

from icd10.config import EXPERIMENTS, get_config
from icd10.data.dataset import load_or_build_dataset

PERCENTILES = (5, 25, 50, 75, 90, 95, 99)
BATCH_SIZE = 1000  # tokenize in batches so only lengths, not token IDs, are kept in memory


def token_lengths(texts: list[str], tokenizer) -> np.ndarray:
    lengths = []
    for start in range(0, len(texts), BATCH_SIZE):
        encoded = tokenizer(texts[start : start + BATCH_SIZE])["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return np.array(lengths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS, help="Experiment defined in icd10/config.py")
    parser.add_argument("--max-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 6144, 8192])
    parser.add_argument("--sample", type=int, default=0, help="Tokenize a random sample of N notes (0 = all).")
    args = parser.parse_args()
    config = get_config(args.experiment)

    df = load_or_build_dataset(config)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"])
    lengths = token_lengths(list(df["text"]), tokenizer)

    print(f"Tokenizer: {config['model']['checkpoint']}; notes: {len(lengths)}")
    print(f"Mean tokens: {lengths.mean():.0f}; max: {lengths.max()}")
    for p in PERCENTILES:
        print(f"  p{p:<3} {np.percentile(lengths, p):>8.0f}")
    print("\nmax_length  notes truncated  tokens lost")
    for max_length in sorted(set(args.max_lengths) | {config["data"]["max_length"]}):
        truncated = (lengths > max_length).mean()
        lost = np.maximum(lengths - max_length, 0).sum() / lengths.sum()
        marker = "  <- config" if max_length == config["data"]["max_length"] else ""
        print(f"{max_length:>10}  {truncated:>15.1%}  {lost:>11.1%}{marker}")


if __name__ == "__main__":
    main()
