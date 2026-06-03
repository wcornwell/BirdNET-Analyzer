#!/usr/bin/env python3
"""Convert per-species embeddings .npz to training cache format.

Usage:
    python embedding_analysis/npz_to_cache.py \
        --input embedding_analysis/reallybig_pelican0-9_embeddings.npz \
        --output embedding_analysis/reallybig_train_cache.npz
"""

import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Per-species embeddings .npz")
    parser.add_argument("--output", required=True, help="Training cache .npz output path")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Hold out this fraction as x_test/y_test (default: 0, use training val_split)")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    species_list = data["species_list"].tolist()

    # Build label list (common names, matching the folder-name convention)
    labels = np.array(species_list, dtype=object)
    n_classes = len(species_list)

    all_x, all_y = [], []
    for i, sp in enumerate(species_list):
        embs = data[f"emb_{sp}"]          # (N, embedding_dim)
        n = len(embs)
        y = np.zeros((n, n_classes), dtype="float32")
        y[:, i] = 1.0
        all_x.append(embs)
        all_y.append(y)

    x = np.concatenate(all_x, axis=0).astype("float32")
    y = np.concatenate(all_y, axis=0).astype("float32")

    # Shuffle
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]

    if args.val_split > 0:
        split = int(len(x) * (1 - args.val_split))
        x_train, y_train = x[:split], y[:split]
        x_test, y_test = x[split:], y[split:]
    else:
        x_train, y_train = x, y
        x_test = np.array([], dtype="float32")
        y_test = np.array([], dtype="float32")

    np.savez(
        args.output,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        labels=labels,
        binary_classification=False,
        multi_label=False,
    )

    print(f"Saved: {args.output}")
    print(f"  {len(species_list)} species, {len(x_train)} train clips"
          + (f", {len(x_test)} test clips" if len(x_test) > 0 else ""))


if __name__ == "__main__":
    main()
