#!/usr/bin/env python3
"""Identify potential misclassifications based on embedding distances.

Usage:
    python embedding_analysis/identify_misclassifications.py \
        --input embedding_analysis/results_name_embeddings.npz \
        --output embedding_analysis/results_name
"""

import os
import sys
import time
import argparse
import numpy as np
import csv
from scipy.spatial.distance import cdist
from collections import Counter

def short_name(species_dir):
    """Extract common name from 'Genus species_Common Name' directory name."""
    parts = species_dir.split("_", 1)
    return parts[1] if len(parts) > 1 else species_dir

def main():
    parser = argparse.ArgumentParser(description="Identify misclassifications from embeddings.")
    parser.add_argument("--input", required=True, help="Path to .npz file containing embeddings.")
    parser.add_argument("--output", required=True, help="Base name for output files (without extension).")
    
    args = parser.parse_args()
    
    data = np.load(args.input, allow_pickle=True)
    
    active_species = data["species_list"].tolist()
    total_clips_processed = int(data["total_clips_processed"][0])
    total_clips_failed = int(data["total_clips_failed"][0])
    model_path = str(data["model_path"][0])
    input_dir = str(data["input_dir"][0])
    
    print(f"Loading embeddings from {args.input}...")
    print(f"Model used: {model_path}")
    print(f"Input dir: {input_dir}")
    print(f"Species: {len(active_species)}")
    
    all_embeddings = {sp: data[f"emb_{sp}"] for sp in active_species}
    all_filenames = {sp: data[f"files_{sp}"].tolist() for sp in active_species}
    centroids = {sp: data[f"centroid_{sp}"] for sp in active_species}
    
    print("\n" + "=" * 90)
    print("CROSS-CLASS MISCLASSIFICATION DETECTION")
    print("=" * 90 + "\n")

    # Pool all embeddings
    all_X = np.vstack([all_embeddings[sp] for sp in active_species])
    all_labs = np.concatenate([[sp] * len(all_embeddings[sp]) for sp in active_species])
    all_fns = sum([all_filenames[sp] for sp in active_species], [])

    print(f"  Computing {len(all_X)} × {len(active_species)} distance matrix ...")
    centroid_matrix = np.array([centroids[sp] for sp in active_species])
    distances = cdist(all_X, centroid_matrix, metric="cosine")

    print(f"  Finding misclassification suspects ...")

    suspects = []
    for i in range(len(all_X)):
        true_sp = all_labs[i]
        true_idx = active_species.index(true_sp)
        dist_own = distances[i, true_idx]

        dists_copy = distances[i].copy()
        dists_copy[true_idx] = np.inf
        nearest_idx = np.argmin(dists_copy)
        dist_nearest = dists_copy[nearest_idx]
        ratio = dist_own / dist_nearest if dist_nearest > 0 else 0

        suspects.append(
            {
                "file": all_fns[i],
                "true_class": short_name(true_sp),
                "true_class_full": true_sp,
                "dist_own": dist_own,
                "nearest_class": short_name(active_species[nearest_idx]),
                "nearest_class_full": active_species[nearest_idx],
                "dist_nearest": dist_nearest,
                "ratio": ratio,
            }
        )

    suspects.sort(key=lambda s: s["ratio"], reverse=True)
    misplaced = [s for s in suspects if s["ratio"] > 1.0]

    print(f"\n  Clips closer to another class: {len(misplaced)} / {len(suspects)}\n")
    print(f"  {'#':<4s} {'File':<60s} {'Current':<30s} {'Suggested':<30s} {'Ratio':<7s}")
    print(f"  {'─' * 4} {'─' * 60} {'─' * 30} {'─' * 30} {'─' * 7}")
    for i, s in enumerate(misplaced[:50]):
        print(f"  {i + 1:<4d} {s['file'][:60]:<60s} {s['true_class'][:30]:<30s} {s['nearest_class'][:30]:<30s} {s['ratio']:<7.3f}")

    # --- Save Results ---
    csv_path = f"{args.output}_misclassification_suspects.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file", "current_class", "current_class_full", "dist_to_own", "nearest_class", "nearest_class_full", "dist_to_nearest", "ratio", "likely_misplaced"]
        )
        for s in suspects:
            writer.writerow(
                [
                    s["file"],
                    s["true_class"],
                    s["true_class_full"],
                    f"{s['dist_own']:.4f}",
                    s["nearest_class"],
                    s["nearest_class_full"],
                    f"{s['dist_nearest']:.4f}",
                    f"{s['ratio']:.4f}",
                    "YES" if s["ratio"] > 1.0 else "",
                ]
            )

    print(f"\nSaved CSV: {csv_path}")

    summary_path = f"{args.output}_misclassification_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 90 + "\n")
        f.write("MISCLASSIFICATION ANALYSIS SUMMARY\n")
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Clip library: {os.path.basename(input_dir)} ({total_clips_processed} clips, {len(active_species)} species)\n")
        f.write("=" * 90 + "\n\n")

        f.write(f"Total clips analysed: {total_clips_processed}\n")
        f.write(f"Clips closer to another class (ratio > 1.0): {len(misplaced)}\n")
        f.write(f"Failed to process: {total_clips_failed}\n\n")

        f.write(f"{'#':<4s} {'File':<60s} {'Current':<30s} {'Suggested':<30s} {'Ratio':<7s}\n")
        f.write(f"{'─' * 4} {'─' * 60} {'─' * 30} {'─' * 30} {'─' * 7}\n")
        for i, s in enumerate(misplaced):
            f.write(f"{i + 1:<4d} {s['file'][:60]:<60s} {s['true_class'][:30]:<30s} {s['nearest_class'][:30]:<30s} {s['ratio']:<7.3f}\n")

        f.write(f"\n\n{'=' * 90}\n")
        f.write("PER-SPECIES BREAKDOWN\n")
        f.write("=" * 90 + "\n\n")

        misclass_by_species = Counter(s["true_class"] for s in misplaced)
        for species, count in misclass_by_species.most_common():
            total_in_class = sum(1 for s in suspects if s["true_class"] == species)
            pct = 100 * count / total_in_class if total_in_class > 0 else 0
            f.write(f"  {species:<40s} {count:>3d} / {total_in_class:>4d} ({pct:.1f}%)\n")

    print(f"Saved Summary: {summary_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
