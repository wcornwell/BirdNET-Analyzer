#!/usr/bin/env python3
"""Extract embeddings from a directory of audio clips using a TFLite model.

Usage:
    python embedding_analysis/extract_embeddings.py \
        --model /path/to/model.tflite \
        --input /path/to/clips \
        --output embedding_analysis/results_name
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# --- Setup ---
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import audio, model  # noqa: E402


def short_name(species_dir):
    """Extract common name from 'Genus species_Common Name' directory name."""
    parts = species_dir.split("_", 1)
    return parts[1] if len(parts) > 1 else species_dir

def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from audio clips.")
    parser.add_argument("--model", required=True, help="Path to TFLite model file.")
    parser.add_argument("--input", required=True, help="Path to directory containing species folders.")
    parser.add_argument("--output", required=True, help="Base name for output files (without extension).")
    parser.add_argument("--sample_rate", type=int, default=48000, help="Sample rate for audio files.")

    args = parser.parse_args()

    # Configure BirdNET
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SAMPLE_RATE = args.sample_rate
    cfg.SIG_OVERLAP = 0
    cfg.MODEL_PATH = os.path.abspath(args.model)

    base_dir = args.input
    output_base = args.output

    species_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")])

    print(f"Model: {args.model}")
    print(f"Input: {base_dir}")
    print(f"Found {len(species_dirs)} species directories\n")

    all_embeddings = {}  # species_dir -> np.array (N, embedding_dim)
    all_filenames = {}   # species_dir -> list of filenames

    total_species = len(species_dirs)
    total_clips_processed = 0
    total_clips_failed = 0
    start_time = time.time()

    for sp_idx, species in enumerate(species_dirs):
        cls_dir = os.path.join(base_dir, species)
        wav_files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg"))])
        sname = short_name(species)

        embs, fnames = [], []
        for fname in wav_files:
            try:
                sig, rate = audio.open_audio_file(os.path.join(cls_dir, fname), sample_rate=cfg.SAMPLE_RATE)
                sig_crop = audio.crop_center(sig, rate, cfg.SIG_LENGTH)
                emb = model.embeddings([sig_crop])
                embs.append(emb[0])
                fnames.append(fname)
            except Exception:
                total_clips_failed += 1

        n_emb = len(embs)
        total_clips_processed += n_emb

        if n_emb > 0:
            all_embeddings[species] = np.array(embs)
            all_filenames[species] = fnames

        elapsed = time.time() - start_time
        rate = total_clips_processed / elapsed if elapsed > 0 else 0

        print(f"  [{sp_idx + 1}/{total_species}] {sname:.<45s} {n_emb:>4d} clips  (total: {total_clips_processed}, {rate:.1f} clips/s, elapsed: {elapsed:.0f}s)")

    elapsed_total = time.time() - start_time
    print("\nEmbedding extraction complete!")
    print(f"  Total clips embedded: {total_clips_processed}")
    print(f"  Failed: {total_clips_failed}")
    print(f"  Species with embeddings: {len(all_embeddings)}")

    # Calculate centroids (mean)
    active_species = sorted(all_embeddings.keys())
    centroids = {sp: all_embeddings[sp].mean(axis=0) for sp in active_species}

    # Save results to .npz for the next script
    npz_path = f"{output_base}_embeddings.npz"
    # We save as a dict of arrays. Since all_filenames can't be easily saved in npz if lengths vary,
    # we'll save objects.
    save_dict = {
        "species_list": np.array(active_species),
        "total_clips_processed": np.array([total_clips_processed]),
        "total_clips_failed": np.array([total_clips_failed]),
        "model_path": np.array([args.model]),
        "input_dir": np.array([base_dir])
    }

    # Add embeddings and centroids to the dict
    for sp in active_species:
        save_dict[f"emb_{sp}"] = all_embeddings[sp]
        save_dict[f"centroid_{sp}"] = centroids[sp]
        save_dict[f"files_{sp}"] = np.array(all_filenames[sp])

    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved binary embeddings: {npz_path}")

    # Save human-readable centroids CSV
    csv_path = f"{output_base}_centroids.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header: Species, Dimension 1, Dimension 2, ...
        emb_dim = next(iter(centroids.values())).shape[0]
        header = ["species"] + [f"d{i}" for i in range(emb_dim)]
        writer.writerow(header)
        for sp in active_species:
            writer.writerow([sp, *centroids[sp].tolist()])

    print(f"Saved centroid CSV: {csv_path}")
    print(f"Done! Total time: {elapsed_total:.1f}s")

if __name__ == "__main__":
    main()
