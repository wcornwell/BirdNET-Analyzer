#!/usr/bin/env python3
"""Extract base-backbone embeddings from a directory of audio clips.

Backbone-agnostic: the embedding backbone is selected by ``--version`` (a birdnet
"acoustic" model version, e.g. "2.4") and pulled via
``birdnet_analyzer.model_utils.get_embeddings_array`` -- the refactor's generalized,
versioned extractor -- instead of reading a frozen ``.tflite``'s penultimate layer.
The embedding geometry (sample rate, segment length) is read from the loaded model,
so this works across foundation-model generations without code changes.

These are the **base** embeddings for the chosen backbone. On V2.4 they are
bit-identical to what the old TFLite ``model.embeddings`` path produced (verified:
cosine 1.000000), so existing centroid/misclassification/UMAP artifacts stay
comparable. ``--model`` is accepted for provenance/back-compat only; it no longer
selects the backbone.

Usage:
    python embedding_analysis/extract_embeddings.py \
        --version 2.4 \
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

import birdnet  # noqa: E402

from birdnet_analyzer import audio  # noqa: E402
from birdnet_analyzer.model_utils import (  # noqa: E402
    get_embeddings_array_with_session,
)

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def short_name(species_dir):
    """Extract common name from 'Genus species_Common Name' directory name."""
    parts = species_dir.split("_", 1)
    return parts[1] if len(parts) > 1 else species_dir


def parse_args():
    p = argparse.ArgumentParser(description="Extract embeddings from audio clips.")
    p.add_argument(
        "--version", default="2.4", help="birdnet acoustic backbone version (e.g. 2.4)."
    )
    p.add_argument(
        "--input", required=True, help="Directory containing species folders."
    )
    p.add_argument(
        "--output", required=True, help="Base name for output files (no extension)."
    )
    p.add_argument(
        "--model",
        default="",
        help="Provenance only (recorded in the npz); does NOT select the backbone.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Encode batch size passed to get_embeddings_array.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load the backbone once; embedding geometry comes from the model itself, not
    # from config globals. One encode_session is reused across every species folder
    # (opening a session per folder would pay its worker-pool setup hundreds of times).
    model = birdnet.load("acoustic", args.version, "tf")
    sample_rate = model.get_sample_rate()
    sig_length = model.get_segment_size_s()

    base_dir = args.input
    output_base = args.output

    species_dirs = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    )

    print(
        f"Backbone: acoustic v{args.version}  "
        f"({sample_rate} Hz, {sig_length}s window)"
    )
    if args.model:
        print(f"Provenance --model: {args.model} (not used for extraction)")
    print(f"Input: {base_dir}")
    print(f"Found {len(species_dirs)} species directories\n")

    all_embeddings = {}  # species_dir -> np.array (N, embedding_dim)
    all_filenames = {}  # species_dir -> list of filenames

    total_species = len(species_dirs)
    total_clips_processed = 0
    total_clips_failed = 0
    start_time = time.time()

    with model.encode_session(batch_size=args.batch_size) as session:
        for sp_idx, species in enumerate(species_dirs):
            cls_dir = os.path.join(base_dir, species)
            wav_files = sorted(
                f for f in os.listdir(cls_dir) if f.lower().endswith(AUDIO_EXTS)
            )
            sname = short_name(species)

            # Decode + center-crop each clip individually so a single corrupt file is
            # skipped (not the whole batch), then embed the folder in one batched call.
            sigs, fnames = [], []
            for fname in wav_files:
                try:
                    sig, rate = audio.open_audio_file(
                        os.path.join(cls_dir, fname), sample_rate=sample_rate
                    )
                    sigs.append(audio.crop_center(sig, rate, sig_length))
                    fnames.append(fname)
                except Exception:
                    total_clips_failed += 1

            if sigs:
                inputs = [(sig, sample_rate) for sig in sigs]
                embs = get_embeddings_array_with_session(session, inputs)
                all_embeddings[species] = np.asarray(embs)
                all_filenames[species] = fnames

            n_emb = len(fnames) if sigs else 0
            total_clips_processed += n_emb

            elapsed = time.time() - start_time
            rate = total_clips_processed / elapsed if elapsed > 0 else 0
            print(
                f"  [{sp_idx + 1}/{total_species}] {sname:.<45s} {n_emb:>4d} clips  "
                f"(total: {total_clips_processed}, {rate:.1f} clips/s, "
                f"elapsed: {elapsed:.0f}s)"
            )

    elapsed_total = time.time() - start_time
    print("\nEmbedding extraction complete!")
    print(f"  Total clips embedded: {total_clips_processed}")
    print(f"  Failed: {total_clips_failed}")
    print(f"  Species with embeddings: {len(all_embeddings)}")

    # Calculate centroids (mean)
    active_species = sorted(all_embeddings.keys())
    centroids = {sp: all_embeddings[sp].mean(axis=0) for sp in active_species}

    # Save results to .npz for the next script. Filenames per species can vary in
    # length, so they are stored as separate per-species arrays.
    npz_path = f"{output_base}_embeddings.npz"
    save_dict = {
        "species_list": np.array(active_species),
        "total_clips_processed": np.array([total_clips_processed]),
        "total_clips_failed": np.array([total_clips_failed]),
        "backbone_version": np.array([args.version]),
        # provenance only; kept for downstream back-compat
        "model_path": np.array([args.model]),
        "input_dir": np.array([base_dir]),
    }

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
        emb_dim = next(iter(centroids.values())).shape[0]
        header = ["species"] + [f"d{i}" for i in range(emb_dim)]
        writer.writerow(header)
        for sp in active_species:
            writer.writerow([sp, *centroids[sp].tolist()])

    print(f"Saved centroid CSV: {csv_path}")
    print(f"Done! Total time: {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
