#!/usr/bin/env python3
"""Extract CUSTOM-CLASSIFIER embeddings -- the trained head's penultimate layer.

This is the space every `reallybig_pelican0-*_embeddings.npz` before 0-29 lives in:
2048-d, specific to one recognizer, reflecting what that recognizer learned.

Not to be confused with its sibling `extract_embeddings.py`, which since f99b435
("Port extract_embeddings.py onto the backbone-agnostic embedding API") returns the
**base V2.4 backbone** features -- 1024-d, identical for every recognizer, because
that refactor demoted `--model` to provenance-only when `model.embeddings` was
dropped from the core. Both are legitimate; they are not interchangeable, and
centroid / misclassification / drift artifacts computed in one are meaningless
against the other:

    extract_embeddings.py       1024-d  base backbone     shared across recognizers
    extract_head_embeddings.py  2048-d  custom head       one recognizer's own space

The head layer is still present in the exported .tflite (input audio -> backbone ->
GLOBAL_AVG_POOL 1024 -> dense_1 relu 2048 -> 430 classes), so this reads it straight
out of the graph rather than depending on a core API. `experimental_preserve_all_tensors`
is what keeps the intermediate readable; without it TFLite reuses the buffer.

Audio handling is deliberately identical to the pre-refactor extractor
(`open_audio_file` -> `crop_center`, one centred window per clip), so output stays
directly comparable to the existing 0-18/0-19 artifacts. Output .npz/_centroids.csv
format is unchanged from that script too.

Usage:
    python embedding_analysis/extract_head_embeddings.py \
        --model  /path/to/pelican0-29.tflite \
        --input  /path/to/reallybig \
        --output /path/to/embeddings/reallybig_pelican0-29
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from birdnet_analyzer import audio  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
BIRDNET_SIG_LENGTH = 3.0


def short_name(species_dir):
    """Extract common name from 'Genus species_Common Name' directory name."""
    parts = species_dir.split("_", 1)
    return parts[1] if len(parts) > 1 else species_dir


def find_head_tensor(interpreter):
    """Index of the head's penultimate activation.

    Located by graph shape rather than a hard-coded index: the last 2-D
    intermediate before the class output. Hard-coding an index would silently
    read the wrong tensor if a recognizer were exported with different hidden
    units or an extra layer.
    """
    out_idx = interpreter.get_output_details()[0]["index"]
    n_classes = interpreter.get_output_details()[0]["shape"][-1]
    cands = [
        d for d in interpreter.get_tensor_details()
        if d["index"] < out_idx
        and len(d["shape"]) == 2
        and d["shape"][0] == 1
        and d["shape"][1] not in (n_classes,)
        and "dense" in d["name"].lower()
        and "relu" in d["name"].lower()
    ]
    if not cands:
        raise SystemExit(
            "Could not find the head's penultimate activation in this model. "
            "Is it a custom classifier exported by train_pelican.sh?"
        )
    return max(cands, key=lambda d: d["index"])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, help="Custom classifier .tflite. Selects the embedding space.")
    p.add_argument("--input", required=True, help="Directory containing species folders.")
    p.add_argument("--output", required=True, help="Base name for output files (no extension).")
    p.add_argument("--sample_rate", type=int, default=48000)
    args = p.parse_args()

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=os.path.abspath(args.model), experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    n_samples = int(in_detail["shape"][-1])
    head = find_head_tensor(interpreter)
    head_idx, emb_dim = head["index"], int(head["shape"][-1])

    base_dir, output_base = args.input, args.output
    species_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    )

    print(f"Model:  {args.model}")
    print(f"Layer:  {head['name'][:70]}  ({emb_dim}-d, tensor {head_idx})")
    print(f"Input:  {base_dir}")
    print(f"Found {len(species_dirs)} species directories\n")

    all_embeddings, all_filenames = {}, {}
    total_clips_processed = total_clips_failed = 0
    start_time = time.time()

    for sp_idx, species in enumerate(species_dirs):
        cls_dir = os.path.join(base_dir, species)
        wav_files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(AUDIO_EXTS))
        embs, fnames = [], []
        for fname in wav_files:
            try:
                sig, rate = audio.open_audio_file(
                    os.path.join(cls_dir, fname), sample_rate=args.sample_rate
                )
                chunk = audio.crop_center(sig, rate, BIRDNET_SIG_LENGTH)
                x = np.zeros(n_samples, dtype="float32")
                x[: min(len(chunk), n_samples)] = chunk[:n_samples]
                interpreter.set_tensor(in_detail["index"], x[None])
                interpreter.invoke()
                embs.append(interpreter.get_tensor(head_idx)[0].copy())
                fnames.append(fname)
            except Exception:
                total_clips_failed += 1

        total_clips_processed += len(embs)
        if embs:
            all_embeddings[species] = np.array(embs)
            all_filenames[species] = fnames

        elapsed = time.time() - start_time
        cps = total_clips_processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{sp_idx + 1}/{len(species_dirs)}] {short_name(species):.<45s} "
            f"{len(embs):>4d} clips  (total: {total_clips_processed}, "
            f"{cps:.1f} clips/s, elapsed: {elapsed:.0f}s)"
        )

    print("\nEmbedding extraction complete!")
    print(f"  Total clips embedded: {total_clips_processed}")
    print(f"  Failed: {total_clips_failed}")
    print(f"  Species with embeddings: {len(all_embeddings)}")

    active_species = sorted(all_embeddings.keys())
    centroids = {sp: all_embeddings[sp].mean(axis=0) for sp in active_species}

    save_dict = {
        "species_list": np.array(active_species),
        "total_clips_processed": np.array([total_clips_processed]),
        "total_clips_failed": np.array([total_clips_failed]),
        "model_path": np.array([args.model]),
        "input_dir": np.array([base_dir]),
        "embedding_space": np.array(["custom_head_penultimate"]),
    }
    for sp in active_species:
        save_dict[f"emb_{sp}"] = all_embeddings[sp]
        save_dict[f"centroid_{sp}"] = centroids[sp]
        save_dict[f"files_{sp}"] = np.array(all_filenames[sp])

    npz_path = f"{output_base}_embeddings.npz"
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved binary embeddings: {npz_path}")

    csv_path = f"{output_base}_centroids.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "short_name"] + [f"d{i}" for i in range(emb_dim)])
        for sp in active_species:
            w.writerow([sp, short_name(sp)] + [f"{v:.6f}" for v in centroids[sp]])
    print(f"Saved centroid CSV: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())
