#!/usr/bin/env python3
"""A/B/C/D test: non-target sounds as non-events vs. as positive classes.

Tests the CLAUDE.md "Design fork" hypothesis — do the geophony (Environment_*) and
anthropophony (Homo sapiens_*) helper classes protect species better as *non-events*
(all-zero hard-negative label rows, like BirdNET's NON_EVENT_CLASSES) than as the
ordinary positive, reportable classes they currently are?

Four arms (species classes never change; `Noise` is always a non-event):

    none         baseline — Environment_* AND Homo sapiens_* are positive classes
    environment  Environment_*  -> non-events; Homo sapiens_* stay classes
    human        Homo sapiens_* -> non-events; Environment_* stay classes
    both         Environment_* AND Homo sapiens_* -> non-events

Why this is cheap (one weekend run, not 4x): `model.embeddings()` returns the BASE
BirdNET feature-extractor output — the exact input the trainable head consumes — and
it is LABEL-INDEPENDENT. So we extract embeddings ONCE (the only slow step) and fit
four cheap heads that differ only in label encoding, reusing the real training
function `model.train_linear_classifier` (same architecture / focal loss / upsampling
/ metric tagging as the production pipeline).

Evaluation uses ONE shared, fixed, stratified held-out split (seed cfg.RANDOM_SEED),
reused across all four arms, so the comparison is apples-to-apples. The held-out set
keeps the geophony/anthropophony clips (own-class rows in `none`, all-zero rows where
converted) so they act as negatives that can leak onto species in every arm.

Single ungated command (matches the Bash(.venv/bin/python *) allow rule):

    .venv/bin/python experiments/geophony_nonevent_ab.py \
        --input  /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig \
        --output /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent

Smoke test:  add  --limit 30 --epochs 2
"""

import argparse
import csv
import hashlib
import os
import sys
import time

import numpy as np

# --- Setup: import the repo's model pipeline ---
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import audio, model  # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")
ARMS = ["none", "environment", "human", "both"]

# Pelican hyperparameters (from train_pelican.sh)
HP = {
    "hidden_units": 2048,
    "dropout": 0.25,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "upsampling_ratio": 0.4,
    "upsampling_mode": "repeat",
    "focal_alpha": 0.25,
    "focal_gamma": 3.0,
    "epochs": 50,
}


# --------------------------------------------------------------------------- #
# Class-role helpers
# --------------------------------------------------------------------------- #
def is_nontarget(name: str) -> bool:
    """Geophony / anthropophony helper class (mirrors model.py:_is_nontarget)."""
    return name.startswith(("Environment_", "Homo sapiens_"))


def is_always_nonevent(name: str) -> bool:
    """A true non-event in every arm (real Noise/other/background/silence folder)."""
    return name.lower() in cfg.NON_EVENT_CLASSES


def converts_to_nonevent(arm: str, name: str) -> bool:
    """Does this class become an all-zero non-event under the given arm?"""
    env = name.startswith("Environment_")
    hum = name.startswith("Homo sapiens_")
    return {
        "none": False,
        "environment": env,
        "human": hum,
        "both": env or hum,
    }[arm]


# --------------------------------------------------------------------------- #
# Embedding extraction (BASE BirdNET model — label-independent, done once)
# --------------------------------------------------------------------------- #
def extract_all(input_dir: str, sample_rate: int, limit):
    """Extract base BirdNET embeddings for every clip under input_dir.

    Returns (X[N,dim] float32, y_class[N] str, files[N] str).
    """
    # Use the BASE BirdNET model so embeddings() returns the training substrate
    # (the index-1 feature tensor of THIS model). A pelican .tflite would yield
    # head-derived features, which is NOT what train.py fits on.
    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SAMPLE_RATE = sample_rate
    cfg.SIG_OVERLAP = 0

    classes = sorted(
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
    )
    print(f"Model (base): {cfg.MODEL_PATH}")
    print(f"Input:        {input_dir}")
    print(f"Found {len(classes)} class directories\n")

    X, y_class, files = [], [], []
    n_fail = 0
    t0 = time.time()
    for ci, cls in enumerate(classes):
        cdir = os.path.join(input_dir, cls)
        wavs = sorted(f for f in os.listdir(cdir) if f.lower().endswith(AUDIO_EXTS))
        if limit:
            wavs = wavs[:limit]
        n = 0
        for fn in wavs:
            try:
                sig, rate = audio.open_audio_file(os.path.join(cdir, fn), sample_rate=cfg.SAMPLE_RATE)
                sig_crop = audio.crop_center(sig, rate, cfg.SIG_LENGTH)
                emb = model.embeddings([sig_crop])
                X.append(emb[0])
                y_class.append(cls)
                files.append(f"{cls}/{fn}")
                n += 1
            except Exception:
                n_fail += 1
        elapsed = time.time() - t0
        rate_cps = len(X) / elapsed if elapsed > 0 else 0
        print(f"  [{ci + 1}/{len(classes)}] {cls:.<55s} {n:>4d}  (total {len(X)}, {rate_cps:.1f}/s, {elapsed:.0f}s)")

    print(f"\nExtraction done: {len(X)} clips, {n_fail} failed, {time.time() - t0:.0f}s")
    return np.asarray(X, dtype="float32"), np.asarray(y_class), np.asarray(files)


def load_or_extract(args):
    cache = f"{args.output}_embeddings_base.npz"
    if os.path.exists(cache) and not args.force_extract:
        print(f"Loading cached base embeddings: {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y_class"], d["files"]
    X, y_class, files = extract_all(args.input, args.sample_rate, args.limit)
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez_compressed(cache, X=X, y_class=y_class, files=files)
    print(f"Saved base embeddings cache: {cache}")
    return X, y_class, files


# --------------------------------------------------------------------------- #
# Shared, fixed, stratified train/held-out split (computed ONCE)
# --------------------------------------------------------------------------- #
def stratified_split(y_class, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_idx, held_idx = [], []
    for cls in np.unique(y_class):
        idx = np.where(y_class == cls)[0]
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * (1 - val_ratio)))
        train_idx.extend(idx[:n_train].tolist())
        held_idx.extend(idx[n_train:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(held_idx))


# --------------------------------------------------------------------------- #
# Per-arm label encoding
# --------------------------------------------------------------------------- #
def arm_labels(arm, y_class):
    """valid_labels for this arm + a name->column map."""
    all_classes = sorted(np.unique(y_class).tolist())
    valid = [
        c for c in all_classes
        if not is_always_nonevent(c) and not converts_to_nonevent(arm, c)
    ]
    return valid, {c: i for i, c in enumerate(valid)}


def encode(y_class, idx, col, n_labels):
    """One-hot rows for kept classes; all-zero rows for non-events/converted."""
    Y = np.zeros((len(idx), n_labels), dtype="float32")
    for r, i in enumerate(idx):
        c = y_class[i]
        if c in col:
            Y[r, col[c]] = 1.0
    return Y


# --------------------------------------------------------------------------- #
# Per-arm metrics on the SHARED held-out set
# --------------------------------------------------------------------------- #
def evaluate_arm(probs, valid, Y_held, held_true_classes, thresholds=(0.25, 0.5)):
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    sp_cols = [i for i, c in enumerate(valid) if not is_nontarget(c)]
    Yb = (Y_held > 0).astype(int)

    # Species PR-AUC (threshold-free, primary)
    aps = [
        average_precision_score(Yb[:, i], probs[:, i])
        for i in sp_cols if Yb[:, i].sum() > 0
    ]
    aupr_macro = float(np.mean(aps)) if aps else 0.0
    sp_true_flat = Yb[:, sp_cols].ravel()
    aupr_micro = (
        float(average_precision_score(sp_true_flat, probs[:, sp_cols].ravel()))
        if sp_true_flat.sum() > 0 else 0.0
    )

    # Non-target leak: held-out clips whose TRUE folder is non-target
    # (Environment_*/Homo sapiens_*/Noise) that fire >= thr on ANY species.
    # Uses true folder only => identical clip set + mask across all arms.
    nt_mask = np.array([
        is_nontarget(c) or is_always_nonevent(c) for c in held_true_classes
    ])
    sp_max = probs[:, sp_cols].max(axis=1) if sp_cols else np.zeros(len(probs))

    row = {
        "n_species": len(sp_cols),
        "n_held": len(probs),
        "n_held_nontarget": int(nt_mask.sum()),
        "species_aupr_macro": aupr_macro,
        "species_aupr_micro": aupr_micro,
    }
    for thr in thresholds:
        pred = (probs[:, sp_cols] >= thr).astype(int)
        pmi, rmi, _, _ = precision_recall_fscore_support(Yb[:, sp_cols], pred, average="micro", zero_division=0)
        pma, rma, _, _ = precision_recall_fscore_support(Yb[:, sp_cols], pred, average="macro", zero_division=0)
        row[f"species_P@{thr:g}"] = float(pmi)
        row[f"species_R@{thr:g}"] = float(rmi)
        row[f"species_Pmacro@{thr:g}"] = float(pma)
        row[f"species_Rmacro@{thr:g}"] = float(rma)
        row[f"nontarget_leak@{thr:g}"] = float((sp_max[nt_mask] >= thr).mean()) if nt_mask.any() else 0.0
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig")
    p.add_argument("--output", default="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    p.add_argument("--epochs", type=int, default=HP["epochs"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--limit", type=int, default=None, help="Max clips per class (smoke test).")
    p.add_argument("--force-extract", action="store_true", help="Re-extract even if cache exists.")
    p.add_argument("--save-models", action="store_true", help="Also save each arm as a deployable .tflite.")
    p.add_argument("--save-heads", action="store_true", help="Save each arm's trained keras head (for OOD scoring).")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # 1) Embeddings (once)
    X, y_class, files = load_or_extract(args)
    print(f"Embeddings: {X.shape[0]} clips x {X.shape[1]} dim, {len(np.unique(y_class))} classes\n")

    # 2) Shared fixed split
    train_idx, held_idx = stratified_split(y_class, args.val_ratio, cfg.RANDOM_SEED)
    held_files = sorted(files[held_idx].tolist())
    split_hash = hashlib.sha256("\n".join(held_files).encode()).hexdigest()[:12]
    # Leakage guard
    assert len(set(train_idx.tolist()) & set(held_idx.tolist())) == 0, "train/held overlap!"
    print(f"Split: {len(train_idx)} train / {len(held_idx)} held-out  (held-set hash {split_hash})\n")

    held_true_classes = y_class[held_idx]
    cfg.MULTI_LABEL = True
    cfg.BINARY_CLASSIFICATION = False

    rows = []
    for arm in args.arms:
        print(f"\n{'=' * 70}\nARM: {arm}\n{'=' * 70}")
        valid, col = arm_labels(arm, y_class)
        n_lab = len(valid)
        n_conv = sum(1 for c in np.unique(y_class) if converts_to_nonevent(arm, c))
        print(f"  {n_lab} output classes ({sum(not is_nontarget(c) for c in valid)} species, "
              f"{sum(is_nontarget(c) for c in valid)} kept helpers); {n_conv} classes -> non-events")

        Y_train = encode(y_class, train_idx, col, n_lab)
        Y_held = encode(y_class, held_idx, col, n_lab)

        # Deterministic head init per arm
        try:
            import keras
            keras.utils.set_random_seed(cfg.RANDOM_SEED)
        except Exception:
            pass

        clf = model.build_linear_classifier(n_lab, X.shape[1], HP["hidden_units"], HP["dropout"])

        # Route train_linear_classifier's own metrics CSV to this arm's base path
        arm_base = f"{args.output}_arm_{arm}"
        cfg.CUSTOM_CLASSIFIER = arm_base
        cfg.MODEL_PATH = arm_base  # both -> single co-located *_validation_metrics.csv
        cfg.LABELS = valid

        clf, _ = model.train_linear_classifier(
            clf,
            X[train_idx], Y_train,
            X[held_idx], Y_held,
            epochs=args.epochs,
            batch_size=HP["batch_size"],
            learning_rate=HP["learning_rate"],
            val_split=0,  # use x_test/y_test (shared held-out) as validation
            upsampling_ratio=HP["upsampling_ratio"],
            upsampling_mode=HP["upsampling_mode"],
            train_with_mixup=False,
            train_with_label_smoothing=False,
            train_with_focal_loss=True,
            focal_loss_gamma=HP["focal_gamma"],
            focal_loss_alpha=HP["focal_alpha"],
            labels=valid,
        )

        probs = clf.predict(X[held_idx], batch_size=HP["batch_size"], verbose=0)
        row = {"arm": arm, "n_classes": n_lab, "n_converted": n_conv, "split_hash": split_hash}
        row.update(evaluate_arm(probs, valid, Y_held, held_true_classes))
        rows.append(row)
        print(f"  -> species AUPRC macro {row['species_aupr_macro']:.4f} | "
              f"leak@0.5 {row['nontarget_leak@0.5']:.4f} | leak@0.25 {row['nontarget_leak@0.25']:.4f}")

        if args.save_heads:
            clf.save(f"{arm_base}_head.keras")
            print(f"  saved {arm_base}_head.keras")

        if args.save_models:
            try:
                model.save_linear_classifier(clf, arm_base + ".tflite", valid, mode="replace")
                print(f"  saved {arm_base}.tflite")
            except Exception as e:
                print(f"  [WARN] could not save tflite ({e})")

    # 3) Cross-arm comparison report
    write_reports(args.output, rows, split_hash)


def write_reports(output, rows, split_hash):
    cols = list(rows[0].keys())
    csv_path = f"{output}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    md_path = f"{output}_summary.md"
    with open(md_path, "w") as f:
        f.write("# Geophony/anthropophony non-event A/B/C/D — results\n\n")
        f.write(f"Shared held-out set hash: `{split_hash}` (identical across arms)\n\n")
        f.write("| arm | species AUPRC (macro) | species AUPRC (micro) | "
                "species P/R @0.5 | non-target leak @0.5 | non-target leak @0.25 |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.writelines(f"| {r['arm']} | {r['species_aupr_macro']:.4f} | {r['species_aupr_micro']:.4f} | "
                    f"{r['species_P@0.5']:.3f}/{r['species_R@0.5']:.3f} | "
                    f"{r['nontarget_leak@0.5']:.4f} | {r['nontarget_leak@0.25']:.4f} |\n" for r in rows)
        # Decision note
        base = next((r for r in rows if r["arm"] == "none"), rows[0])
        f.write("\n## Decision rule\n\n")
        f.write("An arm wins if it **reduces non-target leak** without lowering **species AUPRC** "
                f"vs. baseline `none` (AUPRC macro {base['species_aupr_macro']:.4f}, "
                f"leak@0.5 {base['nontarget_leak@0.5']:.4f}).\n\n")
        for r in rows:
            if r["arm"] == "none":
                continue
            d_aupr = r["species_aupr_macro"] - base["species_aupr_macro"]
            d_leak = r["nontarget_leak@0.5"] - base["nontarget_leak@0.5"]
            verdict = "WIN" if (d_leak < 0 and d_aupr >= -0.002) else ("worse" if d_aupr < -0.002 else "neutral")
            f.write(f"- **{r['arm']}**: ΔAUPRC {d_aupr:+.4f}, Δleak@0.5 {d_leak:+.4f} → {verdict}\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
