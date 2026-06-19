#!/usr/bin/env python3
"""Out-of-distribution leak eval for the geophony non-event A/B/C/D arms.

The in-sample A/B/C/D test (geophony_nonevent_ab.py) showed species AUPRC is flat
across encodings — but it can only see IN-DISTRIBUTION non-target audio. The real
payoff of encoding helpers as non-events (broad, cheap negative diversity) is
rejecting NOVEL non-target sound at deployment, which an in-sample metric can't
measure. This script measures exactly that: it scores the four trained arm heads on
real long-format field recordings the model never saw, and compares how often each
arm spuriously fires SPECIES detections.

Field audio: the Smiths Lake "Powerline Strip" deployment — anthropophony-heavy
(powerline hum), so it directly stresses the human/both arms. The recordings are
unlabeled, so absolute precision isn't available; the signal is the CROSS-ARM DELTA
on identical windows (same audio, same embeddings, four heads). Night/quiet windows
are ~all non-target, so species firings there are mostly leak. A `--dump-top` option
writes the highest-confidence firings (file + timestamp + species) for listen-checks.

Reuses the in-sample run's cached base embeddings (for species-column alignment) and
the saved arm heads (geophony_nonevent_ab.py --save-heads). Single ungated command:

    .venv/bin/python experiments/geophony_nonevent_ood.py \
        --field-dir "/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/Smiths Lake Long Format Recordings/Powerline Strip" \
        --base      /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent \
        --max-files 8 --seconds-per-file 600

Smoke test:  --max-files 1 --seconds-per-file 60
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import audio, model  # noqa: E402
from experiments.geophony_nonevent_ab import ARMS, arm_labels, is_nontarget  # noqa: E402

THRESHOLDS = (0.25, 0.5)


def parse_start_seconds(fname):
    """POWERLINES_20260204_003202.wav -> seconds-into-day of the start time (or None)."""
    try:
        hhmmss = os.path.basename(fname).split("_")[2].split(".")[0]
        return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])
    except Exception:
        return None


def load_arms(base):
    """Return {arm: (head, species_cols, valid_labels)} from the in-sample run."""
    npz = f"{base}_embeddings_base.npz"
    if not os.path.exists(npz):
        sys.exit(f"Missing training embeddings cache: {npz}\nRun geophony_nonevent_ab.py first.")
    y_class = np.load(npz, allow_pickle=True)["y_class"]

    import keras
    arms = {}
    for arm in ARMS:
        head_path = f"{base}_arm_{arm}_head.keras"
        if not os.path.exists(head_path):
            sys.exit(f"Missing head: {head_path}\nRun geophony_nonevent_ab.py --save-heads first.")
        valid, _ = arm_labels(arm, y_class)
        sp_cols = np.array([i for i, c in enumerate(valid) if not is_nontarget(c)])
        arms[arm] = (keras.models.load_model(head_path, compile=False), sp_cols, valid)
        print(f"  arm {arm:<12s} head loaded ({len(valid)} classes, {len(sp_cols)} species cols)")
    return arms


def embed_file(path, sample_rate, seconds_per_file, batch):
    """Window a field file into 3 s frames and return base embeddings (n,1024)."""
    sig, rate = audio.open_audio_file(
        path, sample_rate=sample_rate, duration=seconds_per_file if seconds_per_file else None
    )
    chunks = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
    embs = []
    for i in range(0, len(chunks), batch):
        embs.append(model.embeddings(chunks[i : i + batch]))
    return np.concatenate(embs) if embs else np.zeros((0, 1024), dtype="float32")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--field-dir", required=True)
    p.add_argument("--base", default="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent")
    p.add_argument("--output", default=None, help="Report base (default: <base>_ood).")
    p.add_argument("--max-files", type=int, default=None, help="Subsample N files spread across the sorted (time-ordered) set.")
    p.add_argument("--seconds-per-file", type=int, default=600, help="Score only the first N s of each file (0 = whole file).")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--dump-top", type=int, default=50, help="Dump the N highest-confidence species firings per arm for listen-checks.")
    args = p.parse_args()
    out = args.output or f"{args.base}_ood"

    # Base BirdNET model => base embeddings (the head input)
    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SAMPLE_RATE = args.sample_rate
    cfg.SIG_OVERLAP = 0
    cfg.USE_NOISE = False

    print("Loading arm heads...")
    arms = load_arms(args.base)

    files = sorted(
        os.path.join(args.field_dir, f)
        for f in os.listdir(args.field_dir)
        if f.lower().endswith(".wav")
    )
    if args.max_files and args.max_files < len(files):
        idx = np.unique(np.linspace(0, len(files) - 1, args.max_files).round().astype(int))
        files = [files[i] for i in idx]
    print(f"\nScoring {len(files)} field file(s), {args.seconds_per_file or 'full'} s each\n")

    # Per-arm accumulators
    agg = {arm: {"n_win": 0, "fired": {t: 0 for t in THRESHOLDS}, "dets": {t: 0 for t in THRESHOLDS},
                 "sp_counts": np.zeros(len(arms[arm][1]), dtype=np.int64), "sum_max": 0.0}
           for arm in ARMS}
    top_hits = {arm: [] for arm in ARMS}  # (prob, file, start_s, species)

    t0 = time.time()
    for fi, path in enumerate(files):
        emb = embed_file(path, args.sample_rate, args.seconds_per_file, args.batch)
        nwin = len(emb)
        base_t = parse_start_seconds(path) or 0
        for arm, (head, sp_cols, valid) in arms.items():
            probs = head.predict(emb, batch_size=512, verbose=0)
            sp = probs[:, sp_cols]
            a = agg[arm]
            a["n_win"] += nwin
            a["sum_max"] += float(sp.max(axis=1).sum()) if nwin else 0.0
            a["sp_counts"] += (sp >= min(THRESHOLDS)).sum(axis=0).astype(np.int64)
            for t in THRESHOLDS:
                fired = sp >= t
                a["fired"][t] += int(fired.any(axis=1).sum())
                a["dets"][t] += int(fired.sum())
            # Top firings for listen-checks (window start time within file)
            if args.dump_top and nwin:
                wmax = sp.max(axis=1)
                warg = sp.argmax(axis=1)
                order = np.argsort(wmax)[::-1][: args.dump_top]
                for w in order:
                    if wmax[w] < min(THRESHOLDS):
                        break
                    start_s = w * cfg.SIG_LENGTH
                    top_hits[arm].append((float(wmax[w]), os.path.basename(path), float(start_s), valid[sp_cols[warg[w]]]))
        print(f"  [{fi + 1}/{len(files)}] {os.path.basename(path):<40s} {nwin:>5d} win  ({time.time() - t0:.0f}s)")

    write_reports(out, agg, arms, top_hits, args)


def write_reports(out, agg, arms, top_hits, args):
    rows = []
    for arm in ARMS:
        a = agg[arm]
        hours = a["n_win"] * cfg.SIG_LENGTH / 3600.0
        valid = arms[arm][2]
        sp_cols = arms[arm][1]
        sp_names = [valid[c] for c in sp_cols]
        top_sp_idx = np.argsort(a["sp_counts"])[::-1][:8]
        top_species = "; ".join(f"{sp_names[i]}={int(a['sp_counts'][i])}" for i in top_sp_idx if a["sp_counts"][i] > 0)
        row = {
            "arm": arm,
            "n_windows": a["n_win"],
            "hours": round(hours, 2),
            "mean_max_sp_prob": round(a["sum_max"] / a["n_win"], 4) if a["n_win"] else 0.0,
        }
        for t in THRESHOLDS:
            row[f"windows_fired@{t:g}"] = a["fired"][t]
            row[f"fire_rate_per_hr@{t:g}"] = round(a["fired"][t] / hours, 2) if hours else 0.0
            row[f"dets_per_hr@{t:g}"] = round(a["dets"][t] / hours, 2) if hours else 0.0
        row["top_fired_species"] = top_species
        rows.append(row)

    csv_path = f"{out}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    base = next(r for r in rows if r["arm"] == "none")
    md = f"{out}_summary.md"
    with open(md, "w") as f:
        f.write("# OOD field leak — geophony non-event arms (Powerline Strip)\n\n")
        f.write(f"Unlabeled field audio; lower species fire-rate = better non-target rejection "
                f"(but includes any real birds present). Baseline = `none`.\n\n")
        f.write("| arm | hours | fire-rate/hr @0.5 | dets/hr @0.5 | fire-rate/hr @0.25 | mean max-sp prob |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['arm']} | {r['hours']} | {r['fire_rate_per_hr@0.5']} | {r['dets_per_hr@0.5']} | "
                    f"{r['fire_rate_per_hr@0.25']} | {r['mean_max_sp_prob']} |\n")
        f.write("\n## Δ vs baseline `none` (negative = fewer spurious species firings)\n\n")
        for r in rows:
            if r["arm"] == "none":
                continue
            d5 = r["fire_rate_per_hr@0.5"] - base["fire_rate_per_hr@0.5"]
            d25 = r["fire_rate_per_hr@0.25"] - base["fire_rate_per_hr@0.25"]
            f.write(f"- **{r['arm']}**: Δfire-rate/hr @0.5 {d5:+.2f}, @0.25 {d25:+.2f}\n")
        f.write("\n_Caveat: unlabeled — a reduction is only a true win if the suppressed firings "
                "were non-target. Check `*_top_firings.csv`._\n")
    print(f"Wrote {md}")

    # Top firings for listen-checks
    if args.dump_top:
        tp = f"{out}_top_firings.csv"
        with open(tp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arm", "prob", "file", "start_sec", "species"])
            for arm in ARMS:
                for prob, fn, s, sp in sorted(top_hits[arm], reverse=True)[: args.dump_top]:
                    w.writerow([arm, f"{prob:.4f}", fn, f"{s:.0f}", sp])
        print(f"Wrote {tp}")


if __name__ == "__main__":
    main()
