#!/usr/bin/env python3
"""LABELED-soundscape eval for the geophony non-event A/B/C/D arms.

The decisive next step the A/B (in-sample, flat) and OOD (unlabeled field, `both`
~5% better but not adjudicable) runs both flagged: score the four arm heads on a
soundscape with GROUND TRUTH, so real species precision/recall and TRUE leak are
available — not the unlabeled cross-arm delta the OOD run was stuck with.

Data: `labeled_soundscape/` — two consecutive recordings (2MM03792_..._141450 then
_151402, 24 kHz mono) and `post_transcription_1.csv`, an EXHAUSTIVE transcription
of the ~1 hr spanning the file boundary: file1 is labeled from its first event to
its end, file2 from its start to its last event; the unlabeled head of file1 and
tail of file2 are excluded from scoring. Exhaustive within that span ⇒ absence of a
species annotation = a true negative, which is what makes precision/leak real.

The transcription is point events (`min`,`sec`,common name). Two scoring views:

  1. Micro P/R on the labeled span, with a ±`--tol-windows` slack so a point event
     credits the window(s) it could fall in (timing imprecision + call-bout gaps).
     present = annotated within slack; fired-and-not-present = false positive.

  2. TOLERANCE-FREE LEAK (the clean cross-arm signal): species the labeler never
     logged anywhere in the exhaustive hour are absent for the whole hour, so ANY
     firing on them is a false positive regardless of slack. This is exactly the
     geophony/anthropophony rejection question — does encoding Environment_*/
     Homo sapiens_* as non-events suppress phantom species firings?

Reuses the in-sample run's cached y_class (species-column alignment) and the four
saved arm heads (geophony_nonevent_ab.py --save-heads).

    .venv/bin/python experiments/geophony_nonevent_labeled.py \
        --soundscape-dir labeled_soundscape \
        --base /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent

Smoke test:  --max-seconds 120
"""

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import audio, model  # noqa: E402
from experiments.geophony_nonevent_ab import ARMS, arm_labels, is_nontarget  # noqa: E402

THRESHOLDS = (0.25, 0.5)

# Ordered file list of the soundscape (consecutive in time; "next file" in the
# transcription switches from the first to the second).
SOUNDSCAPE_FILES = ["2MM03792_20250119_141450.wav", "2MM03792_20250119_151402.wav"]

# GT common-name spelling fixes -> the model's common name (post-`_`).
NAME_FIXES = {
    "lewin's honeater": "lewin's honeyeater",
    "eatern yellow robin": "eastern yellow robin",
    "eastern shriketit": "eastern shrike-tit",
    "willie wagtail": "willie-wagtail",
}


def parse_transcription(csv_path):
    """-> (events, n_files). events: list of (file_idx, t_sec, common_lower, uncertain)."""
    events = []
    file_idx = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 4:
                continue
            mn, sc, name = row[1].strip(), row[2].strip(), row[3].strip()
            flag = (row[4] if len(row) > 4 else "").strip().lower()
            if "next file" in flag:
                file_idx = 1  # this row is the first event of file 2
            if not mn and not sc:
                continue
            try:
                t = int(mn) * 60 + int(sc)
            except ValueError:
                continue
            if not name:
                continue  # blank = time tick / unidentified, not a species positive
            uncertain = "?" in flag
            events.append((file_idx, t, NAME_FIXES.get(name.lower(), name.lower()), uncertain))
    return events


def common_name(cls):
    return cls.split("_", 1)[1] if "_" in cls else cls


def load_arms(base):
    """{arm: (head, sp_cols, valid)} from the in-sample run."""
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


def embed_file(path, sample_rate, max_seconds, batch):
    sig, rate = audio.open_audio_file(
        path, sample_rate=sample_rate, duration=max_seconds if max_seconds else None
    )
    chunks = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
    embs = [model.embeddings(chunks[i : i + batch]) for i in range(0, len(chunks), batch)]
    return np.concatenate(embs) if embs else np.zeros((0, 1024), dtype="float32")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape-dir", default=os.path.join(REPO_DIR, "labeled_soundscape"))
    p.add_argument("--base", default="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent")
    p.add_argument("--output", default=None, help="Report base (default: <base>_labeled).")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bin-seconds", type=int, nargs="+", default=[3, 6, 12, 30, 60, 120, 300, 600, 1200],
                   help="Match-window (time-bin) widths to score, in seconds. Wider = more "
                        "forgiving recall. A whole-hour bin is appended automatically as the asymptote.")
    p.add_argument("--max-seconds", type=int, default=0, help="Smoke test: only first N s of each file.")
    args = p.parse_args()
    out = args.output or f"{args.base}_labeled"

    csv_path = os.path.join(args.soundscape_dir, "post_transcription_1.csv")
    events = parse_transcription(csv_path)
    n_unc = sum(e[3] for e in events)
    print(f"Parsed {len(events)} species events ({n_unc} uncertain) from {csv_path}")

    # Exhaustive labeled span per file = [first event, last event] (certain or not).
    span = {}
    for fi in (0, 1):
        ts = [e[1] for e in events if e[0] == fi]
        span[fi] = (min(ts), max(ts))
        print(f"  file{fi+1} labeled span: {span[fi][0]}s ({span[fi][0]//60}:{span[fi][0]%60:02d}) "
              f"-> {span[fi][1]}s ({span[fi][1]//60}:{span[fi][1]%60:02d})")

    # GT species set actually present in the hour (by common name).
    present_names = sorted({e[2] for e in events if not e[3]})
    print(f"  {len(present_names)} distinct certain species in the labeled hour")

    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SAMPLE_RATE = args.sample_rate
    cfg.SIG_OVERLAP = 0
    cfg.USE_NOISE = False
    W = cfg.SIG_LENGTH  # window seconds (3.0)

    print("\nLoading arm heads...")
    arms = load_arms(args.base)
    # common-name -> species-column-index (alignment is identical across arms for species cols).
    valid0, sp_cols0 = arms["none"][2], arms["none"][1]
    name_to_spidx = {common_name(valid0[c]).lower(): j for j, c in enumerate(sp_cols0)}
    n_sp = len(sp_cols0)

    unmatched = sorted(n for n in present_names if n not in name_to_spidx)
    if unmatched:
        print(f"[WARN] {len(unmatched)} annotated species not in model classes (ignored): {unmatched}")
    present_spidx = {name_to_spidx[n] for n in present_names if n in name_to_spidx}
    # species columns NEVER present in the hour -> tolerance-free leak set.
    never_spidx = np.array([j for j in range(n_sp) if j not in present_spidx])
    print(f"  {len(present_spidx)} species cols present, {len(never_spidx)} never-present (leak set)")

    # Collect raw predictions + GT once; metrics are then computed at any bin width.
    # The two consecutive files are stitched into ONE contiguous labeled timeline
    # (each file's time shifted to (t - lo_s) + cumulative-labeled-duration) so a
    # single hour-wide bin spans the whole labeled hour rather than splitting at the
    # file boundary.  store[arm] = list of (sp_probs (n_keep, n_sp), win_start_s).
    store = {arm: [] for arm in ARMS}
    gt_events = []        # (global_t_sec, spidx, uncertain)
    labeled_run = 0.0     # cumulative labeled duration placed on the global timeline

    t0 = time.time()
    for fi, fname in enumerate(SOUNDSCAPE_FILES):
        path = os.path.join(args.soundscape_dir, fname)
        emb = embed_file(path, args.sample_rate, args.max_seconds, args.batch)
        nwin = len(emb)
        lo_s, hi_s = span[fi]
        if args.max_seconds:
            hi_s = min(hi_s, args.max_seconds)
        # window w covers [w*W, w*W + W); keep windows overlapping the labeled span.
        keep = np.array([w for w in range(nwin) if (w * W + W) > lo_s and (w * W) < hi_s], dtype=int)
        print(f"  [{fi+1}/2] {fname:<34s} {nwin} win, {len(keep)} in labeled span ({time.time()-t0:.0f}s)")

        for _, t, name, unc in (e for e in events if e[0] == fi):
            j = name_to_spidx.get(name)
            if j is not None and lo_s <= t < hi_s:
                gt_events.append(((t - lo_s) + labeled_run, j, unc))

        for arm, (head, sp_cols, valid) in arms.items():
            if not len(keep):
                store[arm].append((np.zeros((0, n_sp), "float32"), np.zeros(0)))
                continue
            probs = head.predict(emb[keep], batch_size=512, verbose=0)
            starts = np.clip(keep.astype(float) * W - lo_s, 0, None) + labeled_run
            store[arm].append((probs[:, sp_cols], starts))
        labeled_run += (hi_s - lo_s)

    print(f"  total labeled timeline: {labeled_run:.0f}s ({labeled_run/60:.1f} min)")
    # Add a single whole-hour bin so the curve reaches its asymptote (one window).
    bins = sorted(set(args.bin_seconds) | {int(np.ceil(labeled_run)) + 1})
    never_set = set(never_spidx.tolist())
    all_results = {bs: compute_metrics(bs, store, gt_events, never_set) for bs in bins}
    write_reports(out, all_results, store, gt_events, arms, span, present_names,
                  name_to_spidx, never_spidx, idx_from_spidx(arms), args, labeled_run)


def idx_from_spidx(arms):
    valid0, sp_cols0 = arms["none"][2], arms["none"][1]
    return {j: common_name(valid0[c]) for j, c in enumerate(sp_cols0)}


def compute_metrics(bin_seconds, store, gt_events, never_set):
    """Presence/absence per (time-bin, species) on the stitched labeled timeline.

    A wider bin = a more forgiving match window: the model 'detects' species j in a
    bin if it fires j in ANY 3 s window inside the bin, and the bin counts j present
    if the labeler logged j anywhere in it. At the widest (whole-hour) bin this is
    "did the model detect this species anywhere in the labeled hour." Returns
    {arm: row dict} plus per-species recall, all at this bin width.
    """
    present, ignore = {}, {}
    for t, j, unc in gt_events:
        b = int(t // bin_seconds)
        (ignore if unc else present).setdefault(b, set()).add(j)
    for b in present:  # certainty wins over an uncertain mark in the same bin
        ignore.get(b, set()).difference_update(present[b])

    rows = {}
    rec = {}  # arm -> {spidx: [hit, total]} at higher threshold
    n_bins_total = None
    for arm, files in store.items():
        fired = {}  # bin -> {thr: set of spidx fired somewhere in the bin}
        valid_bins = set()
        for sp, starts in files:
            for ki in range(len(starts)):
                b = int(starts[ki] // bin_seconds)
                valid_bins.add(b)
                cell = fired.setdefault(b, {t: set() for t in THRESHOLDS})
                row = sp[ki]
                for t in THRESHOLDS:
                    hit = np.where(row >= t)[0]
                    if len(hit):
                        cell[t].update(hit.tolist())
        n_bins_total = len(valid_bins)
        acc = {t: {"tp": 0, "fp": 0, "fn": 0, "leak_dets": 0, "leak_bins": 0} for t in THRESHOLDS}
        thi = max(THRESHOLDS)
        rec[arm] = {}
        for b in valid_bins:
            pres = present.get(b, set())
            ign = ignore.get(b, set()) - pres
            for t in THRESHOLDS:
                f = fired.get(b, {}).get(t, set()) - ign
                tp = f & pres
                acc[t]["tp"] += len(tp)
                acc[t]["fp"] += len(f - pres)
                acc[t]["fn"] += len(pres - f)
                leak = f & never_set
                acc[t]["leak_dets"] += len(leak)
                acc[t]["leak_bins"] += 1 if leak else 0
            fhi = fired.get(b, {}).get(thi, set())
            for j in pres:
                d = rec[arm].setdefault(j, [0, 0])
                d[1] += 1
                d[0] += 1 if j in fhi else 0
        hours = n_bins_total * bin_seconds / 3600.0
        row = {"bin_seconds": bin_seconds, "arm": arm, "n_bins": n_bins_total, "hours": round(hours, 3)}
        for t in THRESHOLDS:
            a = acc[t]
            prec = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else float("nan")
            r = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else float("nan")
            f1 = 2 * prec * r / (prec + r) if (prec + r) else float("nan")
            row[f"precision@{t:g}"] = round(prec, 4)
            row[f"recall@{t:g}"] = round(r, 4)
            row[f"f1@{t:g}"] = round(f1, 4)
            row[f"tp@{t:g}"], row[f"fp@{t:g}"], row[f"fn@{t:g}"] = a["tp"], a["fp"], a["fn"]
            row[f"leak_dets_per_hr@{t:g}"] = round(a["leak_dets"] / hours, 2) if hours else 0.0
            row[f"leak_bin_rate@{t:g}"] = round(a["leak_bins"] / n_bins_total, 4) if n_bins_total else 0.0
        rows[arm] = row
    return {"rows": rows, "rec": rec}


def write_reports(out, all_results, store, gt_events, arms, span, present_names,
                  name_to_spidx, never_spidx, idx_to_name, args, labeled_run):
    bins = sorted(all_results)            # bin widths (seconds)
    primary = bins[-1]                    # widest bin = whole-hour window
    rows = [all_results[bs]["rows"][arm] for bs in bins for arm in ARMS]

    csv_path = f"{out}_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    # per-species recall (across arms) at the primary bin width, higher threshold
    thi = max(THRESHOLDS)
    rec = all_results[primary]["rec"]
    sp_path = f"{out}_species_recall.csv"
    all_j = sorted({j for arm in ARMS for j in rec[arm]})
    with open(sp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", f"n_bins({primary}s)"] + [f"recall_{arm}@{thi:g}" for arm in ARMS])
        for j in all_j:
            tot = next((rec[arm][j][1] for arm in ARMS if j in rec[arm]), 0)
            cells = []
            for arm in ARMS:
                d = rec[arm].get(j)
                cells.append(round(d[0] / d[1], 3) if d and d[1] else "")
            w.writerow([idx_to_name[j], tot] + cells)
    print(f"Wrote {sp_path}")

    plot_recall_vs_window(out, all_results, bins)

    def R(bs, arm):
        return all_results[bs]["rows"][arm]
    base = R(primary, "none")
    md = f"{out}_summary.md"
    with open(md, "w") as f:
        f.write("# Labeled-soundscape eval — geophony non-event arms\n\n")
        f.write(f"Exhaustively labeled span: file1 {span[0][0]}–{span[0][1]}s, "
                f"file2 {span[1][0]}–{span[1][1]}s. {len(present_names)} species present; "
                f"{len(never_spidx)} model species never present (tolerance-free leak set). "
                f"Metrics computed by presence/absence per (species, time-bin) at bin widths "
                f"{bins} s — a wider bin is a more forgiving match window.\n\n")
        f.write(f"## Recall vs match-window (bin) size — @{thi:g}\n\n")
        f.write("| bin (s) | " + " | ".join(ARMS) + " |\n|" + "---|" * (len(ARMS) + 1) + "\n")
        for bs in bins:
            f.write(f"| {bs} | " + " | ".join(f"{R(bs, arm)[f'recall@{thi:g}']}" for arm in ARMS) + " |\n")
        f.write(f"\n## P/R at the {primary}s match window\n\n")
        f.write("| arm | P@0.5 | R@0.5 | F1@0.5 | P@0.25 | R@0.25 | F1@0.25 |\n|---|---|---|---|---|---|---|\n")
        for arm in ARMS:
            r = R(primary, arm)
            f.write(f"| {arm} | {r['precision@0.5']} | {r['recall@0.5']} | {r['f1@0.5']} | "
                    f"{r['precision@0.25']} | {r['recall@0.25']} | {r['f1@0.25']} |\n")
        f.write(f"\n## Tolerance-free leak (bin-width-independent denominator differs; {primary}s bins)\n\n")
        f.write("| arm | leak dets/hr @0.5 | leak bin-rate @0.5 | leak dets/hr @0.25 | leak bin-rate @0.25 |\n")
        f.write("|---|---|---|---|---|\n")
        for arm in ARMS:
            r = R(primary, arm)
            f.write(f"| {arm} | {r['leak_dets_per_hr@0.5']} | {r['leak_bin_rate@0.5']} | "
                    f"{r['leak_dets_per_hr@0.25']} | {r['leak_bin_rate@0.25']} |\n")
        f.write(f"\n_See `{os.path.basename(sp_path)}` for per-species recall, "
                f"`{os.path.basename(out)}_recall_vs_window.png` for the curve._\n")
    print(f"Wrote {md}")


def plot_recall_vs_window(out, all_results, bins):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping recall-vs-window plot")
        return
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(5.2 * len(THRESHOLDS), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    markers = {"none": "o", "environment": "s", "human": "^", "both": "D"}
    for ax, t in zip(axes, sorted(THRESHOLDS)):
        for arm in ARMS:
            y = [all_results[bs]["rows"][arm][f"recall@{t:g}"] for bs in bins]
            ax.plot(bins, y, marker=markers.get(arm, "o"), label=arm)
        ax.set_xscale("log")
        ax.set_xticks(bins)
        ax.set_xticklabels([str(b) for b in bins[:-1]] + ["hour"])
        ax.set_xlabel("match window / bin width (s)")
        ax.set_title(f"detection threshold ≥ {t:g}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("recall (presence detected per species×bin)")
    axes[0].legend(title="arm", fontsize=8)
    fig.suptitle("Recall vs match-window size — labeled soundscape", y=1.02)
    fig.tight_layout()
    png = f"{out}_recall_vs_window.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
