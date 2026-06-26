#!/usr/bin/env python3
"""Does low soundscape recall come from FAINT calls under the detector trigger?

The labeled-soundscape eval (geophony_nonevent_labeled.py) found recall ~0.2 even at
a 5-min match window — most annotated calls go undetected. Hypothesis (W.C.): these
are calls audible to the transcriber's ear but too faint to trip the detector. Two
falsifiable predictions, both tested here on the baseline `none` head:

  1. RECOVERABILITY — if missed calls produce weak-but-nonzero activations, recall
     should rise steeply as the detection threshold drops. (recall-vs-threshold)
  2. LOUDNESS — detected calls should be louder than missed ones. Per annotated
     event we compute a vocalisation-band SNR proxy (peak vs median frame energy in
     1.5-9 kHz within the event's 3 s window) and compare detected vs missed, plus a
     detection-rate-vs-SNR curve.

A loudness proxy, not a calibrated SPL: it cannot separate "faint" from "out-of-
domain / masked by chorus" (both give low activation), but it CAN show whether
detection success tracks how much a call sticks out of its background.

    .venv/bin/python experiments/geophony_nonevent_faintness.py
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

import birdnet_analyzer.config as cfg  # noqa: E402
from birdnet_analyzer import audio, model  # noqa: E402
from experiments.geophony_nonevent_ab import arm_labels, is_nontarget  # noqa: E402
from experiments.geophony_nonevent_labeled import (  # noqa: E402
    SOUNDSCAPE_FILES, common_name, parse_transcription,
)

SWEEP = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]


def band_snr(sig, rate, lo=1500, hi=9000, frame=2400, hop=1200):
    """(within-window SNR in dB, abs peak band energy in dB) for one window's signal."""
    if len(sig) < frame:
        sig = np.pad(sig, (0, frame - len(sig)))
    fr = np.lib.stride_tricks.sliding_window_view(sig, frame)[::hop]
    win = np.hanning(frame)
    mag = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    band = (freqs >= lo) & (freqs <= hi)
    e = mag[:, band].sum(axis=1)  # per-frame band energy
    eps = 1e-12
    peak, floor = float(e.max()), float(np.median(e))
    return 10 * np.log10(peak / (floor + eps) + eps), 10 * np.log10(peak + eps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundscape-dir", default=os.path.join(REPO_DIR, "labeled_soundscape"))
    p.add_argument("--base", default="/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/experiments/geophony_nonevent")
    p.add_argument("--output", default=None)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()
    out = args.output or f"{args.base}_faintness"

    events = parse_transcription(os.path.join(args.soundscape_dir, "post_transcription_1.csv"))
    span = {fi: (min(e[1] for e in events if e[0] == fi), max(e[1] for e in events if e[0] == fi))
            for fi in (0, 1)}

    cfg.MODEL_PATH = cfg.BIRDNET_MODEL_PATH
    cfg.SIG_LENGTH = cfg.BIRDNET_SIG_LENGTH
    cfg.SAMPLE_RATE = args.sample_rate
    cfg.SIG_OVERLAP = 0
    cfg.USE_NOISE = False
    W = cfg.SIG_LENGTH

    import keras
    y_class = np.load(f"{args.base}_embeddings_base.npz", allow_pickle=True)["y_class"]
    valid, _ = arm_labels("none", y_class)
    sp_cols = np.array([i for i, c in enumerate(valid) if not is_nontarget(c)])
    name_to_spidx = {common_name(valid[c]).lower(): j for j, c in enumerate(sp_cols)}
    head = keras.models.load_model(f"{args.base}_arm_none_head.keras", compile=False)
    print(f"Baseline head loaded ({len(sp_cols)} species cols)")

    detail = []  # (file, t, species, prob, snr_db, abs_db)
    for fi, fname in enumerate(SOUNDSCAPE_FILES):
        path = os.path.join(args.soundscape_dir, fname)
        sig, rate = audio.open_audio_file(path, sample_rate=args.sample_rate)
        lo_s, hi_s = span[fi]
        nwin = int(len(sig) // (W * rate))
        keep = np.array([w for w in range(nwin) if (w * W + W) > lo_s and (w * W) < hi_s], dtype=int)
        chunks = audio.split_signal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)
        chunks = [chunks[w] for w in keep]
        embs = [model.embeddings(chunks[i:i + args.batch]) for i in range(0, len(chunks), args.batch)]
        sp = head.predict(np.concatenate(embs), batch_size=512, verbose=0)[:, sp_cols] if chunks else np.zeros((0, len(sp_cols)))
        wpos = {int(w): i for i, w in enumerate(keep)}
        print(f"  {fname}: {len(keep)} windows scored")

        for _, t, name, unc in (e for e in events if e[0] == fi):
            if unc:
                continue
            j = name_to_spidx.get(name)
            w = int(t // W)
            if j is None or w not in wpos:
                continue
            prob = float(sp[wpos[w], j])
            seg = sig[int(w * W * rate):int((w * W + W) * rate)]
            snr_db, abs_db = band_snr(seg, rate)
            detail.append((fi, t, name, prob, round(snr_db, 2), round(abs_db, 2)))

    probs = np.array([d[3] for d in detail])
    snr = np.array([d[4] for d in detail])
    print(f"\n{len(detail)} annotated events scored. "
          f"prob: median={np.median(probs):.3f} mean={probs.mean():.3f}; "
          f"frac>0.5={np.mean(probs >= 0.5):.3f} frac>0.25={np.mean(probs >= 0.25):.3f} "
          f"frac>0.05={np.mean(probs >= 0.05):.3f} frac~0(<0.01)={np.mean(probs < 0.01):.3f}")

    with open(f"{out}_event_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_idx", "t_sec", "species", "prob", "snr_db", "abs_band_db"])
        w.writerows(detail)
    print(f"Wrote {out}_event_detail.csv")

    make_plot(out, probs, snr)


def make_plot(out, probs, snr):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib missing — no plot")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # 1. recall (event-level) vs detection threshold
    rec = [float(np.mean(probs >= th)) for th in SWEEP]
    ax[0].plot(SWEEP, rec, marker="o")
    ax[0].set_xscale("log"); ax[0].set_xlabel("detection threshold")
    ax[0].set_ylabel("event-level recall"); ax[0].set_title("1. Recoverability:\nrecall vs threshold")
    ax[0].grid(True, which="both", ls=":", alpha=0.4)
    ax[0].axvline(0.5, color="grey", ls="--", lw=0.8); ax[0].axvline(0.25, color="grey", ls=":", lw=0.8)

    # 2. detection rate vs SNR decile
    order = np.argsort(snr)
    nq = 10
    edges = np.quantile(snr, np.linspace(0, 1, nq + 1))
    cx, det25, det50 = [], [], []
    for k in range(nq):
        lo, hi = edges[k], edges[k + 1]
        m = (snr >= lo) & (snr <= hi) if k == nq - 1 else (snr >= lo) & (snr < hi)
        if m.sum() == 0:
            continue
        cx.append(snr[m].mean())
        det25.append(np.mean(probs[m] >= 0.25))
        det50.append(np.mean(probs[m] >= 0.5))
    ax[1].plot(cx, det25, marker="s", label="≥0.25")
    ax[1].plot(cx, det50, marker="D", label="≥0.5")
    ax[1].set_xlabel("call SNR proxy (dB, peak vs median in-window 1.5–9 kHz)")
    ax[1].set_ylabel("detection rate"); ax[1].set_title("2. Loudness:\ndetection rate vs call SNR")
    ax[1].legend(); ax[1].grid(True, ls=":", alpha=0.4)

    # 3. SNR distribution, detected vs missed (@0.5)
    det = snr[probs >= 0.5]; miss = snr[probs < 0.5]
    ax[2].hist([miss, det], bins=20, label=[f"missed (n={len(miss)})", f"detected (n={len(det)})"],
               color=["#d62728", "#2ca02c"], alpha=0.8)
    ax[2].set_xlabel("call SNR proxy (dB)"); ax[2].set_ylabel("events")
    ax[2].set_title("3. SNR of detected vs missed\n(@0.5)"); ax[2].legend()
    ax[2].grid(True, ls=":", alpha=0.4)

    fig.suptitle("Is low recall driven by faint calls? — baseline head, labeled soundscape", y=1.03)
    fig.tight_layout()
    png = f"{out}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    if len(det) and len(miss):
        print(f"median SNR: detected={np.median(det):.2f} dB, missed={np.median(miss):.2f} dB")


if __name__ == "__main__":
    main()
