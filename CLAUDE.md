# BirdNET-Analyzer — Project Notes for Claude

## Where this repo sits — read before building anything new

**One library, two backbones.** `reallybig` is backbone-agnostic labeled audio living
outside every repo (OneDrive `call_library/`). Only the *embedding caches* and *venvs*
fork per backbone — the library itself never does.

**This repo owns:** the engine — training, analyze, and embedding *extraction*.
**It does NOT own:** curation of `reallybig` (→ `Training_library_assembly_pipeline`),
evaluation (→ `soundscape-eval`), or embedding *analysis/visualisation* (→ `birdnetEmbed`).

⚠️ **Its `.venv` is shared infrastructure.** `Training_library_assembly_pipeline/curation/`
invokes it by absolute path. Keep this on `main` when others are running evals.

**`soundscape-eval` no longer depends on it for most work (2026-08-31).** It vendored the two
audio helpers it used, moved checkpoint resolution to `find_spec` (which never executes this
package), and declares its own deps — so `pip install -e "soundscape-eval[birdnet30]"` scores
BirdNET 3.0 and any raw-audio recognizer with **no engine and no TensorFlow**. It still needs
an editable install of this fork for the **Perch** paths and **bare-head** recognizers, and
that install must be on `main` (whichever branch is checked out is what it gets).

**Why that decoupling happened:** BirdNET 3.0 needed `birdnet==1.1.0` while this fork pinned
`birdnet<1`. **That ended 2026-10-26:** `main` fast-forwarded to the 3.0 sync
(`upstream-sync-2026-10`, tip `3964f89`) and the shared `.venv` was rebuilt on
**`birdnet` 1.3.1** / TF 2.21 (old venv kept as `.venv-pre1007` for rollback). The
`[birdnet30]` env remains the lightweight, engine-free way to score.

⚠️ `labeled_soundscape/` is **evaluation data that should not be in this repo** — pending a
move to `${call_library}/testing_soundscapes/smithslake/`. See ECOACOUSTICS.md.

### Stray local data — CLEARED 2026-08-09, keep it that way

The repo had ~1.55 GB of untracked field data and analysis output dating from before the project
was organized into sibling repos. **All of it has been migrated out and deleted.** The repo is now
only code + `birdnet_analyzer/` checkpoints. **Do not let audio, selection tables, or run outputs
accumulate here again** — everything below has a home in `call_library`.

| Was here | Size | Where it went |
|---|---|---|
| `sl1_segments/` | 904M | **deleted** — see the audition verdict below |
| `sl1/` | 353M | 3 wavs were byte-identical dupes of `testing_soundscapes/smithslake_hours/trial1_wavs/` → deleted; unique `.m4a` + `key_log` moved |
| `smithslakeoutput/` | 220M | R project; `.m4a`×2, key_logs, selection tables, `processing_script.R` extracted → `smithslake_hours/`; `Archive.zip` was a redundant zip of itself |
| `testing/` | 48M | `letnicaug_24` + `myhour` recognizers (**were nowhere else**) → `recognizers/` |
| `cc_output/` | 20M | 730 selection tables → `smithslake_hours/run_records_2024-10_1STSMM2/` |
| `sl1_output/` | 5.7M | selection tables → `run_records_2025-08/selection_tables/` |

⚠️ **`cc_output` was NOT reproducible and was nearly deleted as if it were.** Its 730 tables span
2024-10-18 → 2024-11-23 from recorder `1STSMM2`, and **those source recordings exist nowhere** in
`call_library` or `ecoacoustics` (the `First Smiths Lake Trip` folders hold `SL21_`/`SL25_2026*`
from a later trip). The tables are the only surviving record of that run, so they were preserved.
**Check for the source audio before calling any selection-table folder "reproducible".**

⚠️ **OneDrive placeholders will fake a checksum mismatch.** Verifying the `sl1` wavs against their
migrated copies first reported `DIFFERS` on two 130–170 MB files *with identical byte sizes* —
because the remote copies were dataless cloud stubs (`stat -f%b` → `0 blocks`) and `shasum` read
them mid-hydration. Re-hashing after forcing hydration (`cat file > /dev/null`, then confirm
`stat -f%b` is non-zero) gave **IDENTICAL** for all three. **Before trusting any hash/diff against
`call_library`, check `stat -f%b` — `0` means you are hashing a stub, not the file.**

**Audition verdict on `sl1_segments` (asked and answered 2026-08-09): nothing was worth adding to
`reallybig`.** It was `birdnet_analyzer.segments` output from a 2025-08-28 run — 1,962 clips, 28
folders, capped 100/species. Three reasons it was all deleted:
1. **All 28 species were already in `reallybig` and none were data-poor.** Thinnest was Eastern
   Shrike-tit at 115 clips, above the library's p10 of 79 (distribution: min 33, p10 79, median
   160, p90 312, max 910, n=448).
2. **It was the *global* BirdNET model's output, not a pelican's** — folders were bare en_US
   common names (`Gray Shrikethrush`, `Scarlet Myzomela`) rather than `reallybig`'s
   `Genus species_Common Name`. That naming shape is a **useful tell for which model produced a
   segments folder.**
3. **Confidence-capped at 100/species with the caps saturated at ~1.000** — a model's most
   confident hits on well-covered species is self-training material: no new information, real
   label-noise risk.

**General rule this establishes:** the informative output of an analysis run is its *mistakes and
low-confidence tail*, not its confident hits. For label-quality work use
`curation/identify_misclassifications.py` on embeddings, not a high-confidence segments dump.

Full ownership table, seams, venvs, shared data: **`~/Documents/ecoacoustics/ECOACOUSTICS.md`**.

## Repo structure

This is a fork of [birdnet-team/BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer).

```
git remote upstream  → https://github.com/birdnet-team/BirdNET-Analyzer.git
git remote origin    → https://github.com/wcornwell/BirdNET-Analyzer.git
```

### `.git` size — ~1.2 GB is the floor, anything above that is garbage

**~1.18 GB of it is packed upstream history you cannot remove:** every BirdNET model binary ever
committed (V1.0→V2.4, each in tflite FP32/FP16/INT8 plus SavedModel `variables.data-*`) — the
largest single blob is 69 MB. Shrinking it means rewriting history, which severs shared history
with `upstream` and breaks every future merge. **Don't.**

**Anything beyond that is loose-object garbage and should be collected.** On 2026-08-09 `.git`
was 2.4 GB: 1.24 GB of it was 1,977 **unreachable** blobs — big WAV soundscapes, a `.3gp`, a
379 MB raw float array — `git add`-ed into the index and later unstaged, never committed (zero
dangling commits). This is the residue of local analysis data staged before commit `447b650`
gitignored it. `git gc --prune=now` reclaimed all of it (2.4 GB → 1.2 GB, 8 packs → 1, `fsck`
clean).

⚠️ **Auto-gc will never clean this up on its own.** It triggers on loose-object *count* (default
6,700) and these files are enormous but few (~2,100). So this recurs silently every time large
audio gets staged by accident. **If `.git` is much over 1.2 GB, run `git gc --prune=now`.**

## Branch strategy

### `main` — working branch (use this for all analysis work)

**As of 2026-07-22, `main` uses upstream's `birdnet` library core** — the former
`sync-upstream-refactor` / `refactor-to-main-trial` content, swapped in from the old
TFLite embedding pipeline after the 5-phase pre-merge validation passed (execution log
below). The swap is merge commit `1d2ac30` (first parent `5dcbbe0` = old TFLite main,
second parent `b05939c` = the trial branch); its tree equals `refactor-to-main-trial`
exactly. **The old TFLite `main` is preserved as tag `pre-refactor-main` (`5dcbbe0`).**
Data loading runs at ~40 clips/sec (upstream #939), at parity with the old TFLite loader —
so the swap's justification was "stay current with upstream / shrink the fork," not speed.

⚠️ **macOS: training needs TF-first init.** The birdnet-library loader imports PyArrow,
whose statically-linked absl interposes TensorFlow's and deadlocks `model.fit` at epoch 1
if libarrow binds first. All training entry points route through repo-root **`train_tf_first.py`**
(imports/initialises TensorFlow *before* the trainer pulls in PyArrow); `train_pelican.sh`
already calls it. Distinct from the XLA-init guard in `config.py`.

Local changes on top of upstream (the fork feature layer, re-applied onto the refactor core):
- **Binary upsampling `added_count` fix** — per-class `added_count` (upstream left the binary branch on the shared-counter `len(y_temp)`)
- **Upsampling summary printout** — reference class, target min samples, 5 smallest classes
- **Per-class validation metrics** — `train_linear_classifier()` writes `<model>_validation_metrics.csv`, rows tagged **species** vs **non_target** (`Environment_*`/`Homo sapiens_*`; domestic animals are species), with `overall_*` / **`species_*`** (headline) / `non_target_*` micro+macro summary rows
- **helpers-as-non-events** — `config.NON_EVENT_PREFIXES`/`NON_EVENT_KEEP_CLASSES` + module-level `train.utils.is_non_event()`, wired through `train/core.py`+`cli.py` (see Training workflow; the `train_pelican.sh` DEFAULT)
- **`train_pelican.sh`** — training wrapper over the birdnet-library inline loader, via `train_tf_first.py`; exposes `--report-helpers` / `--keep-airplane-siren` (`--nonevent-helpers` is a no-op default)
- **eBird taxonomy maintenance** — `map_custom_to_global.py`, `update_ebird_taxonomy.py`, `eBird_taxonomy_codes_2025E.json` (`config.CODES_FILE`), custom↔global bridge (see Taxonomy maintenance)
- **`embedding_analysis/extract_embeddings.py`** — backbone-agnostic (`--version`, default 2.4; embeddings via `model_utils.get_embeddings_array_with_session`)
- **`embedding_analysis/extract_head_embeddings.py`** — the *custom head's* penultimate layer (see "Two embedding spaces" below)

### Two embedding spaces — pick deliberately

There are now two extractors and they are **not interchangeable**. Centroids,
misclassification suspects, near-duplicates and drift audits computed in one space are
meaningless against the other, so check which space an `.npz` is in before comparing it
to anything.

| script | dim | what it encodes | comparable across recognizers? |
|---|---|---|---|
| `extract_embeddings.py` | 1024 | base V2.4 backbone | yes — identical for every recognizer |
| `extract_head_embeddings.py` | 2048 | one recognizer's trained head | no — specific to that `.tflite` |

**Every `reallybig_pelican0-*_embeddings.npz` up to and including 0-19 is 2048-d head
space.** Pre-refactor, `extract_embeddings.py --model` *selected* the model
(`cfg.MODEL_PATH = args.model` → `model.embeddings` → the custom classifier's penultimate
layer). `f99b435` demoted `--model` to provenance-only when `model.embeddings` left the
core, so the same script now silently returns base backbone features instead. Its commit
message's "consumers stay compatible" is true of the container format (same keys, same
CSV) but **not of the vector space**, which is the part consumers actually use.

`extract_head_embeddings.py` restores the old behaviour without re-adding the removed API:
it reads the layer straight out of the exported graph (audio → backbone →
`GLOBAL_AVG_POOL` 1024 → `dense_1` relu 2048 → classes) via
`experimental_preserve_all_tensors`, locating it by shape rather than a hard-coded tensor
index. Audio handling matches the pre-refactor extractor (`open_audio_file` →
`crop_center`, one centred window), so output is directly comparable to the 0-18/0-19
artifacts. Verified equivalent two ways: reading the tensor, and projecting the pooled
1024-d output through the head's own `W1`/`b1` (`max|diff| = 7.2e-7`).

> **REMOVED in the swap — do NOT re-add** (see the execution log's "guiding principle"):
> the frozen-backbone `model.embeddings` / `predict` / `predict_with_perch` / `predict_filter`
> cluster. The refactor core is **backbone-agnostic** (`birdnet.load("acoustic", version, "tf")`
> via `model_utils`); those functions were a local bolt-on that re-welded a frozen V2.4 tflite.
> `flat_sigmoid` is kept (upstream + live via `train/utils`). The old macOS **TF XLA deadlock fix**
> in `model.py` is superseded by the XLA guard in `config.py` + the `train_tf_first.py` TF-first init.

### `sync-upstream-refactor` / `refactor-to-main-trial` — merged into `main`, both DELETED

Both branches are **gone as of 2026-07-23** (local and remote), after the upstream sync below
confirmed they had no remaining job. `sync-upstream-refactor` tracked upstream's
birdnet-library core and `main` absorbed its content via `refactor-to-main-trial` (tip
`b05939c`) in the `1d2ac30` swap; its last tip, `9dc98c4`, is **fully contained in `main`**, so
nothing was lost and it is restorable with
`git push origin 9dc98c4:refs/heads/sync-upstream-refactor` if ever needed.

**Why the staging branch was retired:** it existed to absorb the TFLite-vs-birdnet-library
divergence away from `main`. That divergence is gone, and the 2026-07-23 sync demonstrated the
replacement workflow in practice — `git merge upstream/main` on a throwaway branch cut from
`main`, tested, fast-forwarded, branch deleted. **Use that pattern for future syncs**; there is
no longer a long-lived staging branch to route through.

**The only remaining branch is `main`.** Anything else you see is a `remotes/upstream/*`
feature branch belonging to the upstream project, not ours.

⚠️ **`git branch -a` lies about this — it shows ~24 phantom `remotes/origin/*` branches.** They
are **stale local tracking refs** for branches long since deleted on GitHub and never pruned
(`birdnet-lib`, `upsampling-fix`, `sync-upstream-refactor`, …). Cleared with
`git remote prune origin` on 2026-08-09, and `bodangora-library-flag` (merged, tip `0e5556a`,
already in `main`) deleted from origin the same day. **`origin` now has exactly `main` and
`gh-pages`.** Use `git ls-remote --heads origin` for ground truth — never `git branch -a`.

**Syncing `main` with upstream — now a plain merge** (the TFLite divergence that made this
dangerous is gone):
```bash
git fetch upstream
git checkout main
git merge upstream/main
# conflicts likely on the fork feature layer: config.py (NON_EVENT_PREFIXES / CODES_FILE),
# train/utils.py (is_non_event), model.py (validation metrics). Resolve KEEPING our layer.
```

**Upstream sync — DONE 2026-07-23.** The 8 commits `9dc98c4..upstream/main` (through #962
"Improve GUI startup time") were merged on branch `upstream-sync-2026-07`. It went far more
smoothly than the pre-merge note predicted: **one conflict**, the import block at the top of
`model.py` (upstream dropped `config as cfg` + `utils`; resolved by keeping `cfg`, which the
validation-metrics block still uses, and dropping `utils`, whose only use — `save_params_to_file`
— upstream moved out). Everything else auto-merged; the whole fork feature layer survived
(verified by inspection, not assumption: `NON_EVENT_PREFIXES`/`NON_EVENT_KEEP_CLASSES`/`CODES_FILE`,
`is_non_event()`, the `added_count` fix, the upsampling summary, the species/non_target metrics
block, `cfg.CUSTOM_CLASSIFIER = output`, both CLI flags). **Tests: 502 passed / 2 skipped / 0
failures** (up from 474 — upstream added ~28). ruff clean.

**Two upstream behavior changes this pulled in — not merge damage, deliberate upstream moves:**
1. **`<name>_Params.csv` is gone**, replaced by **`<name>.birdnet.train-params.csv`** (one *row*
   per parameter instead of one column), written from `train/utils.py` rather than `model.py`;
   `config.TRAIN_PARAMS_SUFFIX` holds the suffix. Upstream's new `params.py` still reads the old
   file, so historical `pelican0-*_Params.csv` stay loadable. Anything downstream globbing
   `*_Params.csv` needs updating.
2. Upstream's `dev` extra bumps **ruff 0.14.0 → 0.15.10** (we still run 0.14.0 locally, clean).

Local follow-ups committed on top (`b509ddf`): the new params dict is explicit and had **omitted
the helper-mode settings**, so a run's `--non_event_prefixes`/`--keep_as_class` were no longer
recoverable from its artifacts — now recorded from `cfg`; and `train_pelican.sh`'s no-clobber
preflight was still checking the never-written `_Params.csv`, now pointed at the new name.

**Post-merge smoke (2026-07-23, subset — 5 species + 4 helper folders, ~2.3k clips):** both
helper modes end-to-end, exit 0, no macOS deadlock. Default mode → 5 species labels (all 4
helpers correctly neuron-less non-events); `--keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren"`
→ 7 labels with the two carve-outs present and `Environment_*` still absent, and they are tagged
`non_target` (not `species`) in the metrics CSV, which keeps main's full format
(`overall_*`/`species_*`/`non_target_*` summary rows). A **full-`reallybig` run was judged
unnecessary** here: upstream's diff is GUI/logging/Docker by theme and touches `model.py` by only
−11 lines, so there is no plausible species-quality surface — unlike the core swap, which needed
all 5 phases.

**Upstream sync — DONE 2026-10-19 (branch `upstream-sync-2026-10`): the 3.0 release.** All 33
commits `b1d866b..c2c71a1` (#967 → #1005), merged in two steps so #968 and the dependency bump
landed as separate merge commits: `770803b` (through #968/#971) then `86d7f79` (3.0 + ONNX +
`birdnet` 0.2.16 → **1.1.0** + soxr). The parked `upstream-sync-2026-08` (`f82d4fb`) had been
garbage-collected; its resolutions were redone from these notes and are recorded below.

What the release is: BirdNET **3.0** is the default *analysis* model (ONNX), `--sensitivity` is a
no-op on 3.0, Python 3.11+, and the trained-classifier default output is CWD-relative (harmless —
`train_pelican.sh` passes `-o`). **Training and custom classifiers stay on the 2.4 base** in this
release — *superseded a week later by #1007, see "BirdNET 3.0 training — merged" below.* See
upstream's `docs/breaking-changes.rst`.

**Conflicts and how they were resolved (reuse for the next sync):**
- `.github/workflows/*` — take upstream, always. Nothing fork-specific was ever in them.
- `CLAUDE.md` — upstream ships its own now; keep ours (it conflicts every sync — see the open
  follow-up below).
- `__init__.py` — keep BOTH: our TF env-var block first, then upstream's `apply_model_directory()`.
- `model.py` `upsample_core` binary branch — take upstream. #968's per-class
  `missing_samples = min_samples - len(source_indices)` **subsumes our old `added_count` fix**, so
  the fork no longer carries one. ⚠️ Trap: git auto-merges our `added_count += 1` *outside* the
  conflict region — a `NameError` on every binary run. Delete it.
- `audio.py` — keep our `mpeg_safe_open` lock around both `librosa.load` calls, take upstream's
  `res_type="soxr_hq"`. (`resampy` is gone upstream and nothing here needs it any more.)
- `train/utils.py` — adjacency only: keep our species-list / non-event block, take upstream's
  shorter batch-size comment.
- On top: our upsampling summary's `class_counts` aligned to `(y == 1).sum(axis=0)` (#968 made
  `min_samples` count only `== 1`, because multi-label negatives are encoded **-1**).

**Behaviour changes pulled in (deliberate upstream moves, not merge damage):**
1. **#968: pre-split and post-upsampling shuffles removed** (compensated by `shuffle=True` in
   `fit`) ⇒ the train/val partition for a given seed changed. **Pelican numbers are NOT
   seed-comparable across this merge.**
2. **#968: `_check_input_folders` hard-fails** on any class folder with no supported audio (an
   unmaterialised OneDrive folder now aborts a run). It did not trip on `reallybig`.
3. **#996: resampling `kaiser_fast` → `soxr_hq`.** Embeddings of clips recorded **below 32 kHz
   shift** (100 non-48 kHz `reallybig` clips, same 2.4 backbone, old vs new loader — cosine
   by source rate: 44.1/96 kHz 1.000, 32 kHz 0.998, 16 kHz 0.977, 24 kHz 0.959, **22.05 kHz
   0.885, min 0.800**). The difference is in how the empty band above a low-rate file's Nyquist
   is upsampled. ⇒ **embedding `.npz`/centroids extracted before 2026-10-19 are not
   interchangeable with ones extracted after** for low-rate clips — re-extract rather than mix.

**Gate — PASSED 2026-10-19** (all in an isolated worktree `../BirdNET-sync` + its own venv, so
the shared `.venv` and soundscape-eval kept seeing `main` throughout):
- Tests **593 passed / 6 skipped** (all 6 = gradio GUI tests), ruff 0.15.10 clean.
- Full-`reallybig` `run0-20-bn` (same recipe as `run0-19-bn`, 98,651 samples, 464 labels).
  In-sample species macro P/R 0.9808/0.9332 vs 0.9800/0.9339; all Δ ≤ 0.0034. ⚠️ the library
  itself moved between the two runs (1 rename, 14 classes topped up, incl. Treecreeper +168 and
  Grey Butcherbird +188), a small confound.
- soundscape-eval (`configs/{smithslake,fowlers}_run0-20.yaml`): Smiths Lake AUPRC 0.1933 vs
  0.1930, bestF1 0.2486 vs 0.2480; Fowlers AUPRC 0.2152 vs 0.2175, bestF1 0.2693 vs 0.2718.
  Signs flip between soundscapes, per-species churn symmetric (7↑/6↓ and 8↑/10↓ at |ΔAP|>0.01)
  ⇒ seed noise, same verdict shape as Phase 4. Fowlers R@1FP/hr 0.179 vs 0.110 is Bustard-driven
  (indicative only).
- Smoke: `train_pelican.sh` default / `--report-helpers` / `--keep-airplane-siren` /
  `--species_list … --unlisted drop` → 5 / 9 / 7 / 2 labels as expected, Airplane/Siren tagged
  `non_target`; `extract_embeddings.py` (1024-d, `backbone_version=2.4`),
  `extract_head_embeddings.py` (2048-d), `analyze --classifier`; soundscape-eval's Perch and
  bare-head paths against the merged engine.

### BirdNET 3.0 training — merged 2026-10-26 (`d8ea11a`), on `main` since 2026-10-26

Upstream landed it as **#1007** (`70a9b9f`, 2026-10-19). The 18 commits `c2c71a1..8095252`
(#1007 → #1025) were merged on top of the 10-19 sync; `birdnet` → **`>=1.3.1`**.

What it gives us: `train --birdnet {3.0,2.4}`. 3.0 = **32 kHz**, 3 s, **1280-d** embeddings
(extracted on ONNX). A 3.0 `.tflite` takes input `x` = **96000** samples and has named outputs
`predictions` (**probabilities** — sigmoid inside the graph) + `embeddings`; a 2.4 one stays
`input_audio` = 144000 → logits. `analyze` reads the version from the file; a 3.0 custom
classifier always analyses at **batch 1**. **A cache trains on the backbone it was embedded with,
whatever `--birdnet` says** (verified: a 3.0 cache given `--birdnet 2.4` records 3.0).
- ⚠️ **`--birdnet` defaults to the NEWEST model upstream.** `train_pelican.sh` always passes
  it explicitly — **3.0 by default since 2026-10-28** (was 2.4), `--birdnet 2.4` to override —
  and prints `Backbone:` in the banner (support detected by parsing `train/core.py`, no TF
  import; a 2.4-only engine falls back to 2.4). A 3.0 head is a third embedding space —
  incomparable with any 2.4 head, centroid or misclassification `.npz` — so 3.0 runs start a
  new series, `run1-0` (`training_runs/CLAUDE.md`, "Run naming").
- Other training-behaviour changes riding along: `--val_split` must be > 0 (validation always
  from training data; `--test_data` only evaluates); early stopping watches `--autotune_metric`
  (AUPRC) and restores the best epoch; `HardLabelAUC` fixes double-sigmoid AUPRC/AUROC;
  `--label_smoothing` uses the standard formula; new `linear` upsampling mode.
  (`curation/npz_to_cache.py`'s `--val_split 0.0` is local to that script — it builds its own
  `x_test`/`y_test` and never reaches `train` — so it is unaffected.)

**Conflicts (6) and resolutions:** `__init__.py` — git **dropped `import os` while keeping our
`os.environ` lines** (NameError on import); now follows upstream #1023 (cosmetic TF vars live in
`utils.configure_runtime()`, a library import must set no env var — `tests/test_runtime.py`
asserts it) and keeps **only the macOS XLA guard, as a `setdefault`**; its duplicates in
`config.py`/`model.py` are gone (`train_tf_first.py` sets all of them itself before TF).
`audio.py` keep the mpeg lock + take lazy librosa. `config.py`/`core.py` keep both.
`model.py` keep `compute_validation_metrics` + CSV block beside `HardLabelAUC` /
`@_hiding_cuda_devices` / best-epoch restore (our manual CUDA restore dropped; `SCRIPT_DIR` →
`cfg.SCRIPT_DIR`). `train/utils.py` our label guard on `parse_folder_labels`, both
`labels=`/`monitor=`.

**Two breakages outside the conflict regions — check for these again next sync:**
1. `_load_from_cache` returns the backbone **version as an 8th value**; our
   `_scope_cache_to_species_list` must pass it through, or cache + `--species_list` → TypeError.
2. The params file is written via `by_header(TRAIN_PARAMS, …)`, which **silently drops keys not
   in the table**. Our provenance rows (`Non-event prefixes`, `Non-event keep classes`,
   `Species list`, `Species list entries`, `Unlisted handling`) are appended *after* it.

**Validated (sync worktree venv, birdnet 1.3.1, TF 2.21):** 857 passed / 30 skipped (3.0
export tests skip until the 3.0 pb is cached; `tests/train/test_export.py` 10/10 once it is).
Needs upstream's new **`tests/data` submodule** (`git submodule update --init tests/data`,
71 MB) or 2 export tests fail on missing audio. Subset smoke (5 species + Rain/Wind/Airplane/
Siren/Noise, 5 epochs): 2.4 default/airsiren → 5/7 labels; 3.0 default/report-helpers/airsiren/
`--unlisted drop` → 5/9/7/2; 3.0 cache + species list → 2, records 3.0. 3.0 trains ~2× slower
end-to-end than 2.4 on the subset (≈160 s vs 77 s). `analyze --classifier` works on both.

**Full-`reallybig` gate — PASSED 2026-10-26, then landed.** `sync1007-bn24` and `sync1007-bn30`
(same recipe, `train_pelican.sh` defaults; 3.0 via `--save_cache_to`, cache in
`training_runs/out/sync1007/cache30.npz`, 689 MB) scored with `run0-20-bn` in soundscape-eval
(`configs/{smithslake,fowlers,sarah_soundscapes}_sync1007.yaml`). AUPRC / bestF1 / R@1FP/hr:

| model | Smiths Lake (25 spp, 60 min) | Fowlers Gap (38 spp, 400 min) | Sarah's (51 spp, 1720 min) |
|---|---|---|---|
| `sync1007-bn30` (3.0) | **0.230** / **0.283** / 0.061 | **0.263** / **0.329** / 0.118 | **0.250** / **0.312** / **0.276** |
| `sync1007-bn24` (2.4) | 0.182 / 0.238 / 0.141 | 0.223 / 0.281 / 0.102 | 0.210 / 0.261 / 0.184 |
| `run0-20-bn` (2.4) | 0.193 / 0.249 / 0.172 | 0.215 / 0.269 / 0.179 | 0.205 / 0.257 / 0.250 |

- **2.4 no regression:** bn24 vs run0-20-bn per-species AUPRC (>0.02) is 1 better / 5 worse on
  Smiths Lake, 8 / 4 on Fowlers, 11 / 3 on Sarah's, median Δ ≈ 0 on all three — seed noise.
- **3.0 wins ranking on all three soundscapes**, broadly: vs bn24 11/3 (Smiths), 17/5 (Fowlers),
  26/9 (Sarah's) species better/worse. **The operating point is mixed:** in-sample recall@0.5 is
  0.78 vs 0.93, and R@1FP/hr trails run0-20-bn on Smiths Lake and Fowlers (each budget carried by
  one leak species — indicative only) but leads on Sarah's, the largest set and the only one
  where no single species carries the budget. **`train_pelican.sh` switched to 3.0 by
  default on 2026-10-28**, after the calibration check below.
- **Calibration check (2026-10-28) — no global offset; the R@1FP/hr losses are not a reason
  to stay on 2.4.** Recall re-computed at 0.5/1/2/4 FP/hr and with the top 1–2 leak species
  silenced (each model's threshold is set by its own leak, so a uniform score shift cannot
  matter — only the *shape* of the leak tail can):

  | soundscape | model | R@0.5 / 1 / 2 / 4 FP/hr | R@1, top-2 leak silenced |
  |---|---|---|---|
  | Smiths Lake (1.0 h) | 3.0 | 0.061 / 0.061 / 0.104 / 0.190 | 0.061 |
  | | run0-20-bn | 0.172 / 0.172 / 0.172 / 0.190 | 0.190 |
  | Fowlers (6.6 h) | 3.0 | 0.096 / 0.118 / 0.169 / **0.213** | 0.134 |
  | | sync1007-bn24 | 0.086 / 0.102 / 0.118 / 0.126 | 0.179 |
  | | run0-20-bn | 0.179 / 0.179 / 0.188 / 0.192 | 0.208 |
  | Sarah's (28.4 h) | 3.0 | **0.248 / 0.276 / 0.317 / 0.365** | **0.315** |
  | | sync1007-bn24 | 0.177 / 0.184 / 0.192 / 0.210 | 0.280 |
  | | run0-20-bn | 0.229 / 0.250 / 0.272 / 0.292 | 0.290 |

  - **Smiths Lake is one hour, so a 1/hr budget is ONE false alarm** (3.0's was a single
    Large-billed Scrubwren). At 4/hr 3.0 and run0-20-bn tie. Don't read operating-point
    metrics off it.
  - **Fowlers is a real but small strict-budget loss with a different cause per backbone.**
    2.4's leak is almost all **Australian Bustard** (22–26 of its runs at 4/hr; silencing it
    lifts bn24 0.10 → 0.18). 3.0 has no Bustard problem; its leak is spread over ~10 plausible
    confusions (flies, field cricket, sheep), so silencing doesn't help — a broader high tail,
    which costs it at ≤1/hr and wins from ~2–4/hr.
  - **3.0 fixes the Australian Bustard leak** (bn24: 110 runs on Sarah's at 4/hr; absent from
    3.0's top three). **Thick-billed Grasswren and Mottled Bush Cricket leak in every model** —
    library-curation targets regardless of backbone.
  - Script: re-runnable from soundscape-eval with `compare.build_comparison` +
    `metrics.matched_threshold`/`binned` on the three `*_sync1007` manifests.
- 3.0 extraction of full `reallybig` took ~46 min here (not the 2–3 h the pilot suggested);
  scoring a 3.0 tflite at batch 1 ran Fowlers' 400 min in ~7 min.

**Shared `.venv` swap — DONE 2026-10-26.** `.venv` → `.venv-pre1007` (birdnet 0.2.16, TF 2.20);
new `.venv` = `pip install -c <sync venv freeze> -e ".[train,tests,dev,embeddings]"`, i.e. pinned
to the versions the gate ran on. ⚠️ **Unconstrained, pip resolves librosa 1.0 / pandas 3.1 /
python-dateutil 3.0**, which the gate never ran — rebuild with constraints or re-validate.
⚠️ **The pyproject extras are not the whole venv.** It also carries editable
**`../soundscape-eval`** (install `--no-deps`, or pip swaps in score-ensemble from GitHub) and
**`../score-ensemble`**, plus **`resampy==0.4.3`** (soundscape-eval's and a2o-bats' `kaiser_fast`
resampling). `run_dual.sh`'s compare step and `a2o-bats/compete/score_clips.py` run soundscape-eval
from THIS venv. The 2026-10-26 rebuild missed all three until 10-28; restored, and a cached
re-compare under this venv matched the soundscape-eval venv's summaries byte for byte. Before
deleting `.venv-pre1007`, diff its `site-packages/*.dist-info` names against the new venv.
`tests/data` submodule initialised in this checkout. Suite 863 passed / 24 skipped (same as the
sync venv), ruff clean, curation scripts and soundscape-eval (checkpoints via
`SOUNDSCAPE_EVAL_BIRDNET_CHECKPOINTS`) unaffected — a cached re-compare was byte-identical.
Delete `.venv-pre1007` once nothing has needed it for a while.

**Run `./check_upstream.sh` monthly (and at the start of a session here).** Read-only. It prints
behind/ahead, a `merge-tree` conflict dry run, every incoming commit tagged RED (training / model
/ audio / `model_utils` / birdnet pin → full train + soundscape-eval) · AMBER (config / cli /
analyze / embeddings / deps → tests + subset smoke) · GREEN (GUI / docs / CI / lang → tests),
whether upstream still marks 3.0 training unsupported, **unmerged upstream branches touching the
training path** (that is where 3.0 training showed up first), and the birdnet pin vs PyPI.
Merge GREEN/AMBER batches promptly; when a RED commit is in range, merge up to `<sha>^` first so
only the RED part pays for the expensive gate.

**Open follow-ups:** move this file to `.claude/CLAUDE.md` so root `CLAUDE.md` tracks upstream
and stops conflicting; and pull the fork layer (`compute_validation_metrics`, `is_non_event` /
species-list helpers, the four train flags) out of `model.py` / `train/utils.py` / `cli.py` into
a fork-owned module with one-line hooks — 3.0 training lands on exactly those files.

⚠️ **While a sync branch is checked out, `soundscape-eval` sees it** (editable install → whatever
branch is checked out here). Do syncs in a separate worktree + venv, as above.

### Merging the refactor into `main` — testing plan (formulated 2026-07-20; **EXECUTED 2026-07-22, all phases passed → `main` swapped, commit `1d2ac30`**)

**This is a real option now that #939 removed the speed blocker, but it is not additive:**
merging the refactor into `main` means **swapping main's TFLite core for the
birdnet-library core, then re-applying main's feature layer on top.** The two branches
rewrote the same files (`model.py`, `train/utils.py`) in opposite directions.

**Framing:** speed is only at *parity*, so the merge's justification is "stay current with
upstream / shrink the fork," not performance. The bar is therefore **no regression** — on
species P/R, on behavior, or on the test suite. Any regression kills the merge (no speed
win to trade against).

**Main-only features missing from `sync-upstream-refactor` (must be ported first, or they
silently regress):**
- **helpers-as-non-events** (`--non_event_prefixes` / `--keep_as_class`) — ABSENT. This is
  the current `train_pelican.sh` *default*, so losing it is a real behavior regression.
- **`train_pelican.sh`** — ABSENT, and it assumes the no-cache inline TFLite pipeline; the
  refactor uses a `cache_file`/`cache_mode` workflow.
- (species/non_target metric tagging is already present on the refactor branch.)

**Phased plan (do all merge/port work on a throwaway branch off `sync-upstream-refactor`,
e.g. `refactor-to-main-trial` — nothing lands on `main` until Phases 1 and 4 pass):**

| Phase | What | Gate |
|---|---|---|
| **0 — Feature port** (prereq) | Port helpers-as-non-events (`config.NON_EVENT_PREFIXES`/`NON_EVENT_KEEP_CLASSES` + `is_non_event()` through `train/core.py`+`cli.py`); rework `train_pelican.sh` for the refactor loader (decide cache vs inline). | `train --help` shows the flags; `is_non_event()` unit test passes. |
| **1 — Test suite** | `pytest tests/ --timeout=120 -q --ignore=tests/analyze/test_analyze.py` vs main's 363 passed / 4 skipped. | No new failures beyond the two known-broken (test_analyze fixture, test_embeddings `SQLiteUSearchDB`). |
| **2 — Embedding equivalence** (diagnostic, decision-critical) | Same N clips through both pipelines (main TFLite `model.embeddings` vs refactor `encode_session`); compare per-clip cosine sim + max-abs diff. | Not a hard gate. cos > 0.999 = clean swap (heads + shared npz transfer). Drift = retrain-and-regenerate-everything migration (existing pelican `.tflite`, centroids, misclass npz no longer comparable) → raises cost, feeds go/no-go. |
| **3 — Full-scale speed** | End-to-end `train` on the **full `reallybig`** library, both branches, wall-clock. | Refactor within ~1.5× of main (soft). |
| **4 — Model quality** (headline gate) | Train a pelican model on each branch, identical hyperparameters, **full `reallybig`**; compare `species_macro`/`species_micro` P/R, then score both **whole models in `soundscape-eval`** (the arbiter) against the labeled soundscape. | Refactor species P/R ≥ main within seed noise. Any real regression = do not merge. |
| **5 — Workflow smoke** | `train_pelican.sh` in all three modes (default, `--report-helpers`, `--keep-airplane-siren`), `extract_embeddings.py`, a short `analyze` run. | All complete and emit expected artifacts. |

**Go/no-go:** merge only if Phase 1 (no new test breakage) **and** Phase 4 (no species P/R
regression) pass. Phase 2 decides clean-swap vs costly-migration; Phase 3 is informational.
**Decisions locked:** Phases 3–4 run on the **full `reallybig`** library (not a subset).

#### Execution log (branch `refactor-to-main-trial`, off `sync-upstream-refactor`)

- **Phase 0 — feature port: DONE** (commit `ff13e15`). Ported helpers-as-non-events
  (`config.NON_EVENT_PREFIXES`/`NON_EVENT_KEEP_CLASSES`, module-level
  `train.utils.is_non_event()`, `cfg` wiring in `core.train()`, flags in
  `cli.train_parser()`), plus `train_pelican.sh` (identical flags/hyperparameters, runs
  the refactor's inline loader). Unit test `tests/train/test_non_events.py`.
- **Phase 1 — test suite: PASS.** 474 passed, 0 failures (the merged branch has *more*
  tests than main's 363; all green).
- **Phase 2 — embedding equivalence: PASS (clean swap).** 300 clips through both
  pipelines (main TFLite `model.embeddings` vs refactor `encode_session`): **cosine
  = 1.000000, max|diff| = 0.0000, identical L2 norms.** The refactor's tf-SavedModel
  embedding is bit-identical to main's TFLite embedding ⇒ trained heads transfer,
  existing pelican `.tflite`/centroids/misclass `.npz` stay valid. **No
  retrain-and-regenerate-everything cost.**
- **Bugs found + fixed running training end-to-end** (commit `83e7b68`; both were latent
  because training had **never completed on the refactor branch before** — issue #1
  blocked it):
  1. **macOS TF × PyArrow absl deadlock.** The refactor loader imports PyArrow
     (`libarrow`), whose statically-linked absl interposes TensorFlow's; if libarrow
     binds first, TF's eager executor deadlocks on `absl::Notification` during
     `model.fit` (0% CPU hang at epoch 1). **Fix:** launch training via a wrapper that
     imports/initialises TensorFlow *before* PyArrow (`pelican-ab-run/train_wrapper.py`).
     Distinct from the XLA-init guard already in `model.py`. **Any refactor-branch
     training on macOS needs this TF-first init.**
  2. **Validation-metrics CSV broken on the refactor path.** `model.py` referenced `cfg`
     7× but only imported `RANDOM_SEED` → `NameError`, skipped the CSV; and
     `train_linear_classifier` derives the CSV path from `cfg.CUSTOM_CLASSIFIER`, which
     the refactor's param-passing flow doesn't set → CSV misfiled as
     `sequential_validation_metrics.csv`. **Fix:** import `config as cfg` in `model.py`;
     set `cfg.CUSTOM_CLASSIFIER = output` early in `train_model`.
- **Known follow-ups (do NOT block the A/B; fix before an actual main merge):**
  - ~~The refactor's validation-metrics CSV is a **simpler format** than main's (no
    `species_macro`/non-target summary rows).~~ **RESOLVED 2026-07-22 (commit `10a3c11`).**
    Ported main's fuller format: partitions species vs non_target
    (`Environment_`/`Homo sapiens_` prefix), writes `overall_*` + `species_micro/macro`
    + `non_target_micro/macro` summary rows, and tags each per-class row species vs
    non_target (was hardcoded `species`, mistagging helper columns in `--report-helpers`
    mode). Verified on a subset in report-helpers mode. Format now matches main.
  - ~~`model.py` has ~51 ruff/undefined-name errors in the kept TFLite/perch functions.~~
    **RESOLVED 2026-07-21 (commit `c41154d`) by REMOVAL, not porting.** Key realization
    (W.C.): the point of upstream's refactor is a **backbone-agnostic core** — swap the
    embedding model across foundation-model generations via `model_utils.get_embeddings_array(
    signals, version=…)` → `birdnet.load("acoustic", version, "tf")`. Upstream's refactored
    `model.py` has **none** of `embeddings`/`predict`/`predict_with_*`/`load_model` (ends at
    `flat_sigmoid`); those were a **local bolt-on** we carried over, and on this branch they
    were dead (no live callers; `analyze` uses `model_utils.run_inference`) AND broken.
    Reviving them would re-weld a frozen-V2.4-tflite backbone — the exact coupling the
    refactor dissolved. **So we deleted the cluster** → preserves generality, converges
    `model.py` toward upstream (1164→1067 lines), clears all 51 ruff errors (package clean,
    474 tests pass). `flat_sigmoid` kept (in upstream + live via `train/utils`). **Guiding
    principle going forward: stay close to upstream's future-proofed design; don't re-add
    frozen-backbone code.** (Sibling-repo follow-up — `soundscape-eval`'s BirdNET/Perch/geo
    paths off `birdnet_analyzer.model` — **now fully DONE 2026-07-22**, soundscape-eval commit
    `3c8f394`; scoring no longer needs a `main` worktree. See the decoupling entry below.)
- **Phases 3–4 overnight A/B — SET UP & RUNNING.** Launcher: `pelican-ab-run/run_overnight.sh`
  (persistent, session-independent) → sequential full-`reallybig` (~78.8k clips) training
  of `pelicanAB-main` (TFLite, from a `main` worktree at `pelican-ab-run/wt-main`) and
  `pelicanAB-refactor` (birdnet-lib + #939), identical hyperparameters, helpers-as-non-events
  default. Each arm pinned to its tree via `cd`+`PYTHONPATH`; TF-first wrapper; checkpoint +
  no-clobber preflights. Both arms smoke-validated end-to-end (produce `.tflite` +
  `_validation_metrics.csv`). Logs: `pelican-ab-run/overnight_<ts>_{main,refactor}.log`.
- **Phases 3–4 A/B — RAN 2026-07-20 (both arms exit 0).** `pelicanAB-main` (TFLite, 64 min)
  and `pelicanAB-refactor` (birdnet-lib+#939, 31 min); both 430 species labels, 78,764
  samples. **Species P/R is a dead heat:** main macro P/R 0.9881/0.9510, micro 0.9860/0.9587;
  refactor macro 0.9873/0.9476, micro 0.9871/0.9591 — all Δ ≤ 0.0034. Per-class (430): mean
  Δ ≈ 0, **symmetric** churn (recall 62 worse / 55 better >0.05), biggest swings all tiny-N
  species (herons, Australian Bustard) = val-split noise, not regression. **Structural reason
  it must be noise:** Phase 2 proved the head's input embeddings are bit-identical, so any
  model difference can only come from training randomness (seed/shuffle/split). **Caveats:**
  (a) the 64-vs-31-min gap is a **cache-ordering artifact, NOT a speedup** — the main arm ran
  first and paid the cold read of all 78,764 clips (37–140 f/s) while a stray `soundscape_eval.run`
  ate a core; the refactor arm ran second on a warm OS file cache (2,000–3,570 f/s on
  already-read clips) and a free machine. On its own *cold* folders the refactor was 18–28 f/s
  (if anything slightly slower), matching the controlled 40.5-vs-45–50 clips/sec benchmark ⇒
  pipelines are ~parity on compute; (b) per-class deltas compare *different* val splits ⇒ not
  apples-to-apples.
  **Verdict: no regression; migration remains clean-swap viable.**
  **Next (the real gate):** score both saved models in `soundscape-eval` against the labeled
  soundscape (same external test set removes the val-split confound).
- **Phase 4 real gate — RAN 2026-07-21, both soundscapes, PASS.** Scored the two overnight
  `.tflite` models (`configs/pelicanAB.yaml` / `configs/fowlers_pelicanAB.yaml` in
  `soundscape-eval`) via `python -m soundscape_eval.run all`, run with
  `PYTHONPATH=pelican-ab-run/wt-main` since this branch deleted `model.embeddings`/`predict`
  (see above) — the adapter still needs those, so scoring goes through the `main` worktree
  regardless of which branch trained the head (Phase 2's bit-identical-embeddings result is
  exactly why that's valid).
  - **Smiths Lake** (26 species, 60 min labeled): main leads 4/5 metrics — AUPRC 0.177 vs
    0.1725, bestF1 0.2339 vs 0.2225, R@P50 0.1439 vs 0.1369, R@P80 0.1098 vs 0.1076;
    refactor only wins R@1FP/hr (0.2073 vs 0.1768).
  - **Fowlers Gap** (38 species, 400 min labeled, 4 sites/years — larger + independent):
    refactor leads **5/5** metrics — AUPRC 0.2143 vs 0.2076, bestF1 0.266 vs 0.2593, R@P50
    0.1757 vs 0.1728, R@P80 0.1528 vs 0.1376, R@1FP/hr 0.1937 vs 0.1819.
  - **Verdict: the sign flips between the two soundscapes ⇒ the Smiths Lake gap reads as
    val-split/seed noise, not a real regression** — consistent with Phase 2 (bit-identical
    embeddings mean any model delta can only be training randomness) and with the in-sample
    A/B's symmetric per-class churn. No consistent species-quality regression found across
    two independent labeled soundscapes. **Phase 4 gate: PASS.**
  - Configs: `soundscape-eval/configs/{pelicanAB,fowlers_pelicanAB}.yaml`. Outputs:
    `call_library/experiments/model_compare_{pelicanAB,fowlers_pelicanAB}/`.
  - **Outstanding before an actual main merge:** move `soundscape-eval`'s BirdNET adapter off
    `model.embeddings`/`predict` (removed from this branch) onto
    `model_utils.get_embeddings_array`, so scoring doesn't depend on a `main` worktree.
    **RESOLVED 2026-07-22** (soundscape-eval `3c8f394`) — and extended to the Perch/geo paths too;
    scoring no longer needs a `main` worktree. See the decoupling entry below.
  - **Next:** Phase 5 workflow smoke test (`train_pelican.sh` all 3 modes, `extract_embeddings.py`,
    a short `analyze` run).
- **`extract_embeddings.py` ported off `model.embeddings` — DONE (refactor branch).** The
  script didn't exist on `refactor-to-main-trial` (it's a main-only feature that used the
  removed TFLite `model.embeddings`). Re-added it backbone-agnostic: backbone selected by
  `--version` (default `2.4`), embeddings pulled via `model_utils.get_embeddings_array_with_session`
  over **one** reused `encode_session` (opening a session per species folder would pay the
  worker-pool setup ~430×); embedding geometry (SR, window) read from the loaded model, not
  from config globals (the refactor config drops `MODEL_PATH`/`SIG_LENGTH`). `--model` is now
  **provenance-only** (recorded in the npz, does not select the backbone). Output `.npz`/
  `_centroids.csv` format unchanged (added a `backbone_version` key), so
  `curation/identify_misclassifications.py` + birdnetEmbed stay compatible. On V2.4 the
  embeddings are the same base 1024-d features the old path produced (Phase-2 bit-identity),
  and the shared-session vs per-species-session outputs are identical (max|diff| 0.0).
  ruff-clean; smoke-tested on 2 species. ⚠️ The main-branch "Embedding analysis" section below
  still documents main's `--model`/TFLite interface — correct for `main`, superseded on the
  refactor branch by `--version`.
- **Phase 5 scope decision (2026-07-21):** full `reallybig`, all 3 `train_pelican.sh` helper
  modes (default, `--report-helpers`, `--keep-airplane-siren`) run end-to-end — most faithful
  to production, ~1.5–3 hr combined (each mode trains on the full ~78.8k-clip library like
  Phase 3–4 did).
- **Phase 5 workflow smoke — RAN 2026-07-21, PASS (with one transient).** Runner:
  `pelican-ab-run/run_phase5.sh` (persistent; 3 train modes full `reallybig` +
  `extract_embeddings.py` on a 20-species subset + a short `analyze`). Outputs under
  `scratchpad/phase5_<ts>/`. Results:
  - **Bug found + fixed before launch: `train_pelican.sh` still deadlocked on macOS.** The
    Phase-0 port called `python -m birdnet_analyzer.train` directly, so it hit the TF×PyArrow
    absl deadlock at epoch 1 (the same one the overnight A/B dodged via a wrapper). Fixed by
    routing through a new repo-root **`train_tf_first.py`** launcher (imports/initialises
    TensorFlow before the trainer pulls in PyArrow, via `runpy` so all CLI args pass through);
    `train_pelican.sh` now calls it instead of `-m`. Committed `16109ab`. **Any refactor-branch
    training entry point on macOS needs this TF-first init** (distinct from the XLA guard in
    `config.py`). De-risked on a 3-folder subset before the full run.
  - **`train_pelican.sh` default: ✅** 35m41s, 430 species rows (helpers → non-events).
  - **`train_pelican.sh` --report-helpers: ✅** 30m43s, **444** species rows. The 430→444 delta
    is exactly the ~14 `Environment_*`/`Homo sapiens_*` helper folders flipping from non-events
    to positive reported classes — confirms the helper-mode switch works end-to-end.
  - **`train_pelican.sh` --keep-airplane-siren: ✗ transient (NOT re-run, by decision).** Crashed
    with `Bus error: 10` (SIGBUS) at 44s while *decoding audio* (~folder 30, `Ardeotis
    australis`) — before `--keep_as_class`'s label-encoding (the mode's only distinguishing
    behavior) ever runs, and its banner/args were correct. Arms 1–2 loaded the same full library
    fine, so this reads as a transient in the parallel decode (OneDrive on-demand
    file-materialization or memory pressure after two back-to-back full trainings on the
    network-backed mount), not a logic bug in the mode. Not implicated ⇒ no re-run.
  - **`extract_embeddings.py` (ported): ✅** 37s on a 20-species subset → npz + centroids.
  - **`analyze`: ✅** 14s with the default smoke model → output produced.
  - **Verdict: Phase 5 PASS.** TF-first fix validated across two full 30–35 min trainings (no
    deadlock); helper-mode switching, the ported extractor, and `analyze` all validated on full
    `reallybig`. The one failure was a pre-logic transient. Smoke models (`pelican5smoke-*`) were
    throwaway; delete from `recognizers/` after review.
  - **All 5 pre-merge phases now pass** (1 tests, 2 embeddings bit-identical, 3–4 no species
    regression, 4 soundscape-eval gate, 5 workflow).
- **`soundscape-eval` fully decoupled from `birdnet_analyzer.model` — DONE 2026-07-22**
  (soundscape-eval commit `3c8f394`). This was the last thing forcing scoring through a `main`
  worktree. The PerchV2 and geo paths (plus a stray `flat_sigmoid`/label-file coupling in
  `run.py`) followed the same treatment as the earlier BirdNET adapter + `extract_embeddings.py`:
  - `perch_embed.py` resolves the Perch V2 SavedModel from the installed package's bundled
    `checkpoints/perch_v2` (env-overridable via `SOUNDSCAPE_EVAL_PERCH_V2_DIR`), not
    `cfg.PERCH_V2_MODEL_PATH`; adds `perch_predict()` (softmax over Perch's native head) beside
    `perch_embed()`, both off one `serving_default` call — so `PerchV2.score` no longer needs
    `model.predict_with_perch`.
  - `adapters.py` drops `from birdnet_analyzer import model` entirely: geo runs a self-contained
    MData meta-model tflite runner (`_meta_predict`, replacing `model.predict_filter`),
    `calibrate()` uses an inlined `_flat_sigmoid`, and checkpoint-path resolution is centralized
    (`birdnet_checkpoints_dir`/`birdnet_labels_file`, `LOCATION_FILTER_THRESHOLD = 0.03`), reused
    by `BirdNETGlobal`. `PerchV2`/`PerchCustomHead.prepare` no longer set any `cfg` globals.
  - **Verified behavior-identical** to the removed functions against a `main`-like install:
    meta/geo max|diff| 3e-7 (float32), `flat_sigmoid` and Perch native head bit-exact.
  - soundscape-eval now imports only `birdnet_analyzer.config` (for the package location) +
    `model_utils.get_embeddings_array` (bare-head branch only, never hit by the raw-audio pelican
    tflites) ⇒ import-compatible with **both** the TFLite `main` and the refactor. **Scoring no
    longer needs a `main` worktree.** ⇒ **No sibling-repo blockers remain before an actual `main`
    merge.**

---

## Python environment

```bash
.venv/bin/python   # always use this, not system python
```

Install deps:
```bash
.venv/bin/pip install -e ".[train,embeddings,tests]"
```

**Rebuilt 2026-10-26 for the 3.0 landing** (birdnet 1.3.1, TF 2.21, ruff 0.15.10, `embeddings`
extra included) — see "Shared `.venv` swap" under Branch strategy for the constrained install.

Still true, and still the reason to check before changing it: this venv is shared
infrastructure (see the header warning), sibling repos invoke it by absolute path. Reinstall
shape if ever needed:

```bash
.venv/bin/pip install -e ".[train,embeddings,tests,dev]"
```

Linting: the repo enforces clean `ruff` (CI fixed all violations). `ruff` ships in the
`dev` extra (`0.15.10`); run `.venv/bin/ruff check <files>` before committing.

---

## Training workflow (main branch)

Training uses the **birdnet-library inline loader** (as of the 2026-07-22 swap) — embeddings extracted inline (~40 clips/sec via upstream #939), no cache step needed. On macOS it must run via `train_tf_first.py` (TF-first init; `train_pelican.sh` already does this).

```bash
./train_pelican.sh run1-0-bn                  # BirdNET 3.0 backbone (the default since 2026-10-28)
./train_pelican.sh run1-0-bn --epochs 100     # override any flag
./train_pelican.sh run0-21-bn --birdnet 2.4   # a 2.4 head, e.g. to extend the run0-N series
```

**Backbone:** 3.0 by default (32 kHz, 1280-d embeddings, a `.tflite` that outputs
probabilities); `--birdnet 2.4` for the old one. For repeated 3.0 runs on the same library,
extract once with `--save_cache_to <file>.npz` (~46 min for full `reallybig`) and pass that
cache as the input of later runs — a cache trains on the backbone it was embedded with.

Default hyperparameters in `train_pelican.sh` (from pelican0-9_Params.csv, revised by the 2026-11-16 ablation):

| Parameter | Value |
|-----------|-------|
| hidden_units | 2048 |
| dropout | 0.25 |
| batch_size | 32 |
| learning_rate | 0.0001 |
| upsampling | **none** (was `repeat` at ratio 0.4 until 2026-11-16; see "Recipe ablation on 3.0") |
| focal_loss | true |
| focal_loss_alpha | 0.25 |
| focal_loss_gamma | 2.0 (was 3.0; the Smiths Lake gamma sweep — see the comment in `train_pelican.sh`) |
| epochs | 50 |

Override any parameter by appending flags: `./train_pelican.sh pelican0-11 --epochs 100`

Appended flags are passed straight through to `birdnet_analyzer.train`, and argparse uses the last value, so an appended `--epochs 100` overrides the baked-in `--epochs 50`. Note: only flag-style overrides work — don't append a bare positional like `cache.npz`, since `train` accepts a single positional (`INPUT`, already set to the reallybig library) and a second one errors out. `train_pelican.sh` needs no cache — it passes the `reallybig` folder and the loader extracts embeddings inline. The `--cache_file` / `--cache_mode` flags still exist (they came in with the birdnet-library core and now live on `main`); the refactor's cache path only triggers when `INPUT` is a cache file, so the inline route is unaffected.

### Recipe ablation on 3.0 — what earns its place (2026-11-16)

Question: on the 3.0 backbone, does our recipe beat upstream's training as shipped (linear
head, no dropout, no upsampling, no focal loss, helpers as positive classes), and which of our
pieces and upstream's extras earn their place? One full 3.0 extraction of `reallybig`
(`training_runs/out/recipe30/cache30_full.npz`, 101,862 samples, 479 labels), then 11
single-change heads `abl30-*` off it, scored on Smiths Lake, Fowlers and Sarah's
(`soundscape-eval/configs/*_recipe30.yaml`). Driver, scorer and analysis:
`training_runs/configs/recipe30/{run.sh,score.sh,analyse.py}`; table in
`training_runs/out/recipe30/analysis.txt`.

**Noise band** = the spread of `fork` / `fork-s2` / `fork-s3` (same recipe, `--seed` 42/2/3).
Rule: a piece is kept only if removing it costs more than the band on ≥2 of 3 soundscapes,
Sarah's being one; an upstream extra is adopted by the same rule in reverse.

AUPRC_all change vs the seed mean (↑/↓ = beyond the band):

| arm | Smiths Lake | Fowlers | Sarah's | verdict |
|---|---|---|---|---|
| upstream vanilla | −0.021 ↓ | −0.007 ↓ | −0.007 ↓ | **our recipe wins** on all 3 (bestF1 too) |
| − hidden layer (linear head) | −0.019 ↓ | −0.010 ↓ | −0.007 ↓ | **keep** — nearly the whole gap to vanilla |
| − focal loss | +0.002 | +0.001 | +0.003 | **keep** — ranking-neutral, but see below |
| − upsampling (ratio 0) | +0.000 | +0.003 ↑ | +0.003 | **retired** — costs nothing anywhere |
| − helpers as non-events | +0.000 | −0.001 | −0.001 | neutral on every metric |
| + `--label_smoothing` | −0.001 | +0.001 | +0.000 | not adopted |
| + `--mixup` | +0.001 ↑ | +0.005 ↑ | +0.004 ↑ | not adopted — barely above the band, loses R@1/R@4 on Sarah's |
| + `--upsampling_mode linear` | −0.001 | +0.002 | +0.002 | not adopted — R@1/R@4 −0.04 to −0.05 on Sarah's |

- **Focal loss is kept for the operating point, not ranking.** Without it, Sarah's recall falls
  0.061 at 1 FP/hr and 0.047 at 4 FP/hr (2.5× and 11× the seed spread). Not a one-species
  artifact: without focal the scores crowd toward 1 (the 1 FP/hr threshold moves 0.59 → 0.96)
  and the false alarms come from 10 species. Smiths Lake can't judge an operating point (1 h,
  so 1/hr = one false alarm); Fowlers is roughly neutral. Strictly it fails the 2-of-3 rule —
  a judgement call, made because Sarah's is the only soundscape whose budget no single
  species carries.
- **Helpers as non-events stays the default on other grounds** (cleaner output, the earlier
  OOD evidence under "Helper classes as non-events"); this ablation says it is free, not that
  it helps.
- Vanilla looks better at R@1FP/hr on Smiths Lake and Fowlers, but both budgets are carried by
  a few species (Fowlers: sheep 83%). Indicative only.
- Three seeds is a thin band; effects just past its edge (mixup, dropping upsampling) are weak
  evidence. Dropping upsampling was chosen because it is *free*, not because it wins.
- `abl30-fork`'s label set equals `sync1007-bn30`'s exactly (464), which validated the cache
  path's new helper-mode handling (`_scope_cache`, commit `3aeb9ec`).
- Scoring: a 3.0 head scores at batch 1 on one core, so `score.sh` runs 5 `run score` workers
  in parallel (~70 min for 29 pairs vs ~7 h serial). ⚠️ One score cache written to OneDrive
  came out truncated (`BadZipFile` in `run all`); delete it and re-score that pair.

### Site-scoped recognizers — `--species_list` / `--unlisted` (added 2026-08-08)

A recognizer per site (Wild Deserts, Smiths Lake) trained off the one shared `reallybig`,
without copying class folders — copies desync, since a clip fixed in `reallybig` never
reaches them.

```bash
./train_pelican.sh wilddeserts0-1 --species_list /path/to/wilddeserts.txt
./train_pelican.sh wilddeserts0-1 --species_list /path/to/wilddeserts.txt --unlisted drop
```

Format: one scientific binomial per line, `#` comments ignored, matched case-insensitively
against each class folder's `Genus species` prefix. Same file `derive_species_list`
(`training_runs/lib/common.sh`) writes and `perch-head`'s `train_head.py --species-list`
takes, so one list scopes both arms of a dual run.

| `--unlisted` | unlisted classes | cost |
|---|---|---|
| `non_event` (default) | audio kept, all-zero rows, no output neuron — still suppress the retained species | full library decoded |
| `drop` | never scanned or decoded | subset-sized |

**Which is better for field FP rate is not measured yet** — see `training_runs/CLAUDE.md`
§ "Open threads". `non_event` is the default because it is conservative, not because it won.

**Two paths, and the cache one is what makes a model per site cheap:**

- **Cache (primary)** — `_scope_cache_to_species_list` slices `y_train`'s columns. A row
  whose only positive class was dropped becomes all-zero, which *is* the non-event
  encoding, so `non_event` mode is the slice alone. Extract the library once
  (`--save_cache_to`), then every additional site is a column slice, not a re-decode.
- **Folder (fallback)** — `drop` filters `train_folders` at discovery, so `is_binary`,
  `is_multi_label` and the `--test_data` intersection all narrow for free; `non_event`
  leaves the folders in place and lets `is_non_event()` demote them.

⚠️ **Helper folders are exempt from the filter.** `Environment_Rain` has no binomial
(`scientific_prefix` → `environment`), so filtering it on a species list would silently
delete the non-event hard negatives the species classes depend on. `--keep_as_class`
carve-outs stay reported classes.

⚠️ **Not the same flag as `analyze --slist`.** That restricts what an *already-trained*
model reports (and is a genuinely useful, zero-retraining way to cut FPs — it masks output
via `species_ids_whitelist`, works with a custom classifier, and wants full label strings
like `Acanthiza pusilla_Brown Thornbill`, not bare binomials). `train --species_list`
restricts what gets *trained*.

Provenance lands in `<name>.birdnet.train-params.csv` as `Species list` /
`Species list entries` / `Unlisted handling`; `_Labels.txt` is the authoritative resolved
class set. **Inert by default** — no list, no filtering, identical label set.

### Helper classes as non-events (DEFAULT in `train_pelican.sh`)

> **Dual-arm run contract:** `training_runs/CLAUDE.md` (the `run_dual.sh` orchestrator) relies on
> this being the default, and mirrors it on the Perch side. The run-level defaults —
> `Environment_*`/`Homo sapiens_*` as non-events, and training on **all** `reallybig`
> classes — live in `~/Documents/ecoacoustics/training_runs/CLAUDE.md` § "Run defaults". Keep the two
> in sync if this default changes.

Formalizes the geophony/anthropophony non-event experiments (the "Design fork" /
`both` decision below) into the training pipeline. Two `birdnet_analyzer.train`
flags drive it (added to main; see the non-event mechanism in the metrics section):

| Flag | Effect |
|------|--------|
| `--non_event_prefixes "Environment_,Homo sapiens_"` | Comma-separated class-name **prefixes** whose folders are trained as **non-events** — all-zero hard-negative label rows, **no output neuron** (mechanically identical to a `Noise`/`NON_EVENT_CLASSES` folder). They protect species as hard negatives without being reportable classes or polluting the species metric. **Engine default is empty** (no behavior change); `train_pelican.sh` injects this value by default (see below). |
| `--keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren"` | Comma-separated **exact** class names that stay positive, reportable classes even when they match a `--non_event_prefixes` prefix (the episodic anthropophony exception). |

Implementation: `config.NON_EVENT_PREFIXES` / `NON_EVENT_KEEP_CLASSES` →
`train/utils.py::is_non_event()` (extends the exact-match `NON_EVENT_CLASSES` check
used for `valid_labels` and the label-vector encoding) → wired via `train/core.py`
+ `cli.py`. Converted classes simply drop out of `valid_labels`, so they get no
column in `<model>_validation_metrics.csv`.

**`train_pelican.sh` makes helpers-as-non-events the default** (the `both` arm). All
`Environment_*`/`Homo sapiens_*` classes become non-events with no extra flag; turning
them back into modelled-and-reported positive classes is now the **active choice**:

```bash
# DEFAULT: ALL Environment_*/Homo sapiens_* -> non-events, no exceptions.
./train_pelican.sh pelican0-19
# implicitly adds: --non_event_prefixes "Environment_,Homo sapiens_"

# Opt out: keep helpers as positive, reported classes (the old default):
./train_pelican.sh pelican0-19 --report-helpers
# injects nothing — helpers train as ordinary positive classes.

# Keep episodic anthropophony (Airplane, Siren) reportable, rest stay non-events:
./train_pelican.sh pelican0-19 --keep-airplane-siren
# adds: --keep_as_class "Homo sapiens_Airplane,Homo sapiens_Siren"
```

- **`--report-helpers`** is the active opt-out: helpers become ordinary positive,
  reported classes (no `--non_event_prefixes` injected).
- **`--keep-airplane-siren`** carves out the two episodic anthropophony classes you
  may still want reported while the rest stay non-events.
- **`--nonevent-helpers`** is still accepted as a **no-op** (it is now the default),
  for backward compatibility with older commands.
- An **explicit `--non_event_prefixes`** suppresses the baked-in default, so any other
  split (e.g. geophony-only) still works:
  `./train_pelican.sh pelican0-19 --non_event_prefixes "Environment_"`.

The run banner prints a `Helpers:` line stating which mode is active.

---

## Recognizer outputs

Trained models go to:
```
/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/
```

Each run produces:
- `<name>.tflite` — the classifier
- `<name>_Labels.txt`
- `<name>.birdnet.train-params.csv` — the run's hyperparameters, **one row per parameter**.
  Renamed from `<name>_Params.csv` (one column per parameter) by the 2026-07-23 upstream sync;
  models trained before that still carry the old file, and upstream's `params.py` reads both.
  Also records `Non-event prefixes` / `Non-event keep classes` (our addition), so a run's
  helper mode is recoverable from its artifacts.
- `<name>_sample_counts.csv`
- `<name>_validation_metrics.csv` — per-species precision/recall (our addition)

---

## Taxonomy maintenance (keeping class names current)

Class names in `reallybig` drift as taxonomy is revised (e.g. the 2025 Litoria
generic split → Rhyaconastes/Rawlinsonia/Pengilleyia/…). The workflow is designed
to be **low-maintenance: one *live* authority per group, followed mechanically,
with no yearly snapshot file and no override list that goes stale.** All checkers
are **report-only** — renaming a folder is a human decision (folder names are baked
into trained models; a rename means a retrain).

**Authority per group:**
- **Birds → eBird**, Australian English common names (`locale=en_AU`). Custom bird
  classes track *current* eBird and update yearly; they are **not** locked to the
  frozen global BirdNET model.
- **Everything else → ALA** (`namematching-ws.ala.org.au`, free/no-auth) — the only
  source unifying frogs/cicadas/katydids/mammals. Birds are excluded from the ALA
  checker up front by eBird membership, so the two reports cleanly partition.

**Tools:**

| Tool | Repo | Job |
|---|---|---|
| `curation/check_taxonomy_drift.py` | assembly | non-bird classes vs live **ALA** |
| `curation/check_ebird_drift.py` | assembly | bird classes vs live **eBird** (en_AU) |
| `map_custom_to_global.py` | here | bridge custom↔global model by **eBird species code** |
| `update_ebird_taxonomy.py` | here | resync the global model's aux eBird code JSON |

Run (from BirdNET-Analyzer, via this repo's `.venv`):
```bash
.venv/bin/python /Users/z3484779/Documents/ecoacoustics/Training_library_assembly_pipeline/curation/check_taxonomy_drift.py
.venv/bin/python /Users/z3484779/Documents/ecoacoustics/Training_library_assembly_pipeline/curation/check_ebird_drift.py
.venv/bin/python map_custom_to_global.py          # → custom_to_global_lookup.csv
.venv/bin/python update_ebird_taxonomy.py         # only when swapping the global model
```

**Custom ↔ global model bridge.** The **global** BirdNET model
(`BirdNET_GLOBAL_6K_V2.4`) has frozen labels locked to one eBird edition; the custom
pelican model's bird names run ahead of it. The **eBird species code is invariant
across editions** (`Calyptorhynchus funereus` → `Zanda funerea` both stay `ytbcoc1`),
so `map_custom_to_global.py` joins them on the code — the bridge holds no matter how
far display names diverge. `update_ebird_taxonomy.py` regenerates
`eBird_taxonomy_codes_<edition>.json` and repoints `config.py`; it is tied to the
frozen global model (aux code lookup for Raven output only — does **not** feed the
model or the species list), so rerun it only on a global-model swap, not yearly.

**Design rules that keep it maintenance-free (no rotting overrides):**
- **Change-detection by stable AFD `taxonConceptID` GUID** — report when the
  authority's *answer changes*, not merely when a folder name differs.
- **Strip ALA subgenus parens** `Genus (Subgenus) species` before comparing.
- **Same-accepted-name collision → keep classes DISTINCT** (domestic dog vs dingo,
  the one real special case), never merge; auto-re-checks if ALA re-splits.
- **Open-nomenclature aware**: `Genus spp.` = genus-level class, `Genus sp.` =
  undescribed species, single token = higher taxon (e.g. `Diptera`) — resolved at
  their own rank, **not** mis-flagged. Only a described binomial ALA can't place is a
  genuine truncation (needs a one-time manual GUID attach).
- **`curation/taxonomy_notes.csv`** escape hatch is **self-expiring** — each note is
  pinned to ALA's answer at note time, so it re-alerts the moment ALA moves. Starts
  empty; the rules above cover the current library.
- Human/environmental helpers (`Environment_*`, `Homo sapiens_*`, `Noise`) are
  excluded from both checkers.

The persistent GUID map lives with the other shared data at
`recognizers/ala_taxon_ids.csv`. Applying a batch of renames writes a
`curation/taxonomy_rename_undo_<ts>.sh` (reverse `mv`s) for reversibility. Known
residuals needing a human: `Rawlinsonia revelata` → `Litoria revelata` (one-time GUID
attach — ALA doesn't index that synonym) and `Ovis aries` (ALA `namematching` gap).

---

## Interpreting metrics: non-target classes & species confusion

**The project goal is high precision AND recall on the SPECIES classes.** The
`Environment_*` / `Homo sapiens_*` / `Noise` classes are **helpers** — they exist
to keep non-target sound off the species classes. They are means, not ends, so
`<model>_validation_metrics.csv` must be read in that light, not as N symmetric
per-class scores.

### How BirdNET actually handles non-events (verified in code)

- `config.py` `NON_EVENT_CLASSES = ["noise","other","background","silence"]`,
  matched by **exact lowercase equality**. Routed (with the prefix mechanism below)
  through `train/utils.py::is_non_event()`, used at both the `valid_labels` build and
  the label-vector encoding. By itself, only a folder literally named
  noise/other/background/silence is a non-event.
- A non-event clip is encoded as an **all-zero label row** (`load_data` in
  `train/utils.py`; non-events detected at `model.py:228` via `sum(row)==0`) — "none
  of the classes present." It gets **no output neuron**, is never reported, and acts
  purely as a **hard negative** that pushes every class down on that input.
- **Consequence for our geophony (default):** `Environment_Rain`, `Environment_Wind`,
  `Homo sapiens_Airplane`, … do **not** match `NON_EVENT_CLASSES`, so by default they
  are trained as **ordinary positive, reportable classes** — mechanically identical to
  a species. The memory's "Rain/Wind = continuous non-event reject" intent is **not**
  reflected in the *default* encoding; only the real `Noise` folder is a true non-event.
- **➜ Now switchable:** `--non_event_prefixes` / `--keep_as_class` (preset
  `train_pelican.sh --nonevent-helpers`, see Training workflow) convert helper classes
  to non-events by prefix — this is the implementation of the `both` decision below.
- The head is **per-class sigmoid (multi-label), not softmax**. So there is **no
  cross-class suppression at inference** — a high `Environment_Rain` score does
  nothing to veto a co-firing species on the same 3 s window. Helpers protect
  species **only** through the training-time hard-negative effect (rain clips are
  negatives for species), which both the positive-class and the non-event encoding
  provide. ⇒ **Judge a helper by its effect on species precision, not by its own
  recall.** (e.g. pelican0-14 `Homo sapiens_Music` recall 0.68 but ~0% of its clips
  confuse with any species = a perfect helper.)
- **Design fork (matches the continuous-vs-episodic taxonomy):** episodic
  anthropophony you want reported (Airplane/Helicopter/Siren/Human Voice/Music) →
  keep as positive classes. Continuous geophony you never want reported
  (Rain/Wind/Thunder/Surf/Stream) → **convert to non-events** (all-zero hard
  negatives) so they protect species without cluttering output or polluting the
  metric. Note: the "split diffuse Noise into tight classes" rationale is weaker for
  pure rejection, since an all-zero target has no positive neuron to struggle with a
  diffuse distribution. **The fork is now wired into training** — see
  `--non_event_prefixes` / `--keep_as_class` / `--nonevent-helpers` (Training
  workflow); choosing which prefixes/exceptions to pass *is* picking the fork.
> **Methodology note (2026-06-29):** the `experiments/geophony_nonevent_*.py`
> A/B scripts cited below have been **removed**. They fit 4 heads on one shared
> base-embedding extraction to compare arms cheaply — superseded by training and
> saving **whole separate models** (`train_pelican.sh`, helpers-as-non-event now the
> default) and comparing them in the dedicated **`soundscape-eval`** repo
> (`/Users/z3484779/Documents/ecoacoustics/soundscape-eval`). The findings below are retained as
> the empirical record that motivated the default; re-run via whole-model eval to
> refresh them.

- **Tested (2026-06-19, removed `geophony_nonevent_ab.py`).** A/B/C/D over one
  shared held-out split (16,147 clips), 4 heads on one base-embedding extraction:
  none (baseline) / environment→non-event / human→non-event / both→non-event.
  **Result: species AUPRC is flat to 4 dp across all four (0.9885–0.9886)** — the
  encoding is *safe*, it costs nothing on the headline objective. Non-target leak@0.5
  moved sub-1pp and not monotonically: `environment`-only slightly **worse**
  (1.73→1.95%, consistent with converted classes losing their upsampling hard-neg
  exposure), `human` flat, `both` slightly **best** (1.62% @0.5, 8.00% @0.25, species
  P 0.989/R 0.955). The `both` edge is **within single-seed noise — do not over-read**.
  ⇒ **The species metric cannot adjudicate this fork; decide on the other axes**
  (output cleanliness, edge-device rain rejection, cheap scalable negative diversity),
  knowing the choice is free on species. The mechanism prediction held: species get
  the hard-negative signal under *either* encoding, so near-parity was expected.
  **What this run can't see:** rejection of *novel/OOD* non-target audio — needs an
  out-of-distribution negative eval (FSD50K/ESC-50/Freesound), the logical next test.
  Outputs: `call_library/experiments/geophony_nonevent_{comparison.csv,summary.md}`.
- **OOD field test (2026-06-19, removed `geophony_nonevent_ood.py`).** Scored the
  4 saved arm heads (`--save-heads`) on 6.33 hr of *novel* long-format field audio
  (Smiths Lake "Powerline Strip", anthropophony-heavy; 19/24 files — 5 skipped as
  `audioread NoBackendError`, all the Feb 04–05 batch) over the common 423 species
  columns. Species spurious-fire-rate/hr: none 173.21 / environment 173.05 / human
  172.26 / **both 165.00** @0.5 (and @0.25: 797.5/798.6/817.3/**759.8**).
  **Only `both` improves — ~5% fewer spurious firings at *both* thresholds, and
  super-additively** (environment-only and human-only are flat; human-only is *worse*
  @0.25). This is the diversity prediction: the *combined* negative space rejects
  novel non-target audio better than either half. **Caveat — audio is UNLABELED**, so
  the net 5% is not all leak: per-species deltas (`both−none`) show the one clean leak
  win is *Tyto longimembris* (Australasian Grass-Owl) −187 (~23%; ~130 firings/hr at a
  powerline strip is ecologically implausible = hum/wind/insect leak), but cricket
  (−320) and cicada (−313) drop while katydid *Pseudorhynchus* rises (+221) — partly
  detection-mass *redistribution* among the continuous-stridulation insect "species,"
  not pure leak removal. Real passerine firings (Golden Whistler, E. Spinebill, Lewin's
  Honeyeater @0.92–0.98) look retained. ⇒ **Directionally supports `both`→non-events
  (free on species in-sample, modestly better OOD rejection, cleaner output), but
  suggestive not conclusive.** Outputs: `geophony_nonevent_ood_{comparison.csv,
  summary.md,top_firings.csv}` (top-firings carry file+timestamp+species for
  listen-checks).
- **DECISION (lean): `both` → non-events** — convert all `Environment_*` and
  `Homo sapiens_*` to non-events (except any episodic anthropophony you want *reported*,
  e.g. Airplane/Siren, kept as positive classes). Free on the species objective,
  marginally better at OOD rejection, removes geophony/anthropophony clutter from
  output. **Now the `train_pelican.sh` DEFAULT** — `./train_pelican.sh <name>` converts
  all `Environment_,Homo sapiens_`, no exceptions (the exact tested `both` arm); add
  `--keep-airplane-siren` to keep Airplane/Siren reportable, or `--report-helpers` to
  opt out entirely (helpers as positive, reported classes — the old default).
  **Next step: validate on a LABELED soundscape** (ground-truth annotations →
  real species precision/recall + true leak, vs the unlabeled field proxy here). Then
  the scaling lever: fold FSD50K/ESC-50/Freesound non-target corpora into the `both`
  non-event sink (clean negatives only — contamination = mislabeled positives hurt
  species recall) to grow the OOD-rejection upside that in-sample metrics can't see.
- **LABELED-soundscape eval → moved to `soundscape-eval`** (`/Users/z3484779/Documents/
  soundscape-eval`). The retired `geophony_nonevent_labeled.py` scored the 4 shared
  arm heads on `labeled_soundscape/` (two consecutive 2MM03792 recordings + an
  exhaustive point-event transcription, so precision/leak are *real*); its headline
  finding stands: **species recall is low (~0.2) even at a 5-min match window** — most
  annotated calls go undetected, which dominated the arm comparison. The companion
  `geophony_nonevent_faintness.py` chased the W.C. hypothesis that the misses are
  **faint calls below the detector trigger** (recall-vs-threshold + a vocalisation-band
  SNR proxy). Both are gone; this evaluation now belongs to `soundscape-eval`, run
  against **whole saved models**. ⇒ the `both` vs baseline verdict on labeled data is
  **not yet conclusive** — recall/faintness is the current blocker; re-establish it in
  `soundscape-eval`.

### Species↔species confusion is often real biology

Confusion between congeners can be genuine signal, not model error: **acoustic
partitioning / reproductive character displacement** — selection only pushes calls
apart where confusion would cost a mating/territorial response. Species separated by
**space (parapatry/allopatry), breeding time/phenology, diel calling window, or
other prezygotic channel** are under no pressure to diverge acoustically, so their
calls stay similar and an acoustic classifier *should* confuse them.

- **Architecture implication:** don't force acoustic separation biology never
  encoded — let acoustics reach the confusable group, then split with a
  **spatiotemporal prior at inference** (deployment site + date/season + time-of-day).
  Generalizes BirdNET's range filter to phenology + diel timing.
- **Metric implication:** confusion cost should be **co-occurrence-weighted** — a
  confusion between species that never share place AND time is nearly free
  (resolvable downstream); sympatric synchronous-breeder confusion is the expensive
  one. Flat macro-recall over-penalizes the former.
- **Two low-recall signatures** — read recall as a tuple `(recall, n, intra-cohesion,
  leak-shape)`: **data-limited** = low n + low cohesion + diffuse leak + non-sibling
  nearest centroid (e.g. Australian Bustard n=27, cohesion 0.61, scattershot leak →
  more/better recordings if gettable); **acoustically-capped** = high n + tight
  cohesion + sibling-specific leak (e.g. the Yoyetta/Atrapsalta/Haemopsalta/Palapsalta
  cicadas, parapatric + temporally partitioned → data won't separate; use a
  spatiotemporal prior + co-occurrence-aware scoring, or accept). Cicada clip IDs look
  like iNaturalist obs IDs (carry lat/lon + date) so segregation is testable. Calidris
  sandpipers (Sharp-tailed vs Curlew) are the counter-case: co-occur in AU flocks but
  breed allopatrically in the Arctic, so AU non-breeding calls are conservative — a
  place filter won't fix them.

### Per-class P/R computation — current method & its limits

**`model.compute_validation_metrics(y_true, y_pred, labels)`** — module-level and pure,
called by `train_linear_classifier` after `fit`. `y_pred = (y_prob >= threshold)` with
`threshold = 0.5`, then **per-output-column TP/FP/FN counted directly** — **per-class
binary at a fixed 0.5 threshold** (the correct multi-label framing, *not* a 438-way
argmax). Micro comes from the summed counts, macro from the per-class mean; both are
also computed over the species/non_target column subsets. Returns
`(per_class, summaries)`: one `(type, label, precision, recall)` per column, plus a dict
of the six `overall_*`/`species_*`/`non_target_*` summaries whose insertion order *is*
the CSV row order.

> **Was `sklearn.precision_recall_fscore_support`, inline in `train_linear_classifier`;
> replaced 2026-08-07 (bug fix).** sklearn infers the target type, and a **single-column**
> indicator matrix `(n, 1)` infers as **binary, not multilabel** — it then scores the
> negative class as a second class. Effects: micro/macro inflated (counts true negatives
> as hits), and `average=None` returns **2** rows for 1 class, so per-class CSV rows
> shift off their labels (real label ← negative-class scores, plus a phantom `class_1`).
> This hit **single-species classifiers**, and — less obviously — **any** run where the
> species *or* non_target partition was exactly one column wide, since the subset
> summaries took the same path on an otherwise multi-class model. Verified equal to
> sklearn at ≥2 columns, so **all historical pelican numbers are unaffected**; the CSV
> format is unchanged. Regression tests: `tests/train/test_validation_metrics.py`.
> Side effect: sklearn (an *undeclared* transitive dep) is no longer imported here, so
> the old "skip metrics" fallback — which also leaked `CUDA_VISIBLE_DEVICES` by returning
> early — is gone.
>
> **Why the computation is a separate pure function:** so it can be tested without
> training. See the `model.fit`-under-pytest warning in the Tests section.

Weaknesses to fix:
1. **Single arbitrary 0.5 threshold** that also **doesn't match the deployed
   operating point** (inference applies `flat_sigmoid(sensitivity,bias)`,
   `model.py:1258`). → report **threshold-free PR-AUC / average-precision** and/or
   **recall@fixed-precision** instead of one-point P/R.
2. ~~CSV tags every row `type="species"`, including helpers.~~ **DONE
   (2026-06-19):** rows tagged species vs `non_target` (= `Environment_*`/
   `Homo sapiens_*`), and `species_micro`/`species_macro` summary rows added as the
   headline objective (pelican0-14 re-aggregated: species macro P 0.988 / R 0.948 vs
   all-class R 0.942). Takes effect on the next retrain (pelican0-15+).
3. **Non-event classes are excluded from the metric** (no column) but their clips
   still penalize species precision as false positives — good, keep.
4. ~~**Verify no train/val leakage from `repeat` upsampling**~~ **CHECKED 2026-08-31 — it
   does NOT leak.** The split runs FIRST and upsampling receives only the training half:
   `model.py:672/676` (`random_split` / `random_multilabel_split`) then `:681`
   (`upsampling(x_train, y_train, ...)`). A duplicated clip therefore cannot reach
   validation. Worth having settled, since every pelican run up to 2026-11-16 upsampled
   (`--upsampling_mode repeat --upsampling_ratio 0.4` in `train_pelican.sh`; dropped since),
   so a leak here would have inflated the recall in every `_validation_metrics.csv` we have.
   **Upstream keeps the same order** (`upstream/main` `model.py:517/521` then `:531`), so
   the parked #968 rewrite does not put this at risk — re-check only if the *call order* in
   `train_model` ever changes, not merely because the upsampling internals do.

---

## Embedding analysis

This repo owns **embedding extraction only** (`extract_embeddings.py`, coupled to
the fast TFLite `model.py` pipeline). The rest of the embedding workflow moved to
sibling projects (reorg 2026-06-19):
- **curation** (misclassification worklists, downsample/cap, npz→cache, relabel) →
  `Training_library_assembly_pipeline/curation/` — run via **this repo's `.venv` by
  absolute path** (no separate venv).
- **visualisation** (categorised UMAP / confusability / centroid plots) →
  `birdnetEmbed` R package (`~/Documents/ecoacoustics/birdnetEmbed`, `traitecoevo/birdnetEmbed`).

Shared embeddings are **DATA, not a repo internal**: the `.npz` + `_centroids.csv` +
candidate/misclass CSVs live in **`call_library/embeddings/`** (next to `reallybig`
+ `recognizers`), read by both curation and birdnetEmbed. Only
`extract_embeddings.py` stays in `embedding_analysis/` (plus historical
downsample/relabel run-records).

**Extract (once per model)** → write the npz into the shared data store:
```bash
.venv/bin/python embedding_analysis/extract_embeddings.py \
    --model /path/to/recognizers/pelican0-14.tflite \
    --input /path/to/reallybig \
    --output /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/embeddings/reallybig_pelican0-14
```

**Misclassification analysis** (now in the assembly repo's `curation/`):
```bash
.venv/bin/python /Users/z3484779/Documents/ecoacoustics/Training_library_assembly_pipeline/curation/identify_misclassifications.py \
    --input /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/embeddings/reallybig_pelican0-14_embeddings.npz \
    --output /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/embeddings/reallybig_pelican0-14
```

**Categorisation + plots** → `birdnetEmbed` (R), one command per model:
```bash
BIRDNET_PYTHON=/Users/z3484779/Documents/ecoacoustics/BirdNET-Analyzer/.venv/bin/python \
BIRDNET_TAXONOMY_CACHE=/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/ala_taxonomy_cache.csv \
  Rscript /Users/z3484779/Documents/ecoacoustics/birdnetEmbed/scripts/analyse_model.R \
    /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/embeddings/reallybig_pelican0-14_embeddings.npz pelican0-14
```

> Use **literal absolute paths**, not `$(pwd)`/`~` (command substitution always
> prompts and can't be allowlisted).

**ALA taxonomy cache (`BIRDNET_TAXONOMY_CACHE`).** `analyse_model.R` resolves each
class's scientific name → broad category via `galah` (Atlas of Living Australia).
That lookup is now cached to the CSV named in `BIRDNET_TAXONOMY_CACHE`
(`recognizers/ala_taxonomy_cache.csv`, shared across all models). Only **new
classes** absent from the cache are queried; a run with no new classes makes **no
network call and needs neither `galah` nor an ALA email**. Consequence: `galah`
+ network + `galah_config(email=…)` are only needed the **first time a new class
appears** (then the cache row — taxonomy or `NA` for unmatched — is permanent).
Unmatched names (`Environment`, `Homo sapiens`, `Noise`, renamed genera like
`Tachyspiza`) are still fixed by `category_overrides.csv`, applied *after*
taxonomy. Delete a row (or the file) to force a re-query.

> Adding `BIRDNET_TAXONOMY_CACHE=…` changes the command prefix. A dedicated allow
> rule for that exact shape is already in `.claude/settings.local.json` (the
> `BIRDNET_PYTHON=… BIRDNET_TAXONOMY_CACHE=… Rscript *` rule), so the cached-taxonomy
> command runs prompt-free. If you reorder env vars or add another the prefix won't
> match — either keep this exact ordering or `export BIRDNET_TAXONOMY_CACHE` in your
> shell profile so the command reverts to the plain `BIRDNET_PYTHON=… Rscript …` shape.

So the **per-model recipe** is: (1) `extract_embeddings.py` here → `.npz` in
`call_library/embeddings/`, then (2) `curation/identify_misclassifications.py`
(label-quality suspects), then (3) `analyse_model.R` (UMAP + confusability).
See `Training_library_assembly_pipeline/CLAUDE.md` and `birdnetEmbed/CLAUDE.md`.

### Pre-approved workflow commands (permission ↔ command sync)

These steps run **without a permission gate** via wildcard rules in
`.claude/settings.local.json`. The rules fix the command prefix and wildcard the
tail, so the **model version lives entirely inside the `*`** — bumping
`pelican0-13` → `pelican0-14` needs no new approval. Keep commands in these
shapes (literal absolute paths, no `$(...)`/`~`) so they keep matching.

| Workflow step | Allow rule that covers it |
|---|---|
| `extract_embeddings.py` (curation tools `identify_misclassifications.py`/`npz_to_cache.py`/`downsample_class.py` now in the assembly repo) | `Bash(.venv/bin/python *)` |
| `analyse_model.R` (UMAP + confusability) | `Bash(BIRDNET_PYTHON=/Users/z3484779/Documents/ecoacoustics/BirdNET-Analyzer/.venv/bin/python Rscript *)` |
| `train_pelican.sh <name> [flags]` | `Bash(./train_pelican.sh *)` |
| test suite | `Bash(.venv/bin/pytest *)` |

Two hard limits no allow rule can bypass (always prompt, by design): commands
containing **command substitution** `$(...)` / backticks, and **writes to
`.claude/settings*.json`** (the privilege-escalation surface). Avoid `$(pwd)` in
workflow commands for this reason.

---

## Non-target / geophony class build → moved to `Training_library_assembly_pipeline`

The non-target sink/event class build (rain/wind/thunder/stream/surf/airplane/…
from FSD50K + Freesound + ESC-50) and all library-curation tooling moved out of this
repo in the 2026-06-19 reorg. It now lives in
**`~/Documents/ecoacoustics/Training_library_assembly_pipeline/`** (`geophony/`, `species/`,
`curation/`). See that repo's `CLAUDE.md` for the per-class build recipe, naming
conventions (`Environment_<Type>` / `Homo sapiens_<Type>`), and the 350-cap rule.
This repo keeps only `extract_embeddings.py` (extraction); curation reads/writes the
shared `call_library/embeddings/` npz via this repo's `.venv`.

## Key data paths

| What | Path |
|------|------|
| Training library | `/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig` |
| Recognizers | `/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/` |
| Embeddings (shared data) | `/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/embeddings/` |
| Library assembly + curation | `/Users/z3484779/Documents/ecoacoustics/Training_library_assembly_pipeline/` |

---

## Running analysis

```bash
.venv/bin/python -c "
from birdnet_analyzer.analyze import analyze
analyze(
    '/path/to/audio/folder',
    output='/path/to/output',
    classifier='/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/pelican0-9.tflite',
)
"
```

Or via CLI:
```bash
.venv/bin/python -m birdnet_analyzer.analyze /path/to/audio \
    --classifier /Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/pelican0-9.tflite \
    --output /path/to/output
```

---

## Tests

```bash
.venv/bin/pytest tests/ --timeout=120 -q --ignore=tests/analyze/test_analyze.py
# 863 passed, 24 skipped in ~6 min (checked 2026-10-26 on main after the 3.0 landing;
# needs the `embeddings` extra and `git submodule update --init tests/data`)
```

⚠️ **Since #1007 the suite takes minutes legitimately** — `tests/train/test_export.py` builds and
runs real 2.4/3.0 exports (~6 min alone). So wall time no longer tells a hang from slowness:
**check CPU time.** A healthy run burns CPU (~245 s CPU over ~7 min); a deadlock sits near 0%
CPU. `--timeout` will *not* save you from a real deadlock.

**Never call `model.fit` (or anything that trains) from a test.** Under pytest it hits the
macOS TF × PyArrow absl deadlock: an earlier test imports the birdnet loader → PyArrow,
whose statically-linked absl interposes TensorFlow's, so eager execution parks forever in
`absl::Notification` inside `TFE_Execute`. The `train_tf_first.py` launcher fixes this for
*training entry points* but **cannot help under pytest**, which controls import order. Such
a test passes in isolation (TF binds first) and hangs only in the full suite — and
pytest-timeout's thread-based method can't interrupt a C-level condvar, so it hangs
*silently and indefinitely* rather than failing. Test training-adjacent logic by extracting
it into a pure function (as `compute_validation_metrics` was) and calling that.

To diagnose a suspected hang: `sample <pid> 3 -mayDie` and look for `absl::Notification`
under `TFE_Execute` on the main thread. A tell-tale sign is CPU time far below wall time
(e.g. 10 s of CPU over 30 min).

Still excluded: `tests/analyze/test_analyze.py` — 2 of its 6 tests
(`test_analyze_with_real_custom_classifier[_and_species_list]`) need a
`CustomClassifier.tflite` fixture that isn't in the repo (upstream issue); the other 4 pass.

The 6 skips are all `tests/gui/` tests that skip because `gradio` isn't installed in this
`.venv` (we don't use the GUI).
Install the `gui-tests` extra if you ever need them.

**Resolved by the 2026-07-23 upstream sync:** the old `perch_hoplite` `SQLiteUsearchDB`
capitalization drift — `birdnet_analyzer/embeddings/utils.py` is gone and every remaining
reference uses the correct `SQLiteUSearchDB`, so `tests/embeddings/` passes.

`tests/train/` covers the training pipeline (including `test_non_events.py`) and passes.
