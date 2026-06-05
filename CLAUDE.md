# BirdNET-Analyzer — Project Notes for Claude

## Repo structure

This is a fork of [birdnet-team/BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer).

```
git remote upstream  → https://github.com/birdnet-team/BirdNET-Analyzer.git
git remote origin    → https://github.com/wcornwell/BirdNET-Analyzer.git
```

## Branch strategy

### `main` — working branch (use this for all analysis work)

Stays close to upstream but includes local improvements and uses the **TFLite-based** embedding extraction pipeline (fast, ~40 clips/sec on reallybig).

Local changes on top of upstream:
- **macOS TF XLA deadlock fix** — `os.environ["TF_XLA_FLAGS"]` etc. set at top of `model.py` before imports; prevents deadlock when TensorFlow initialises concurrently with PyArrow/tqdm
- **Binary upsampling `added_count` fix** — upstream fixed the multilabel branch but left the binary branch using `len(y_temp)` (shared counter bug); our fix uses a per-class `added_count`
- **Upsampling summary printout** — `upsampling()` prints reference class, target min samples, and the 5 smallest classes before training
- **Per-class validation metrics** — `train_linear_classifier()` computes precision/recall per species after training and writes `<model>_validation_metrics.csv`
- **`predict_with_perch` and `embeddings` functions** — kept in `model.py` for perch support and the embedding analysis scripts
- **`train_pelican.sh`** — training script using TFLite-extracted embeddings cache (see Training workflow below)
- **`embedding_analysis/`** — scripts for embedding extraction and misclassification detection

### `sync-upstream-refactor` — tracking upstream's birdnet-library refactor

Follows the major upstream refactor that replaced core analysis with the `birdnet` pip library. **Do not use for production analysis work** — data loading during training is ~40x slower (uses birdnet library's SavedModel for embedding extraction, ~1 clip/sec vs TFLite's ~40 clips/sec). This is accepted; no TFLite workarounds are added to this branch.

Local additions preserved in this branch (same as main):
- macOS TF XLA deadlock fix
- Binary upsampling fix
- Upsampling summary
- Per-class validation metrics
- `predict_with_perch` / `embeddings`

**Syncing this branch with upstream:**
```bash
git fetch upstream
git checkout sync-upstream-refactor
git merge upstream/main
# resolve conflicts as needed (model.py and train/utils.py are most likely)
```

**Syncing main with upstream** — ⚠️ **do NOT `git merge upstream/main` anymore.**

Upstream commit `0132334` ("Replace core with `birdnet` library", #867, 2026-04-30)
is now the *base* of upstream/main. Its parent (`3286d78`, #896) is the last
upstream commit that is in main. Merging upstream/main — or advancing main to
`0132334` or anything after it — pulls in the birdnet-library refactor, which
main deliberately does not use (it keeps the fast TFLite pipeline). That is the
purpose of the `sync-upstream-refactor` branch instead.

Checked 2026-06-05: of the 10 commits after the refactor on upstream/main, none
are safely cherry-pickable onto main — the genuine bugfixes either target
refactor-only code (#900 `save_as_rtable`; #902 search/`threaded_brute_search`)
or are already present in main (#902 eval-path fix). The rest conflict because
they build on the refactored tree (#901 cli docs, #903 CI, #905 detached
classifier, #907 settings move). Only `FUNDING.yml` updates apply cleanly and
those are irrelevant to a fork. Re-evaluate after each upstream release, but
default assumption: **post-refactor upstream commits do not belong on main.**

---

## Python environment

```bash
.venv/bin/python   # always use this, not system python
```

Install deps:
```bash
.venv/bin/pip install -e ".[train,tests]"
```

---

## Training workflow (main branch)

Training uses the **old TFLite-based pipeline** — fast embedding extraction built in (~40 clips/sec), no cache step needed.

```bash
./train_pelican.sh pelican0-10
./train_pelican.sh pelican0-10 --epochs 100   # override any flag
```

Default hyperparameters in `train_pelican.sh` (from pelican0-9_Params.csv):

| Parameter | Value |
|-----------|-------|
| hidden_units | 2048 |
| dropout | 0.25 |
| batch_size | 32 |
| learning_rate | 0.0001 |
| upsampling_mode | repeat |
| upsampling_ratio | 0.4 |
| focal_loss | true |
| focal_loss_alpha | 0.25 |
| focal_loss_gamma | 3.0 |
| epochs | 50 |

Override any parameter by appending flags: `./train_pelican.sh pelican0-11 --epochs 100`

Appended flags are passed straight through to `birdnet_analyzer.train`, and argparse uses the last value, so an appended `--epochs 100` overrides the baked-in `--epochs 50`. Note: only flag-style overrides work — don't append a bare positional like `cache.npz`, since `train` accepts a single positional (`INPUT`, already set to the reallybig library) and a second one errors out. The main-branch pipeline needs no cache; the `--cache_file` / `--cache_mode` workflow belongs to the slower `sync-upstream-refactor` branch.

---

## Recognizer outputs

Trained models go to:
```
/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/
```

Each run produces:
- `<name>.tflite` — the classifier
- `<name>_Labels.txt`
- `<name>_Params.csv`
- `<name>_sample_counts.csv`
- `<name>_validation_metrics.csv` — per-species precision/recall (our addition)

---

## Embedding analysis

This repo owns **embedding extraction** (the part coupled to the fast TFLite
`model.py` pipeline). Downstream **analysis and visualisation** — taxonomic
categorisation, UMAP/PCA/t-SNE reduction, misclassification detection, centroid
plots — live in the separate **`birdnetEmbed`** R package
(`~/Documents/birdnetEmbed`, GitHub `traitecoevo/birdnetEmbed`), which consumes
the `.npz`/`.csv` files produced here. Keep that seam: extraction here, analysis
there. See `birdnetEmbed/CLAUDE.md`.

Scripts in `embedding_analysis/`:

| Script | Purpose |
|--------|---------|
| `extract_embeddings.py` | Extract TFLite embeddings + per-class centroids from a clip library → `*_embeddings.npz`, `*_centroids.csv` |
| `identify_misclassifications.py` | Find clips closer to another class centroid (label-quality suspects). Standalone Python; also available as `birdnetEmbed::identify_misclassifications()` |
| `npz_to_cache.py` | Convert per-species .npz to flat training cache |

**Extract (once per model):**
```bash
.venv/bin/python embedding_analysis/extract_embeddings.py \
    --model /path/to/recognizers/pelican0-10.tflite \
    --input /path/to/reallybig \
    --output embedding_analysis/reallybig_pelican0-10
```

**Misclassification analysis (standalone Python):**
```bash
.venv/bin/python embedding_analysis/identify_misclassifications.py \
    --input embedding_analysis/reallybig_pelican0-10_embeddings.npz \
    --output embedding_analysis/reallybig_pelican0-10
```

**Categorised centroid plot** → done in `birdnetEmbed` (R), e.g.:
```r
library(birdnetEmbed)
x    <- load_embeddings_npz("embedding_analysis/reallybig_pelican0-10_embeddings.npz")
cats <- categorize_labels(x,                      # galah/ALA taxonomy
          overrides = system.file("extdata", "category_overrides.csv",
                                  package = "birdnetEmbed"))
plot_centroids(x, categories = cats, label = "pelican0-10")
```

---

## Key data paths

| What | Path |
|------|------|
| Training library | `/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig` |
| Recognizers | `/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/recognizers/` |
| Embeddings / caches | `embedding_analysis/` (in repo) |

---

## Running analysis

```bash
.venv/bin/python -c "
from birdnet_analyzer.analyze import analyze
analyze(
    '/path/to/audio/folder',
    output='/path/to/output',
    classifier='recognizers/pelican0-9.tflite',
)
"
```

Or via CLI:
```bash
.venv/bin/python -m birdnet_analyzer.analyze /path/to/audio \
    --classifier recognizers/pelican0-9.tflite \
    --output /path/to/output
```

---

## Tests

```bash
.venv/bin/pytest tests/ --timeout=120 -q --ignore=tests/analyze/test_analyze.py
# 454 passed — test_analyze.py excluded because it needs a test fixture
# (CustomClassifier.tflite) that isn't in the repo; upstream issue
```
