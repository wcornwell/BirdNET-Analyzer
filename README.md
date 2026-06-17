<div align="center">
  <h1>BirdNET-Analyzer</h1>
    <a href="https://birdnet-team.github.io/BirdNET-Analyzer/">
        <img src="https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/docs/_static/logo_birdnet_big.png?raw=true" width="300" alt="BirdNET-Logo" />
    </a>
</div>
<br>
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![OS](https://badgen.net/badge/OS/Linux%2C%20Windows%2C%20macOS/blue)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

</div>

---

> **⚠ Fork Notice** — this is a research fork of [birdnet-team/BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer). This README covers only what the fork adds; for installation, full documentation, the GUI, and model downloads, see the [upstream project](https://github.com/birdnet-team/BirdNET-Analyzer).

## About this fork

A research fork of BirdNET-Analyzer focused on **training and inspecting custom
classifiers**. Where upstream ships a general-purpose bird model, this fork adds the
pieces you need to build your own recognizer from a local clip library and to judge how
well it separates classes — per-class validation metrics, fixes and reporting for class
upsampling, and embedding extraction for inspecting the feature space (see
[What's different from upstream](#whats-different-from-upstream) below).

It is used here to train the Australian-fauna **"pelican" recognizer series**, and the
`main` branch deliberately keeps the fast **TFLite-based** embedding pipeline
(~40 clips/sec) rather than upstream's `birdnet`-library refactor.

Downstream embedding **analysis and visualisation** (taxonomic categorisation, UMAP/PCA/t-SNE
reduction, misclassification detection, centroid plots) lives in the companion R package
[`birdnetEmbed`](https://github.com/traitecoevo/birdnetEmbed), which consumes the `.npz`/`.csv`
files produced here. The seam is deliberate: extraction here, analysis there.

### What's different from upstream

1. **Per-class validation metrics** — after training a custom classifier, per-class precision and recall are computed on the validation set; the worst-10 classes are printed and a full `_validation_metrics.csv` is written alongside the model. ([diff](https://github.com/wcornwell/BirdNET-Analyzer/commit/f41b970))
2. **Upsampling bug fix** — `upsample_core` used a shared global counter (`len(y_temp)`) that prevented minority classes after the first from being upsampled; the fix uses a per-class counter. ([diff](https://github.com/wcornwell/BirdNET-Analyzer/commit/e8f7e80))
3. **Upsampling summary** — the upsampling step prints a human-readable summary (reference class, target sample count, classes needing upsampling, 5 smallest classes by name).
4. **macOS TF XLA deadlock fix** — `TF_XLA_FLAGS` etc. set before TensorFlow imports in `model.py` to avoid a deadlock when TF initialises alongside PyArrow/tqdm.

### Where the pieces are

| Area | Location |
|------|----------|
| Train a recognizer (TFLite pipeline) | [`train_pelican.sh`](train_pelican.sh) → `birdnet_analyzer.train` |
| Embedding **extraction** | [`embedding_analysis/`](embedding_analysis/) (this repo) |
| Embedding **analysis / plots** | [`birdnetEmbed`](https://github.com/traitecoevo/birdnetEmbed) R package |
| Recognizer outputs (`.tflite`, labels, metrics) | OneDrive `call_library/recognizers/` |

### Training a custom classifier

Point the trainer at a folder whose subfolders are the class labels (one per species/call
type); each subfolder holds that class's audio clips:

```bash
.venv/bin/python -m birdnet_analyzer.train /path/to/clip-library \
    -o my-classifier.tflite \
    --epochs 50 --hidden_units 2048 --dropout 0.25 \
    --upsampling_mode repeat --upsampling_ratio 0.4
```

Alongside `my-classifier.tflite` the fork also writes `my-classifier_validation_metrics.csv`
(per-class precision/recall) and prints the worst-performing classes and an upsampling
summary — see [What's different from upstream](#whats-different-from-upstream).

### Embedding extraction (`embedding_analysis/`)

| Script | Purpose |
|--------|---------|
| `extract_embeddings.py` | Extract TFLite embeddings + per-class centroids from a clip library → `*_embeddings.npz`, `*_centroids.csv` |
| `identify_misclassifications.py` | Find clips closer to another class's centroid than their own (also in `birdnetEmbed`) |
| `npz_to_cache.py` | Convert per-species `.npz` to a flat training cache |

```bash
# extract embeddings for a model (once per model)
.venv/bin/python embedding_analysis/extract_embeddings.py \
    --model /path/to/recognizers/pelican0-10.tflite \
    --input /path/to/call_library/reallybig \
    --output embedding_analysis/reallybig_pelican0-10
```

The resulting `.npz`/`.csv` are then fed to [`birdnetEmbed`](https://github.com/traitecoevo/birdnetEmbed)
for categorisation (galah/ALA taxonomy + manual overrides), dimensionality reduction, and
the categorised centroid plot — see that package's README.

---

## Upstream BirdNET-Analyzer

This fork is built on the upstream project, developed by the [K. Lisa Yang Center for Conservation Bioacoustics](https://www.birds.cornell.edu/ccb/) at the Cornell Lab of Ornithology with Chemnitz University of Technology. For installation, full documentation, the GUI, and pre-trained models, see upstream:

- **Repository**: <https://github.com/birdnet-team/BirdNET-Analyzer>
- **Documentation**: <https://birdnet-team.github.io/BirdNET-Analyzer/>
- **Models** (Zenodo): <https://zenodo.org/records/15050749>

If you use BirdNET in your research, please cite:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## License

- **Source code**: [MIT License](https://opensource.org/licenses/MIT)
- **Models**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) — all educational and research use counts as non-commercial.
