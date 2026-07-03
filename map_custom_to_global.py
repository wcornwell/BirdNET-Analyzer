#!/usr/bin/env python3
"""Bridge the custom pelican model's bird classes to the frozen GLOBAL BirdNET model.

The two models sit on DIFFERENT eBird editions and that is fine:
  - The GLOBAL model (BirdNET_GLOBAL_6K_V2.4) has FROZEN labels, locked to the
    edition it was trained on — it never moves.
  - The CUSTOM model's bird classes track CURRENT eBird (updated yearly; see
    Training_library_assembly_pipeline/curation/check_ebird_drift.py), so their
    scientific/common names drift ahead of the global model over time.

The stable bridge between them is the eBird SPECIES CODE, which is invariant
across editions by design (e.g. 'ytbcoc1' stays 'ytbcoc1' when the name goes
Calyptorhynchus funereus -> Zanda funerea). This tool emits a lookup CSV
keyed on that code so a custom class can always be related to its global-model
counterpart, no matter how far the display names have diverged.

Resolution per custom bird class:
  1. Exact label already in the global model  -> same code (custom == global edition).
  2. Else resolve the custom scientific name against CURRENT eBird -> code ->
     reverse-lookup the global label that carries that code (custom is newer).
  3. Else unmappable (reported) — not in eBird, or a code eBird has since retired.

Human ('Homo sapiens_*') and environmental ('Environment_*', 'Noise',
'Diptera_Flies') labels are excluded — they are not eBird taxa. Non-bird
biological classes (frogs/insects/mammals) simply aren't in eBird and drop out
of the bird bridge automatically.

Read-only. Emits custom_to_global_lookup.csv.

Usage:
    .venv/bin/python map_custom_to_global.py
    .venv/bin/python map_custom_to_global.py --labels /path/to/pelican0-16_Labels.txt
"""

import argparse
import csv
import json
from pathlib import Path

import requests

TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json&locale=en_AU"

REPO_ROOT = Path(__file__).parent
GLOBAL_CODES_FILE = REPO_ROOT / "birdnet_analyzer" / "eBird_taxonomy_codes_2025E.json"
DEFAULT_LIBRARY_DIR = Path(
    "/Users/z3484779/Library/CloudStorage/OneDrive-UNSW/call_library/reallybig"
)
DEFAULT_OUTPUT = REPO_ROOT / "custom_to_global_lookup.csv"

HELPER_PREFIXES = ("Environment_", "Homo sapiens_")
HELPER_EXACT = {"Noise", "Diptera_Flies"}


def is_helper(label: str) -> bool:
    return label.startswith(HELPER_PREFIXES) or label in HELPER_EXACT


def custom_classes(labels_path: Path | None, library_dir: Path) -> list[str]:
    if labels_path:
        return [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]
    return sorted(p.name for p in library_dir.iterdir() if p.is_dir())


def latest_global_codes_file() -> Path:
    """Prefer the newest eBird_taxonomy_codes_*.json present (edition-agnostic)."""
    candidates = sorted((REPO_ROOT / "birdnet_analyzer").glob("eBird_taxonomy_codes_*.json"))
    return candidates[-1] if candidates else GLOBAL_CODES_FILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Custom model Labels.txt (default: live reallybig folder names)")
    parser.add_argument("--library-dir", type=Path, default=DEFAULT_LIBRARY_DIR)
    parser.add_argument("--global-codes", type=Path, default=None,
                        help="Global model eBird codes JSON (default: newest present)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    codes_file = args.global_codes or latest_global_codes_file()
    global_codes = json.loads(codes_file.read_text())  # bidirectional: label<->code
    print(f"Global model codes: {codes_file.name}")

    print("Fetching current eBird taxonomy (en_AU)...")
    taxonomy = requests.get(TAXONOMY_URL, timeout=60).json()
    ebird_by_sci = {r["sciName"]: r for r in taxonomy}
    print(f"  {len(taxonomy)} taxa fetched.")

    labels = custom_classes(args.labels, args.library_dir)
    source = args.labels.name if args.labels else "reallybig folders"
    print(f"{len(labels)} custom classes from {source}.")

    rows, unmappable = [], 0
    for label in labels:
        if is_helper(label):
            continue
        sci = label.split("_", 1)[0]

        # 1. exact match to a frozen global label
        code = global_codes.get(label)
        # 2. else resolve current eBird sciName -> code
        if not code:
            r = ebird_by_sci.get(sci)
            code = r["speciesCode"] if r else None
        if not code:
            continue  # not an eBird taxon (frog/insect/mammal) -> not part of the bird bridge

        global_label = global_codes.get(code)  # reverse lookup (JSON is bidirectional)
        if global_label == label:
            status = "same_label"
        elif global_label:
            status = "renamed_since_global"
        else:
            status = "not_in_global_model"
            unmappable += 1

        rows.append({
            "custom_class": label,
            "ebird_code": code,
            "global_label": global_label or "",
            "status": status,
        })

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["custom_class", "ebird_code", "global_label", "status"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["custom_class"]))

    renamed = sum(1 for r in rows if r["status"] == "renamed_since_global")
    print(f"\n{len(rows)} bird classes bridged to the global model via eBird code:")
    print(f"  {sum(1 for r in rows if r['status'] == 'same_label')} identical label")
    print(f"  {renamed} renamed since the global edition (code still bridges them)")
    print(f"  {unmappable} not present in the global model (custom-only species)")
    if renamed:
        print("\nRenamed-but-bridged examples:")
        for r in [r for r in rows if r["status"] == "renamed_since_global"][:10]:
            print(f"  {r['custom_class']}  =[{r['ebird_code']}]=  {r['global_label']}")
    print(f"\nLookup written to {args.output}")


if __name__ == "__main__":
    main()
