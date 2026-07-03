#!/usr/bin/env python3
"""Refresh the eBird species-code lookup file from the current published eBird taxonomy.

`eBird_taxonomy_codes_<edition>.json` is a bidirectional lookup between
BirdNET's frozen label strings ("Genus species_Common Name", one per line of
`BirdNET_GLOBAL_6K_V2.4_Labels.txt`) and eBird's 6-letter species codes. It is
purely auxiliary metadata (used only to print a species code column in Raven
output, see `analyze/utils.py::generate_raven_table`) — it does NOT feed the
model or determine the species list, so refreshing it needs no retraining.

`BirdNET_GLOBAL_6K_V2.4_Labels.txt` itself is NEVER modified here: it's the
model's fixed output-neuron order, baked in at training time by the upstream
BirdNET team. Renaming an entry there would desync it from the checkpoint.

eBird species codes are the stable identifier across taxonomy revisions (that
is their purpose), so refresh matches old codes to new taxonomy rows first,
falling back to scientific-name match only if a code was retired. Any label
whose eBird name has since changed is left exactly as-is in the label file
(untouched) but reported, same drift-report pattern as
`Training_library_assembly_pipeline/curation/check_taxonomy_drift.py` for the
non-bird ALA side.

Usage:
    .venv/bin/python update_ebird_taxonomy.py
"""

import json
import re
from pathlib import Path

import requests

TAXONOMY_VERSIONS_URL = "https://api.ebird.org/v2/ref/taxonomy/versions"
TAXONOMY_URL = "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json"

REPO_ROOT = Path(__file__).parent
PACKAGE_DIR = REPO_ROOT / "birdnet_analyzer"
LABELS_FILE = PACKAGE_DIR / "checkpoints" / "V2.4" / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
CONFIG_FILE = PACKAGE_DIR / "config.py"
OLD_CODES_FILE = PACKAGE_DIR / "eBird_taxonomy_codes_2024E.json"
DRIFT_REPORT = REPO_ROOT / "ebird_taxonomy_drift_report.csv"


def latest_edition() -> str:
    versions = requests.get(TAXONOMY_VERSIONS_URL, timeout=15).json()
    latest = next(v for v in versions if v["latest"])
    year = int(latest["authorityVer"])
    return f"{year}E"


def fetch_taxonomy() -> list[dict]:
    return requests.get(TAXONOMY_URL, timeout=60).json()


def main():
    edition = latest_edition()
    new_codes_file = PACKAGE_DIR / f"eBird_taxonomy_codes_{edition}.json"
    print(f"Latest published eBird taxonomy edition: {edition}")
    if new_codes_file.exists():
        print(f"{new_codes_file.name} already up to date — nothing to do.")
        return

    print("Fetching full eBird taxonomy...")
    taxonomy = fetch_taxonomy()
    by_code = {row["speciesCode"]: row for row in taxonomy}
    by_sciname = {row["sciName"]: row for row in taxonomy}
    print(f"  {len(taxonomy)} taxa fetched.")

    old_codes = json.loads(OLD_CODES_FILE.read_text())
    labels = [line.strip() for line in LABELS_FILE.read_text().splitlines() if line.strip()]
    print(f"{len(labels)} labels in the frozen model label file.")

    new_codes = {}
    drift = []
    unresolved = []
    for label in labels:
        sci, _, common = label.partition("_")
        old_code = old_codes.get(label)

        row = by_code.get(old_code) if old_code else None
        match_method = "code" if row else None
        if row is None:
            row = by_sciname.get(sci)
            match_method = "sciname" if row else None

        if row is None:
            unresolved.append(label)
            if old_code:
                new_codes[label] = old_code
                new_codes[old_code] = label
            continue

        new_code = row["speciesCode"]
        new_codes[label] = new_code
        new_codes[new_code] = label

        if match_method == "sciname" and new_code != old_code:
            drift.append((label, sci, common, row["sciName"], row["comName"], old_code, new_code, "code changed"))
        elif row["sciName"] != sci or row["comName"] != common:
            drift.append((label, sci, common, row["sciName"], row["comName"], old_code, new_code, "name changed upstream"))

    new_codes_file.write_text(json.dumps(new_codes, indent=2, ensure_ascii=False))
    print(f"Wrote {new_codes_file.name} ({len(labels)} labels, all preserved as-is).")

    config_text = CONFIG_FILE.read_text()
    updated_config = re.sub(
        r'CODES_FILE: str = os\.path\.join\(SCRIPT_DIR, "eBird_taxonomy_codes_\S+\.json"\)',
        f'CODES_FILE: str = os.path.join(SCRIPT_DIR, "{new_codes_file.name}")',
        config_text,
    )
    updated_config = re.sub(
        r"# We use the \d{4} eBird taxonomy for species names \(Clements list\)",
        f"# We use the {edition[:-1]} eBird taxonomy for species names (Clements list)",
        updated_config,
    )
    CONFIG_FILE.write_text(updated_config)
    print(f"Updated {CONFIG_FILE.relative_to(REPO_ROOT)} to point at {new_codes_file.name}.")

    if drift:
        with DRIFT_REPORT.open("w") as f:
            f.write("label,old_sciname,old_comname,new_sciname,new_comname,old_code,new_code,reason\n")
            for label, sci, common, new_sci, new_common, old_code, new_code, reason in drift:
                f.write(f'"{label}","{sci}","{common}","{new_sci}","{new_common}","{old_code}","{new_code}","{reason}"\n')
        print(f"\n{len(drift)} label(s) have drifted from the current eBird taxonomy (label file left untouched):")
        for label, sci, common, new_sci, new_common, old_code, new_code, reason in drift[:20]:
            print(f"  [{reason}] {sci} {common!r} -> {new_sci} {new_common!r}")
        if len(drift) > 20:
            print(f"  ... and {len(drift) - 20} more, see {DRIFT_REPORT.name}")
    else:
        print("\nNo drift — every label still matches its current eBird name.")

    if unresolved:
        print(f"\n{len(unresolved)} label(s) had no match at all in the new taxonomy (code kept as-is, needs manual check):")
        for label in unresolved:
            print(f"  {label}")


if __name__ == "__main__":
    main()
