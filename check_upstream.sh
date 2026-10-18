#!/usr/bin/env bash
# Report how far this fork has drifted from upstream BirdNET-Analyzer. Read-only.
#
# Usage:
#   ./check_upstream.sh            (compare main against upstream/main)
#   ./check_upstream.sh <ref>      (compare another branch, e.g. a sync branch)
#
# Prints, in order:
#   1. behind/ahead counts and the age of the merge base;
#   2. the files a merge would conflict on (git merge-tree dry run, no checkout);
#   3. every incoming commit tagged RED / AMBER / GREEN by the paths it touches:
#        RED    training, model, audio, model_utils, or the birdnet pin
#               -> full reallybig train + soundscape-eval before main moves
#        AMBER  config, cli, analyze, embeddings, other dependencies
#               -> test suite + subset smoke
#        GREEN  GUI, lang, docs, CI, Docker only
#               -> test suite
#   4. the BirdNET 3.0 TRAINING WATCH: upstream ships "training does not support
#      3.0" markers today; the moment they disappear from upstream/main, 3.0 training
#      has (probably) landed, and it will land on the fork layer's files;
#   5. the newest birdnet on PyPI vs upstream's pin, since 3.0 training may arrive
#      library-first.
#
# Changes nothing: fetches upstream, never checks out, merges or writes the tree.

set -euo pipefail

REF="${1:-main}"
UP="upstream/main"

cd "$(dirname "$0")"

git fetch upstream --quiet

base=$(git merge-base "$REF" "$UP")
behind=$(git rev-list --count "$REF..$UP")
ahead=$(git rev-list --count "$UP..$REF")

echo "== $REF vs $UP"
echo "  behind: $behind   ahead: $ahead"
echo "  merge base: $(git log -1 --format='%h %cs %s' "$base")"
echo "  upstream tip: $(git log -1 --format='%h %cs %s' "$UP")"

if [[ "$behind" -eq 0 ]]; then
    echo "  up to date."
fi

echo
echo "== Merge dry run"
if conflicts=$(git merge-tree --write-tree --name-only --no-messages "$REF" "$UP"); then
    echo "  clean (no conflicts)"
else
    # First line is the would-be tree id; the rest are conflicted paths.
    echo "$conflicts" | tail -n +2 | sed 's/^/  CONFLICT  /'
fi

echo
echo "== Incoming commits by risk"
red=0
while read -r sha; do
    [[ -z "$sha" ]] && continue
    files=$(git diff-tree --no-commit-id --name-only -r "$sha")
    tier=GREEN
    if grep -qE '^birdnet_analyzer/(train/|model\.py|model_utils\.py|audio\.py)' <<<"$files" ||
        git show "$sha" -- pyproject.toml | grep -qE '^[-+].*"birdnet[=<>~ ]'; then
        tier=RED
    elif grep -qE '^(pyproject\.toml|birdnet_analyzer/(config|cli|utils)\.py|birdnet_analyzer/(analyze|embeddings|segments|species)/)' <<<"$files"; then
        tier=AMBER
    fi
    [[ "$tier" == RED ]] && red=$((red + 1))
    printf '  %-5s %s\n' "$tier" "$(git log -1 --format='%h %cs %s' "$sha")"
done < <(git rev-list --reverse "$REF..$UP")
[[ "$behind" -gt 0 ]] && echo "  ($red RED of $behind)"

echo
echo "== BirdNET 3.0 training watch"
markers=$(git grep -c -iE 'training does not support|3\.0 training is not yet supported' \
    "$UP" -- birdnet_analyzer/model_utils.py docs/breaking-changes.rst 2>/dev/null |
    awk -F: '{s += $NF} END {print s + 0}')
if [[ "$markers" -gt 0 ]]; then
    echo "  upstream still says 3.0 training is unsupported ($markers markers)."
else
    echo "  !! the 'no 3.0 training' markers are GONE from $UP."
    echo "  !! 3.0 training has probably landed -- read the incoming train/ and"
    echo "  !! model_utils.py changes before anything else; they hit the fork layer."
fi

echo
echo "== birdnet library"
pin=$(git show "$UP:pyproject.toml" | grep -oE '"birdnet[=<>~][^"]*"' | head -1)
echo "  upstream pin: ${pin:-unknown}"
venv_pip="./.venv/bin/pip"
if [[ -x "$venv_pip" ]]; then
    latest=$("$venv_pip" index versions birdnet 2>/dev/null | head -1 || true)
    echo "  PyPI: ${latest:-unavailable}"
    echo "  this venv: $("$venv_pip" show birdnet 2>/dev/null | awk '/^Version/ {print $2}')"
fi
