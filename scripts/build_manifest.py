"""Generate the fixed benchmark defined in the project proposal."""

import json
from collections import Counter
from itertools import permutations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every fret vector uses string order G, C, E, A.
CHORDS = {
    "C": [0, 0, 0, 3],
    "G": [0, 2, 3, 2],
    "Am": [2, 0, 0, 0],
    "F": [2, 0, 1, 0],
}

PATTERNS = {
    "P1": ["D", ".", "D", ".", "D", ".", "D", "."],
    "P2": ["D", "U", "D", "U", "D", "U", "D", "U"],
    "P3": ["D", ".", "D", "U", ".", "U", "D", "U"],
}

TEMPOS = [70, 90, 110]
HELD_OUT_TRANSITIONS = {("Am", "G"), ("F", "C")}


def build_manifest():
    cells = []

    for (source, destination), pattern, bpm in product(
        permutations(CHORDS, 2), PATTERNS, TEMPOS
    ):
        if (source, destination) in HELD_OUT_TRANSITIONS:
            split = "heldout_transition"
        elif destination == "F" and pattern == "P3":
            split = "heldout_pair"
        else:
            split = "train"

        cells.append({
            "id": f"{source}_to_{destination}_{pattern}_{bpm}",
            "source": source,
            "destination": destination,
            "pattern": pattern,
            "bpm": bpm,
            "split": split,
        })

    counts = Counter(cell["split"] for cell in cells)

    assert len(cells) == 108
    assert len({cell["id"] for cell in cells}) == 108
    assert counts == {
        "train": 81,
        "heldout_transition": 18,
        "heldout_pair": 9,
    }

    training_cells = [
        cell for cell in cells if cell["split"] == "train"
    ]

    for cell in training_cells:
        assert (
            cell["source"], cell["destination"]
        ) not in HELD_OUT_TRANSITIONS
        assert not (
            cell["destination"] == "F"
            and cell["pattern"] == "P3"
        )

    # All individual factors must remain represented in training.
    for field, expected in (
        ("source", set(CHORDS)),
        ("destination", set(CHORDS)),
        ("pattern", set(PATTERNS)),
        ("bpm", set(TEMPOS)),
    ):
        assert {cell[field] for cell in training_cells} == expected

    return {
        "schema_version": 1,
        "string_order": ["G", "C", "E", "A"],
        "open_string_midi": [67, 60, 64, 69],
        "chord_frets": CHORDS,
        "patterns": PATTERNS,
        "tempos_bpm": TEMPOS,
        "counts": dict(counts),
        "cells": cells,
    }


if __name__ == "__main__":
    manifest = build_manifest()
    output = ROOT / "manifests" / "benchmark_v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest["counts"], indent=2))
    print(f"Wrote {output}")
