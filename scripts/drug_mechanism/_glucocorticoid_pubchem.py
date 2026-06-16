from __future__ import annotations

import csv
from pathlib import Path


PUBCHEM_GR_SCREEN_TSV = (
    Path(__file__).resolve().parents[2]
    / "data" / "drug_mechanism" / "00_pubchem_glucocorticoid_screen.tsv"
)

PUBCHEM_GR_POSITIVE_LABELS = frozenset({
    "pubchem_GR_or_corticosteroid_candidate",
})


def load_pubchem_gr_set(
    path: Path = PUBCHEM_GR_SCREEN_TSV,
    positive_labels: frozenset[str] = PUBCHEM_GR_POSITIVE_LABELS,
) -> set[str]:
    """Return drug/component names classified as GR/corticosteroid by PubChem."""
    if not path.exists():
        raise FileNotFoundError(
            f"PubChem GR screen TSV not found: {path}. "
            "Run scripts/drug_mechanism/00_pubchem_glucocorticoid_screen.py first."
        )

    names = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("pubchem_annotation") not in positive_labels:
                continue
            for key in ("drug_name", "component_name"):
                name = (row.get(key) or "").strip()
                if name:
                    names.add(name)
    return names
