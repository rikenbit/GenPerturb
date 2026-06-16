#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import load_panel_drug_names


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "drug_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TSV = OUT_DIR / "00_pubchem_glucocorticoid_screen.tsv"
OUT_TXT = OUT_DIR / "00_pubchem_glucocorticoid_evidence.txt"
CACHE_DIR = OUT_DIR / "pubchem_cache"

CID_OVERRIDES = {
    "Beclomethasone dipropionate": "21700",
    "Budesonide": "5281004",
}

POSITIVE_TERMS = [
    "glucocorticoid receptor agonist",
    "glucocorticoid receptor",
    "glucocorticoid",
    "corticosteroid hormone receptor agonist",
    "corticosteroid",
    "anti-inflammatory corticosteroid",
    "nr3c1",
]

STRONG_POSITIVE_TERMS = [
    "glucocorticoid receptor agonist",
    "glucocorticoid receptor",
    "corticosteroid hormone receptor agonist",
    "anti-inflammatory corticosteroid",
    "nr3c1",
]

GR_SPECIFIC_TERMS = [
    "glucocorticoid receptor agonist",
    "glucocorticoid receptor",
    "nr3c1",
]

DEFINITION_PATTERNS = [
    r"\bis an? [^.]{0,80}\bglucocorticoid\b",
    r"\bis an? [^.]{0,80}\bcorticosteroid\b",
    r"\bis an? [^.]{0,80}\bcorticosteroid hormone\b",
    r"\bsynthetic [^.]{0,60}\bglucocorticoid\b",
    r"\bsynthetic [^.]{0,60}\bcorticosteroid\b",
    r"\btopical [^.]{0,60}\bglucocorticoid\b",
    r"\btopical [^.]{0,60}\bcorticosteroid\b",
    r"\binhaled [^.]{0,60}\bcorticosteroid\b",
    r"\bhas a role as an? [^.]{0,80}\bglucocorticoid\b",
    r"\bmechanism of action [^.]{0,120}\bcorticosteroid hormone receptor agonist\b",
    r"\bh02ab\s*-\s*glucocorticoids\b",
]

CAUTION_PATTERNS = [
    r"\bslight glucocorticoid activity\b",
    r"\bweak[^.]{0,40}\bglucocorticoid receptor\b",
    r"\bweak binding[^.]{0,40}\bglucocorticoid\b",
]

COMBINATION_PRODUCT_PATTERNS = [
    r"\bis a combination of\b",
    r"\bcombination of\b[^.]{0,120}\bcorticosteroid\b",
    r"\bin combination with\b[^.]{0,120}\bcorticosteroid\b",
    r"\bfixed[- ]dose combination\b[^.]{0,120}\bcorticosteroid\b",
]

NEGATING_PATTERNS = [
    r"\bno [^.]{0,80}\bglucocorticoid\b",
    r"\bnot [^.]{0,80}\bglucocorticoid\b",
    r"\bdoes not [^.]{0,80}\bglucocorticoid\b",
]

INDIRECT_TERMS = [
    "11 beta-hydroxysteroid dehydrogenase",
    "11-beta-hydroxysteroid dehydrogenase",
    "11beta-hydroxysteroid dehydrogenase",
    "11β-hydroxysteroid dehydrogenase",
    "glycyrrhizin",
    "glycyrrhetinic",
]

NEGATIVE_CONTEXT_TERMS = [
    "non-corticosteroid",
    "without corticosteroid",
]


USE_CURL = False
CACHE_ONLY = False


def fetch_text(url: str, cache_path: Path, delay: float = 0.18) -> str:
    if cache_path.exists():
        return cache_path.read_text()
    if CACHE_ONLY:
        raise RuntimeError("CACHE_MISS")
    if USE_CURL:
        last_error = None
        for attempt in range(3):
            proc = subprocess.run(
                ["curl", "-fsSL", "--max-time", "20", url],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode == 0:
                text = proc.stdout
                break
            last_error = proc.stderr.strip() or f"curl exit {proc.returncode}"
            time.sleep(1.0 + attempt)
        else:
            raise RuntimeError(last_error)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "GenPerturb-PubChem-screen/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    time.sleep(delay)
    return text


def clean_component_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+_", "_", cleaned)
    cleaned = re.sub(r"_\s+", "_", cleaned)
    return cleaned


def candidate_components(drug_name: str) -> list[str]:
    parts = [clean_component_name(drug_name)]
    if "_" in drug_name:
        parts.extend(clean_component_name(part) for part in drug_name.split("_"))
    seen = set()
    out = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return out


def name_variants(name: str) -> list[str]:
    variants = []
    no_parens = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if no_parens:
        variants.append(no_parens)
    variants.append(name)
    replacements = {
        " DiHCl": " dihydrochloride",
        " 2HCl": " dihydrochloride",
        " HCl": " hydrochloride",
    }
    for old, new in replacements.items():
        if old in name:
            variants.append(name.replace(old, new))
    out = []
    seen = set()
    for variant in variants:
        variant = clean_component_name(variant)
        if variant and variant not in seen:
            seen.add(variant)
            out.append(variant)
    return out


def resolve_cid(name: str) -> tuple[str, str, str]:
    if name in CID_OVERRIDES:
        return CID_OVERRIDES[name], name, ""
    for variant in name_variants(name):
        quoted = urllib.parse.quote(variant)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quoted}/cids/TXT"
        cache = CACHE_DIR / "cid" / f"{safe_filename(variant)}.txt"
        try:
            text = fetch_text(url, cache).strip()
        except urllib.error.HTTPError:
            continue
        except Exception as exc:
            return "", variant, f"ERROR: {exc}"
        cid = text.splitlines()[0].strip() if text else ""
        if cid.isdigit():
            return cid, variant, ""
    return "", name, "NO_CID"


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:160] or "empty"


def flatten_sections(node) -> list[str]:
    texts = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "String" and isinstance(value, str):
                texts.append(value)
            elif key == "TOCHeading" and isinstance(value, str):
                texts.append(f"[{value}]")
            else:
                texts.extend(flatten_sections(value))
    elif isinstance(node, list):
        for item in node:
            texts.extend(flatten_sections(item))
    return texts


def fetch_pubchem_text(cid: str) -> tuple[str, str]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    cache = CACHE_DIR / "pug_view" / f"{cid}.json"
    try:
        raw = fetch_text(url, cache)
        data = json.loads(raw)
    except Exception as exc:
        return "", f"ERROR: {exc}"
    texts = flatten_sections(data)
    return "\n".join(texts), ""


def matched_terms(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if term.lower() in lower]


def make_snippet(text: str, terms: list[str], width: int = 180) -> str:
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in terms if lower.find(term.lower()) >= 0]
    if not positions:
        return ""
    pos = min(positions)
    start = max(0, pos - width // 2)
    end = min(len(text), pos + width)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    return snippet


def classify(text: str) -> tuple[str, list[str], str]:
    pos = matched_terms(text, POSITIVE_TERMS)
    indirect = matched_terms(text, INDIRECT_TERMS)
    negative = matched_terms(text, NEGATIVE_CONTEXT_TERMS)
    lower = text.lower()
    gr_specific = matched_terms(text, GR_SPECIFIC_TERMS)
    definition_hit = any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in DEFINITION_PATTERNS)
    caution_hit = any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in CAUTION_PATTERNS)
    negating_hit = any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in NEGATING_PATTERNS)
    combo_product = any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in COMBINATION_PRODUCT_PATTERNS)

    blocked_by_combination = combo_product and not gr_specific
    blocked_by_negation = negating_hit and not definition_hit
    self_defined = definition_hit and not blocked_by_combination and not blocked_by_negation
    gr_positive = bool(gr_specific) and not blocked_by_combination and not blocked_by_negation

    if self_defined:
        label = "pubchem_GR_or_corticosteroid_candidate"
        confidence = "keyword_positive"
    elif caution_hit:
        label = "pubchem_glucocorticoid_like_caution"
        confidence = "keyword_caution"
    elif gr_positive:
        label = "pubchem_GR_or_corticosteroid_candidate"
        confidence = "keyword_positive"
    elif indirect:
        label = "pubchem_indirect_glucocorticoid_related"
        confidence = "keyword_indirect"
    else:
        label = "no_pubchem_keyword_match"
        confidence = "keyword_negative"
    terms = pos + indirect + negative
    return label, terms, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drug",
        action="append",
        default=[],
        help=(
            "Drug/component name to screen. Repeat to pass multiple names. "
            "If omitted, the complete experiment drug panel (unique single-drug "
            "perturbation names from the TF-MoDISco matrix) is screened."
        ),
    )
    parser.add_argument(
        "--use-curl",
        action="store_true",
        help="Fetch PubChem URLs through curl instead of Python urllib.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not call PubChem; classify only already cached responses.",
    )
    return parser.parse_args()


def input_drug_names(args: argparse.Namespace) -> list[str]:
    names = args.drug if args.drug else load_panel_drug_names()
    out = []
    seen = set()
    for name in names:
        cleaned = clean_component_name(name)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return out


def main() -> None:
    global USE_CURL, CACHE_ONLY
    args = parse_args()
    USE_CURL = args.use_curl
    CACHE_ONLY = args.cache_only

    drugs = input_drug_names(args)
    rows = []
    for drug_name in drugs:
        for component in candidate_components(drug_name):
            cid, query_name, cid_status = resolve_cid(component)
            if cid:
                text, text_status = fetch_pubchem_text(cid)
                label, terms, confidence = classify(text)
                status = text_status or "OK"
                snippet = make_snippet(text, terms)
            else:
                label = "no_pubchem_cid"
                confidence = "not_evaluated"
                terms = []
                status = cid_status
                snippet = ""
            rows.append(
                {
                    "drug_name": drug_name,
                    "component_name": component,
                    "pubchem_query_name": query_name,
                    "cid": cid,
                    "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else "",
                    "pubchem_annotation": label,
                    "confidence": confidence,
                    "matched_terms": ";".join(terms),
                    "status": status,
                    "evidence_snippet": snippet,
                }
            )

    fieldnames = [
        "drug_name",
        "component_name",
        "pubchem_query_name",
        "cid",
        "pubchem_url",
        "pubchem_annotation",
        "confidence",
        "matched_terms",
        "status",
        "evidence_snippet",
    ]
    with OUT_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    positive_rows = [
        row
        for row in rows
        if row["pubchem_annotation"] != "no_pubchem_keyword_match"
        and row["pubchem_annotation"] != "no_pubchem_cid"
    ]
    with OUT_TXT.open("w") as handle:
        handle.write("PubChem glucocorticoid keyword evidence\n")
        handle.write(f"Date: {date.today().isoformat()}\n")
        if args.drug:
            handle.write("Input: --drug arguments\n")
        else:
            handle.write("Input: complete experiment drug panel (TF-MoDISco single-drug perturbations)\n")
        handle.write(f"Rows screened: {len(rows)}\n")
        handle.write(f"Positive/indirect rows: {len(positive_rows)}\n\n")
        for row in positive_rows:
            handle.write(f"- drug_name: {row['drug_name']}\n")
            handle.write(f"  component_name: {row['component_name']}\n")
            handle.write(f"  CID: {row['cid']}\n")
            handle.write(f"  URL: {row['pubchem_url']}\n")
            handle.write(f"  annotation: {row['pubchem_annotation']}\n")
            handle.write(f"  matched_terms: {row['matched_terms']}\n")
            handle.write(f"  evidence_snippet: {row['evidence_snippet']}\n\n")

    print(f"Wrote {OUT_TSV}")
    print(f"Wrote {OUT_TXT}")
    print(f"Rows screened: {len(rows)}")
    print(f"Positive/indirect rows: {len(positive_rows)}")


if __name__ == "__main__":
    main()
