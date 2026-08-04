#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
import importlib.metadata as metadata

expected_version = "2.5.2"
version = metadata.version("modisco")
if version != expected_version:
    raise SystemExit(f"Expected modisco {expected_version}, found {version}; refusing to patch")

dist = metadata.distribution("modisco")
site_root = Path(dist.locate_file(""))
target = site_root / "modiscolite" / "descriptive_report.py"
text = target.read_text()

replacements = [
    (
        "if match_key in matches and matches[match_key]:\n"
        "                match_name = matches[match_key].strip()",
        "if match_key in matches and isinstance(matches[match_key], str) and matches[match_key]:\n"
        "                match_name = matches[match_key].strip()",
    ),
]

changed = False
for old, new in replacements:
    count_new = text.count(new)
    count_old = text.count(old)
    if count_old:
        text = text.replace(old, new)
        changed = True
    elif count_new:
        continue
    else:
        raise SystemExit(f"Expected TomTom match guard pattern not found in {target}")

if changed:
    target.write_text(text)
    print(f"Patched {target}")
else:
    print(f"Already patched: {target}")
PY
