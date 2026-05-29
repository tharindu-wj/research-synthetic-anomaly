"""CLI: pre-flight validate a generated hashmap TSV before shipping.

Mirrors the checks ADKGD's loader will do on its side (vocab match,
real-graph collision) plus structural checks (column count, slot
label, per-positive completeness, slot-vs-corruption consistency).
Exits non-zero if any check fails - safe to chain after
`generate_gan_hashmap.py` (which auto-runs this) or to invoke
manually after manual edits.

Usage:
    python scripts/validate_hashmap.py outputs/dummy_gan_hashmap.tsv \\
        --data data/dummy_kg
    python scripts/validate_hashmap.py outputs/gan_hashmap.tsv \\
        --data data/FB15K
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_data.loader import load_kg_union                            # noqa: E402
from validation.preflight_hashmap import validate_hashmap_tsv       # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="Path to the hashmap TSV to validate.")
    parser.add_argument("--data", required=True, type=Path,
                        help="Dataset directory (provides the union vocab & triple set).")
    args = parser.parse_args()

    kg = load_kg_union(args.data)
    report = validate_hashmap_tsv(args.tsv, kg)
    print(report.render())
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
