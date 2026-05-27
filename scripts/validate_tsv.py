"""CLI: pre-flight validate a generated anomalies TSV before shipping.

Mirrors the checks ADKGD's `Reader.load_gan_negatives()` does on its end
(unknown-vocab drops, real-graph collision drops) plus a uniqueness
check (mode-collapse guard). Exits non-zero if any check fails - safe to
chain after `build_random_baseline_tsv.py` or `generate_gan_tsv.py`.

Usage:
    python scripts/validate_tsv.py outputs/dummy_random.tsv \\
        --data data/dummy_kg --target 200
    python scripts/validate_tsv.py outputs/gan_negatives.tsv \\
        --data data/FB15K --target 400000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_data.loader import load_kg_union          # noqa: E402
from validation.preflight import validate_tsv     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="Path to the TSV to validate.")
    parser.add_argument("--data",   required=True, type=Path,
                        help="Dataset directory (provides the union vocab & triple set).")
    parser.add_argument("--target", type=int, default=400_000,
                        help="Required minimum line count. Default 400,000.")
    parser.add_argument("--min-unique-ratio", type=float, default=0.95,
                        help="Required unique/total ratio. Default 0.95.")
    parser.add_argument("--max-unknown-vocab-rate", type=float, default=0.05,
                        help="Max fraction of lines with unknown strings. Default 0.05.")
    parser.add_argument("--max-collision-rate", type=float, default=0.05,
                        help="Max fraction of lines colliding with the real graph. Default 0.05.")
    args = parser.parse_args()

    kg = load_kg_union(args.data)
    report = validate_tsv(
        args.tsv,
        kg,
        target_pool_size=args.target,
        min_unique_ratio=args.min_unique_ratio,
        max_unknown_vocab_rate=args.max_unknown_vocab_rate,
        max_collision_rate=args.max_collision_rate,
    )
    print(report.render())
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
