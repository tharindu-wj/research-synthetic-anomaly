"""CLI: produce a random-baseline anomalies TSV ready for ADKGD.

This is the Phase-1 deliverable: a stub file the user can SCP into the
ADKGD repo *immediately*, before the real GAN exists. It proves that:
  * the TSV format matches what `Reader.load_gan_negatives` expects,
  * the vocab strings round-trip correctly,
  * the cluster-side `--neg_source gan` plumbing all works.

It also serves as an honest control in the final metric comparison: if
the GAN's TSV outperforms a TSV produced here, the lift comes from
structure rather than a fresh random seed.

Usage:
    python scripts/build_random_baseline_tsv.py \\
        --data data/dummy_kg --target 200  --out outputs/dummy_random.tsv
    python scripts/build_random_baseline_tsv.py \\
        --data data/FB15K            --target 400000 --out outputs/random_baseline.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baseline_random.random_anomaly_generator import generate_random_anomalies  # noqa: E402
from kg_data.loader import load_kg_union                                        # noqa: E402
from sampling.tsv_writer import write_triples_tsv                               # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",   required=True, type=Path,
                        help="Dataset directory with train.txt / valid.txt / test.txt.")
    parser.add_argument("--target", required=True, type=int,
                        help="Target number of unique synthetic triples.")
    parser.add_argument("--out",    required=True, type=Path,
                        help="Output TSV path.")
    parser.add_argument("--seed",   type=int, default=0,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    print(f"[random_baseline] loading union KG from {args.data} ...", flush=True)
    kg = load_kg_union(args.data)
    print(
        f"[random_baseline] entities={len(kg.entity_id_to_row):,} "
        f"relations={len(kg.relation_id_to_channel):,} "
        f"union_triples={len(kg.triples_idx):,}",
        flush=True,
    )

    print(f"[random_baseline] generating {args.target:,} unique random triples ...", flush=True)
    random_generator = np.random.default_rng(args.seed)
    pool = generate_random_anomalies(
        num_entities=len(kg.entity_id_to_row),
        num_relations=len(kg.relation_id_to_channel),
        real_triple_set=kg.triple_set_idx,
        target_pool_size=args.target,
        random_generator=random_generator,
    )
    print(f"[random_baseline] generated {len(pool):,} triples.", flush=True)

    print(f"[random_baseline] writing TSV to {args.out} ...", flush=True)
    lines_written = write_triples_tsv(args.out, pool, kg)
    print(f"[random_baseline] done. {lines_written:,} lines written.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
