"""CLI: load a trained GAN checkpoint, sample a pool of synthetic anomalies,
write the ADKGD-format TSV, and run preflight validation.

Usage (dummy KG smoke):
    python scripts/generate_gan_tsv.py --data data/dummy_kg \\
        --ckpt outputs/checkpoints/dummy.pt \\
        --target 50 --out outputs/dummy_gan.tsv --device cpu

Usage (FB15k-237):
    python scripts/generate_gan_tsv.py --data data/FB15K \\
        --ckpt outputs/checkpoints/fb15k.pt \\
        --target 400000 --out outputs/gan_negatives.tsv --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kg_data.loader import load_kg_union                   # noqa: E402
from sampling.sample_anomalies import (                    # noqa: E402
    load_checkpoint,
    sample_anomaly_pool,
)
from sampling.tsv_writer import write_triples_tsv          # noqa: E402
from validation.preflight import validate_tsv              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",   required=True, type=Path,
                        help="Dataset directory with train.txt / valid.txt / test.txt.")
    parser.add_argument("--ckpt",   required=True, type=Path,
                        help="Path to a checkpoint saved by scripts/train_gan.py.")
    parser.add_argument("--out",    required=True, type=Path,
                        help="Output TSV path (will be created/overwritten).")
    parser.add_argument("--target", type=int, required=True,
                        help="Target unique-pool size. Use 400000 for FB15k-237.")
    parser.add_argument("--samples-per-triple", type=int, default=2)
    parser.add_argument("--max-passes",         type=int, default=5)
    parser.add_argument("--batch-size",         type=int, default=256)
    parser.add_argument("--gumbel-temperature", type=float, default=0.5,
                        help="Lower => sharper / closer to argmax. Default 0.5.")
    parser.add_argument("--no-masked-single-position", action="store_true",
                        help="Disable the default single-position masked decoding "
                             "(which mirrors TRIC's substitution semantics: exactly "
                             "one of h/r/t changes, forced to differ from input). "
                             "Use only for diagnostic comparisons; vanilla argmax "
                             "produces near-zero non-collision yields.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Don't run validate_tsv after writing. Use when "
                             "target is below preflight defaults.")
    parser.add_argument("--preflight-min-unique-ratio",   type=float, default=0.95)
    parser.add_argument("--preflight-max-unknown-vocab-rate", type=float, default=0.05)
    parser.add_argument("--preflight-max-collision-rate", type=float, default=0.05)
    args = parser.parse_args()

    if args.device is None:
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_string = args.device
    device = torch.device(device_string)
    print(f"[generate_gan_tsv] device: {device}", flush=True)

    print(f"[generate_gan_tsv] loading union KG from {args.data} ...", flush=True)
    kg = load_kg_union(args.data)
    print(
        f"[generate_gan_tsv] entities={len(kg.entity_id_to_row):,} "
        f"relations={len(kg.relation_id_to_channel):,} "
        f"union_triples={len(kg.triples_idx):,}",
        flush=True,
    )

    print(f"[generate_gan_tsv] loading checkpoint {args.ckpt} ...", flush=True)
    bundle = load_checkpoint(args.ckpt, device=device)
    if bundle['config'].embedding_dim and (
        bundle['generator'].num_entities != len(kg.entity_id_to_row)
        or bundle['generator'].num_relations != len(kg.relation_id_to_channel)
    ):
        print(
            "[generate_gan_tsv] WARNING: checkpoint vocab sizes "
            f"(E={bundle['generator'].num_entities}, "
            f"R={bundle['generator'].num_relations}) do not match the dataset "
            f"(E={len(kg.entity_id_to_row)}, "
            f"R={len(kg.relation_id_to_channel)}). "
            "Strings emitted may be misaligned. Use the dataset the GAN was trained on.",
            flush=True,
        )

    print(
        f"[generate_gan_tsv] sampling pool: target={args.target:,} "
        f"samples_per_triple={args.samples_per_triple} "
        f"max_passes={args.max_passes} batch_size={args.batch_size}",
        flush=True,
    )
    pool = sample_anomaly_pool(
        generator=bundle['generator'],
        entity_embedding=bundle['entity_embedding'],
        relation_embedding=bundle['relation_embedding'],
        clean_triples=list(kg.triples_idx),
        real_triple_set=kg.triple_set_idx,
        target_pool_size=args.target,
        samples_per_triple=args.samples_per_triple,
        max_passes=args.max_passes,
        batch_size=args.batch_size,
        gumbel_temperature=args.gumbel_temperature,
        masked_single_position=not args.no_masked_single_position,
        device=device,
        random_seed=args.seed,
    )
    print(f"[generate_gan_tsv] pool size: {len(pool):,}", flush=True)

    print(f"[generate_gan_tsv] writing TSV to {args.out} ...", flush=True)
    lines_written = write_triples_tsv(args.out, pool, kg)
    print(f"[generate_gan_tsv] wrote {lines_written:,} lines.", flush=True)

    if args.skip_preflight:
        return 0

    print("[generate_gan_tsv] running pre-flight validator ...", flush=True)
    report = validate_tsv(
        args.out, kg,
        target_pool_size=args.target,
        min_unique_ratio=args.preflight_min_unique_ratio,
        max_unknown_vocab_rate=args.preflight_max_unknown_vocab_rate,
        max_collision_rate=args.preflight_max_collision_rate,
    )
    print(report.render())
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
