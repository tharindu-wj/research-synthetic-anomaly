"""CLI: load a trained GAN checkpoint, export the 6-column hashmap TSV
consumed by ADKGD, then run preflight validation.

Usage (dummy KG smoke, CPU):
    python scripts/generate_gan_hashmap.py \\
        --data data/dummy_kg \\
        --ckpt outputs/checkpoints/dummy.pt \\
        --out  outputs/dummy_gan_hashmap.tsv \\
        --device cpu

Usage (FB15K-237, V100):
    python scripts/generate_gan_hashmap.py \\
        --data data/FB15K \\
        --ckpt outputs/checkpoints/fb15k.pt \\
        --out  outputs/gan_hashmap.tsv \\
        --device cuda

Output: one row per unique real triple. Six tab-separated columns:
    orig_h <TAB> orig_r <TAB> orig_t <TAB> neg_h <TAB> neg_r <TAB> neg_t
Each row's negative differs from the original at exactly ONE of the
three positions (head, rel, or tail), picked uniformly at random per
positive at export time. See src/sampling/hashmap_export.py for the
full on-disk contract.
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

from kg_data.loader import load_kg_union                                # noqa: E402
from sampling.hashmap_export import (                                   # noqa: E402
    export_hashmap_tsv,
    load_checkpoint,
)
from validation.preflight_hashmap import validate_hashmap_tsv           # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",   required=True, type=Path,
                        help="Dataset directory with train.txt / valid.txt / test.txt.")
    parser.add_argument("--ckpt",   required=True, type=Path,
                        help="Path to a checkpoint saved by scripts/train_gan.py.")
    parser.add_argument("--out",    required=True, type=Path,
                        help="Output hashmap TSV path (will be created/overwritten).")
    parser.add_argument("--batch-size",         type=int, default=256)
    parser.add_argument("--gumbel-temperature", type=float, default=0.5,
                        help="Lower => sharper / closer to argmax. Default 0.5.")
    parser.add_argument("--max-retries",        type=int, default=20,
                        help="Per-item retries before uniform-random fallback.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu. Default: cuda if available else cpu.")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Don't run validate_hashmap_tsv after writing.")
    args = parser.parse_args()

    if args.device is None:
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_string = args.device
    device = torch.device(device_string)
    print(f"[generate_gan_hashmap] device: {device}", flush=True)

    print(f"[generate_gan_hashmap] loading union KG from {args.data} ...", flush=True)
    kg = load_kg_union(args.data)
    print(
        f"[generate_gan_hashmap] entities={len(kg.entity_id_to_row):,} "
        f"relations={len(kg.relation_id_to_channel):,} "
        f"union_triples={len(kg.triples_idx):,}",
        flush=True,
    )

    print(f"[generate_gan_hashmap] loading checkpoint {args.ckpt} ...", flush=True)
    bundle = load_checkpoint(args.ckpt, device=device)
    if (
        bundle['generator'].num_entities != len(kg.entity_id_to_row)
        or bundle['generator'].num_relations != len(kg.relation_id_to_channel)
    ):
        print(
            "[generate_gan_hashmap] WARNING: checkpoint vocab sizes "
            f"(E={bundle['generator'].num_entities}, "
            f"R={bundle['generator'].num_relations}) do not match the dataset "
            f"(E={len(kg.entity_id_to_row)}, "
            f"R={len(kg.relation_id_to_channel)}). "
            "Strings emitted may be misaligned. Use the dataset the GAN was trained on.",
            flush=True,
        )

    print(
        f"[generate_gan_hashmap] exporting hashmap: "
        f"batch_size={args.batch_size} max_retries={args.max_retries} "
        f"gumbel_temperature={args.gumbel_temperature}",
        flush=True,
    )
    stats = export_hashmap_tsv(
        generator=bundle['generator'],
        entity_embedding=bundle['entity_embedding'],
        relation_embedding=bundle['relation_embedding'],
        kg=kg,
        output_path=args.out,
        gumbel_temperature=args.gumbel_temperature,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        device=device,
        random_seed=args.seed,
    )
    print(f"[generate_gan_hashmap] export complete: {stats.render()}", flush=True)

    if args.skip_preflight:
        return 0

    print("[generate_gan_hashmap] running pre-flight validator ...", flush=True)
    report = validate_hashmap_tsv(args.out, kg)
    print(report.render())
    return 0 if report.is_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
