"""Operation-mix categorisation: classify each generated corruption as TRIC-style edit."""
from typing import Dict, List, Sequence, Tuple

Triple = Tuple[int, int, int]


def categorise_corruption(
    clean_kg: Sequence[Triple], corrupted_kg: Sequence[Triple],
) -> Dict[str, int]:
    """Classify each changed triple between clean and corrupted KGs.

    Returns dict with keys: change_relation, swap_subject, swap_object,
    multi_op, no_change. Counts assume same length (corruption is in-place).
    """
    counts = dict(
        change_relation=0,
        swap_subject=0,
        swap_object=0,
        multi_op=0,
        no_change=0,
    )
    if len(corrupted_kg) != len(clean_kg):
        return counts
    for clean_tri, corr_tri in zip(clean_kg, corrupted_kg):
        if clean_tri == corr_tri:
            counts["no_change"] += 1
            continue
        h,  r,  t  = clean_tri
        h2, r2, t2 = corr_tri
        if h == h2 and t == t2 and r != r2:    counts["change_relation"] += 1
        elif r == r2 and t == t2 and h != h2:  counts["swap_subject"]    += 1
        elif h == h2 and r == r2 and t != t2:  counts["swap_object"]     += 1
        else:                                   counts["multi_op"]       += 1
    return counts


def summarise_operation_mix(
    clean_kg: Sequence[Triple],
    corrupted_kgs: Sequence[Sequence[Triple]],
    *,
    show_first_n: int = 5,
) -> Dict[str, int]:
    """Print a per-sample table + a TOTAL row + a distinct-count line.

    Returns the totals dict for downstream use.
    """
    print(f'{"Sample":<8} {"chg_rel":<10} {"swp_sub":<10} {"swp_obj":<10} {"multi":<10} {"identity":<10}')
    print("-" * 60)

    totals = dict(change_relation=0, swap_subject=0, swap_object=0, multi_op=0, no_change=0)
    distinct = set()
    for k, corr_kg in enumerate(corrupted_kgs):
        c = categorise_corruption(clean_kg, corr_kg)
        for key in totals:
            totals[key] += c[key]
        distinct.add(tuple(sorted(corr_kg)))
        if k < show_first_n:
            print(f"{k:<8} {c['change_relation']:<10} {c['swap_subject']:<10} "
                  f"{c['swap_object']:<10} {c['multi_op']:<10} {c['no_change']:<10}")
    print("-" * 60)
    print(f"TOTAL    {totals['change_relation']:<10} {totals['swap_subject']:<10} "
          f"{totals['swap_object']:<10} {totals['multi_op']:<10} {totals['no_change']:<10}")
    print(f"\nDistinct corrupted KGs across K={len(corrupted_kgs)}: {len(distinct)}")
    return totals


def print_sample_details(
    clean_kg: Sequence[Triple],
    corrupted_kgs: Sequence[Sequence[Triple]],
    *,
    node_labels: Dict[int, str],
    relation_names: Sequence[str],
    num_samples: int = 3,
) -> None:
    """Print the actual changed triples for the first `num_samples` corrupted KGs."""
    print("\n=== Detailed view ===")
    for k in range(min(num_samples, len(corrupted_kgs))):
        print(f"\n--- Sample {k} ---")
        corr_kg = corrupted_kgs[k]
        any_change = False
        for clean_tri, corr_tri in zip(clean_kg, corr_kg):
            if clean_tri == corr_tri:
                continue
            any_change = True
            h,  r,  t  = clean_tri
            h2, r2, t2 = corr_tri
            clean_str = f"{node_labels[h]} --{relation_names[r]}--> {node_labels[t]}"
            corr_str  = f"{node_labels[h2]} --{relation_names[r2]}--> {node_labels[t2]}"
            if   h == h2 and t == t2 and r != r2:   op = "[change_relation]"
            elif r == r2 and t == t2 and h != h2:   op = "[swap_subject]"
            elif h == h2 and r == r2 and t != t2:   op = "[swap_object]"
            else:                                    op = "[multi_op]"
            print(f"  CHANGED:  {clean_str}  ==>  {corr_str}  {op}")
        if not any_change:
            print("  (no triples changed - identity sample)")
