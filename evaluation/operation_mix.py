"""Categorise each generated corruption as a TRIC-style edit.

After inference we have K corrupted KGs. For each one we want to know:
  * how many `change_relation` edits did the GAN produce?
  * how many `swap_subject` / `swap_object`?
  * any unrecognised edits (multi-position changes)?
  * how many triples were left unchanged?

This module compares each corrupted KG against the clean KG and tallies the
operation counts. The result tells us whether the GAN learned a clean,
TRIC-compatible corruption pattern or whether it's hallucinating odd edits.
"""
from typing import Dict, List, Sequence, Tuple


# (head_index, relation_index, tail_index)
Triple = Tuple[int, int, int]


def categorise_corruption(
    clean_kg: Sequence[Triple],
    corrupted_kg: Sequence[Triple],
) -> Dict[str, int]:
    """Classify each changed triple between a clean KG and a corrupted KG.

    Assumes both KGs have the same length and are aligned position-by-position
    (which is what our two-stage inference guarantees - unselected triples
    pass through unchanged at their original position).

    Returns a dict with these keys, summed across all triples:
        change_relation : (h, r, t) -> (h,  r', t),   r' != r
        swap_subject    : (h, r, t) -> (h', r,  t),   h' != h
        swap_object     : (h, r, t) -> (h,  r,  t'),  t' != t
        multi_op        : two or more positions changed - unrecognised pattern
        no_change       : triple unchanged
    """
    operation_counts = dict(
        change_relation=0,
        swap_subject=0,
        swap_object=0,
        multi_op=0,
        no_change=0,
    )
    # Defensive: if KGs differ in length, we can't compare position-by-position.
    # Return an all-zero dict to signal malformed input.
    if len(corrupted_kg) != len(clean_kg):
        return operation_counts

    for clean_triple, corrupted_triple in zip(clean_kg, corrupted_kg):
        if clean_triple == corrupted_triple:
            operation_counts["no_change"] += 1
            continue
        clean_head_index, clean_relation_index, clean_tail_index = clean_triple
        new_head_index,   new_relation_index,   new_tail_index   = corrupted_triple

        # Identify which of (head, relation, tail) changed
        head_changed     = clean_head_index     != new_head_index
        relation_changed = clean_relation_index != new_relation_index
        tail_changed     = clean_tail_index     != new_tail_index

        if not head_changed and not tail_changed and relation_changed:
            operation_counts["change_relation"] += 1
        elif not relation_changed and not tail_changed and head_changed:
            operation_counts["swap_subject"] += 1
        elif not head_changed and not relation_changed and tail_changed:
            operation_counts["swap_object"] += 1
        else:
            operation_counts["multi_op"] += 1
    return operation_counts


def summarise_operation_mix(
    clean_kg: Sequence[Triple],
    corrupted_kgs: Sequence[Sequence[Triple]],
    *,
    num_samples_to_print: int = 5,
) -> Dict[str, int]:
    """Print a per-sample table + a TOTAL row + a distinct-output count.

    Returns the totals dict for downstream use.
    """
    # Table header
    print(
        f'{"Sample":<8} {"chg_rel":<10} {"swp_sub":<10} '
        f'{"swp_obj":<10} {"multi":<10} {"identity":<10}'
    )
    print("-" * 60)

    operation_totals = dict(
        change_relation=0, swap_subject=0, swap_object=0,
        multi_op=0, no_change=0,
    )
    distinct_corrupted_kgs: set = set()

    for sample_index, corrupted_kg in enumerate(corrupted_kgs):
        sample_counts = categorise_corruption(clean_kg, corrupted_kg)
        for operation_name in operation_totals:
            operation_totals[operation_name] += sample_counts[operation_name]

        # Two corrupted KGs are "the same" iff they contain the same set of
        # triples (order doesn't matter at the KG level). We use sorted tuple
        # as a canonical hashable representation.
        distinct_corrupted_kgs.add(tuple(sorted(corrupted_kg)))

        if sample_index < num_samples_to_print:
            print(
                f"{sample_index:<8} {sample_counts['change_relation']:<10} "
                f"{sample_counts['swap_subject']:<10} "
                f"{sample_counts['swap_object']:<10} "
                f"{sample_counts['multi_op']:<10} "
                f"{sample_counts['no_change']:<10}"
            )

    print("-" * 60)
    print(
        f"TOTAL    {operation_totals['change_relation']:<10} "
        f"{operation_totals['swap_subject']:<10} "
        f"{operation_totals['swap_object']:<10} "
        f"{operation_totals['multi_op']:<10} "
        f"{operation_totals['no_change']:<10}"
    )
    print(f"\nDistinct corrupted KGs across K={len(corrupted_kgs)}: "
          f"{len(distinct_corrupted_kgs)}")
    return operation_totals


def print_sample_details(
    clean_kg: Sequence[Triple],
    corrupted_kgs: Sequence[Sequence[Triple]],
    *,
    node_labels: Dict[int, str],
    relation_names: Sequence[str],
    num_samples_to_print: int = 3,
) -> None:
    """Print human-readable details for the first `num_samples_to_print` corrupted KGs.

    For each corrupted KG, prints each CHANGED triple as a "before -> after"
    line tagged with the inferred operation type:

        CHANGED:  Alice --lives_in--> Australia  ==>  Alice --married_to--> Australia  [change_relation]
    """
    print("\n=== Detailed view ===")
    for sample_index in range(min(num_samples_to_print, len(corrupted_kgs))):
        print(f"\n--- Sample {sample_index} ---")
        corrupted_kg = corrupted_kgs[sample_index]
        anything_changed_in_this_sample = False
        for clean_triple, corrupted_triple in zip(clean_kg, corrupted_kg):
            if clean_triple == corrupted_triple:
                continue
            anything_changed_in_this_sample = True
            clean_head_index, clean_relation_index, clean_tail_index = clean_triple
            new_head_index,   new_relation_index,   new_tail_index   = corrupted_triple

            clean_string_form = (
                f"{node_labels[clean_head_index]} "
                f"--{relation_names[clean_relation_index]}--> "
                f"{node_labels[clean_tail_index]}"
            )
            corrupted_string_form = (
                f"{node_labels[new_head_index]} "
                f"--{relation_names[new_relation_index]}--> "
                f"{node_labels[new_tail_index]}"
            )

            # Tag with the inferred operation
            if (clean_head_index == new_head_index
                    and clean_tail_index == new_tail_index
                    and clean_relation_index != new_relation_index):
                operation_tag = "[change_relation]"
            elif (clean_relation_index == new_relation_index
                    and clean_tail_index == new_tail_index
                    and clean_head_index != new_head_index):
                operation_tag = "[swap_subject]"
            elif (clean_head_index == new_head_index
                    and clean_relation_index == new_relation_index
                    and clean_tail_index != new_tail_index):
                operation_tag = "[swap_object]"
            else:
                operation_tag = "[multi_op]"
            print(f"  CHANGED:  {clean_string_form}  ==>  {corrupted_string_form}  "
                  f"{operation_tag}")
        if not anything_changed_in_this_sample:
            print("  (no triples changed - identity sample)")
