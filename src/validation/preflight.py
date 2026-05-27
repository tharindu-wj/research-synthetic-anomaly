"""Pre-flight validation of a generated `gan_negatives.tsv` before shipping.

The downstream ADKGD repo will silently drop any triple in the TSV whose
strings aren't in its vocabulary, OR that already exists in the real graph.
A TSV with lots of drops wastes the negative pool and degrades the
experiment. This module catches the problem here rather than after SCP.

Five checks (mirroring ADKGD's loader behaviour exactly):

  1. **Parse**:  every line splits into exactly three tab-separated columns.
  2. **Volume**: line count >= `target_pool_size`.
  3. **Uniqueness**: `unique_lines / total_lines` >= `min_unique_ratio`
     (default 0.95) - guards against mode collapse.
  4. **Vocab coverage**: every entity and relation string appears in the
     KG's union vocabulary - exactly what ADKGD checks via its own
     `ent2id`/`rel2id` (which it builds from the same source files).
  5. **Real-graph collision**: no emitted triple sits in
     `kg.triple_set_idx` - ADKGD filters these defensively, but a high
     collision rate means a wasted pool.

The intent is a clean "Verdict: OK to use." for a healthy TSV; any
failure should be loud enough to abort the surrounding script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from kg_data.loader import KnowledgeGraph


@dataclass
class PreflightReport:
    """Result of `validate_tsv`. `is_ok` aggregates all checks."""
    tsv_path:               str
    total_lines:            int = 0
    parse_failures:         int = 0
    unique_lines:           int = 0
    uniqueness_ratio:       float = 0.0
    unknown_entity_lines:   int = 0
    unknown_relation_lines: int = 0
    real_graph_collisions:  int = 0

    # Thresholds the report was evaluated against (for transparency)
    target_pool_size:       int = 0
    min_unique_ratio:       float = 0.95
    max_unknown_vocab_rate: float = 0.05
    max_collision_rate:     float = 0.05

    sample_lines:           List[str] = field(default_factory=list)
    is_ok:                  bool = False
    failure_reasons:        List[str] = field(default_factory=list)

    def render(self) -> str:
        """Human-readable summary - matches the ADKGD-side validator's tone."""
        lines: List[str] = []
        lines.append(f"Pre-flight report for: {self.tsv_path}")
        lines.append(f"  Total lines parsed:         {self.total_lines:,}")
        lines.append(f"  Parse failures:             {self.parse_failures:,}")
        lines.append(
            f"  Unique lines:               {self.unique_lines:,}"
            f"  (ratio {self.uniqueness_ratio:.4f},"
            f" target >= {self.min_unique_ratio:.2f})"
        )
        lines.append(
            f"  Unknown-entity drops:       {self.unknown_entity_lines:,}"
            f"  (ADKGD would silently drop these)"
        )
        lines.append(
            f"  Unknown-relation drops:     {self.unknown_relation_lines:,}"
        )
        lines.append(
            f"  Real-graph collisions:      {self.real_graph_collisions:,}"
            f"  (ADKGD would also drop these defensively)"
        )
        lines.append(f"  Target pool size:           {self.target_pool_size:,}")
        if self.sample_lines:
            lines.append("  Sample (first 10 lines, eyeball type-coherence):")
            for sample_line in self.sample_lines:
                lines.append(f"    {sample_line}")
        if self.is_ok:
            lines.append("Verdict: OK to use.")
        else:
            lines.append("Verdict: NOT OK - " + "; ".join(self.failure_reasons))
        return "\n".join(lines)


def validate_tsv(
    tsv_path: str | Path,
    kg: KnowledgeGraph,
    *,
    target_pool_size: int = 400_000,
    min_unique_ratio: float = 0.95,
    max_unknown_vocab_rate: float = 0.05,
    max_collision_rate: float = 0.05,
    sample_size: int = 10,
) -> PreflightReport:
    """Run all five pre-flight checks against `tsv_path`."""
    tsv_path = Path(tsv_path)
    report = PreflightReport(
        tsv_path=str(tsv_path),
        target_pool_size=target_pool_size,
        min_unique_ratio=min_unique_ratio,
        max_unknown_vocab_rate=max_unknown_vocab_rate,
        max_collision_rate=max_collision_rate,
    )

    if not tsv_path.exists():
        report.failure_reasons.append(f"file does not exist: {tsv_path}")
        return report

    seen_lines: set = set()
    parsed_triples: List[Tuple[str, str, str]] = []
    with tsv_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            stripped_line = raw_line.rstrip("\n").rstrip("\r")
            if not stripped_line.strip():
                continue
            report.total_lines += 1
            if len(report.sample_lines) < sample_size:
                report.sample_lines.append(stripped_line)
            columns = stripped_line.split("\t")
            if len(columns) != 3:
                report.parse_failures += 1
                continue
            parsed_triples.append((columns[0], columns[1], columns[2]))
            seen_lines.add(stripped_line)

    report.unique_lines = len(seen_lines)
    if report.total_lines > 0:
        report.uniqueness_ratio = report.unique_lines / report.total_lines

    # Vocab and collision checks - O(n) over parsed triples.
    for head_string, relation_string, tail_string in parsed_triples:
        head_known     = head_string     in kg.entity_id_to_row
        tail_known     = tail_string     in kg.entity_id_to_row
        relation_known = relation_string in kg.relation_id_to_channel
        if not (head_known and tail_known):
            report.unknown_entity_lines += 1
            continue
        if not relation_known:
            report.unknown_relation_lines += 1
            continue
        # Both vocab checks passed - test real-graph collision.
        candidate_triple_idx = (
            kg.entity_id_to_row[head_string],
            kg.relation_id_to_channel[relation_string],
            kg.entity_id_to_row[tail_string],
        )
        if candidate_triple_idx in kg.triple_set_idx:
            report.real_graph_collisions += 1

    # Verdict
    if report.parse_failures > 0:
        report.failure_reasons.append(f"{report.parse_failures} lines failed to parse")
    if report.total_lines < target_pool_size:
        report.failure_reasons.append(
            f"only {report.total_lines:,} lines (target {target_pool_size:,})"
        )
    if report.uniqueness_ratio < min_unique_ratio:
        report.failure_reasons.append(
            f"uniqueness {report.uniqueness_ratio:.3f} "
            f"below threshold {min_unique_ratio:.2f}"
        )
    if report.total_lines > 0:
        unknown_vocab_rate = (
            (report.unknown_entity_lines + report.unknown_relation_lines)
            / report.total_lines
        )
        collision_rate = report.real_graph_collisions / report.total_lines
        if unknown_vocab_rate > max_unknown_vocab_rate:
            report.failure_reasons.append(
                f"unknown-vocab rate {unknown_vocab_rate:.3f} "
                f"above threshold {max_unknown_vocab_rate:.2f}"
            )
        if collision_rate > max_collision_rate:
            report.failure_reasons.append(
                f"real-graph collision rate {collision_rate:.3f} "
                f"above threshold {max_collision_rate:.2f}"
            )
    report.is_ok = not report.failure_reasons
    return report
