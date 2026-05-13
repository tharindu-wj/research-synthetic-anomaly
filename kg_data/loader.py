"""Unified KG loader: works for the dummy KG and FB15k-237 (same TSV layout).

Expected directory layout::

    <dataset_dir>/
        train.txt              REQUIRED - tab-separated head, relation, tail
        valid.txt              OPTIONAL (may be empty)
        test.txt               OPTIONAL (may be empty)
        entity_metadata.txt    OPTIONAL - entity_id <TAB> display_name <TAB> type
        relation_metadata.txt  OPTIONAL - relation_id <TAB> display_name

Missing metadata files are handled gracefully:
- display_name defaults to the entity_id / relation_id itself
- type defaults to 'unknown'

The loader is triple-level - no adjacency tensor is built. Callers that need
adjacency can call `triples_to_adjacency_tensor(kg.triples_idx, ...)` on
demand (e.g. for visualisation).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


Triple = Tuple[int, int, int]


@dataclass
class KnowledgeGraph:
    """Everything a downstream consumer needs about a loaded KG (triple-level)."""
    # Raw string triples - original form from the TSV
    triples: List[Tuple[str, str, str]]
    valid_triples: List[Tuple[str, str, str]]
    test_triples: List[Tuple[str, str, str]]

    # Index triples - the form downstream code (TRIC, dataset builder, model)
    # consumes. (h_idx, r_idx, t_idx) using the vocabularies below.
    triples_idx: List[Triple]
    triple_set_idx: Set[Triple]              # O(1) "edge exists" check

    entity_id_to_row: Dict[str, int]
    relation_id_to_channel: Dict[str, int]

    node_labels: Dict[int, str]              # row_idx -> display name
    node_types: Dict[int, str]               # row_idx -> entity type
    relation_names: List[str]                # channel_idx -> display name
    dataset_name: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.entity_id_to_row)

    @property
    def num_relations(self) -> int:
        return len(self.relation_id_to_channel)


def _read_triples_tsv(path: Path) -> List[Tuple[str, str, str]]:
    """Read tab-separated (h, r, t) string triples. Returns [] if file missing or empty."""
    if not path.exists():
        return []
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"{path}: expected 3 tab-separated fields, "
                    f"got {len(parts)} in line: {line!r}"
                )
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def _read_metadata(path: Path, expected_cols: int) -> List[List[str]]:
    """Read a tab-separated metadata file. Returns [] if missing."""
    if not path.exists():
        return []
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            while len(parts) < expected_cols:
                parts.append("")
            rows.append(parts[:expected_cols])
    return rows


def load_kg(dataset_dir: str | Path) -> KnowledgeGraph:
    """Load a KG from a directory of TSV files (FB15k-237-compatible layout)."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    train_triples = _read_triples_tsv(dataset_dir / "train.txt")
    valid_triples = _read_triples_tsv(dataset_dir / "valid.txt")
    test_triples  = _read_triples_tsv(dataset_dir / "test.txt")
    if not train_triples:
        raise ValueError(f"No training triples found at {dataset_dir / 'train.txt'}")

    # Build entity / relation vocabularies from the training split.
    entity_ids_in_order: List[str] = []
    relation_ids_in_order: List[str] = []
    seen_e, seen_r = set(), set()
    for h, r, t in train_triples:
        for e in (h, t):
            if e not in seen_e:
                seen_e.add(e)
                entity_ids_in_order.append(e)
        if r not in seen_r:
            seen_r.add(r)
            relation_ids_in_order.append(r)

    entity_id_to_row       = {eid: i for i, eid in enumerate(entity_ids_in_order)}
    relation_id_to_channel = {rid: i for i, rid in enumerate(relation_ids_in_order)}

    # Optional metadata
    entity_meta_rows = _read_metadata(dataset_dir / "entity_metadata.txt", expected_cols=3)
    eid_to_display: Dict[str, str] = {}
    eid_to_type:    Dict[str, str] = {}
    for eid, disp, etype in entity_meta_rows:
        eid_to_display[eid] = disp or eid
        eid_to_type[eid]    = etype or "unknown"

    rel_meta_rows = _read_metadata(dataset_dir / "relation_metadata.txt", expected_cols=2)
    rid_to_display: Dict[str, str] = {}
    for rid, disp in rel_meta_rows:
        rid_to_display[rid] = disp or rid

    node_labels    = {row: eid_to_display.get(eid, eid)        for eid, row in entity_id_to_row.items()}
    node_types     = {row: eid_to_type.get(eid, "unknown")     for eid, row in entity_id_to_row.items()}
    relation_names = [rid_to_display.get(rid, rid) for rid in relation_ids_in_order]

    # Vocab-mapped triple form (what downstream code uses)
    triples_idx: List[Triple] = [
        (entity_id_to_row[h], relation_id_to_channel[r], entity_id_to_row[t])
        for h, r, t in train_triples
    ]
    triple_set_idx: Set[Triple] = set(triples_idx)

    return KnowledgeGraph(
        triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        triples_idx=triples_idx,
        triple_set_idx=triple_set_idx,
        entity_id_to_row=entity_id_to_row,
        relation_id_to_channel=relation_id_to_channel,
        node_labels=node_labels,
        node_types=node_types,
        relation_names=relation_names,
        dataset_name=dataset_dir.name,
    )
