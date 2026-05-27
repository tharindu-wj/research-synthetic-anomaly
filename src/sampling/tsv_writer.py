"""Round-trip integer-indexed triples back to string IDs and emit the TSV.

The downstream ADKGD repo consumes the file at this exact format:
  * UTF-8
  * LF (`\n`) line endings - NOT CRLF (we're on Windows so explicit
    `newline="\n"` is required to suppress Python's universal-newline
    translation)
  * No header row
  * Three tab-separated columns: `head_id \t relation_id \t tail_id`
  * No trailing whitespace on any line

`ADKGD.Reader.load_gan_negatives()` maps each string through its own
`ent2id` / `rel2id` (built from the same `train.txt`/`valid.txt`/`test.txt`
files), so the strings we emit MUST be the original string IDs - not
display names, not pretty-printed labels. We invert
`kg.entity_id_to_row` and `kg.relation_id_to_channel` to recover them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

from kg_data.loader import KnowledgeGraph


Triple = Tuple[int, int, int]


def _invert(row_to_id_map: Dict[str, int]) -> Dict[int, str]:
    return {row_index: identifier for identifier, row_index in row_to_id_map.items()}


def write_triples_tsv(
    output_path: str | Path,
    triples: Iterable[Triple],
    kg: KnowledgeGraph,
) -> int:
    """Write `triples` (integer-indexed) to `output_path` as ADKGD-format TSV.

    Returns the number of lines written. Raises `KeyError` if a triple
    references an index outside the KG's vocabulary - which would indicate
    an upstream bug, not a recoverable condition.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_to_entity_id    = _invert(kg.entity_id_to_row)
    channel_to_relation = _invert(kg.relation_id_to_channel)

    lines_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for head_index, relation_index, tail_index in triples:
            head_string     = row_to_entity_id[head_index]
            relation_string = channel_to_relation[relation_index]
            tail_string     = row_to_entity_id[tail_index]
            output_file.write(f"{head_string}\t{relation_string}\t{tail_string}\n")
            lines_written += 1
    return lines_written
