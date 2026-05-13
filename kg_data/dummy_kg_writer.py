"""(Re)generate the dummy KG TSV files at datasets/dummy_kg/.

Run this module (or call write_dummy_kg()) when you want to recreate the
dummy KG files from this single source of truth. The committed TSV files
already exist; this module is here so the dummy data is regeneratable.

The dummy KG design:
    6 persons   - Alice, Bob, Carol, Dave, Eve, Frank
    4 countries - Australia, Japan, Brazil, France
    3 relations - born_in, married_to, lives_in
    18 directed triples in total

All entity ids and relation ids carry a "/dummy/" prefix to keep them visually
distinct from FB15k-237's "/m/..." ids, while still being valid Freebase-style
strings.
"""
from pathlib import Path
from typing import Iterable


# ── Single source of truth for the dummy KG ──────────────────────────────

# (entity_id, display_name, entity_type)
_DUMMY_ENTITIES = [
    ("/dummy/Alice",     "Alice",     "Person"),
    ("/dummy/Bob",       "Bob",       "Person"),
    ("/dummy/Carol",     "Carol",     "Person"),
    ("/dummy/Dave",      "Dave",      "Person"),
    ("/dummy/Eve",       "Eve",       "Person"),
    ("/dummy/Frank",     "Frank",     "Person"),
    ("/dummy/Australia", "Australia", "Country"),
    ("/dummy/Japan",     "Japan",     "Country"),
    ("/dummy/Brazil",    "Brazil",    "Country"),
    ("/dummy/France",    "France",    "Country"),
]

# (relation_id, display_name)
_DUMMY_RELATIONS = [
    ("/dummy/born_in",    "born_in"),
    ("/dummy/married_to", "married_to"),
    ("/dummy/lives_in",   "lives_in"),
]

# Triples, written in (display_name, relation_display_name, display_name) form
# for readability; the writer translates them to the /dummy/<id> form below.
_DUMMY_TRIPLES = [
    # born_in
    ("Alice", "born_in", "Australia"),
    ("Bob",   "born_in", "Australia"),
    ("Carol", "born_in", "Japan"),
    ("Dave",  "born_in", "Brazil"),
    ("Eve",   "born_in", "France"),
    ("Frank", "born_in", "Australia"),
    # married_to  (symmetric: every marriage shows up in both directions)
    ("Alice", "married_to", "Bob"),
    ("Bob",   "married_to", "Alice"),
    ("Carol", "married_to", "Dave"),
    ("Dave",  "married_to", "Carol"),
    ("Eve",   "married_to", "Frank"),
    ("Frank", "married_to", "Eve"),
    # lives_in  (some people moved since birth)
    ("Alice", "lives_in", "Australia"),
    ("Bob",   "lives_in", "Australia"),
    ("Carol", "lives_in", "Australia"),   # moved from Japan
    ("Dave",  "lives_in", "Japan"),       # moved from Brazil
    ("Eve",   "lives_in", "Brazil"),      # moved from France
    ("Frank", "lives_in", "France"),      # moved from Australia
]


def _write_tsv(path: Path, rows: Iterable[Iterable[str]]) -> None:
    """Write `rows` (iterable of iterables) as a tab-separated text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write("\t".join(row) + "\n")


def write_dummy_kg(output_directory: str | Path = "datasets/dummy_kg") -> Path:
    """Write the dummy KG TSV files into `output_directory`.

    Creates the directory if it doesn't exist. Files written:
      train.txt              triples (18 rows)
      valid.txt              empty (placeholder for FB15k-237 compatibility)
      test.txt               empty
      entity_metadata.txt    entity_id, display_name, type
      relation_metadata.txt  relation_id, display_name
    """
    output_directory = Path(output_directory)

    # Translation tables: display_name -> /dummy/<id> form
    display_name_to_entity_id = {
        display_name: entity_id for entity_id, display_name, _ in _DUMMY_ENTITIES
    }
    display_name_to_relation_id = {
        display_name: relation_id for relation_id, display_name in _DUMMY_RELATIONS
    }

    # Translate the triples from display-name form to /dummy/<id> form
    triple_rows_in_id_form = [
        (
            display_name_to_entity_id[head_display_name],
            display_name_to_relation_id[relation_display_name],
            display_name_to_entity_id[tail_display_name],
        )
        for head_display_name, relation_display_name, tail_display_name in _DUMMY_TRIPLES
    ]
    _write_tsv(output_directory / "train.txt", triple_rows_in_id_form)

    # Empty placeholders (FB15k-237 has three splits; we mirror that layout)
    (output_directory / "valid.txt").write_text("", encoding="utf-8")
    (output_directory / "test.txt").write_text("",  encoding="utf-8")

    _write_tsv(output_directory / "entity_metadata.txt",   _DUMMY_ENTITIES)
    _write_tsv(output_directory / "relation_metadata.txt", _DUMMY_RELATIONS)

    return output_directory


if __name__ == "__main__":
    written_path = write_dummy_kg()
    print(f"Wrote dummy KG to {written_path.resolve()}")
