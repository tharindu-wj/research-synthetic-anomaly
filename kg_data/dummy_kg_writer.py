"""Regenerate the dummy KG TSV files at datasets/dummy_kg/.

Run this script (or call write_dummy_kg()) to (re)create the dummy dataset
in FB15k-237-compatible TSV format. The committed files already exist; this
module is here so the dummy data is regeneratable from a single source of
truth.

Design:
    6 persons   - Alice, Bob, Carol, Dave, Eve, Frank
    4 countries - Australia, Japan, Brazil, France
    3 relations - born_in, married_to, lives_in
    18 directed triples
"""
from pathlib import Path
from typing import Iterable


# Single source of truth for the dummy KG.
_ENTITIES = [
    # (entity_id, display_name, type)
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

_RELATIONS = [
    # (relation_id, display_name)
    ("/dummy/born_in",     "born_in"),
    ("/dummy/married_to",  "married_to"),
    ("/dummy/lives_in",    "lives_in"),
]

_TRIPLES = [
    # born_in
    ("Alice", "born_in", "Australia"),
    ("Bob",   "born_in", "Australia"),
    ("Carol", "born_in", "Japan"),
    ("Dave",  "born_in", "Brazil"),
    ("Eve",   "born_in", "France"),
    ("Frank", "born_in", "Australia"),
    # married_to (symmetric)
    ("Alice", "married_to", "Bob"),
    ("Bob",   "married_to", "Alice"),
    ("Carol", "married_to", "Dave"),
    ("Dave",  "married_to", "Carol"),
    ("Eve",   "married_to", "Frank"),
    ("Frank", "married_to", "Eve"),
    # lives_in
    ("Alice", "lives_in", "Australia"),
    ("Bob",   "lives_in", "Australia"),
    ("Carol", "lives_in", "Australia"),    # moved from Japan
    ("Dave",  "lives_in", "Japan"),        # moved from Brazil
    ("Eve",   "lives_in", "Brazil"),       # moved from France
    ("Frank", "lives_in", "France"),       # moved from Australia
]


def _write_tsv(path: Path, rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")


def write_dummy_kg(out_dir: str | Path = "datasets/dummy_kg") -> Path:
    """Write the dummy KG TSV files to `out_dir`. Returns the path written to."""
    out_dir = Path(out_dir)

    name_to_eid = {disp: eid for eid, disp, _ in _ENTITIES}
    name_to_rid = {disp: rid for rid, disp in _RELATIONS}

    triple_rows = [
        (name_to_eid[h], name_to_rid[r], name_to_eid[t])
        for h, r, t in _TRIPLES
    ]
    _write_tsv(out_dir / "train.txt", triple_rows)

    # Empty split files (placeholders for FB15k-237 compatibility)
    (out_dir / "valid.txt").write_text("", encoding="utf-8")
    (out_dir / "test.txt").write_text("",  encoding="utf-8")

    _write_tsv(out_dir / "entity_metadata.txt",  _ENTITIES)
    _write_tsv(out_dir / "relation_metadata.txt", _RELATIONS)

    return out_dir


if __name__ == "__main__":
    path = write_dummy_kg()
    print(f"Wrote dummy KG to {path.resolve()}")
