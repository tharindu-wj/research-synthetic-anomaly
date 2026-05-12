"""Dummy knowledge graph for pipeline testing.

Provides a small, fully interpretable KG with 10 nodes so every edge
can be verified by eye. Returns the SAME data structures as the FB15k-237
pipeline so it plugs straight into the dataset builder and CNN.

Design:
    6 people  — Alice, Bob, Carol, Dave, Eve, Frank
    4 countries — Australia, Japan, Brazil, France
    3 relations — born_in, married_to, lives_in

Key design decisions:
    - Entity IDs start at 100 (not 0) to catch id-vs-row-index bugs.
    - Relation IDs start at 50 (not 0) to catch id-vs-channel-index bugs.
    - Return format matches build_kg_subgraph() exactly (tuple, not dict).
    - Separate function for name/label metadata.
"""
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────
# Offsets ensure entity_id ≠ row_index and relation_id ≠ channel_index,
# matching the real FB15k-237 pipeline where IDs are arbitrary integers.
_ENTITY_ID_OFFSET   = 100
_RELATION_ID_OFFSET = 50

_ENTITIES = [
    # (name, type, entity_id)
    ('Alice',     'Person',  _ENTITY_ID_OFFSET + 0),
    ('Bob',       'Person',  _ENTITY_ID_OFFSET + 1),
    ('Carol',     'Person',  _ENTITY_ID_OFFSET + 2),
    ('Dave',      'Person',  _ENTITY_ID_OFFSET + 3),
    ('Eve',       'Person',  _ENTITY_ID_OFFSET + 4),
    ('Frank',     'Person',  _ENTITY_ID_OFFSET + 5),
    ('Australia', 'Country', _ENTITY_ID_OFFSET + 6),
    ('Japan',     'Country', _ENTITY_ID_OFFSET + 7),
    ('Brazil',    'Country', _ENTITY_ID_OFFSET + 8),
    ('France',    'Country', _ENTITY_ID_OFFSET + 9),
]

_RELATIONS = [
    # (name, relation_id)
    ('born_in',     _RELATION_ID_OFFSET + 0),
    ('married_to',  _RELATION_ID_OFFSET + 1),
    ('lives_in',    _RELATION_ID_OFFSET + 2),
]

_TRIPLES = [
    # born_in  (person → country they were born in)
    ('Alice', 'born_in', 'Australia'),
    ('Bob',   'born_in', 'Australia'),
    ('Carol', 'born_in', 'Japan'),
    ('Dave',  'born_in', 'Brazil'),
    ('Eve',   'born_in', 'France'),
    ('Frank', 'born_in', 'Australia'),

    # married_to  (symmetric: both directions)
    ('Alice', 'married_to', 'Bob'),
    ('Bob',   'married_to', 'Alice'),
    ('Carol', 'married_to', 'Dave'),
    ('Dave',  'married_to', 'Carol'),
    ('Eve',   'married_to', 'Frank'),
    ('Frank', 'married_to', 'Eve'),

    # lives_in  (person → country they currently live in)
    ('Alice', 'lives_in', 'Australia'),
    ('Bob',   'lives_in', 'Australia'),
    ('Carol', 'lives_in', 'Australia'),    # moved from Japan
    ('Dave',  'lives_in', 'Japan'),        # moved from Brazil
    ('Eve',   'lives_in', 'Brazil'),       # moved from France
    ('Frank', 'lives_in', 'France'),       # moved from Australia
]


def build_dummy_kg():
    """Build a 10-node, 3-relation knowledge graph by hand.

    Returns the SAME tuple signature as build_kg_subgraph():
        (adjacency_tensor, entity_id_to_row, relation_id_to_channel)

    Entity IDs start at 100 and relation IDs at 50, so the id→index
    mappings are non-trivial (matching how FB15k-237 works).

    Returns
    -------
    adjacency_tensor      : np.ndarray, shape (10, 10, 3), float32
    entity_id_to_row      : dict[int, int] — e.g. {100: 0, 101: 1, ...}
    relation_id_to_channel: dict[int, int] — e.g. {50: 0, 51: 1, 52: 2}
    """
    N = len(_ENTITIES)
    R = len(_RELATIONS)

    # --- ID → index mappings (non-trivial, just like FB15k-237) ---
    entity_name_to_id  = {name: eid for name, _, eid in _ENTITIES}
    entity_id_to_row   = {eid: row for row, (_, _, eid) in enumerate(_ENTITIES)}

    relation_name_to_id    = {name: rid for name, rid in _RELATIONS}
    relation_id_to_channel = {rid: ch for ch, (_, rid) in enumerate(_RELATIONS)}

    # --- Build adjacency tensor ---
    adjacency_tensor = np.zeros((N, N, R), dtype=np.float32)

    for head_name, rel_name, tail_name in _TRIPLES:
        h_id = entity_name_to_id[head_name]
        t_id = entity_name_to_id[tail_name]
        r_id = relation_name_to_id[rel_name]

        row     = entity_id_to_row[h_id]
        col     = entity_id_to_row[t_id]
        channel = relation_id_to_channel[r_id]
        adjacency_tensor[row, col, channel] = 1.0

    total_edges = int(adjacency_tensor.sum())
    density     = total_edges / (N * (N - 1) * R)

    print(f'Dummy KG built: {N} entities, {R} relations, {total_edges} directed edges')
    print(f'Tensor shape: {adjacency_tensor.shape}  (N, N, R)')
    print(f'Density: {density:.4f}')
    print(f'Entity IDs: {sorted(entity_id_to_row.keys())}  (non-trivial, offset={_ENTITY_ID_OFFSET})')
    print(f'Relation IDs: {sorted(relation_id_to_channel.keys())}  (non-trivial, offset={_RELATION_ID_OFFSET})')

    return adjacency_tensor, entity_id_to_row, relation_id_to_channel


def get_dummy_kg_metadata():
    """Return the human-readable name and type metadata for the dummy KG.

    Separated from build_dummy_kg() to keep the return signature identical
    to build_kg_subgraph(). Call this for visualisation and diagnostics.

    Returns
    -------
    relation_id_to_name : dict[int, str] — e.g. {50: 'born_in', 51: 'married_to', ...}
    entity_id_to_name   : dict[int, str] — e.g. {100: 'Alice', 101: 'Bob', ...}
    node_labels         : dict[int, str] — row_index → display name
    node_types          : dict[int, str] — row_index → entity type ('Person', 'Country')
    """
    relation_id_to_name = {rid: name for name, rid in _RELATIONS}
    entity_id_to_name   = {eid: name for name, _, eid in _ENTITIES}
    node_labels         = {row: name for row, (name, _, _) in enumerate(_ENTITIES)}
    node_types          = {row: etype for row, (_, etype, _) in enumerate(_ENTITIES)}

    return relation_id_to_name, entity_id_to_name, node_labels, node_types
