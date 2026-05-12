"""Knowledge-graph subgraph extraction and adjacency-tensor construction.

Mirrors the role of `clean_adjacency_matrix = nx.to_numpy_array(source_graph)`
in the Facebook pipeline, but for multi-relational data.

The output is a (N, N, R) binary tensor where:
    A[i, j, r] = 1  iff  triple (entity_at_row_i, relation_at_channel_r, entity_at_row_j) exists.
This tensor is what the rest of the pipeline (dataset_builder, generator) consumes.

Two important differences vs the Facebook adjacency matrix:
    1. The tensor is NOT symmetric — KG edges are directional.
       (Alice, born_in, Australia) != (Australia, born_in, Alice)
    2. Multiple slices can have an edge at the same (i, j) — relations co-exist.
"""
import numpy as np


def find_well_connected_ego_entity(triples, relation_subset, num_nodes=10, min_neighbours=None):
    """Find an entity that has at least `min_neighbours` distinct 1-hop neighbours
    reachable via the curated relation subset.

    Strategy: count, for each entity, how many distinct neighbours it has via
    edges whose relation is in the subset. Pick the entity with the highest
    such count. This guarantees the resulting subgraph will actually have
    enough nodes — picking a famous entity by hand can fail if its edges all
    use rare relations.

    Parameters
    ----------
    triples         : np.ndarray, shape (num_triples, 3)
    relation_subset : array-like of int — the curated relation IDs
    num_nodes       : int — target subgraph size N
    min_neighbours  : int or None — if None, requires `num_nodes - 1` neighbours

    Returns
    -------
    ego_entity_id : int
    """
    if min_neighbours is None:
        min_neighbours = num_nodes - 1   # ego + N-1 neighbours = N total

    relation_subset_set = set(int(r) for r in relation_subset)

    # Build a per-entity neighbour set, restricted to in-vocab relations
    entity_to_neighbours = {}
    for head, relation, tail in triples:
        if int(relation) not in relation_subset_set:
            continue
        h, t = int(head), int(tail)
        # Both endpoints can act as "ego" — track neighbours in both directions
        entity_to_neighbours.setdefault(h, set()).add(t)
        entity_to_neighbours.setdefault(t, set()).add(h)

    # Pick the entity with the most neighbours (deterministic — tie-break on id)
    best_entity, best_count = None, -1
    for entity, neighbours in entity_to_neighbours.items():
        if len(neighbours) > best_count:
            best_entity, best_count = entity, len(neighbours)

    if best_count < min_neighbours:
        raise ValueError(
            f'No entity has {min_neighbours}+ in-vocab neighbours. '
            f'Best candidate has {best_count}. '
            f'Try a larger relation_subset or smaller num_nodes.'
        )
    return best_entity


def build_kg_subgraph(triples, ego_entity_id, relation_subset, num_nodes=10):
    """Extract a 1-hop subgraph around `ego_entity_id` and build the (N, N, R) tensor.

    Steps:
      1. Find all 1-hop neighbours of the ego, reachable via the relation subset.
      2. Take the first (num_nodes - 1) neighbours by entity-id ordering — same
         "first N" sampling rule the Facebook plan uses, deterministic for repro.
      3. Keep all triples (h, r, t) where h, t, r are all in the subgraph.
      4. Build the (N, N, R) adjacency tensor.

    Parameters
    ----------
    triples         : np.ndarray, shape (num_triples, 3)
    ego_entity_id   : int
    relation_subset : array-like of int — must match the curated vocabulary
    num_nodes       : int — total nodes including the ego

    Returns
    -------
    adjacency_tensor      : np.ndarray, shape (num_nodes, num_nodes, R), float32
        adjacency_tensor[i, j, r] = 1 iff (entities[i], relations[r], entities[j]) is a triple.
    entity_id_to_row      : dict[int, int] — entity_id → row/col index 0..N-1
    relation_id_to_channel: dict[int, int] — relation_id → channel index 0..R-1
    """
    relation_subset_array = np.asarray(relation_subset, dtype=np.int64)
    relation_subset_set   = set(int(r) for r in relation_subset_array)

    # --- Step 1: collect 1-hop neighbours via in-vocab edges ---
    ego_int = int(ego_entity_id)
    neighbours = set()
    for head, relation, tail in triples:
        if int(relation) not in relation_subset_set:
            continue
        h, t = int(head), int(tail)
        if h == ego_int and t != ego_int:
            neighbours.add(t)
        elif t == ego_int and h != ego_int:
            neighbours.add(h)

    # --- Step 2: take the first (N-1) neighbours, deterministic ordering ---
    sorted_neighbours      = sorted(neighbours)
    selected_neighbours    = sorted_neighbours[: num_nodes - 1]
    if len(selected_neighbours) < num_nodes - 1:
        raise ValueError(
            f'Ego {ego_int} has only {len(selected_neighbours)} in-vocab neighbours, '
            f'need {num_nodes - 1}.'
        )

    selected_entity_ids = [ego_int] + selected_neighbours      # ego at row 0
    entity_id_to_row    = {eid: row for row, eid in enumerate(selected_entity_ids)}
    selected_entity_set = set(selected_entity_ids)

    # Channel mapping — preserve the curated vocabulary order
    relation_id_to_channel = {int(r): c for c, r in enumerate(relation_subset_array)}

    # --- Step 3: filter triples to subgraph membership ---
    num_relations    = len(relation_subset_array)
    adjacency_tensor = np.zeros((num_nodes, num_nodes, num_relations), dtype=np.float32)

    edges_added = 0
    for head, relation, tail in triples:
        h, r, t = int(head), int(relation), int(tail)
        if h not in selected_entity_set or t not in selected_entity_set:
            continue
        if r not in relation_subset_set:
            continue
        row     = entity_id_to_row[h]
        col     = entity_id_to_row[t]
        channel = relation_id_to_channel[r]
        adjacency_tensor[row, col, channel] = 1.0
        edges_added += 1

    return adjacency_tensor, entity_id_to_row, relation_id_to_channel


def summarise_kg_subgraph(adjacency_tensor, relation_id_to_channel, relation_id_to_name):
    """Diagnostic print analogous to summarise_source_graph.

    Reports: shape, total edges, density (per-channel and overall), and which
    relations actually appear in the subgraph (many curated relations will have
    zero edges in any single subgraph — that's expected).
    """
    num_nodes      = adjacency_tensor.shape[0]
    num_relations  = adjacency_tensor.shape[2]
    total_edges    = int(adjacency_tensor.sum())
    max_possible   = num_nodes * (num_nodes - 1) * num_relations   # directed, no self-loops
    overall_density = total_edges / max_possible if max_possible else 0.0

    print(f'Subgraph tensor shape: {adjacency_tensor.shape}  (N, N, R)')
    print(f'Total directed edges:  {total_edges}')
    print(f'Overall density:       {overall_density:.4f}  (vs Facebook ~0.5)')
    print(f'Edges per channel:')

    # Reverse the channel mapping so we can print rel_id beside channel
    channel_to_relation_id = {c: r for r, c in relation_id_to_channel.items()}
    edges_per_channel = adjacency_tensor.sum(axis=(0, 1)).astype(int)
    for channel_index, edge_count in enumerate(edges_per_channel):
        rel_id = channel_to_relation_id[channel_index]
        name   = relation_id_to_name.get(rel_id, f'relation_{rel_id}')
        marker = '   ' if edge_count > 0 else ' . '   # dot = empty channel
        print(f'  {marker} channel {channel_index:>2d}  rel_id {rel_id:>4d}  '
              f'{int(edge_count):>3d} edges  {name[:50]}')


def _extract_entity_type(freebase_path, role='head'):
    """Extract an entity type label from a Freebase relation path.

    Freebase paths follow the pattern '/domain/type/property'.
    Compound (CVT) paths look like '/d1/t1/p1./d2/t2/p2'.

    For HEAD entities:  returns the type (2nd segment of the first component).
        '/people/person/nationality'  →  'Person'
        '/film/actor/film./film/performance/film'  →  'Actor'

    For TAIL entities:  returns the property (last segment of the last component),
    which describes what kind of thing the tail is.
        '/people/person/nationality'  →  'Nationality'
        '/film/actor/film./film/performance/film'  →  'Film'

    Parameters
    ----------
    freebase_path : str
    role          : 'head' or 'tail'

    Returns
    -------
    str or None — the inferred type, title-cased with underscores as spaces
    """
    if not freebase_path.startswith('/'):
        return None

    segments = freebase_path.split('.')

    if role == 'head':
        # First component's type: '/domain/TYPE/property' → TYPE
        parts = [p for p in segments[0].split('/') if p]
        if len(parts) >= 2:
            return parts[1].replace('_', ' ').title()

    elif role == 'tail':
        # Last component's property: '/domain/type/PROPERTY' → PROPERTY
        parts = [p for p in segments[-1].split('/') if p]
        if parts:
            return parts[-1].replace('_', ' ').title()

    return None


def infer_entity_type_labels(
    adjacency_tensor,
    entity_id_to_row,
    relation_id_to_channel,
    relation_id_to_name,
):
    """Infer human-readable type labels for entities from their Freebase relations.

    For each entity in the subgraph, examines the relations it participates in:
    - As HEAD: extracts the entity type from the relation path
      (e.g., head of '/people/person/nationality' → 'Person')
    - As TAIL: extracts the property name from the relation path
      (e.g., tail of '/people/person/nationality' → 'Nationality')

    If an entity has multiple HEAD-inferred types, picks the most frequent one.
    Falls back to TAIL-inferred types if no HEAD edges exist.

    Labels are made unique with counters when multiple entities share a type:
    'Person' if unique, 'Person 1', 'Person 2', etc. if multiple.

    Parameters
    ----------
    adjacency_tensor      : np.ndarray, shape (N, N, R)
    entity_id_to_row      : dict[int, int]
    relation_id_to_channel: dict[int, int]
    relation_id_to_name   : dict[int, str]

    Returns
    -------
    node_labels : dict[int, str] — row_index → human-readable label
    node_types  : dict[int, str] — row_index → raw type (without counter)
    """
    num_nodes     = adjacency_tensor.shape[0]
    num_relations = adjacency_tensor.shape[2]

    channel_to_relation_id = {ch: rid for rid, ch in relation_id_to_channel.items()}

    # --- Infer a raw type for each node ---
    raw_types = {}
    for row_idx in range(num_nodes):

        # Collect types from outgoing edges (entity is HEAD)
        head_types = []
        for ch_idx in range(num_relations):
            if adjacency_tensor[row_idx, :, ch_idx].sum() > 0:
                rid = channel_to_relation_id[ch_idx]
                rel_name = relation_id_to_name.get(rid, '')
                t = _extract_entity_type(rel_name, role='head')
                if t:
                    head_types.append(t)

        # Collect types from incoming edges (entity is TAIL)
        tail_types = []
        for ch_idx in range(num_relations):
            if adjacency_tensor[:, row_idx, ch_idx].sum() > 0:
                rid = channel_to_relation_id[ch_idx]
                rel_name = relation_id_to_name.get(rid, '')
                t = _extract_entity_type(rel_name, role='tail')
                if t:
                    tail_types.append(t)

        # Prefer head types, fall back to tail types
        if head_types:
            raw_types[row_idx] = max(set(head_types), key=head_types.count)
        elif tail_types:
            raw_types[row_idx] = max(set(tail_types), key=tail_types.count)
        else:
            raw_types[row_idx] = 'Entity'

    # --- Count how many nodes share each type ---
    type_totals = {}
    for row_idx in range(num_nodes):
        t = raw_types[row_idx]
        type_totals[t] = type_totals.get(t, 0) + 1

    # --- Assign labels with counters for disambiguation ---
    type_counters = {}
    node_labels = {}
    node_types  = {}
    for row_idx in range(num_nodes):
        t = raw_types[row_idx]
        node_types[row_idx] = t
        if type_totals[t] == 1:
            node_labels[row_idx] = t
        else:
            type_counters[t] = type_counters.get(t, 0) + 1
            node_labels[row_idx] = f'{t} {type_counters[t]}'

    return node_labels, node_types


def visualise_kg_subgraph(
    adjacency_tensor,
    entity_id_to_row,
    relation_id_to_channel,
    relation_id_to_name,
    entity_id_to_name=None,
    node_labels=None,
    node_types=None,
    exclude_relations=None,
    figsize=(14, 10),
    seed=42,
):
    """Visualise the multi-relational KG subgraph with human-readable labels.

    Creates a single-panel directed graph with:
    - Nodes coloured by inferred entity type (from Freebase relation paths)
    - Edges coloured by relation type, with curvature to separate overlapping edges
    - A side legend showing both relation colours and entity type colours

    Parameters
    ----------
    adjacency_tensor      : np.ndarray, shape (N, N, R)
    entity_id_to_row      : dict[int, int] — entity_id → row/col index
    relation_id_to_channel: dict[int, int] — relation_id → channel index
    relation_id_to_name   : dict[int, str] — relation_id → human-readable name
    entity_id_to_name     : dict[int, str] or None — entity_id → Freebase MID / name
    node_labels           : dict[int, str] or None — row_index → display name.
                            If provided, bypasses automatic type inference.
    node_types            : dict[int, str] or None — row_index → entity type for colouring.
                            If provided, bypasses automatic type inference.
    exclude_relations     : list[str] or None — case-insensitive keywords; any relation
                            whose name contains one of these strings is hidden from the
                            graph. Nodes that become isolated are also hidden.
    figsize               : tuple — figure size
    seed                  : int — layout seed for reproducibility
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    try:
        from .kg_loader import shorten_freebase_path
    except ImportError:
        def shorten_freebase_path(path, max_parts=2):
            return path

    num_nodes     = adjacency_tensor.shape[0]
    num_relations = adjacency_tensor.shape[2]

    # --- Reverse mappings ---
    channel_to_relation_id = {ch: rid for rid, ch in relation_id_to_channel.items()}

    # --- Determine which channels to exclude ---
    exclude_relations = exclude_relations or []
    exclude_keywords = [kw.lower() for kw in exclude_relations]

    excluded_channels = set()
    for ch_idx in range(num_relations):
        rid = channel_to_relation_id[ch_idx]
        raw_name = relation_id_to_name.get(rid, f'relation_{rid}').lower()
        if any(kw in raw_name for kw in exclude_keywords):
            excluded_channels.add(ch_idx)

    # --- Build a filtered adjacency tensor for type inference ---
    filtered_tensor = adjacency_tensor.copy()
    for ch_idx in excluded_channels:
        filtered_tensor[:, :, ch_idx] = 0.0

    # --- Node labels + types: use overrides or infer from relations ---
    if node_labels is not None and node_types is not None:
        _node_labels = dict(node_labels)
        _node_types  = dict(node_types)
    else:
        _node_labels, _node_types = infer_entity_type_labels(
            filtered_tensor, entity_id_to_row, relation_id_to_channel, relation_id_to_name
        )

    # --- Build relation labels + colour map ---
    edge_cmap = plt.cm.get_cmap('tab20', num_relations)
    relation_colours = {}
    relation_labels  = {}
    for ch_idx in range(num_relations):
        rid = channel_to_relation_id[ch_idx]
        raw_name = relation_id_to_name.get(rid, f'relation_{rid}')
        relation_labels[ch_idx] = shorten_freebase_path(raw_name)
        relation_colours[ch_idx] = edge_cmap(ch_idx)

    # --- Build the edge lists (excluding filtered channels) ---
    edge_list_by_channel = {ch: [] for ch in range(num_relations)}
    for ch_idx in range(num_relations):
        if ch_idx in excluded_channels:
            continue
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_tensor[i, j, ch_idx] > 0:
                    edge_list_by_channel[ch_idx].append((i, j))

    # --- Find nodes that still have visible edges ---
    visible_nodes = set()
    for ch_idx, edges in edge_list_by_channel.items():
        for src, dst in edges:
            visible_nodes.add(src)
            visible_nodes.add(dst)
    visible_nodes = sorted(visible_nodes)

    if not visible_nodes:
        print('No visible nodes after filtering — nothing to draw.')
        return

    # --- Node type → colour mapping (only for visible nodes) ---
    visible_types = sorted(set(_node_types[n] for n in visible_nodes))
    node_type_cmap = plt.cm.get_cmap('Set2', max(len(visible_types), 3))
    type_to_colour = {t: node_type_cmap(i) for i, t in enumerate(visible_types)}
    node_colour_list = [type_to_colour[_node_types[n]] for n in visible_nodes]

    # --- Build the NetworkX DiGraph (visible nodes only) ---
    G = nx.DiGraph()
    for n in visible_nodes:
        G.add_node(n)
    for ch_idx, edges in edge_list_by_channel.items():
        for src, dst in edges:
            G.add_edge(src, dst, relation=ch_idx)

    # --- Layout ---
    pos = nx.spring_layout(G, seed=seed, k=2.5)

    # --- Figure ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw nodes coloured by type
    nx.draw_networkx_nodes(G, pos, nodelist=visible_nodes, ax=ax, node_size=900,
                           node_color=node_colour_list, edgecolors='#2C3E50',
                           linewidths=1.5, alpha=0.95)
    visible_labels = {n: _node_labels[n] for n in visible_nodes}
    nx.draw_networkx_labels(G, pos, labels=visible_labels, ax=ax,
                            font_size=7, font_color='black', font_weight='bold')

    # Draw edges coloured by relation type
    relation_legend_patches = []
    for ch_idx in range(num_relations):
        if ch_idx in excluded_channels:
            continue
        edges = edge_list_by_channel[ch_idx]
        if not edges:
            continue
        colour = relation_colours[ch_idx]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=[colour],
            width=2.5, alpha=0.85,
            arrows=True, arrowsize=18,
            connectionstyle=f'arc3,rad={0.1 + ch_idx * 0.06}',
            min_source_margin=18, min_target_margin=18,
        )
        relation_legend_patches.append(
            mpatches.Patch(color=colour, label=f'{relation_labels[ch_idx]}')
        )

    # --- Build combined legend ---
    type_legend_patches = [
        mpatches.Patch(color=type_to_colour[t], label=t) for t in visible_types
    ]

    spacer = mpatches.Patch(color='none', label='')
    all_patches = (
        [mpatches.Patch(color='none', label='── Entity Types ──')]
        + type_legend_patches
        + [spacer]
        + [mpatches.Patch(color='none', label='── Relations ──')]
        + relation_legend_patches
    )

    ax.legend(
        handles=all_patches,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        framealpha=0.95,
        title='Legend',
        title_fontsize=10,
        borderaxespad=0,
    )

    # --- Title ---
    title = 'KG Subgraph (directed, multi-relational)'
    if exclude_relations:
        title += f'\n(excluding: {", ".join(exclude_relations)})'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.show()
