"""Knowledge graph visualization utilities.

Two public functions are provided:

    plot_dataset_sample   — 2×2 figure: clean + corrupted graphs and heatmaps
    plot_gan_comparison   — 2×3 figure: clean / rule-corrupted / GAN-corrupted

Both functions accept adjacency matrices in (N, N, R) numpy format (values in
[0, 1]) and plain Python dicts for labels and types, so all tensor/permutation
handling stays in the notebook.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bipartite_pos(node_types: dict) -> dict:
    """Build a fixed bipartite layout: Persons on the left, Countries on the right.

    Parameters
    ----------
    node_types : dict[int, str]  row_index → entity type string

    Returns
    -------
    dict[int, tuple[float, float]]  node → (x, y) position
    """
    person_rows  = [n for n, t in node_types.items() if t == 'Person']
    country_rows = [n for n, t in node_types.items() if t == 'Country']

    pos = {}
    for idx, n in enumerate(person_rows):
        pos[n] = (-1.0, 1.0 - idx * (2.0 / max(len(person_rows) - 1, 1)))
    for idx, n in enumerate(country_rows):
        pos[n] = (1.0, 1.0 - idx * (2.0 / max(len(country_rows) - 1, 1)))
    return pos


def _build_edge_lists(adj_nxnxr: np.ndarray) -> dict:
    """Return per-channel edge lists from an (N, N, R) adjacency array."""
    N, _, R = adj_nxnxr.shape
    edges = {ch: [] for ch in range(R)}
    for ch in range(R):
        for i in range(N):
            for j in range(N):
                if adj_nxnxr[i, j, ch] > 0.5:
                    edges[ch].append((i, j))
    return edges


def _make_colour_maps(node_types: dict, num_relations: int):
    """Return (type_to_colour, node_colour_list, relation_colours).

    Colours are deterministic — same types and channel counts always produce
    the same palette.
    """
    visible_types   = sorted(set(node_types.values()))
    type_cmap       = plt.cm.get_cmap('Set2', max(len(visible_types), 3))
    type_to_colour  = {t: type_cmap(i) for i, t in enumerate(visible_types)}
    node_colour_list = [type_to_colour[node_types[n]]
                        for n in range(len(node_types))]

    edge_cmap         = plt.cm.get_cmap('tab20', num_relations)
    relation_colours  = {ch: edge_cmap(ch) for ch in range(num_relations)}

    return type_to_colour, node_colour_list, relation_colours


def _draw_graph_ax(ax, adj_nxnxr, title, pos, node_colour_list,
                   node_labels, relation_colours):
    """Draw one directed multi-relational graph onto an existing Axes."""
    N = adj_nxnxr.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    edge_lists = _build_edge_lists(adj_nxnxr)
    for edges in edge_lists.values():
        G.add_edges_from(edges)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900,
                           node_color=node_colour_list, edgecolors='#2C3E50',
                           linewidths=1.5, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=7, font_color='black', font_weight='bold')

    num_relations = adj_nxnxr.shape[2]
    for ch in range(num_relations):
        if not edge_lists[ch]:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=edge_lists[ch], ax=ax,
                               edge_color=[relation_colours[ch]],
                               width=2.5, alpha=0.85,
                               arrows=True, arrowsize=18,
                               connectionstyle=f'arc3,rad={0.1 + ch * 0.06}',
                               min_source_margin=18, min_target_margin=18)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')


def _draw_heatmap_ax(ax, mat, tick_labels, title, vmax, cmap='Blues'):
    """Draw a summed-relation heatmap onto an existing Axes."""
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=9, fontweight='bold')
    n = mat.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _legend_handles(type_to_colour, relation_colours, relation_names):
    """Build a combined entity-type + relation legend handle list."""
    type_patches = [mpatches.Patch(color=c, label=t)
                    for t, c in type_to_colour.items()]
    rel_patches  = [mpatches.Patch(color=relation_colours[ch], label=relation_names[ch])
                    for ch in range(len(relation_names))]
    return (
        [mpatches.Patch(color='none', label='── Entity Types ──')]
        + type_patches
        + [mpatches.Patch(color='none', label='')]
        + [mpatches.Patch(color='none', label='── Relations ──')]
        + rel_patches
    )


# ── Public API ────────────────────────────────────────────────────────────────

def plot_dataset_sample(
    clean_nxnxr: np.ndarray,
    corr_nxnxr: np.ndarray,
    node_labels: dict,
    node_types: dict,
    relation_names: list,
    num_corruptions: int = 2,
    figsize: tuple = (14, 10),
):
    """2×2 figure: clean and corrupted graph views + adjacency heatmaps.

    Parameters
    ----------
    clean_nxnxr     : np.ndarray (N, N, R), values in [0, 1]
    corr_nxnxr      : np.ndarray (N, N, R), values in [0, 1]
    node_labels     : dict[int, str]  row_index → display name
                      Pass permutation-adjusted labels when showing dataset samples.
    node_types      : dict[int, str]  row_index → entity type
                      Pass permutation-adjusted types when showing dataset samples.
    relation_names  : list[str]  one name per channel, e.g. ['born_in', 'married_to', ...]
    num_corruptions : int  shown in the corrupted graph title
    figsize         : tuple  passed to plt.subplots
    """
    N = clean_nxnxr.shape[0]
    R = clean_nxnxr.shape[2]

    type_to_colour, node_colour_list, relation_colours = _make_colour_maps(node_types, R)
    pos         = _bipartite_pos(node_types)
    tick_labels = [node_labels[i] for i in range(N)]
    vmax        = max(clean_nxnxr.sum(axis=-1).max(), 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Row 0: graphs
    _draw_graph_ax(axes[0, 0], clean_nxnxr,
                   'Clean graph', pos, node_colour_list, node_labels, relation_colours)
    _draw_graph_ax(axes[0, 1], corr_nxnxr,
                   f'Corrupted graph ({num_corruptions} ops)',
                   pos, node_colour_list, node_labels, relation_colours)

    axes[0, 1].legend(
        handles=_legend_handles(type_to_colour, relation_colours, relation_names),
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95, title='Legend', title_fontsize=9, borderaxespad=0,
    )

    # Row 1: heatmaps
    _draw_heatmap_ax(axes[1, 0], clean_nxnxr.sum(axis=-1),
                     tick_labels, 'Clean matrix (Σ relations)', vmax)
    _draw_heatmap_ax(axes[1, 1], corr_nxnxr.sum(axis=-1),
                     tick_labels, 'Corrupted matrix (Σ relations)', vmax)

    plt.tight_layout()
    plt.show()


def plot_gan_comparison(
    clean_nxnxr: np.ndarray,
    rule_nxnxr: np.ndarray,
    gan_nxnxr: np.ndarray,
    node_labels: dict,
    node_types: dict,
    relation_names: list,
    num_corruptions: int = 2,
    figsize: tuple = (18, 11),
):
    """2×3 figure: clean / rule-corrupted / GAN-corrupted graphs and heatmaps.

    Parameters
    ----------
    clean_nxnxr     : np.ndarray (N, N, R), values in [0, 1]
    rule_nxnxr      : np.ndarray (N, N, R), values in [0, 1]  rule-based corruption
    gan_nxnxr       : np.ndarray (N, N, R), values in {0, 1}  GAN output after thresholding
    node_labels     : dict[int, str]  row_index → display name
    node_types      : dict[int, str]  row_index → entity type
    relation_names  : list[str]  one name per channel
    num_corruptions : int  shown in rule-corrupted graph title
    figsize         : tuple  passed to plt.subplots
    """
    N = clean_nxnxr.shape[0]
    R = clean_nxnxr.shape[2]

    type_to_colour, node_colour_list, relation_colours = _make_colour_maps(node_types, R)
    pos         = _bipartite_pos(node_types)
    tick_labels = [node_labels[i] for i in range(N)]
    vmax        = max(clean_nxnxr.sum(axis=-1).max(), 1)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 0: graphs
    _draw_graph_ax(axes[0, 0], clean_nxnxr,
                   'Clean graph', pos, node_colour_list, node_labels, relation_colours)
    _draw_graph_ax(axes[0, 1], rule_nxnxr,
                   f'TRIC-corrupted (ref, {num_corruptions} ops)',
                   pos, node_colour_list, node_labels, relation_colours)
    _draw_graph_ax(axes[0, 2], gan_nxnxr,
                   'GAN-corrupted graph', pos, node_colour_list, node_labels, relation_colours)

    axes[0, 2].legend(
        handles=_legend_handles(type_to_colour, relation_colours, relation_names),
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95, title='Legend', title_fontsize=9, borderaxespad=0,
    )

    # Row 1: heatmaps
    for ax, mat, title in zip(
        axes[1],
        [clean_nxnxr.sum(axis=-1), rule_nxnxr.sum(axis=-1), gan_nxnxr.sum(axis=-1)],
        ['Clean matrix (Σ relations)', 'TRIC-corrupted matrix', 'GAN-corrupted matrix'],
    ):
        _draw_heatmap_ax(ax, mat, tick_labels, title, vmax)
        ax.set_xlabel('Entity j', fontsize=8)
        ax.set_ylabel('Entity i', fontsize=8)

    plt.tight_layout()
    plt.show()
