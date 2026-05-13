"""Knowledge-graph visualisation helpers (multi-panel figures).

Two public functions:

    plot_dataset_sample   2x2 figure: clean + corrupted graphs, then their heatmaps
    plot_gan_comparison   2x3 figure: clean / rule-corrupted / GAN-corrupted

Both take adjacency tensors in `(num_nodes, num_nodes, num_relations)` numpy
form with values in `[0, 1]`, plus plain Python dicts for labels and types.
That keeps all tensor/permutation handling outside this file.

The internal helpers (prefixed with `_`) draw the bits and pieces of each
figure. Layout is "bipartite" by default - persons on the left, countries
on the right - which works well for the dummy KG.
"""
from typing import Dict, List, Sequence, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bipartite_layout(node_types: Dict[int, str]) -> Dict[int, Tuple[float, float]]:
    """Build a bipartite (two-column) layout: Persons on the left, Countries on the right.

    Returns a dict ``{row_index: (x, y)}``.
    """
    person_row_indices  = [row for row, t in node_types.items() if t == "Person"]
    country_row_indices = [row for row, t in node_types.items() if t == "Country"]

    layout_positions: Dict[int, Tuple[float, float]] = {}
    for stacked_index, row_index in enumerate(person_row_indices):
        layout_positions[row_index] = (
            -1.0,
            1.0 - stacked_index * (2.0 / max(len(person_row_indices) - 1, 1)),
        )
    for stacked_index, row_index in enumerate(country_row_indices):
        layout_positions[row_index] = (
            1.0,
            1.0 - stacked_index * (2.0 / max(len(country_row_indices) - 1, 1)),
        )
    return layout_positions


def _build_edge_lists_per_channel(adjacency_tensor: np.ndarray) -> Dict[int, List[tuple]]:
    """Return ``{channel_index: [(src, dst), ...]}`` from an (N, N, R) array."""
    num_nodes     = adjacency_tensor.shape[0]
    num_relations = adjacency_tensor.shape[2]
    edges_per_channel: Dict[int, List[tuple]] = {
        channel_index: [] for channel_index in range(num_relations)
    }
    for channel_index in range(num_relations):
        for source_row in range(num_nodes):
            for target_row in range(num_nodes):
                if adjacency_tensor[source_row, target_row, channel_index] > 0.5:
                    edges_per_channel[channel_index].append((source_row, target_row))
    return edges_per_channel


def _make_colour_maps(
    node_types: Dict[int, str], num_relations: int,
) -> Tuple[Dict[str, tuple], List[tuple], Dict[int, tuple]]:
    """Build colour palettes for node types and relations.

    Returns
    -------
    type_to_colour            : {type_string: rgba}
    node_colour_for_each_row  : list of rgba, indexed by row
    relation_colour_for_each_channel : {channel_index: rgba}
    """
    unique_node_types  = sorted(set(node_types.values()))
    type_cmap          = plt.cm.get_cmap("Set2", max(len(unique_node_types), 3))
    type_to_colour     = {
        node_type: type_cmap(type_index)
        for type_index, node_type in enumerate(unique_node_types)
    }
    node_colour_for_each_row = [
        type_to_colour[node_types[row_index]]
        for row_index in range(len(node_types))
    ]

    edge_cmap = plt.cm.get_cmap("tab20", num_relations)
    relation_colour_for_each_channel = {
        channel_index: edge_cmap(channel_index)
        for channel_index in range(num_relations)
    }

    return type_to_colour, node_colour_for_each_row, relation_colour_for_each_channel


def _draw_graph_on_axis(
    axis,
    adjacency_tensor: np.ndarray,
    title: str,
    node_positions: Dict[int, Tuple[float, float]],
    node_colour_for_each_row: List[tuple],
    node_labels: Dict[int, str],
    relation_colour_for_each_channel: Dict[int, tuple],
) -> None:
    """Draw one directed multi-relational graph onto an existing matplotlib axis."""
    num_nodes = adjacency_tensor.shape[0]
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(range(num_nodes))
    edges_per_channel = _build_edge_lists_per_channel(adjacency_tensor)
    for channel_edges in edges_per_channel.values():
        directed_graph.add_edges_from(channel_edges)

    nx.draw_networkx_nodes(
        directed_graph, node_positions, ax=axis, node_size=900,
        node_color=node_colour_for_each_row, edgecolors="#2C3E50",
        linewidths=1.5, alpha=0.95,
    )
    nx.draw_networkx_labels(
        directed_graph, node_positions, labels=node_labels, ax=axis,
        font_size=7, font_color="black", font_weight="bold",
    )

    num_relations = adjacency_tensor.shape[2]
    for channel_index in range(num_relations):
        channel_edges = edges_per_channel[channel_index]
        if not channel_edges:
            continue
        nx.draw_networkx_edges(
            directed_graph, node_positions, edgelist=channel_edges, ax=axis,
            edge_color=[relation_colour_for_each_channel[channel_index]],
            width=2.5, alpha=0.85,
            arrows=True, arrowsize=18,
            # arc3,rad=... curves each channel slightly differently so
            # multiple edges between the same nodes don't overlap visually.
            connectionstyle=f"arc3,rad={0.1 + channel_index * 0.06}",
            min_source_margin=18, min_target_margin=18,
        )
    axis.set_title(title, fontsize=9, fontweight="bold")
    axis.axis("off")


def _draw_heatmap_on_axis(
    axis,
    matrix: np.ndarray,
    tick_labels: List[str],
    title: str,
    max_value: float,
    colour_map_name: str = "Blues",
) -> None:
    """Draw a summed-relation heatmap onto an existing matplotlib axis."""
    image = axis.imshow(matrix, cmap=colour_map_name, vmin=0, vmax=max_value)
    axis.set_title(title, fontsize=9, fontweight="bold")
    num_rows = matrix.shape[0]
    axis.set_xticks(range(num_rows))
    axis.set_yticks(range(num_rows))
    axis.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(tick_labels, fontsize=7)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _build_legend_handles(
    type_to_colour: Dict[str, tuple],
    relation_colour_for_each_channel: Dict[int, tuple],
    relation_names: Sequence[str],
) -> List[mpatches.Patch]:
    """Build a combined "entity types + relations" legend handle list."""
    entity_type_patches = [
        mpatches.Patch(color=colour, label=type_name)
        for type_name, colour in type_to_colour.items()
    ]
    relation_patches = [
        mpatches.Patch(
            color=relation_colour_for_each_channel[channel_index],
            label=relation_names[channel_index],
        )
        for channel_index in range(len(relation_names))
    ]
    return (
        [mpatches.Patch(color="none", label="-- Entity Types --")]
        + entity_type_patches
        + [mpatches.Patch(color="none", label="")]
        + [mpatches.Patch(color="none", label="-- Relations --")]
        + relation_patches
    )


# ── Public API ────────────────────────────────────────────────────────────────

def plot_dataset_sample(
    clean_adjacency_nxnxr: np.ndarray,
    corrupted_adjacency_nxnxr: np.ndarray,
    node_labels: Dict[int, str],
    node_types: Dict[int, str],
    relation_names: Sequence[str],
    num_corruptions: int = 2,
    figsize: tuple = (14, 10),
) -> None:
    """Show one (clean, corrupted) training-sample pair as a 2x2 figure.

    Layout:
        row 0: clean graph         |  corrupted graph
        row 1: clean heatmap       |  corrupted heatmap

    Parameters
    ----------
    clean_adjacency_nxnxr, corrupted_adjacency_nxnxr
        (N, N, R) arrays in [0, 1].
    node_labels
        row_index -> display string. Pass permutation-adjusted labels when
        showing a sample that has been entity-permuted.
    node_types
        row_index -> entity type, same caveat as above.
    relation_names
        list, indexed by channel.
    num_corruptions
        Number used in the corrupted graph's title.
    figsize
        Forwarded to plt.subplots.
    """
    num_nodes     = clean_adjacency_nxnxr.shape[0]
    num_relations = clean_adjacency_nxnxr.shape[2]

    type_to_colour, node_colour_for_each_row, relation_colour_for_each_channel = (
        _make_colour_maps(node_types, num_relations)
    )
    node_positions = _bipartite_layout(node_types)
    tick_labels    = [node_labels[row_index] for row_index in range(num_nodes)]
    heatmap_vmax   = max(clean_adjacency_nxnxr.sum(axis=-1).max(), 1)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Row 0: graphs
    _draw_graph_on_axis(
        axes[0, 0], clean_adjacency_nxnxr,
        "Clean graph",
        node_positions, node_colour_for_each_row, node_labels,
        relation_colour_for_each_channel,
    )
    _draw_graph_on_axis(
        axes[0, 1], corrupted_adjacency_nxnxr,
        f"Corrupted graph ({num_corruptions} ops)",
        node_positions, node_colour_for_each_row, node_labels,
        relation_colour_for_each_channel,
    )

    # Legend on the right-hand graph
    axes[0, 1].legend(
        handles=_build_legend_handles(type_to_colour, relation_colour_for_each_channel, relation_names),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95,
        title="Legend", title_fontsize=9, borderaxespad=0,
    )

    # Row 1: heatmaps (sum over the R axis so a heatmap shows "any-relation" edges)
    _draw_heatmap_on_axis(
        axes[1, 0], clean_adjacency_nxnxr.sum(axis=-1),
        tick_labels, "Clean matrix (sum over relations)", heatmap_vmax,
    )
    _draw_heatmap_on_axis(
        axes[1, 1], corrupted_adjacency_nxnxr.sum(axis=-1),
        tick_labels, "Corrupted matrix (sum over relations)", heatmap_vmax,
    )

    plt.tight_layout()
    plt.show()


def plot_gan_comparison(
    clean_adjacency_nxnxr: np.ndarray,
    rule_adjacency_nxnxr: np.ndarray,
    gan_adjacency_nxnxr: np.ndarray,
    node_labels: Dict[int, str],
    node_types: Dict[int, str],
    relation_names: Sequence[str],
    num_corruptions: int = 2,
    figsize: tuple = (18, 11),
) -> None:
    """Compare three KG views side-by-side: clean, rule-corrupted, GAN-corrupted.

    Layout:
        row 0: clean graph         |  rule-corrupted graph     |  GAN-corrupted graph
        row 1: clean heatmap       |  rule-corrupted heatmap   |  GAN-corrupted heatmap

    Parameters
    ----------
    clean_adjacency_nxnxr, rule_adjacency_nxnxr, gan_adjacency_nxnxr
        All three are (N, N, R) arrays.
    node_labels, node_types, relation_names
        Same as plot_dataset_sample.
    num_corruptions
        Used in the rule-corrupted graph's title.
    figsize
        Forwarded to plt.subplots.
    """
    num_nodes     = clean_adjacency_nxnxr.shape[0]
    num_relations = clean_adjacency_nxnxr.shape[2]

    type_to_colour, node_colour_for_each_row, relation_colour_for_each_channel = (
        _make_colour_maps(node_types, num_relations)
    )
    node_positions = _bipartite_layout(node_types)
    tick_labels    = [node_labels[row_index] for row_index in range(num_nodes)]
    heatmap_vmax   = max(clean_adjacency_nxnxr.sum(axis=-1).max(), 1)

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Row 0: graphs
    _draw_graph_on_axis(
        axes[0, 0], clean_adjacency_nxnxr, "Clean graph",
        node_positions, node_colour_for_each_row, node_labels,
        relation_colour_for_each_channel,
    )
    _draw_graph_on_axis(
        axes[0, 1], rule_adjacency_nxnxr,
        f"TRIC-corrupted (ref, {num_corruptions} ops)",
        node_positions, node_colour_for_each_row, node_labels,
        relation_colour_for_each_channel,
    )
    _draw_graph_on_axis(
        axes[0, 2], gan_adjacency_nxnxr, "GAN-corrupted graph",
        node_positions, node_colour_for_each_row, node_labels,
        relation_colour_for_each_channel,
    )

    axes[0, 2].legend(
        handles=_build_legend_handles(type_to_colour, relation_colour_for_each_channel, relation_names),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95,
        title="Legend", title_fontsize=9, borderaxespad=0,
    )

    # Row 1: heatmaps
    heatmap_axes_and_data = zip(
        axes[1],
        [
            clean_adjacency_nxnxr.sum(axis=-1),
            rule_adjacency_nxnxr.sum(axis=-1),
            gan_adjacency_nxnxr.sum(axis=-1),
        ],
        [
            "Clean matrix (sum over relations)",
            "TRIC-corrupted matrix",
            "GAN-corrupted matrix",
        ],
    )
    for heatmap_axis, heatmap_matrix, heatmap_title in heatmap_axes_and_data:
        _draw_heatmap_on_axis(
            heatmap_axis, heatmap_matrix, tick_labels, heatmap_title, heatmap_vmax,
        )
        heatmap_axis.set_xlabel("Entity j", fontsize=8)
        heatmap_axis.set_ylabel("Entity i", fontsize=8)

    plt.tight_layout()
    plt.show()
