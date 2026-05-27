"""Whole-KG visualisation with NetworkX.

Single-panel directed multi-relational graph view, with nodes coloured by
entity type and edges coloured by relation. Used in the notebook's
"visualise the loaded KG" step.

We use NetworkX (a Python graph library) for layout + drawing. NetworkX
needs an adjacency representation, so callers pass in the (N, N, R) tensor
form. The triple_only data pipeline does NOT use this tensor anywhere else;
it's built on demand only for plotting.
"""
from typing import Dict, List, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualise_kg(
    adjacency_tensor: np.ndarray,
    node_labels: Dict[int, str],
    node_types: Dict[int, str],
    relation_names: Sequence[str],
    title: str = "Knowledge Graph",
    figsize: tuple = (8, 6),
    layout_seed: int = 42,
) -> None:
    """Draw a multi-relational directed KG with NetworkX.

    Parameters
    ----------
    adjacency_tensor
        (N, N, R) array in {0, 1}. `adjacency_tensor[i, j, r] == 1` means
        there's a directed edge from node i to node j via relation r.
    node_labels
        row_index -> display string. Used as the label inside each node.
    node_types
        row_index -> type string (e.g. "Person", "Country"). Used to colour
        nodes - same type = same colour.
    relation_names
        list indexed by channel. Used in the legend.
    title
        Figure title.
    figsize
        Passed to matplotlib's `plt.subplots`.
    layout_seed
        Seed for NetworkX's spring layout (so the same graph draws the
        same way every time).
    """
    num_nodes     = adjacency_tensor.shape[0]
    num_relations = adjacency_tensor.shape[2]

    # ── Node colour map: one colour per unique entity type ──────────────
    unique_node_types     = sorted(set(node_types.values()))
    node_type_colour_map  = plt.cm.get_cmap("Set2", max(len(unique_node_types), 3))
    type_to_colour        = {
        node_type: node_type_colour_map(type_index)
        for type_index, node_type in enumerate(unique_node_types)
    }
    node_colour_for_each_row = [
        type_to_colour[node_types[row_index]]
        for row_index in range(num_nodes)
    ]

    # ── Relation colour map: one colour per relation channel ────────────
    relation_colour_map = plt.cm.get_cmap("tab20", num_relations)
    relation_colours = {
        channel_index: relation_colour_map(channel_index)
        for channel_index in range(num_relations)
    }

    # ── Per-channel edge lists from the adjacency tensor ────────────────
    edges_for_each_channel: Dict[int, List[tuple]] = {
        channel_index: [] for channel_index in range(num_relations)
    }
    for channel_index in range(num_relations):
        head_rows, tail_rows = np.where(adjacency_tensor[:, :, channel_index] > 0.5)
        edges_for_each_channel[channel_index] = list(
            zip(head_rows.tolist(), tail_rows.tolist())
        )

    # ── Build the NetworkX DiGraph ──────────────────────────────────────
    # A DiGraph is a directed graph; nodes are integers, edges have attrs.
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(range(num_nodes))
    for channel_index, channel_edges in edges_for_each_channel.items():
        for source_node, target_node in channel_edges:
            directed_graph.add_edge(source_node, target_node, relation=channel_index)

    # spring_layout positions nodes by simulating a physics system
    # (attractive force between connected nodes, repulsive between all pairs).
    node_positions = nx.spring_layout(directed_graph, seed=layout_seed, k=2.5)

    fig, axis = plt.subplots(figsize=figsize)

    # Draw nodes (circles), coloured by type
    nx.draw_networkx_nodes(
        directed_graph, node_positions, nodelist=list(range(num_nodes)),
        ax=axis, node_size=900,
        node_color=node_colour_for_each_row, edgecolors="#2C3E50",
        linewidths=1.5, alpha=0.95,
    )
    # Draw the entity-name labels inside the nodes
    nx.draw_networkx_labels(
        directed_graph, node_positions, labels=node_labels, ax=axis,
        font_size=7, font_color="black", font_weight="bold",
    )

    # Draw edges, one channel at a time so each gets its own colour.
    # The `connectionstyle="arc3,rad=..."` argument curves each channel
    # slightly differently so multiple edges between the same nodes don't
    # overlap.
    relation_legend_patches = []
    for channel_index in range(num_relations):
        channel_edges = edges_for_each_channel[channel_index]
        if not channel_edges:
            continue
        edge_colour = relation_colours[channel_index]
        nx.draw_networkx_edges(
            directed_graph, node_positions, edgelist=channel_edges, ax=axis,
            edge_color=[edge_colour], width=2.5, alpha=0.85,
            arrows=True, arrowsize=18,
            connectionstyle=f"arc3,rad={0.1 + channel_index * 0.06}",
            min_source_margin=18, min_target_margin=18,
        )
        relation_legend_patches.append(
            mpatches.Patch(color=edge_colour, label=relation_names[channel_index])
        )

    # ── Legend: entity-type colours on top, relation colours below ──────
    entity_type_legend_patches = [
        mpatches.Patch(color=type_to_colour[node_type], label=node_type)
        for node_type in unique_node_types
    ]
    legend_spacer = mpatches.Patch(color="none", label="")
    legend_patches = (
        [mpatches.Patch(color="none", label="-- Entity Types --")]
        + entity_type_legend_patches
        + [legend_spacer]
        + [mpatches.Patch(color="none", label="-- Relations --")]
        + relation_legend_patches
    )
    axis.legend(
        handles=legend_patches,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95,
        title="Legend", title_fontsize=10, borderaxespad=0,
    )
    axis.set_title(title, fontsize=13, fontweight="bold", pad=15)
    axis.axis("off")
    plt.tight_layout()
    plt.show()
