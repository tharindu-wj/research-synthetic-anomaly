"""Whole-KG visualisation with NetworkX.

Single-panel directed multi-relational graph view, with nodes coloured by
entity type and edges coloured by relation. Used in the notebook's
"visualise the loaded KG" step.
"""
from typing import Dict, List, Optional, Sequence

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
    seed: int = 42,
) -> None:
    """Draw a multi-relational directed KG.

    Parameters
    ----------
    adjacency_tensor : (N, N, R) np.ndarray, values in {0, 1}
    node_labels      : row_index -> display name
    node_types       : row_index -> entity type
    relation_names   : list, indexed by channel
    title            : figure title
    figsize          : passed to plt.subplots
    seed             : layout seed for reproducibility
    """
    num_nodes     = adjacency_tensor.shape[0]
    num_relations = adjacency_tensor.shape[2]

    # Node-type colour map
    types_in_order   = sorted(set(node_types.values()))
    type_cmap        = plt.cm.get_cmap("Set2", max(len(types_in_order), 3))
    type_to_colour   = {t: type_cmap(i) for i, t in enumerate(types_in_order)}
    node_colour_list = [type_to_colour[node_types[n]] for n in range(num_nodes)]

    # Relation colour map
    edge_cmap        = plt.cm.get_cmap("tab20", num_relations)
    relation_colours = {ch: edge_cmap(ch) for ch in range(num_relations)}

    # Build per-channel edge lists
    edge_list_by_channel: Dict[int, List[tuple]] = {ch: [] for ch in range(num_relations)}
    for ch in range(num_relations):
        rows, cols = np.where(adjacency_tensor[:, :, ch] > 0.5)
        edge_list_by_channel[ch] = list(zip(rows.tolist(), cols.tolist()))

    # NetworkX digraph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for ch, edges in edge_list_by_channel.items():
        for src, dst in edges:
            G.add_edge(src, dst, relation=ch)

    pos = nx.spring_layout(G, seed=seed, k=2.5)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(range(num_nodes)), ax=ax, node_size=900,
        node_color=node_colour_list, edgecolors="#2C3E50",
        linewidths=1.5, alpha=0.95,
    )
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=7, font_color="black", font_weight="bold",
    )

    relation_legend_patches = []
    for ch in range(num_relations):
        edges = edge_list_by_channel[ch]
        if not edges:
            continue
        colour = relation_colours[ch]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=[colour], width=2.5, alpha=0.85,
            arrows=True, arrowsize=18,
            connectionstyle=f"arc3,rad={0.1 + ch * 0.06}",
            min_source_margin=18, min_target_margin=18,
        )
        relation_legend_patches.append(
            mpatches.Patch(color=colour, label=relation_names[ch])
        )

    type_legend_patches = [
        mpatches.Patch(color=type_to_colour[t], label=t) for t in types_in_order
    ]
    spacer = mpatches.Patch(color="none", label="")
    legend_patches = (
        [mpatches.Patch(color="none", label="-- Entity Types --")]
        + type_legend_patches
        + [spacer]
        + [mpatches.Patch(color="none", label="-- Relations --")]
        + relation_legend_patches
    )
    ax.legend(
        handles=legend_patches, loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=8, framealpha=0.95, title="Legend", title_fontsize=10,
        borderaxespad=0,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
