"""Two-stage inference: turn a per-triple generator into a graph-level corruptor.

  Stage A (selection):     pick which triples to corrupt (random selection).
  Stage B (transformation): for each picked triple, sample a latent z and
                            run the generator; argmax the three softmax heads.

Repeat K times to get a population of distinct corrupted KGs.
"""
from typing import List, Sequence, Tuple

import numpy as np
import torch


Triple = Tuple[int, int, int]   # (h_row, r_channel, t_row)


def generate_k_corrupted_kgs(
    generator_model: torch.nn.Module,
    entity_embedding: torch.nn.Module,
    relation_embedding: torch.nn.Module,
    clean_triples: Sequence[Triple],
    *,
    K: int = 20,
    num_corruptions: int = 2,
    latent_dim: int = 8,
    tau: float = 0.5,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
) -> List[List[Triple]]:
    """Generate K distinct corrupted versions of the clean KG.

    Stage A: randomly select `num_corruptions` triples to corrupt.
    Stage B: for each selected triple, sample z and forward G; argmax the three heads.

    Returns
    -------
    list of length K. Each element is a list[Triple] representing one corrupted KG.
    Unselected triples appear unchanged in the output.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_total = len(clean_triples)
    n_corrupt = min(num_corruptions, n_total)

    generator_model.eval()
    out: List[List[Triple]] = []
    with torch.no_grad():
        for _ in range(K):
            selected = set(int(i) for i in rng.choice(n_total, size=n_corrupt, replace=False))
            out_kg: List[Triple] = []
            for i, (h, r, t) in enumerate(clean_triples):
                if i not in selected:
                    out_kg.append((h, r, t))
                    continue
                h_emb = entity_embedding(torch.tensor([h], device=device))
                r_emb = relation_embedding(torch.tensor([r], device=device))
                t_emb = entity_embedding(torch.tensor([t], device=device))
                clean_triple_emb = torch.stack([h_emb, r_emb, t_emb], dim=1)   # (1, 3, d)
                z = torch.randn(1, latent_dim, device=device)
                g_out = generator_model(clean_triple_emb, z, tau=tau, return_soft=False)
                out_kg.append((
                    int(g_out["h_logits"].argmax(dim=-1).item()),
                    int(g_out["r_logits"].argmax(dim=-1).item()),
                    int(g_out["t_logits"].argmax(dim=-1).item()),
                ))
            out.append(out_kg)
    return out


def kg_triples_to_nxnxr(
    triples: Sequence[Triple], num_nodes: int, num_relations: int,
) -> np.ndarray:
    """Convert a list of (h, r, t) row/channel indices back to (N, N, R) float {0, 1}.

    Useful for handing GAN output to a plotter that expects an adjacency tensor.
    """
    adj = np.zeros((num_nodes, num_nodes, num_relations), dtype=np.float32)
    for h, r, t in triples:
        if h != t:
            adj[h, t, r] = 1.0
    return adj
