from __future__ import annotations

from collections.abc import Callable

from ete3 import Tree

from plm_phylo_weighting.utils.cleaning import safe_normalize_dict


def compute_branch_sharing_weights(newick: str, normalize_sum: float | None = None) -> dict[str, float]:
    tree = Tree(newick, format=1)
    weights = {leaf.name: 0.0 for leaf in tree.iter_leaves()}

    for node in tree.traverse("postorder"):
        if node.is_root():
            continue

        leaves = node.get_leaves()
        n_leaves = len(leaves)
        branch_length = float(node.dist)

        for leaf in leaves:
            weights[leaf.name] += branch_length / n_leaves

    if normalize_sum is not None:
        weights = safe_normalize_dict(weights, normalize_sum)

    return weights


def compute_mean_tree_distance_weights(newick: str, normalize_sum: float | None = None) -> dict[str, float]:
    tree = Tree(newick, format=1)
    leaves = list(tree.iter_leaves())
    weights: dict[str, float] = {}

    for leaf in leaves:
        distances = [tree.get_distance(leaf, other) for other in leaves if leaf.name != other.name]
        weights[leaf.name] = float(sum(distances) / len(distances)) if distances else 1.0

    if normalize_sum is not None:
        weights = safe_normalize_dict(weights, normalize_sum)

    return weights


def compute_terminal_branch_weights(newick: str, normalize_sum: float | None = None) -> dict[str, float]:
    tree = Tree(newick, format=1)
    weights = {leaf.name: float(leaf.dist) for leaf in tree.iter_leaves()}

    if normalize_sum is not None:
        weights = safe_normalize_dict(weights, normalize_sum)

    return weights


PHYLO_METHODS: dict[str, Callable[[str, float | None], dict[str, float]]] = {
    "branch_sharing": compute_branch_sharing_weights,
    "mean_tree_distance": compute_mean_tree_distance_weights,
    "terminal_branch": compute_terminal_branch_weights,
}
