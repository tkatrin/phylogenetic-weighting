from __future__ import annotations

import numpy as np

from plm_phylo_weighting.utils.cleaning import safe_normalize_array, safe_normalize_dict


def compute_taxonomic_weights(
    sequence_to_group: dict[str, str],
    normalize_sum: float | None = None,
) -> dict[str, float]:
    group_sizes: dict[str, int] = {}

    for group in sequence_to_group.values():
        group_sizes[group] = group_sizes.get(group, 0) + 1

    weights = {
        sequence: 1.0 / group_sizes[group]
        for sequence, group in sequence_to_group.items()
    }

    if normalize_sum is not None:
        weights = safe_normalize_dict(weights, normalize_sum)

    return weights


def compute_henikoff_weights(
    alignment: dict[str, str],
    normalize_sum: float | None = None,
    ignore_chars: set[str] | None = None,
) -> dict[str, float]:
    ignore_chars = ignore_chars or {"-", "."}
    names = list(alignment.keys())

    if not names:
        return {}

    sequences = [alignment[name] for name in names]
    aln_len = len(sequences[0])
    raw_weights = np.zeros(len(names), dtype=float)

    for pos in range(aln_len):
        column = [seq[pos] for seq in sequences]
        valid_symbols = [symbol for symbol in column if symbol not in ignore_chars]

        if not valid_symbols:
            continue

        unique_symbols = sorted(set(valid_symbols))
        n_unique = len(unique_symbols)

        counts: dict[str, int] = {}
        for symbol in valid_symbols:
            counts[symbol] = counts.get(symbol, 0) + 1

        for i, symbol in enumerate(column):
            if symbol in ignore_chars:
                continue
            raw_weights[i] += 1.0 / (n_unique * counts[symbol])

    if normalize_sum is not None:
        raw_weights = safe_normalize_array(raw_weights, normalize_sum)

    return dict(zip(names, raw_weights))


def compute_identity_from_aligned_pair(seq_a: str, seq_b: str, ignore_gaps: bool = True) -> float:
    matches = 0
    total = 0

    for a, b in zip(seq_a, seq_b):
        if ignore_gaps and (a in {"-", "."} or b in {"-", "."}):
            continue
        total += 1
        if a == b:
            matches += 1

    if total == 0:
        return 0.0

    return matches / total


def compute_clustering_weights(
    alignment: dict[str, str],
    threshold: float,
    normalize_sum: float | None = None,
) -> dict[str, float]:
    names = list(alignment.keys())
    graph = {name: {name} for name in names}

    for i, first in enumerate(names):
        for j in range(i + 1, len(names)):
            second = names[j]
            identity = compute_identity_from_aligned_pair(alignment[first], alignment[second])
            if identity >= threshold:
                graph[first].add(second)
                graph[second].add(first)

    seen: set[str] = set()
    weights: dict[str, float] = {}

    for node in graph:
        if node in seen:
            continue

        stack = [node]
        seen.add(node)
        component: list[str] = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        value = 1.0 / len(component)
        for sequence in component:
            weights[sequence] = value

    if normalize_sum is not None:
        weights = safe_normalize_dict(weights, normalize_sum)

    return weights
