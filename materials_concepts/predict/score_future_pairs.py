"""Score currently unconnected concept pairs for prospective hypothesis generation.

This script is intentionally label-free: it does not evaluate AUC because the
future has not happened yet. It samples concept pairs that are unconnected at a
cutoff year, scores them with the trained semantic model and/or GNN, blends the
scores, and writes the top-ranked pairs to CSV.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any

import networkx as nx
import fire
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from materials_concepts.model.gnn.train_pyg import (
    DotDecoder,
    EdgeMLPDecoder,
    Graph,
    SAGEEncoder,
    _parse_config_str,
    _require_pyg,
    _select_device,
    build_edge_index_for_year,
    load_node_feature_matrix,
)
from materials_concepts.utils.utils import parse_concept_list


class SemanticPairNetwork(nn.Module):
    """Same MLP architecture as materials_concepts.model.combi.train.BaselineNetwork.

    Defined locally so prospective scoring does not depend on train.py's global
    logger state.
    """

    def __init__(self, layer_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=False):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        layers.pop()  # remove final dropout
        layers.pop()  # remove final relu
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_compressed(path: str | Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_compressed(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f)


def setup_logger(log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("score_future_pairs")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def load_lookup(path: str | Path) -> pd.DataFrame:
    lookup = pd.read_csv(path)
    if "id" not in lookup or "concept" not in lookup:
        raise ValueError("Lookup file must contain columns 'id' and 'concept'.")
    lookup = lookup[["id", "concept"]].copy()
    lookup["id"] = lookup["id"].astype(int)
    return lookup


def embedding_ids(embeddings: dict) -> set[int]:
    return {int(key) for key in embeddings.keys()}


def parse_optional_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_string_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[,\n;|]+", text) if part.strip()]


def split_modules(value) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in re.split(r"[,\n;|]+", str(value)) if part.strip()}


def concept_to_id_lookup(lookup: pd.DataFrame) -> dict[str, int]:
    return dict(zip(lookup["concept"].str.lower(), lookup["id"].astype(int), strict=False))


def resolve_concept_labels(
    labels: list[str],
    *,
    lookup: pd.DataFrame,
    logger: logging.Logger,
) -> set[int]:
    by_label = concept_to_id_lookup(lookup)
    out: set[int] = set()
    missing: list[str] = []
    for label in labels:
        key = str(label).strip().lower()
        if not key:
            continue
        node_id = by_label.get(key)
        if node_id is None:
            missing.append(label)
        else:
            out.add(int(node_id))
    if missing:
        logger.warning(
            "Could not resolve %d anchor concept(s): %s",
            len(missing),
            ", ".join(missing[:20]),
        )
    return out


def load_anchor_labels(anchor_concepts: str | None, anchor_file: str | None) -> list[str]:
    labels = parse_string_list(anchor_concepts)
    if anchor_file:
        path = Path(anchor_file)
        if not path.exists():
            raise FileNotFoundError(f"Anchor concept file does not exist: {anchor_file}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            column = "concept" if "concept" in df else df.columns[0]
            labels.extend(df[column].dropna().astype(str).tolist())
        else:
            labels.extend(
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip()
            )
    return labels


def collect_concept_ids_from_works(
    *,
    works_path: str,
    lookup: pd.DataFrame,
    concept_column: str,
    modules: set[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    logger: logging.Logger,
) -> set[int]:
    """Collect graph node IDs from works matching optional modules/year filters."""
    if not works_path:
        raise ValueError("works_path is required for module/emerging candidate modes.")

    by_label = concept_to_id_lookup(lookup)
    usecols = [concept_column]
    if modules is not None:
        usecols.append("modules")
    if year_min is not None or year_max is not None:
        usecols.append("publication_year")

    logger.info(
        "Collecting concept IDs from %s | modules=%s | years=%s-%s",
        works_path,
        sorted(modules) if modules else None,
        year_min,
        year_max,
    )
    df = pd.read_csv(works_path, usecols=lambda col: col in set(usecols), low_memory=False)
    out: set[int] = set()
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Collecting concepts"):
        row_dict = row._asdict()
        if modules is not None and not (split_modules(row_dict.get("modules")) & modules):
            continue
        if year_min is not None or year_max is not None:
            try:
                year = int(row_dict.get("publication_year"))
            except (TypeError, ValueError):
                continue
            if year_min is not None and year < year_min:
                continue
            if year_max is not None and year > year_max:
                continue

        for concept in parse_concept_list(row_dict.get(concept_column)):
            node_id = by_label.get(str(concept).lower())
            if node_id is not None:
                out.add(int(node_id))
    logger.info("Collected %d concept ID(s)", len(out))
    return out


def collect_emerging_concept_ids(
    *,
    works_path: str,
    lookup: pd.DataFrame,
    concept_column: str,
    cutoff_year: int,
    emerging_start_year: int,
    emerging_min_count: int,
    modules: set[str] | None,
    logger: logging.Logger,
) -> set[int]:
    """Return concepts whose first observed year is recent enough."""
    if not works_path:
        raise ValueError("works_path is required for emerging candidate mode.")

    by_label = concept_to_id_lookup(lookup)
    first_year: dict[int, int] = {}
    counts: dict[int, int] = {}
    usecols = [concept_column, "publication_year", "modules"]

    logger.info(
        "Collecting emerging concepts from %s | first_year >= %d | cutoff <= %d",
        works_path,
        emerging_start_year,
        cutoff_year,
    )
    df = pd.read_csv(works_path, usecols=lambda col: col in set(usecols), low_memory=False)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Finding emerging concepts"):
        row_dict = row._asdict()
        if modules is not None and not (split_modules(row_dict.get("modules")) & modules):
            continue
        try:
            year = int(row_dict.get("publication_year"))
        except (TypeError, ValueError):
            continue
        if year > cutoff_year:
            continue
        for concept in parse_concept_list(row_dict.get(concept_column)):
            node_id = by_label.get(str(concept).lower())
            if node_id is None:
                continue
            node_id = int(node_id)
            first_year[node_id] = min(first_year.get(node_id, year), year)
            counts[node_id] = counts.get(node_id, 0) + 1

    out = {
        node_id
        for node_id, year in first_year.items()
        if year >= emerging_start_year and counts.get(node_id, 0) >= emerging_min_count
    }
    logger.info("Collected %d emerging concept ID(s)", len(out))
    return out


def filter_ids_by_graph_state(
    ids: set[int],
    *,
    past_graph: nx.Graph,
    allowed_ids: set[int] | None,
    min_degree: int,
    max_degree: int | None,
) -> set[int]:
    out: set[int] = set()
    for node_id in ids:
        if allowed_ids is not None and node_id not in allowed_ids:
            continue
        if node_id not in past_graph:
            continue
        degree = int(past_graph.degree[node_id])
        if degree < min_degree:
            continue
        if max_degree is not None and degree > max_degree:
            continue
        out.add(int(node_id))
    return out


def node_distance(
    past_graph: nx.Graph,
    u: int,
    v: int,
    *,
    cutoff: int | None = None,
) -> int | None:
    try:
        distances = nx.single_source_shortest_path_length(
            past_graph,
            int(u),
            cutoff=cutoff,
        )
        distance = distances.get(int(v))
        return int(distance) if distance is not None else None
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def annotate_pair_distances(
    past_graph: nx.Graph,
    pairs: np.ndarray,
    *,
    cutoff: int | None = None,
    logger: logging.Logger | None = None,
) -> list[int]:
    """Annotate pair distances while reusing graph traversals across repeated nodes.

    Top candidate lists often contain many pairs that share the same anchor/source
    concept. Calling shortest-path search once per pair repeats almost identical
    NetworkX work thousands of times. This helper greedily chooses high-frequency
    endpoints and annotates all still-unprocessed pairs touching that endpoint
    from one bounded BFS.
    """
    if len(pairs) == 0:
        return []

    endpoint_to_indices: dict[int, set[int]] = {}
    for idx, (u, v) in enumerate(pairs):
        endpoint_to_indices.setdefault(int(u), set()).add(idx)
        endpoint_to_indices.setdefault(int(v), set()).add(idx)

    distances = [-1] * len(pairs)
    remaining = set(range(len(pairs)))
    traversals = 0

    with tqdm(total=len(pairs), desc="Annotating top-pair distances") as pbar:
        while remaining:
            best_node = None
            best_indices: set[int] | None = None
            best_count = 0
            for node, indices in endpoint_to_indices.items():
                count = len(indices & remaining)
                if count > best_count:
                    best_node = node
                    best_indices = indices
                    best_count = count

            if best_node is None or best_indices is None:
                break

            active_indices = best_indices & remaining
            if best_node in past_graph:
                try:
                    node_distances = nx.single_source_shortest_path_length(
                        past_graph,
                        best_node,
                        cutoff=cutoff,
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    node_distances = {}
                traversals += 1
            else:
                node_distances = {}

            for idx in active_indices:
                u = int(pairs[idx, 0])
                v = int(pairs[idx, 1])
                other = v if u == best_node else u
                distance = node_distances.get(other)
                distances[idx] = int(distance) if distance is not None else -1

            remaining -= active_indices
            pbar.update(len(active_indices))

    if logger is not None:
        logger.info(
            "Finished graph-distance annotations for %d pair(s) using %d bounded BFS traversal(s)",
            len(pairs),
            traversals,
        )
    return distances


def passes_distance_filter(
    *,
    past_graph: nx.Graph,
    u: int,
    v: int,
    distance_min: int | None,
    distance_max: int | None,
) -> bool:
    if distance_min is None and distance_max is None:
        return True
    distance = node_distance(past_graph, u, v, cutoff=distance_max)
    if distance is None:
        return False
    if distance_min is not None and distance < distance_min:
        return False
    if distance_max is not None and distance > distance_max:
        return False
    return True


def sample_unconnected_pairs(
    *,
    graph: Graph,
    cutoff_year: int,
    num_pairs: int,
    seed: int,
    min_degree: int = 1,
    max_degree: int | None = None,
    allowed_ids: set[int] | None = None,
    logger: logging.Logger,
) -> np.ndarray:
    """Sample undirected node pairs that are not connected by cutoff_year."""
    rng = random.Random(seed)
    vertices = graph.get_vertices(
        until_year=cutoff_year,
        min_degree=min_degree,
        max_degree=max_degree,
    )
    vertices = [int(v) for v in vertices]
    if allowed_ids is not None:
        vertices = [v for v in vertices if v in allowed_ids]
    vertices = sorted(set(vertices))

    if len(vertices) < 2:
        raise ValueError(
            f"Need at least two candidate vertices; got {len(vertices)} after filters."
        )

    past_graph = graph.get_nx_graph(cutoff_year)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(num_pairs * 200, 100_000)

    pbar = tqdm(total=num_pairs, desc="Sampling unconnected pairs")
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(vertices, 2)
        if u > v:
            u, v = v, u
        if (u, v) in pairs:
            continue
        if past_graph.has_edge(u, v):
            continue
        pairs.add((u, v))
        pbar.update(1)
    pbar.close()

    if len(pairs) < num_pairs:
        logger.warning(
            "Requested %d pairs but sampled only %d after %d attempts.",
            num_pairs,
            len(pairs),
            attempts,
        )

    out = np.asarray(sorted(pairs), dtype=np.int64)
    logger.info("Sampled %d candidate pair(s)", len(out))
    return out


def sample_source_target_pairs(
    *,
    past_graph: nx.Graph,
    source_ids: set[int],
    target_ids: set[int],
    num_pairs: int,
    seed: int,
    distance_min: int | None,
    distance_max: int | None,
    target_max_degree: int | None,
    max_pairs_per_source: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Sample unconnected pairs from a source-target candidate space."""
    rng = random.Random(seed)
    sources = sorted(source_ids)
    targets_all = sorted(target_ids)
    if not sources:
        raise ValueError("No source candidate nodes available after filters.")
    if not targets_all:
        raise ValueError("No target candidate nodes available after filters.")

    target_degree_ok = {}
    for target in targets_all:
        if target_max_degree is None:
            target_degree_ok[target] = True
        elif target in past_graph:
            target_degree_ok[target] = int(past_graph.degree[target]) <= int(target_max_degree)
        else:
            target_degree_ok[target] = False

    pairs: set[tuple[int, int]] = set()
    pbar = tqdm(total=num_pairs, desc="Sampling candidate pairs")
    rounds_without_progress = 0

    while len(pairs) < num_pairs and rounds_without_progress < 3:
        before_round = len(pairs)
        source_order = list(sources)
        rng.shuffle(source_order)
        for u in source_order:
            if len(pairs) >= num_pairs:
                break

            if distance_min is not None or distance_max is not None:
                distances = nx.single_source_shortest_path_length(
                    past_graph,
                    u,
                    cutoff=distance_max,
                )
                eligible = [
                    v
                    for v, distance in distances.items()
                    if v in target_ids
                    and v != u
                    and target_degree_ok.get(v, False)
                    and not past_graph.has_edge(u, v)
                    and (distance_min is None or distance >= distance_min)
                    and (distance_max is None or distance <= distance_max)
                ]
            else:
                eligible = [
                    v
                    for v in targets_all
                    if v != u
                    and target_degree_ok.get(v, False)
                    and not past_graph.has_edge(u, v)
                ]

            if not eligible:
                continue
            rng.shuffle(eligible)
            for v in eligible[: int(max_pairs_per_source)]:
                a, b = (u, v) if u <= v else (v, u)
                if (a, b) in pairs:
                    continue
                pairs.add((a, b))
                pbar.update(1)
                if len(pairs) >= num_pairs:
                    break

        rounds_without_progress = (
            rounds_without_progress + 1 if len(pairs) == before_round else 0
        )

    pbar.close()

    if len(pairs) < num_pairs:
        logger.warning(
            "Requested %d source-target pairs but sampled only %d.",
            num_pairs,
            len(pairs),
        )

    out = np.asarray(sorted(pairs), dtype=np.int64)
    logger.info("Sampled %d source-target candidate pair(s)", len(out))
    return out


def exhaustive_source_target_pairs(
    *,
    past_graph: nx.Graph,
    source_ids: set[int],
    target_ids: set[int],
    num_pairs: int,
    distance_min: int | None,
    distance_max: int | None,
    target_max_degree: int | None,
    logger: logging.Logger,
) -> np.ndarray:
    """Enumerate source-target candidate pairs up to num_pairs."""
    pairs: list[tuple[int, int]] = []
    targets = sorted(target_ids)
    target_degree_ok = {
        target: (
            target in past_graph
            and (target_max_degree is None or int(past_graph.degree[target]) <= int(target_max_degree))
        )
        for target in targets
    }
    seen: set[tuple[int, int]] = set()

    for u in tqdm(sorted(source_ids), desc="Enumerating candidate pairs"):
        if distance_min is not None or distance_max is not None:
            distances = nx.single_source_shortest_path_length(
                past_graph,
                u,
                cutoff=distance_max,
            )
            candidate_targets = [
                v
                for v, distance in distances.items()
                if v in target_ids
                and v != u
                and target_degree_ok.get(v, False)
                and not past_graph.has_edge(u, v)
                and (distance_min is None or distance >= distance_min)
                and (distance_max is None or distance <= distance_max)
            ]
        else:
            candidate_targets = targets

        for v in candidate_targets:
            if v == u or not target_degree_ok.get(v, False):
                continue
            if past_graph.has_edge(u, v):
                continue
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))
            if len(pairs) >= num_pairs:
                logger.warning(
                    "Stopping exhaustive enumeration at --num_candidates=%d.",
                    num_pairs,
                )
                return np.asarray(pairs, dtype=np.int64)

    out = np.asarray(pairs, dtype=np.int64)
    logger.info("Enumerated %d candidate pair(s)", len(out))
    return out


def count_source_target_pairs(
    *,
    past_graph: nx.Graph,
    source_ids: set[int],
    target_ids: set[int],
    distance_min: int | None,
    distance_max: int | None,
    target_max_degree: int | None,
    logger: logging.Logger,
) -> int:
    """Count eligible source-target pairs after graph/distance/degree filters."""
    targets = set(target_ids)
    seen: set[tuple[int, int]] = set()
    count = 0

    target_degree_ok = {
        target: (
            target in past_graph
            and (target_max_degree is None or int(past_graph.degree[target]) <= int(target_max_degree))
        )
        for target in targets
    }

    for u in tqdm(sorted(source_ids), desc="Counting eligible candidate pairs"):
        if distance_min is not None or distance_max is not None:
            distances = nx.single_source_shortest_path_length(
                past_graph,
                u,
                cutoff=distance_max,
            )
            candidate_iter = (
                v
                for v, distance in distances.items()
                if v in targets
                and v != u
                and target_degree_ok.get(v, False)
                and not past_graph.has_edge(u, v)
                and (distance_min is None or distance >= distance_min)
                and (distance_max is None or distance <= distance_max)
            )
        else:
            candidate_iter = (
                v
                for v in targets
                if v != u
                and target_degree_ok.get(v, False)
                and not past_graph.has_edge(u, v)
            )

        for v in candidate_iter:
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            count += 1

    logger.info("Eligible candidate pairs after all filters: %d", count)
    return count


def build_candidate_pairs(
    *,
    candidate_mode: str,
    graph: Graph,
    lookup: pd.DataFrame,
    works_path: str,
    concept_column: str,
    cutoff_year: int,
    num_pairs: int,
    seed: int,
    min_degree: int,
    max_degree: int | None,
    target_max_degree: int | None,
    allowed_ids: set[int] | None,
    anchor_concepts: str | None,
    anchor_file: str | None,
    source_modules: str | None,
    target_modules: str | None,
    emerging_start_year: int,
    emerging_min_count: int,
    distance_min: int | None,
    distance_max: int | None,
    exhaustive_candidates: bool,
    count_candidates: bool,
    max_pairs_per_source: int,
    logger: logging.Logger,
) -> np.ndarray:
    mode = str(candidate_mode).strip().lower()
    past_graph = graph.get_nx_graph(cutoff_year)

    all_vertices = set(
        int(v)
        for v in graph.get_vertices(
            until_year=cutoff_year,
            min_degree=min_degree,
            max_degree=max_degree,
        )
    )
    if allowed_ids is not None:
        all_vertices &= allowed_ids

    if mode == "random":
        return sample_unconnected_pairs(
            graph=graph,
            cutoff_year=cutoff_year,
            num_pairs=num_pairs,
            seed=seed,
            min_degree=min_degree,
            max_degree=max_degree,
            allowed_ids=allowed_ids,
            logger=logger,
        )

    modules_source = set(parse_string_list(source_modules)) or None
    modules_target = set(parse_string_list(target_modules)) or None

    if mode == "anchor":
        labels = load_anchor_labels(anchor_concepts, anchor_file)
        source_ids = resolve_concept_labels(labels, lookup=lookup, logger=logger)
        target_ids = set(all_vertices)
    elif mode == "module_cross":
        if not modules_source or not modules_target:
            raise ValueError(
                "candidate_mode='module_cross' requires --source_modules and --target_modules."
            )
        source_ids = collect_concept_ids_from_works(
            works_path=works_path,
            lookup=lookup,
            concept_column=concept_column,
            modules=modules_source,
            year_min=None,
            year_max=cutoff_year,
            logger=logger,
        )
        target_ids = collect_concept_ids_from_works(
            works_path=works_path,
            lookup=lookup,
            concept_column=concept_column,
            modules=modules_target,
            year_min=None,
            year_max=cutoff_year,
            logger=logger,
        )
    elif mode == "emerging":
        modules = modules_source
        source_ids = collect_emerging_concept_ids(
            works_path=works_path,
            lookup=lookup,
            concept_column=concept_column,
            cutoff_year=cutoff_year,
            emerging_start_year=emerging_start_year,
            emerging_min_count=emerging_min_count,
            modules=modules,
            logger=logger,
        )
        if modules_target:
            target_ids = collect_concept_ids_from_works(
                works_path=works_path,
                lookup=lookup,
                concept_column=concept_column,
                modules=modules_target,
                year_min=None,
                year_max=cutoff_year,
                logger=logger,
            )
        else:
            target_ids = set(all_vertices)
    elif mode == "distance":
        source_ids = set(all_vertices)
        target_ids = set(all_vertices)
        if distance_min is None and distance_max is None:
            raise ValueError("candidate_mode='distance' requires --distance_min or --distance_max.")
    else:
        raise ValueError(
            "candidate_mode must be one of: random, anchor, module_cross, emerging, distance"
        )

    source_ids = filter_ids_by_graph_state(
        source_ids,
        past_graph=past_graph,
        allowed_ids=allowed_ids,
        min_degree=min_degree,
        max_degree=max_degree,
    )
    target_ids = filter_ids_by_graph_state(
        target_ids,
        past_graph=past_graph,
        allowed_ids=allowed_ids,
        min_degree=min_degree,
        max_degree=max_degree,
    )
    logger.info(
        "Candidate mode=%s | sources=%d | targets=%d | distance=%s-%s | target_max_degree=%s",
        mode,
        len(source_ids),
        len(target_ids),
        distance_min,
        distance_max,
        target_max_degree,
    )

    if count_candidates:
        possible = count_source_target_pairs(
            past_graph=past_graph,
            source_ids=source_ids,
            target_ids=target_ids,
            distance_min=distance_min,
            distance_max=distance_max,
            target_max_degree=target_max_degree,
            logger=logger,
        )
        logger.info(
            "Using up to %d of %d eligible candidate pair(s).",
            min(int(num_pairs), int(possible)),
            int(possible),
        )

    if exhaustive_candidates:
        return exhaustive_source_target_pairs(
            past_graph=past_graph,
            source_ids=source_ids,
            target_ids=target_ids,
            num_pairs=num_pairs,
            distance_min=distance_min,
            distance_max=distance_max,
            target_max_degree=target_max_degree,
            logger=logger,
        )

    return sample_source_target_pairs(
        past_graph=past_graph,
        source_ids=source_ids,
        target_ids=target_ids,
        num_pairs=num_pairs,
        seed=seed,
        distance_min=distance_min,
        distance_max=distance_max,
        target_max_degree=target_max_degree,
        max_pairs_per_source=max_pairs_per_source,
        logger=logger,
    )


def load_or_sample_pairs(
    *,
    candidate_pairs_path: str | None,
    candidate_mode: str,
    graph: Graph,
    lookup: pd.DataFrame,
    works_path: str,
    concept_column: str,
    cutoff_year: int,
    num_pairs: int,
    seed: int,
    min_degree: int,
    max_degree: int | None,
    target_max_degree: int | None,
    distance_min: int | None,
    distance_max: int | None,
    exhaustive_candidates: bool,
    anchor_concepts: str | None,
    anchor_file: str | None,
    source_modules: str | None,
    target_modules: str | None,
    emerging_start_year: int,
    emerging_min_count: int,
    count_candidates: bool,
    max_pairs_per_source: int,
    allowed_ids: set[int] | None,
    logger: logging.Logger,
) -> np.ndarray:
    if candidate_pairs_path and Path(candidate_pairs_path).exists():
        logger.info("Loading candidate pairs from %s", candidate_pairs_path)
        obj = load_compressed(candidate_pairs_path)
        if isinstance(obj, dict) and "pairs" in obj:
            return np.asarray(obj["pairs"], dtype=np.int64)
        return np.asarray(obj, dtype=np.int64)

    pairs = build_candidate_pairs(
        candidate_mode=candidate_mode,
        graph=graph,
        lookup=lookup,
        works_path=works_path,
        concept_column=concept_column,
        cutoff_year=cutoff_year,
        num_pairs=num_pairs,
        seed=seed,
        min_degree=min_degree,
        max_degree=max_degree,
        target_max_degree=target_max_degree,
        distance_min=distance_min,
        distance_max=distance_max,
        exhaustive_candidates=exhaustive_candidates,
        count_candidates=count_candidates,
        anchor_concepts=anchor_concepts,
        anchor_file=anchor_file,
        source_modules=source_modules,
        target_modules=target_modules,
        emerging_start_year=emerging_start_year,
        emerging_min_count=emerging_min_count,
        max_pairs_per_source=max_pairs_per_source,
        allowed_ids=allowed_ids,
        logger=logger,
    )
    if candidate_pairs_path:
        save_compressed(
            {
                "pairs": pairs,
                "candidate_mode": candidate_mode,
                "cutoff_year": cutoff_year,
                "num_pairs_requested": num_pairs,
                "seed": seed,
                "min_degree": min_degree,
                "max_degree": max_degree,
                "target_max_degree": target_max_degree,
                "distance_min": distance_min,
                "distance_max": distance_max,
                "exhaustive_candidates": exhaustive_candidates,
                "source_modules": source_modules,
                "target_modules": target_modules,
                "emerging_start_year": emerging_start_year,
                "emerging_min_count": emerging_min_count,
            },
            candidate_pairs_path,
        )
        logger.info("Saved candidate pairs to %s", candidate_pairs_path)
    return pairs


def score_semantic(
    *,
    pairs: np.ndarray,
    model_path: str,
    embeddings_path: str,
    layers: list[int],
    dropout: float,
    batch_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> np.ndarray:
    logger.info("Loading semantic embeddings from %s", embeddings_path)
    embeddings = load_compressed(embeddings_path)

    logger.info("Loading semantic model from %s", model_path)
    model = SemanticPairNetwork(layers, dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), batch_size), desc="Semantic scoring"):
            chunk = pairs[start : start + batch_size]
            features = []
            for u, v in chunk:
                emb_u = np.asarray(embeddings[int(u)], dtype=np.float32)
                emb_v = np.asarray(embeddings[int(v)], dtype=np.float32)
                features.append(np.concatenate([emb_u, emb_v]))
            x = torch.tensor(np.asarray(features), dtype=torch.float32, device=device)
            preds = model(x).detach().cpu().numpy().reshape(-1)
            scores.extend(preds.tolist())

    return np.asarray(scores, dtype=np.float32)


def score_gnn(
    *,
    pairs: np.ndarray,
    graph_path: str,
    features_path: str,
    model_path: str,
    cutoff_year: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    logger: logging.Logger,
) -> np.ndarray:
    _require_pyg()
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    logger.info("Loading GNN model from %s", model_path)
    ckpt = torch.load(model_path, map_location=device)
    ckpt_model_cfg = ckpt.get("model") or {}
    ckpt_sampling_cfg = ckpt.get("sampling") or {}
    ckpt_features_cfg = ckpt.get("features") or {}

    model_cfg = _parse_config_str(
        None,
        defaults={
            "hidden_dim": int(ckpt_model_cfg.get("hidden_dim", 128)),
            "out_dim": int(ckpt_model_cfg.get("out_dim", 128)),
            "dropout": float(ckpt_model_cfg.get("dropout", 0.1)),
            "decoder": str(ckpt_model_cfg.get("decoder", "mlp")),
            "decoder_hidden_dim": int(ckpt_model_cfg.get("decoder_hidden_dim", 256)),
            "decoder_dropout": float(ckpt_model_cfg.get("decoder_dropout", 0.1)),
        },
    )
    sampling_cfg = _parse_config_str(
        None,
        defaults={
            "fanout1": int(ckpt_sampling_cfg.get("fanout1", 20)),
            "fanout2": int(ckpt_sampling_cfg.get("fanout2", 15)),
        },
    )
    features_cfg = _parse_config_str(
        None,
        defaults={
            "log1p": bool(ckpt_features_cfg.get("log1p", True)),
            "zscore": bool(ckpt_features_cfg.get("zscore", True)),
            "eps": float(ckpt_features_cfg.get("eps", 1e-6)),
        },
    )

    logger.info("Loading GNN node features from %s", features_path)
    v_features = load_node_feature_matrix(features_path, name="v_features", logger=logger)
    logger.info(
        "GNN node features loaded: shape=%s dtype=%s",
        tuple(v_features.shape),
        v_features.dtype,
    )
    if bool(features_cfg.get("log1p", True)):
        if float(np.min(v_features)) < 0.0:
            logger.warning("v_features contains negative values; skipping log1p transform")
        else:
            logger.info("Applying log1p transform to GNN node features")
            v_features = np.log1p(v_features)
    if bool(features_cfg.get("zscore", True)):
        logger.info("Applying z-score normalization to GNN node features")
        mean = v_features.mean(axis=0, keepdims=True)
        std = v_features.std(axis=0, keepdims=True)
        eps = float(features_cfg.get("eps", 1e-6))
        v_features = (v_features - mean) / (std + eps)

    num_nodes = int(v_features.shape[0])
    logger.info("Loading graph for GNN scoring from %s", graph_path)
    graph = Graph(graph_path)
    logger.info("Building GNN edge_index for cutoff_year=%s", cutoff_year)
    edge_index = build_edge_index_for_year(graph, int(cutoff_year), num_nodes=num_nodes)
    logger.info(
        "GNN edge_index built: shape=%s num_edges=%d",
        tuple(edge_index.shape),
        int(edge_index.shape[1]) if edge_index.ndim == 2 else 0,
    )
    logger.info("Creating PyG Data object for %d node(s)", num_nodes)
    pyg_data = Data(
        x=torch.from_numpy(v_features),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    logger.info("Preparing %d candidate pair(s) for GNN scoring", len(pairs))
    edge_label_index = torch.from_numpy(pairs.T).contiguous()
    edge_label = torch.zeros(len(pairs), dtype=torch.float32)
    logger.info(
        "Creating LinkNeighborLoader: batch_size=%d, fanout=(%d, %d), num_workers=%d",
        int(batch_size),
        int(sampling_cfg["fanout1"]),
        int(sampling_cfg["fanout2"]),
        int(num_workers),
    )
    loader = LinkNeighborLoader(
        pyg_data,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        num_neighbors=[int(sampling_cfg["fanout1"]), int(sampling_cfg["fanout2"])],
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
    )
    logger.info("LinkNeighborLoader created")

    logger.info("Initializing GNN encoder/decoder")
    encoder = SAGEEncoder(
        in_dim=int(v_features.shape[1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(model_cfg["out_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    decoder_kind = str(model_cfg.get("decoder", "mlp")).lower()
    if decoder_kind == "mlp":
        decoder = EdgeMLPDecoder(
            emb_dim=int(model_cfg["out_dim"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["decoder_dropout"]),
        ).to(device)
    elif decoder_kind == "dot":
        decoder = DotDecoder().to(device)
    else:
        raise ValueError("Unsupported GNN decoder in checkpoint.")

    logger.info("Loading GNN checkpoint weights")
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    scores: list[float] = []
    logger.info("Starting GNN scoring over %d candidate pair(s)", len(pairs))
    with torch.no_grad():
        for batch in tqdm(loader, desc="GNN scoring", leave=False):
            batch = batch.to(device, non_blocking=True)
            z = encoder(batch.x, batch.edge_index)
            u = batch.edge_label_index[0]
            v = batch.edge_label_index[1]
            logits = decoder(z[u], z[v])
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            scores.extend(probs.tolist())

    logger.info("Finished GNN scoring: %d score(s)", len(scores))
    return np.asarray(scores, dtype=np.float32)


def main(
    graph_path="data/graph/surgery.edges.pkl",
    lookup_path="data/table/lookup/surgery.lookup.csv",
    works_path="data/table/surgery.pubmed.gpt-5-nano.llm.csv",
    concept_column="llm_concepts",
    output_path="data/model/surgery/future/top_pairs_2026plus.csv",
    candidate_pairs_path="data/model/surgery/future/candidate_pairs_2025.pkl.gz",
    scores_path="data/model/surgery/future/scored_pairs_2026plus.pkl.gz",
    candidate_mode="random",
    cutoff_year=2025,
    num_candidates=1_000_000,
    top_k=10_000,
    seed=42,
    min_degree=1,
    max_degree=None,
    target_max_degree=None,
    distance_min=None,
    distance_max=None,
    exhaustive_candidates=False,
    anchor_concepts=None,
    anchor_file=None,
    source_modules=None,
    target_modules=None,
    emerging_start_year=2021,
    emerging_min_count=3,
    count_candidates=False,
    max_pairs_per_source=100,
    semantic_model_path="data/model/surgery/semantic/pubmedbert_2022.for_2026plus.pt",
    semantic_embeddings_path="data/model/surgery/semantic/pubmedbert_2025.current.pkl.gz",
    semantic_layers="[1536, 1024, 819, 10, 1]",
    semantic_dropout=0.1,
    gnn_model_path="data/model/surgery/gnn/graphsage_2022/model.pt",
    gnn_features_path="data/model/surgery/gnn/features_2025.pkl.gz",
    blend_weight_gnn=0.2,
    semantic_batch_size=10_000,
    gnn_batch_size=16_384,
    num_workers=8,
    device_name="auto",
    log_file="logs/surgery/future/score_future_pairs.log",
):
    """Score unconnected cutoff-year pairs and save the top prospective hypotheses."""
    logger = setup_logger(log_file)
    device = _select_device(device_name)
    logger.info("device: %s", device)
    logger.info("cutoff_year: %s", cutoff_year)
    logger.info("num_candidates: %s", num_candidates)
    logger.info("candidate_mode: %s", candidate_mode)

    max_degree = parse_optional_int(max_degree)
    target_max_degree = parse_optional_int(target_max_degree)
    distance_min = parse_optional_int(distance_min)
    distance_max = parse_optional_int(distance_max)
    exhaustive_candidates = parse_bool(exhaustive_candidates)
    count_candidates = parse_bool(count_candidates)

    graph = Graph(graph_path)
    lookup = load_lookup(lookup_path)
    id_to_concept = dict(zip(lookup["id"], lookup["concept"], strict=False))

    allowed_ids: set[int] | None = None
    semantic_embeddings = None
    if semantic_embeddings_path and Path(semantic_embeddings_path).exists():
        semantic_embeddings = load_compressed(semantic_embeddings_path)
        allowed_ids = embedding_ids(semantic_embeddings)

    if gnn_features_path and Path(gnn_features_path).exists():
        # Keep candidates inside the GNN feature matrix too if using GNN scores.
        gnn_features = load_compressed(gnn_features_path)
        if isinstance(gnn_features, dict) and "v_features" in gnn_features:
            n_gnn = len(gnn_features["v_features"])
            gnn_ids = set(range(n_gnn))
            allowed_ids = gnn_ids if allowed_ids is None else allowed_ids & gnn_ids

    pairs = load_or_sample_pairs(
        candidate_pairs_path=candidate_pairs_path,
        candidate_mode=candidate_mode,
        graph=graph,
        lookup=lookup,
        works_path=works_path,
        concept_column=concept_column,
        cutoff_year=int(cutoff_year),
        num_pairs=int(num_candidates),
        seed=int(seed),
        min_degree=int(min_degree),
        max_degree=max_degree,
        target_max_degree=target_max_degree,
        distance_min=distance_min,
        distance_max=distance_max,
        exhaustive_candidates=exhaustive_candidates,
        anchor_concepts=anchor_concepts,
        anchor_file=anchor_file,
        source_modules=source_modules,
        target_modules=target_modules,
        emerging_start_year=int(emerging_start_year),
        emerging_min_count=int(emerging_min_count),
        count_candidates=count_candidates,
        max_pairs_per_source=int(max_pairs_per_source),
        allowed_ids=allowed_ids,
        logger=logger,
    )

    semantic_scores = None
    if semantic_model_path and semantic_embeddings_path:
        layers = [int(x) for x in str(semantic_layers).strip("[]").split(",")]
        semantic_scores = score_semantic(
            pairs=pairs,
            model_path=semantic_model_path,
            embeddings_path=semantic_embeddings_path,
            layers=layers,
            dropout=float(semantic_dropout),
            batch_size=int(semantic_batch_size),
            device=device,
            logger=logger,
        )

    gnn_scores = None
    if gnn_model_path and gnn_features_path:
        gnn_scores = score_gnn(
            pairs=pairs,
            graph_path=graph_path,
            features_path=gnn_features_path,
            model_path=gnn_model_path,
            cutoff_year=int(cutoff_year),
            batch_size=int(gnn_batch_size),
            num_workers=int(num_workers),
            device=device,
            logger=logger,
        )

    if semantic_scores is None and gnn_scores is None:
        raise ValueError("At least one of semantic_model_path or gnn_model_path is required.")

    if semantic_scores is not None and gnn_scores is not None:
        w = float(blend_weight_gnn)
        blend_scores = w * gnn_scores + (1.0 - w) * semantic_scores
    elif semantic_scores is not None:
        blend_scores = semantic_scores
    else:
        blend_scores = gnn_scores

    logger.info("Ranking %d scored candidate pair(s)", len(blend_scores))
    order = np.argsort(-blend_scores, kind="mergesort")
    top_idx = order[: int(top_k)]
    logger.info("Preparing top-%d output table", len(top_idx))

    logger.info("Loading cutoff graph for output annotations")
    past_graph = graph.get_nx_graph(int(cutoff_year))
    logger.info("Computing degree annotations for top-%d pair(s)", len(top_idx))
    top_degrees_a = [int(past_graph.degree[int(i)]) if int(i) in past_graph else 0 for i in pairs[top_idx, 0]]
    top_degrees_b = [int(past_graph.degree[int(i)]) if int(i) in past_graph else 0 for i in pairs[top_idx, 1]]
    logger.info("Computing graph-distance annotations for top-%d pair(s)", len(top_idx))
    top_distances = annotate_pair_distances(
        past_graph,
        pairs[top_idx],
        cutoff=distance_max,
        logger=logger,
    )

    logger.info("Building top-pairs dataframe")
    df = pd.DataFrame(
        {
            "rank": np.arange(1, len(top_idx) + 1),
            "concept_a_id": pairs[top_idx, 0],
            "concept_b_id": pairs[top_idx, 1],
            "concept_a": [id_to_concept.get(int(i), "<unknown>") for i in pairs[top_idx, 0]],
            "concept_b": [id_to_concept.get(int(i), "<unknown>") for i in pairs[top_idx, 1]],
            "degree_a": top_degrees_a,
            "degree_b": top_degrees_b,
            "graph_distance": top_distances,
            "score": blend_scores[top_idx],
        }
    )
    if semantic_scores is not None:
        df["semantic_score"] = semantic_scores[top_idx]
    if gnn_scores is not None:
        df["gnn_score"] = gnn_scores[top_idx]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing top-pairs CSV to %s", output_path)
    df.to_csv(output_path, index=False)
    logger.info("Saved top %d pairs to %s", len(df), output_path)

    if scores_path:
        logger.info("Writing full scored-pairs cache to %s", scores_path)
        save_compressed(
            {
                "pairs": pairs,
                "score": blend_scores,
                "semantic_score": semantic_scores,
                "gnn_score": gnn_scores,
                "cutoff_year": cutoff_year,
                "blend_weight_gnn": blend_weight_gnn,
                "candidate_mode": candidate_mode,
                "distance_min": distance_min,
                "distance_max": distance_max,
                "min_degree": min_degree,
                "max_degree": max_degree,
                "target_max_degree": target_max_degree,
            },
            scores_path,
        )
        logger.info("Saved full sampled scores to %s", scores_path)


if __name__ == "__main__":
    fire.Fire(main)

