import logging
from datetime import date
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm

from materials_concepts.model.graph import Graph
from materials_concepts.utils.utils import save_compressed, setup_logger

logger = setup_logger(
    logging.getLogger(__name__), file="logs/pre_compute.log", level=logging.DEBUG
)


def get_node_features(graph_path, years, binary):
    logging.debug("Building graph")
    graph = Graph(graph_path)
    num_nodes = int(max(graph.vertices)) + 1

    degree_columns = []
    two_hop_columns = []

    for year in tqdm(years):
        logging.debug("Calculating features for year %s", year)
        edge_cutoff = date(int(year), 12, 31)
        adj = Graph.build_adj_matrix(
            graph.get_until(edge_cutoff),
            binary=binary,
            dim=num_nodes,
        )

        degrees = calc_degs(adj)

        # The original implementation used ``adj ** 2`` and then summed columns.
        # For column sums this is equivalent to ``adj.T @ degree``:
        #
        #   1^T A^2 = (1^T A) A
        #
        # This avoids materialising A^2, which can be much denser than A and can
        # exhaust memory for later-year biomedical graphs.
        degrees_squared = np.asarray(adj.T.dot(degrees)).ravel()

        degree_columns.append(degrees)
        two_hop_columns.append(degrees_squared)

    v_features = np.column_stack(degree_columns + two_hop_columns)
    print(v_features.shape)

    return v_features


def calc_degs(adj):
    return np.array(adj.sum(0))[0]


def main(
    graph_path="data/graph/edges_medium.pkl",
    output_path="data/model/combi/matrices_2016.pkl.gz",
    binary=True,
    years=[2010, 2013, 2016],
):
    logging.info("Calculating embeddings...")
    v_features = get_node_features(graph_path, years=years, binary=binary)

    logging.info("Saving matrices...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    store = {
        "binary": binary,
        "years": years,
        "v_features": v_features,
    }

    logging.info(f"Saving features of nodes ({years}) to {output_path}")
    save_compressed(store, output_path)


if __name__ == "__main__":
    fire.Fire(main)

