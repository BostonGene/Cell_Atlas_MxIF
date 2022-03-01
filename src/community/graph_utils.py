from beartype import beartype
import cv2
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
import torch.nn.functional as F
from typing import Callable, Union


@beartype
def generate_graph_adj_matrix(
    dataset: pd.DataFrame,
    center_x_col_name: str = "center_x",
    center_y_col_name: str = "center_y",
    compute_edge_lengths: bool = True,
) -> csr_matrix:
    """Generates graph based on contour centroids and edge distance matrix.
       Nodes are ordered according to dataset row order, so mapping is one to one.
    :param dataset: pd.DataFrame, which contains coordinates of contour centroids
    and columns of features to add for each node in graph.
    :param center_x_col_name: name of column in dataset, which contains contour centroid x coordinate
    :param center_y_col_name: name of column in dataset, which contains contour centroid y coordinate
    :param compute_edge_lengths: compute length(L2) of connected nodes
    :return: CSR sparse adjacency matrix and optional edge lengths array
    """
    coordinates_columns_difference = {center_x_col_name, center_y_col_name}.difference(
        dataset.columns
    )
    if coordinates_columns_difference:
        raise KeyError(
            f"Centroid columns must be present in dataset, keys not found: {coordinates_columns_difference}"
        )

    n_points = dataset.shape[0]
    point_coordinates = dataset[[center_x_col_name, center_y_col_name]].values
    del_triangulation = Delaunay(point_coordinates, qhull_options="QJ")
    indptr, indices = del_triangulation.vertex_neighbor_vertices

    adjacency_matrix = csr_matrix(
        (
            np.ones_like(indices, dtype=np.int32),
            indices,
            indptr,
        ),
        shape=(n_points, n_points),
    )

    if compute_edge_lengths:
        adj_row_idx, adj_col_idx = adjacency_matrix.nonzero()

        edge_lengths = np.linalg.norm(
            point_coordinates[adj_row_idx] - point_coordinates[adj_col_idx],
            axis=1,
            keepdims=True,
        ).ravel()

        adjacency_matrix = csr_matrix(
            (
                edge_lengths,
                indices,
                indptr,
            ),
            shape=(n_points, n_points),
        )
        return adjacency_matrix

    return adjacency_matrix


@beartype
def threshold_graph_edges_by_distance(
    adjacency_matrix: csr_matrix,
    distance_threshold: Union[int, float],
) -> csr_matrix:
    """
    Removes edges by distance threshold
    :param adjacency_matrix: graph sparse matrix with stored distances
    :param distance_threshold: threshold for cutoff
    """
    if not isinstance(adjacency_matrix, csr_matrix):
        raise ValueError(
            f"Adjacency matrix have to be scipy`s CSR matrix! Currently: {type(adjacency_matrix)}"
        )

    adjacency_matrix[adjacency_matrix > distance_threshold] = 0
    adjacency_matrix.eliminate_zeros()
    return adjacency_matrix


def calculate_edge_length_statistic(
    adjacency_matrix: csr_matrix,
    distance_treshold: float,
    stat_function: Callable[[np.array], float] = np.median,
):
    """
    Calculates node median distance to neighbours
    :param adjacency_matrix: Graph sparse matrix with stored distances
    :param distance_treshold: Default value for nods without neighbors
    :param stat_function: Statistic function
    :return: Aggregated statistic by edge
    """
    adj_row_idx, adj_col_idx = adjacency_matrix.nonzero()
    nonzero_data = np.array(adjacency_matrix[adj_row_idx, adj_col_idx].data).ravel()
    split_indexes = np.cumsum(np.bincount(adj_row_idx))[:-1]
    return np.array(
        [
            stat_function(x) if x.size else distance_treshold
            for x in np.split(nonzero_data, split_indexes)
        ]
    )


@beartype
def count_mask_pixels_in_radius(
    mask: np.ndarray,
    coordinates: np.ndarray,
    radius: int = 100,
) -> np.ndarray:
    """Calculates amount of mask pixels within certain radius.
    :param mask: np.ndarray, two-dimensional mask with
                 0-1 values.
    :param coordinates: np.ndarray, cells centroids
                        size of Nx2, where N is amount
                        of cells.
    :param radius: int, radius of interest.
    :return: np.ndarray, normalized amount of
             pixels.
    """
    if len(mask.shape) != 2:
        raise ValueError('Mask isn\'t two-dimensional.')

    if len(coordinates.shape) != 2:
        raise ValueError(f'Unknown coordinates format. Must be Nx2, '
                         f'now {coordinates.shape}')
    if coordinates.shape[1] != 2:
        raise ValueError(f'Unknown coordinates format. Second dimension'
                         f' must equal 2, now {coordinates.shape[1]}')
    if (coordinates < 0).any():
        raise ValueError('All coordinates must be non-negative.')

    height, width = mask.shape

    radius = int(np.round(radius))
    size = 2 * radius + 1

    neighborhood = np.zeros((coordinates.shape[0], 1),
                            np.float64)

    mask_values = set(np.unique(mask))
    if mask_values == {0}:
        return neighborhood

    if mask_values != {0, 1}:
        raise ValueError('Mask has to be binary.')

    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel,
               center=(radius, radius),
               radius=radius,
               color=1,
               thickness=-1)
    kernel_num_pixels = np.sum(kernel)

    for index, coordinate in enumerate(coordinates):
        coordinate = np.round(coordinate).astype(np.int32)

        min_x = np.abs(min(0, coordinate[0] - radius))
        min_y = np.abs(min(0, coordinate[1] - radius))

        dx = width - coordinate[0]
        if dx > radius:
            dx = radius + 1
        dy = height - coordinate[1]
        if dy > radius:
            dy = radius + 1

        max_x = dx + radius
        max_y = dy + radius

        min_x_m = max(0, coordinate[0] - radius)
        min_y_m = max(0, coordinate[1] - radius)
        max_x_m, max_y_m = coordinate + radius + 1
        if max_x_m > width:
            max_x_m = width
        if max_y_m > height:
            max_y_m = height
        neighborhood[index] = np.sum(mask[min_y_m:max_y_m, min_x_m:max_x_m] *
                                     kernel[min_y:max_y, min_x:max_x])
        if (max_x - min_x) != size or (max_y - min_y) != size:
            subkernel_num_pixels = kernel[min_y:max_y, min_x:max_x].sum()
            neighborhood[index] = neighborhood[index] / subkernel_num_pixels
        else:
            neighborhood[index] = neighborhood[index] / kernel_num_pixels

    return neighborhood
