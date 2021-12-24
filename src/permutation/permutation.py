from typing import Optional, Union

from beartype import beartype
import mxifpublic.permutation.permutation_c as permutation_c
import numpy as np
import numpy.typing as npt


@beartype
def adjacency_label_permutation(
    adjacency: Union[list, np.ndarray],
    cell_type: npt.NDArray[np.uint32],
    num_permutation: int = 10000,
    n_classes: Optional[int] = None,
    threads: int = 1,
    seed: int = 42,
):
    """
    Implementation of https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5617107/.
    :param adjacency: adjacency matrix in form of list of lists.
    Must be list of np.arrays or np.array of np.arrays. Each adjacency must be dtype == np.uint32.
    :param cell_type: np.array of cell type. dtype == np.uint32. Usage of LabelEncoder() from sklearn is recommended.
    :param num_permutation: Number of permutations for test.
    :param n_classes: Number of unique cells.
    :param threads: Define number of threads. If threads < 0 use all available.
    :param seed: Define seed. If seed < 0 will use time as seed.
    :return: Return contact matrix.
    """
    if not all([isinstance(x, np.ndarray) for x in adjacency]):
        raise TypeError("All elements in adjacency must be np.ndarray")

    if any([x.dtype != np.uint32 for x in adjacency]):
        raise TypeError("All elements in adjacency must be np.uint32 dtype.")

    if len(adjacency) != len(cell_type):
        raise Exception("adjacency and cell_type have to be same length")

    if n_classes is None:
        n_classes = np.max(cell_type) + 1

    return permutation_c.adjacency_label_permutation_c(
        list(adjacency), cell_type, n_classes, num_permutation, threads, seed
    )
