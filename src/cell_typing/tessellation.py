from typing import List

from beartype import beartype
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm


@beartype
def tessellate(image: npt.NDArray[np.uint8],
               mask: npt.NDArray[np.uint8],
               side_size: int = 16) -> npt.NDArray[np.float32]:
    """Function that splits image into non-intersecting squares
    and calculates mean mask content.
    :param image: mask of desired marker to be split.
    :param mask:  mask for removing artefact or empty areas.
    0 must correspond to an area that shouldn't be included.
    :param side_size: side of tessellation square.
    :return: tessellated image.
    """
    if set(np.unique(image)) - {0, 255}:
        raise ValueError('Image must contain only 0 and 255.')
    if set(np.unique(mask)) - {0, 255}:
        raise ValueError('Mask must contain only 0 and 255.')
    if image.ndim != 2:
        raise ValueError('Image must have two dimensions.')
    if mask.ndim != 2:
        raise ValueError('Mask must have two dimensions.')

    image = np.where(image == 255, 1, 0)
    mask = np.where(mask == 255, 1, 0)

    x_tesselation = np.arange(0, image.shape[0], side_size, dtype='int32')
    y_tesselation = np.arange(0, image.shape[1], side_size, dtype='int32')
    reduced_image = np.add.reduceat(np.add.reduceat(image, x_tesselation, axis=0),
                                    y_tesselation, axis=1)
    reduced_mask = np.add.reduceat(np.add.reduceat(mask, x_tesselation, axis=0),
                                   y_tesselation, axis=1)
    reduced_mask = reduced_mask == (side_size ** 2)

    return (reduced_image / (side_size ** 2) * reduced_mask).astype(np.float32)


@beartype
def gather_parameters(masks: List[npt.NDArray[np.uint8]],
                      marker_names: List[str],
                      artefact_mask: npt.NDArray[np.uint8],
                      side_size: int = 16,
                      verbose: bool = False) -> pd.DataFrame:
    """Function that gathers marker mean percentages in each tessellation square
    for set of markers.
    :param masks: list of marker masks.
    :param artefact_mask: mask for removing artefact or empty areas.
    0 must correspond area that shouldn't be included.
    :param side_size: side of tessellation square.
    :param verbose: whether display progress bar or not.
    :return: pd.DataFrame, gathered tessellation percentages.
    """
    image_shape = artefact_mask.shape
    tessellations = []
    for mask, marker_name in tqdm(zip(masks, marker_names),
                                  disable=(not verbose),
                                  total=len(marker_names)):
        if mask.shape != image_shape:
            raise ValueError(f'Mask for {marker_name} has '
                             f'shape that don\'t match artefact mask.')
        try:
            tessellation = tessellate(mask, artefact_mask, side_size)
        except ValueError as exception:
            raise ValueError(f'Got `{exception}` for {marker_name}.')
        tessellations.append(tessellation)
    artefact_mask = tessellate(artefact_mask, artefact_mask, side_size)
    artefact_mask = (artefact_mask == 1).astype('uint8')

    expressions = []
    (x_coordinates,
     y_coordinates) = np.meshgrid(np.arange(artefact_mask.shape[1]),
                                  np.arange(artefact_mask.shape[0]))
    for mask in tessellations:
        expressions.append((mask * artefact_mask).ravel())
    table = np.vstack((x_coordinates.ravel(),
                       y_coordinates.ravel(),
                       artefact_mask.ravel(),
                       *expressions)).T
    parameters = pd.DataFrame(table, columns=['x', 'y', 'mask'] + marker_names)
    return parameters[parameters['mask'] == 1].reset_index(drop=True).drop('mask', axis=1)
