from typing import Dict, List, Optional

from PIL import ImageColor
from beartype import beartype
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from .utils import change_lightness, raise_for_incorrect_color


@beartype
def cell_typing_plot(
    cells: pd.Series,
    contours: List[npt.NDArray[np.int32]],
    palette: Dict[str, str],
    height: int,
    width: int,
    offset_x: int = 0,
    offset_y: int = 0,
    dark_background: bool = False,
    mask: Optional[npt.NDArray[np.uint8]] = None,
    contour_color_shift: Optional[float] = None,
) -> npt.NDArray[np.uint8]:
    """Draws a cell typing plot, returns raster image of defined shape.
    :param cells: Series with cell labels as values and
                  cell contour index as index.
    :param contours: list of cells contours. Each contour
                     must be np.int32 array.
    :param palette: dict with cell labels as keys and hex
                    colors as values.
    :param height: height of image. If you want to draw a whole image,
                   you should pass an image.shape[0] here.
    :param width: width of image. If you want to draw a whole image,
                   you should pass an image.shape[1] here.
    :param offset_x: offset along the first axis.
    :param offset_y: offset along the second axis.
    :param dark_background: defines whether to draw on black background
                            or white. It's ignored if mask parameter is
                            specified.
    :param mask: np.uint8 array with shape equal to desired image
                 shape. It can also be croped into image shape if it's
                 possible. Used as background if specified.
    :param contour_color_shift: parameter to specify different color
                                lightness of cell contours. Must be between
                                zero and one. If set to zero corresponds to
                                black color, if set to one corresponds to
                                white color.
    :return: image of defined shape (numpy array of uint8).
    """

    difference = set(cells.unique()) - set(palette.keys())
    if difference:
        raise ValueError(f"Cell types don't match palette for {difference}!")

    cells = cells.rename_axis("cell_index")
    cells = cells.reset_index(name="object_type")
    cells['cell_index'] = cells['cell_index'].astype(int)

    if cells['cell_index'].max() >= len(contours):
        raise ValueError(
                f"Incorrect indexes or contours: maximum "
                f"contour index ({cells['cell_index'].max()+1})"
                f" is greater than overall contours count "
                f"({len(contours)}).",
        )

    offset = (-offset_y, -offset_x)
    if (np.array(offset) > 0).any():
        raise ValueError(f"Offset values can't be negative! "
                         f"Now {offset_x=}, {offset_y=}.")

    shape = (height, width, 3)
    if (np.array(shape) <= 0).any():
        raise ValueError(
           f"Image shape can't be less than 1 along one of the dimensions! Current shape: {shape}.",
        )

    if mask is not None:
        if mask.shape == shape:
            pass
        elif mask.ndim != 3:
            raise ValueError("Mask must must be three dimensional.")
        elif mask.shape[-1] != 3:
            raise ValueError("Mask must have three channels.")
        elif (np.array(mask.shape[:2]) < np.array(shape[:2]) - offset[::-1]).any():
            raise ValueError("Not possible to match mask and image shapes.")
        else:
            mask = mask[offset_x : offset_x + height, offset_y : offset_y + width]
    rgb_palette = {}
    raise_for_incorrect_color(palette)
    for object_type in palette:
        color = ImageColor.getcolor(palette[object_type], "RGB")
        rgb_palette[object_type] = color

    if mask is not None:
        image = mask.copy()
    elif dark_background:
        image = np.zeros(shape, "uint8")
    else:
        image = np.full(shape, 255, "uint8")

    upper_constraint = np.array([offset_y + width, offset_x + height])
    lower_constraint = np.array([offset_y, offset_x])
    contours_indexes = [
        ix
        for ix, cnt in enumerate(contours)
        if (cnt.min(axis=0) < upper_constraint).all() and (cnt.max(axis=0) > lower_constraint).all()
    ]
    contours = np.array(contours, dtype='object')
    cells = cells[cells['cell_index'].isin(contours_indexes)]
    for object_type, object_data in cells.groupby("object_type"):
        type_indexes = object_data["cell_index"].to_list()
        contours_to_draw = contours[type_indexes]
        image = cv2.fillPoly(image, contours_to_draw, rgb_palette[object_type], offset=offset)

    if contour_color_shift is not None:
        shifted_palette = {
            object_type: change_lightness(rgb_palette[object_type], contour_color_shift) for object_type in palette
        }

        for object_type, object_data in cells.groupby("object_type"):
            type_indexes = object_data["cell_index"].to_list()
            contours_to_draw = contours[type_indexes]
            image = cv2.drawContours(image, contours_to_draw, -1, shifted_palette[object_type], offset=offset)

    return image


@beartype
def tessellation_plot(
    cells: pd.Series,
    x_coordinates: pd.Series,
    y_coordinates: pd.Series,
    palette: Dict[str, str],
    height: int,
    width: int,
    resize: Optional[int] = None,
    dark_background=False,
) -> npt.NDArray[np.uint8]:
    """Function for plotting tessellations.
    :param cells: Series with tessellations labels.
    :param x_coordinates: Series with corresponding to cells X coordinates.
    :param y_coordinates: Series with corresponding to cells Y coordinates.
    :param palette: color mapping for unique instances of cells.
    :param height: output height of the generated image.
    :param width: output width of the generated image.
    :param resize: upscale factor (typically corresponds to side size of tessellation square).
    :param dark_background: whether plot on black background or white.
    :return: plot with tessellations.
    """
    difference = set(cells.unique()) - set(palette.keys())
    if difference:
        raise ValueError(f"Cell types don't match palette for {difference}!")
    raise_for_incorrect_color(palette)

    cells = cells.reset_index(drop=True)
    cells.name = 'cell_type'
    x_coordinates = x_coordinates.astype(int).reset_index(drop=True)
    x_coordinates.name = 'X'
    y_coordinates = y_coordinates.astype(int).reset_index(drop=True)
    y_coordinates.name = 'Y'
    cells = pd.concat([cells, x_coordinates, y_coordinates],
                      axis=1)
    if len(cells) != len(cells.dropna()):
        raise ValueError("Either cells, x_coordinates or y_coordinates "
                         "contain NaN, or their shapes are inconsistent.")
    shape = (height, width, 3)
    if (np.array(shape) <= 0).any():
        raise ValueError(
           f"Image shape can't be less than 1 along one of the dimensions! Current shape: {shape}.",
        )
    if dark_background:
        image = np.zeros(shape, "uint8")
    else:
        image = np.full(shape, 255, "uint8")

    if resize is None:
        resize = 1
    if resize < 1:
        raise ValueError(f"Can only upscale image. Increase resize, now {resize}")
    cells = cells[(cells['Y'] < resize * (height - 1)) &
                  (cells['X'] < resize * (width - 1))]
    if dark_background:
        colored_image = np.zeros((shape[0] // resize, shape[1] // resize, 3), "uint8")
    else:
        colored_image = np.full((shape[0] // resize, shape[1] // resize, 3), 255, "uint8")
    for cell_type, cell_data in cells.groupby('cell_type'):
        colored_image[(cell_data['Y'], cell_data['X'])] = ImageColor.getcolor(palette[cell_type], 'RGB')

    if resize > 1:
        colored_image = cv2.resize(colored_image, (colored_image.shape[1] * resize,
                                                   colored_image.shape[0] * resize),
                                   interpolation=cv2.INTER_NEAREST).astype('uint8')
    image[:colored_image.shape[0], :colored_image.shape[1]] = colored_image
    return image
