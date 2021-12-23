from typing import List, Union

from beartype import beartype
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class CellsDataset(Dataset):
    """Dataset class for cell expression prediction. Generates
    separate cell instance images (nuclei, marker, mask of segment) for desired
    segment indexes.
    :param nuclei_image: np.ndarray, nuclei (typically DAPI or Hoechst) image,
    must have two dimensions (HxW). Must be uint8 or uint16.
    :param marker_image: np.ndarray, marker image (CD3, CD20, Ki67, etc),
    must have two dimensions (HxW). Must be uint8 or uint16.
    :param contours: List[np.ndarray], list of cells contours (opencv contour format).
    :param indexes: List[int], list of cells indexes to process.
    :param cell_shape: int, size of image on train stage in pixels.
    """
    @beartype
    def __init__(self,
                 nuclei_image: Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]],
                 marker_image: Union[npt.NDArray[np.uint8], npt.NDArray[np.uint16]],
                 contours: List[np.ndarray],
                 indexes: List[int],
                 cell_shape: int = 128):
        """Constructor method.
        """
        self.nuclei_image = nuclei_image.copy()
        self.marker_image = marker_image.copy()
        self.contours = contours
        self.cell_shape = cell_shape
        self.indexes = indexes

        if self.nuclei_image.ndim != 2:
            raise ValueError(f'Nuclei image must have two '
                             f'dimensions, now: {self.nuclei_image.ndim}.')

        if self.marker_image.ndim != 2:
            raise ValueError(f'Nuclei image must have two '
                             f'dimensions, now: {self.marker_image.ndim}.')

        self.nuclei_image = self.nuclei_image / np.iinfo(self.nuclei_image.dtype).max
        self.nuclei_image = np.pad(self.nuclei_image, ((cell_shape, cell_shape),
                                                       (cell_shape, cell_shape)),
                                   mode='symmetric')
        self.nuclei_image = self.nuclei_image[:, :, np.newaxis]

        self.marker_image = self.marker_image / np.iinfo(self.marker_image.dtype).max
        self.marker_image = np.pad(self.marker_image,
                                   ((self.cell_shape, self.cell_shape),
                                    (self.cell_shape, self.cell_shape)),
                                   mode='symmetric')
        self.marker_image = self.marker_image[:, :, np.newaxis]
        if self.nuclei_image.shape != self.marker_image.shape:
            raise ValueError('Width and height of marker and nuclei images must be the same.')

    def __len__(self):
        return len(self.indexes)

    @beartype
    def __getitem__(self,
                    index: int):
        contour_index = self.indexes[index]
        contour = self.contours[contour_index]
        min_x, min_y = contour.min(axis=0)
        max_x, max_y = contour.max(axis=0)

        cell_mask = np.zeros((self.cell_shape, self.cell_shape), dtype='float32')
        start_x = min_x + int((max_x - min_x) / 2) - int(self.cell_shape / 2) + self.cell_shape
        start_y = min_y + int((max_y - min_y) / 2) - int(self.cell_shape / 2) + self.cell_shape

        offset = (-start_x+self.cell_shape, -start_y+self.cell_shape)

        cell_mask = cv2.fillPoly(cell_mask, [contour], 1.0, offset=offset)
        cell_mask = cell_mask[:, :, np.newaxis]
        nuclei = self.nuclei_image[start_y: start_y+self.cell_shape,
                                   start_x: start_x+self.cell_shape]
        marker = self.marker_image[start_y: start_y+self.cell_shape,
                                   start_x: start_x+self.cell_shape]
        cell = np.dstack([nuclei, marker, cell_mask])

        cell = torch.Tensor(cell)
        cell = cell.permute(2, 0, 1)

        return cell, np.array([contour_index])


class CellPredictor:
    """Class for marker expression prediction. Uses
    neural network to process each cell separately.
    :param model: model for prediction.
    :param device: torch.device, device to compute on.
    :param dataloader: torch.DataLoader, dataloader instance (wrapped around CellsDataset).
    """
    @beartype
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 dataloader: DataLoader):
        """Constructor method.
        """
        self.cell_dataloader = dataloader
        self.model = model
        self.device = device

    @beartype
    def apply(self,
              marker_name: str) -> pd.DataFrame:
        """Main prediction function. Generates batches and passes them through
        network.
        :param marker_name: str, marker name.
        :return: pd.DataFrame, dataframe with two columns, indexes for cells
                 ('index') and probability of zero class ('marker_name_probability').

        """
        self.model.to(self.device)
        self.model.eval()
        annotations = []
        for (cells, indexes) in self.cell_dataloader:
            with torch.no_grad():
                predicted = self.model(cells.to(self.device))
                predicted = torch.nn.functional.softmax(predicted, dim=1)
                predicted_labels = predicted.cpu().numpy()
                for predicted_label, ix in zip(predicted_labels, indexes):
                    label = predicted_label[0]
                    annotation = {
                        'index': ix.numpy()[0],
                        f'{marker_name}_probability': label,
                    }
                    annotations.append(annotation)

        return pd.DataFrame(annotations)
