from typing import Optional

import gin
import numpy as np
import torch


def batch_project(P: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of 3D points P into 2D using camera matrix K.

    Args:
        P (torch.Tensor): A batch of 3D points with shape (batch_size, num_points, 3).
        K (torch.Tensor): A batch of camera matrices with shape (batch_size, 3, 3).

    Returns:
        torch.Tensor: Projected 2D points with shape (batch_size, num_points, 2).
    """
    P_projected_ = torch.bmm(K, P.permute(0, 2, 1))
    P_projected = P_projected_[:, :2, :] / P_projected_[:, 2:, :]
    return P_projected.permute(0, 2, 1)


@gin.configurable
class ImageProcessor:
    def __init__(self):
        """
        Image processor pipeline defining image pre/post processing.
        """
        pass

    @staticmethod
    def preprocess(image: np.ndarray, input_size: Optional[int] = None) -> torch.Tensor:
        """
        Convert an image into a tensor with the right shape for model input.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            input_size (Optional[int]): If specified, resize the image to (input_size, input_size).

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image_ = torch.from_numpy(image).float() / 255.0
        image_ = image_.permute(2, 0, 1).unsqueeze(0)

        if input_size is not None:
            image_ = torch.nn.functional.interpolate(
                image_,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        return image_.squeeze(0)

    @staticmethod
    def unprocess(image: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch image tensor back to a NumPy uint8 array.

        Args:
            image (torch.Tensor): The image tensor to convert.

        Returns:
            np.ndarray: The unprocessed image as a NumPy array.
        """
        image_ = image.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
        return image_.astype(np.uint8)
