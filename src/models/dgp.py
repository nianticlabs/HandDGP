from typing import Tuple

import torch
from loguru import logger


def construct_AB(
    kpts_2d: torch.Tensor, kpts_3d: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the A and B matrices where we minimize ||AX - B|| to solve for the transformation matrix.

    Args:
        kpts_2d (torch.Tensor): 2D keypoints tensor of shape (B, N, 2).
        kpts_3d (torch.Tensor): Corresponding 3D keypoints tensor of shape (B, N, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The constructed A and B matrices.
    """
    batch_size, num_keypoints = kpts_2d.shape[:2]

    # Initialize A and B matrices
    A = torch.zeros((batch_size, num_keypoints * 2, 3), device=kpts_2d.device)
    B = torch.zeros((batch_size, num_keypoints * 2, 1), device=kpts_2d.device)

    # Construct A and B row by row
    A[:, 0::2, 0] = -1.0
    A[:, 1::2, 1] = -1.0
    A[:, 0::2, 2] = kpts_2d[:, :, 0]
    A[:, 1::2, 2] = kpts_2d[:, :, 1]

    B[:, 0::2, 0] = kpts_3d[:, :, 0] - kpts_3d[:, :, 2] * kpts_2d[:, :, 0]
    B[:, 1::2, 0] = kpts_3d[:, :, 1] - kpts_3d[:, :, 2] * kpts_2d[:, :, 1]

    return A, B


def dgp(
    kpts_3d: torch.Tensor,
    kpts_2d_img: torch.Tensor,
    weights: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Implements the Differentiable Global Positioning (DGP) module.

    Args:
        kpts_3d (torch.Tensor): 3D keypoints tensor of shape (B, N, 3).
        kpts_2d_img (torch.Tensor): 2D image keypoints tensor of shape (B, N, 2).
        weights (torch.Tensor): Weights for each keypoint, of shape (B, N).
        intrinsics (torch.Tensor): Camera intrinsics matrix of shape (B, 3, 3).

    Returns:
        torch.Tensor: The estimated translation vector of shape (B, 3, 1).
    """
    # Compute the inverse of the intrinsics matrix
    intrinsics_inv = torch.linalg.pinv(intrinsics)

    # Convert 2D image keypoints to homogeneous coordinates
    ones = torch.ones(
        (kpts_2d_img.shape[0], kpts_2d_img.shape[1], 1), device=kpts_2d_img.device
    )
    kpts_2d_hom = torch.cat([kpts_2d_img, ones], dim=2)

    # Normalize 2D keypoints
    kpts_2d_norm = torch.bmm(intrinsics_inv, kpts_2d_hom.permute(0, 2, 1)).permute(
        0, 2, 1
    )

    # Construct A and B matrices
    A, B = construct_AB(kpts_2d_norm, kpts_3d)

    # Adjust weights for each keypoint (repeat for each dimension)
    weights_expanded = torch.repeat_interleave(weights, 2, dim=-1)
    W05 = torch.diag_embed(weights_expanded)

    # Apply weights to A and B
    A_weighted = torch.bmm(W05, A)
    B_weighted = torch.bmm(W05, B)

    # Solve for X in AX = B using least squares
    try:
        transl = torch.linalg.lstsq(A_weighted, B_weighted).solution
    except RuntimeError as e:
        logger.warning("lstsq failed, falling back to pinv method: " + str(e))
        AT = A_weighted.transpose(1, 2)
        ATA = torch.bmm(AT, A_weighted)

        # Regularize to prevent singularity
        regularization = 1e-4 * ATA.mean() * torch.rand_like(ATA)
        ATA_inv = torch.linalg.pinv(ATA + regularization)

        transl = torch.bmm(ATA_inv, torch.bmm(AT, B_weighted))

    return transl.squeeze(-1)
