from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
from third_party.HandMesh.mobrecon.models.modules import linear_layer
from third_party.HandMesh.mobrecon.tools.kinematics import mano_to_mpii
from torchvision import transforms

from src.models import PARAMS
from src.models.decoder_3d import Decoder3D
from src.models.dgp import dgp

__all__ = ["HandDGP"]


class HandDGP(nn.Module):
    def __init__(
        self,
        batch_size: int,
        latent_size: int = 256,
        spiral_len: Tuple[int, int, int, int] = (9, 9, 9, 9),
        spiral_dilation: Tuple[int, int, int, int] = (1, 1, 1, 1),
        spiral_out_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        imagenet_pretrain: Union[bool, Path] = True,
        variant: str = "resnet50",
        input_size: int = 224,
    ):
        """
        HandDGP model class.

        Args:
            batch_size (int): Batch size for processing.
            latent_size (int): Latent size for the 3D decoder.
            spiral_len (Tuple[int, int, int, int]): Lengths of the spirals at different layers.
            spiral_dilation (Tuple[int, int, int, int]): Dilation values for the spirals.
            spiral_out_channels (Tuple[int, int, int, int]): Number of output channels at different spiral layers.
            imagenet_pretrain (Union[bool, Path]): Use ImageNet pretrained weights or a custom path to weights.
            variant (str): Variant of the ResNet model (e.g., "resnet50").
            input_size (int): Size of the input images.
        """
        super().__init__()

        if variant not in ["resnet50", "resnet101"]:
            raise ValueError("Expected ResNet variant to be 'resnet50' or 'resnet101'.")

        model_type = PARAMS[variant]["model"]
        num_ch = PARAMS[variant]["num_ch"]

        self.batch_size = batch_size
        self.num_keypoints = 21

        # Backbone network initialization
        self.backbone = timm.create_model(
            model_type,
            pretrained=imagenet_pretrain,
            features_only=True,
            num_classes=0,
            global_pool="",
        )

        # 3D Decoder
        self.decoder3d_latent = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=latent_size, kernel_size=1),
            nn.ReLU(),
        )

        # 2D Decoder
        self.decoder2d_latent = nn.Conv2d(
            in_channels=num_ch, out_channels=self.num_keypoints, kernel_size=1
        )

        self.decoder2d = nn.Sequential(
            linear_layer((input_size // 32) ** 2, input_size, bn=False),
            linear_layer(input_size, input_size // 2, bn=False),
            linear_layer(input_size // 2, 2, bn=False, relu=False),
        )

        self.decoder3d = Decoder3D(
            latent_size,
            spiral_out_channels,
            self.num_keypoints,
            batch_size,
            spiral_len=spiral_len,
            spiral_dilation=spiral_dilation,
        )

        # Normalizer for input images
        self.normaliser = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Weights Decoder
        weights_dim = 2048 // self.num_keypoints
        self.decoderw_latent = nn.Sequential(
            nn.Conv2d(in_channels=num_ch, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=weights_dim, kernel_size=1),
        )

        self.decoderw = nn.Sequential(
            nn.Linear(weights_dim * self.num_keypoints, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_keypoints),
        )

        self.weight_activation = nn.Sigmoid()

        # Joint regression matrix
        j_reg = np.load("third_party/HandMesh/template/j_reg.npy")
        self.j_reg = nn.Parameter(
            torch.from_numpy(mano_to_mpii(j_reg)).float().unsqueeze(0),
            requires_grad=False,
        )

    def _single_forward(
        self,
        x: torch.Tensor,
        intrinsics: torch.Tensor,
        hand_scale: Union[torch.Tensor, float] = 0.2,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Performs a forward pass through the model for a single batch.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            intrinsics (torch.Tensor): Camera intrinsics tensor of shape (B, 3, 3).
            hand_scale (Union[torch.Tensor, float]): Scaling factor for the hand model.

        Returns:
            Tuple containing 2D keypoints, 3D vertices, camera-space vertices, camera-space keypoints,
            2D vertices projected from 3D, 2D keypoints projected from 3D, translation vector, weights, and weight logits.
        """
        current_batch = x.shape[0]
        image_size = x.shape[-1]

        # Normalize input images
        x = self.normaliser(x)
        features = self.backbone(x)

        # Process 3D and 2D features
        latent_3d = self.decoder3d_latent(features[-1])
        latent_2d = self.decoder2d_latent(features[-1]).view(
            current_batch, self.num_keypoints, -1
        )

        # Decode keypoints and 3D vertices
        kpts_2d = self.decoder2d(latent_2d)
        verts3d = self.decoder3d(kpts_2d, latent_3d)
        kpts_3d = torch.bmm(
            self.j_reg.repeat(current_batch, 1, 1), verts3d * hand_scale
        )

        # Convert keypoints to image space
        kpts_2d_img = kpts_2d * image_size

        # Compute translation
        latent_w = self.decoderw_latent(features[-1])
        weights_features = self.grid_sample(latent_w, kpts_2d)
        weight_logits = self.decoderw(weights_features)
        weights = self.weight_activation(weight_logits)

        transl = dgp(kpts_3d, kpts_2d_img, weights, intrinsics[:, :3, :3])

        # Compute camera space 3D vertices and keypoints
        verts3d_cs = verts3d * hand_scale + transl.unsqueeze(1)
        kpts3d_cs = torch.bmm(self.j_reg.repeat(current_batch, 1, 1), verts3d_cs)

        # Project vertices and keypoints from 3D to 2D
        vertices_2d_from_3d = self.batch_project(verts3d_cs, intrinsics[:, :3, :3])
        kpts_2d_from_3d = self.batch_project(kpts3d_cs, intrinsics[:, :3, :3])

        return (
            kpts_2d,
            verts3d,
            verts3d_cs,
            kpts3d_cs,
            vertices_2d_from_3d / image_size,
            kpts_2d_from_3d / image_size,
            transl,
            weights,
            weight_logits,
        )

    def grid_sample(self, feat: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Samples the feature map using UV coordinates.

        Args:
            feat (torch.Tensor): Feature map tensor.
            uv (torch.Tensor): UV coordinates tensor.

        Returns:
            torch.Tensor: Sampled features.
        """
        uv2 = torch.clamp((uv - 0.5) * 2, -1.0, 1.0).unsqueeze(2)
        samples = torch.nn.functional.grid_sample(
            feat, uv2, padding_mode="zeros", mode="bilinear", align_corners=True
        )

        return samples.squeeze(-1).reshape(uv2.shape[0], -1)

    @staticmethod
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

    def forward(
        self,
        x: torch.Tensor,
        intrinsics: torch.Tensor,
        hand_scale: Union[torch.Tensor, float] = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the HandDGP model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).
            intrinsics (torch.Tensor): Camera intrinsics tensor of shape (B, 3, 3).
            hand_scale (Union[torch.Tensor, float]): Scaling factor for the hand model.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the model's outputs.
        """
        (
            keypoints2d,
            vertices3d,
            vertices3d_cs,
            kpts3d_cs,
            verts2d_from_3d,
            keypoints2d_from_3d,
            translation,
            weights,
            weight_logits,
        ) = self._single_forward(x, intrinsics, hand_scale)

        return {
            "keypoints2D": keypoints2d,
            "mesh_vertices3D": vertices3d,
            "mesh_vertices2D_from_3D": verts2d_from_3d,
            "keypoints2D_from_3D": keypoints2d_from_3d,
            "translation": translation,
            "mesh_vertices3D_cs": vertices3d_cs,
            "keypoints3D_cs": kpts3d_cs,
            "weights": weights,
            "weight_logits": weight_logits,
        }
