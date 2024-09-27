from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from third_party.HandMesh.conv.dsconv import DSConv
from third_party.HandMesh.mobrecon.models.modules import conv_layer
from third_party.HandMesh.utils.read import spiral_tramsform


class UpsampleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, indices: torch.Tensor):
        super(UpsampleConv, self).__init__()
        self.conv = DSConv(in_channels, out_channels, indices)
        self.conv.reset_parameters()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, up_transform: torch.Tensor) -> torch.Tensor:
        sliced_transform = up_transform[: x.shape[0], :, :]
        x2 = torch.bmm(sliced_transform, x)
        x3 = self.conv(x2)
        out = self.activation(x3)
        return out


class Decoder3D(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        out_channels: List[int],
        keypoints_num: int,
        batch_size: int,
        spiral_len: Tuple[int, int, int, int] = (9, 9, 9, 9),
        spiral_dilation: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ):
        super(Decoder3D, self).__init__()
        self.batch_size = batch_size
        self.spiral_indices, self.up_transform = self._initialize_transforms(
            spiral_len, spiral_dilation
        )
        self.num_vert = [u.shape[1] for u in self.up_transform] + [
            self.up_transform[-1].shape[2]
        ]

        self.preprocess_layer = conv_layer(
            feat_dim, out_channels[-1], 1, bn=False, relu=False
        )
        self.spiral_layers = self._create_spiral_layers(out_channels)
        self.final_layer = DSConv(out_channels[0], 3, self.spiral_indices[0])

        self.upsample_matrix = nn.Parameter(
            torch.ones([self.num_vert[-1], keypoints_num]) * 0.01, requires_grad=True
        )

    def _initialize_transforms(
        self,
        spiral_len: Tuple[int, int, int, int],
        spiral_dilation: Tuple[int, int, int, int],
    ):
        """
        Initialize the spiral indices and upsample transformations.

        Args:
            spiral_len (Tuple[int, int, int, int]): Length of the spiral at each layer.
            spiral_dilation (Tuple[int, int, int, int]): Dilation of the spiral at each layer.

        Returns:
            Tuple: A tuple containing the spiral indices and the upsample transformations.
        """
        spiral_indices, _, _, tmp = spiral_tramsform(
            Path("third_party") / "HandMesh" / "template" / "transform.pkl",
            None,
            None,
            spiral_len,
            spiral_dilation,
        )

        # Convert upsample transforms to dense and batch them
        up_transform = [
            torch.stack(
                [torch.from_numpy(sparse_matrix.todense(order="C"))] * self.batch_size,
                dim=0,
            )
            for sparse_matrix in tmp["up_transform"]
        ]

        up_transform = nn.ParameterList(
            [nn.Parameter(t, requires_grad=False) for t in up_transform]
        )
        return spiral_indices, up_transform

    def _create_spiral_layers(self, out_channels: List[int]) -> nn.ModuleList:
        """
        Create spiral layers for the decoder.

        Args:
            out_channels (List[int]): List of output channels for each layer.

        Returns:
            nn.ModuleList: A module list containing all spiral layers.
        """
        spiral_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            in_ch = out_channels[-idx] if idx > 0 else out_channels[-idx - 1]
            out_ch = out_channels[-idx - 1]
            spiral_layers.append(
                UpsampleConv(in_ch, out_ch, self.spiral_indices[-idx - 1])
            )
        return spiral_layers

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Remove the batch dimension for the up_transform state dict."""
        destination = super(Decoder3D, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        for k in list(destination.keys()):
            if "up_transform" in k:
                destination[k] = destination[k][0]
        return destination

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Broadcast the up_transform back to full batch size when loading."""
        for k, v in state_dict.items():
            if "up_transform" in k:
                state_dict[k] = torch.stack([v] * self.batch_size, dim=0)
        super(Decoder3D, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs
        )

    def grid_sample(
        self, feat: torch.Tensor, uv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Grid sample the feature map with the uv map.

        Args:
            feat (torch.Tensor): The feature map tensor.
            uv (torch.Tensor): The uv map tensor.

        Returns:
            Tuple: Sampled feature tensor, original feature tensor, and uv map.
        """
        uv = uv.unsqueeze(2)
        samples = torch.nn.functional.grid_sample(
            feat, uv, padding_mode="zeros", mode="bilinear", align_corners=True
        )
        return samples[:, :, :, 0].permute(0, 2, 1), samples, uv

    def put_keypoints_in_m1_to_1(self, keypoints_01: torch.Tensor) -> torch.Tensor:
        """
        Convert keypoints from [0, 1] range to [-1, 1] range.

        Args:
            keypoints_01 (torch.Tensor): Keypoints in [0, 1] range.

        Returns:
            torch.Tensor: Keypoints in [-1, 1] range.
        """
        return torch.clamp((keypoints_01 - 0.5) * 2, -1.0, 1.0)

    def forward(self, keypoints_01: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            keypoints_01 (torch.Tensor): Keypoints in [0, 1] range.
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: The decoded output tensor.
        """
        x0 = self.preprocess_layer(x)
        keypoints_m1_1 = self.put_keypoints_in_m1_to_1(keypoints_01)
        x2, _, _ = self.grid_sample(x0, keypoints_m1_1)

        x3 = torch.bmm(self.upsample_matrix.repeat(x2.size(0), 1, 1).to(x2.device), x2)

        for i, spiral_layer in enumerate(self.spiral_layers):
            x3 = spiral_layer(x3, self.up_transform[len(self.spiral_layers) - i - 1])

        out = self.final_layer(x3)
        return out
