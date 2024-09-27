from pathlib import Path
from typing import Dict, Optional, Tuple

import gin
import torch

from src.models.handdgp import HandDGP
from src.runners.runner import Runner


@gin.configurable
class HandDGPRunner(Runner):
    def __init__(
        self,
        batch_size: int = 32,
        latent_size: int = 256,
        spiral_len: Tuple[int, int, int, int] = (9, 9, 9, 9),
        spiral_dilation: Tuple[int, int, int, int] = (1, 1, 1, 1),
        spiral_out_channels: Tuple[int, int, int, int] = (32, 64, 128, 256),
        imagenet_pretrain: bool = True,
        ckpt_pretrain: Optional[Path] = None,
        variant: str = "resnet50",
        input_size: int = 224,
    ):
        """
        Initializes the HandDGPRunner class.

        Args:
            batch_size (int): Batch size for the model.
            latent_size (int): Size of the latent vector in the model.
            spiral_len (Tuple[int, int, int, int]): Lengths of the spirals at different layers.
            spiral_dilation (Tuple[int, int, int, int]): Dilation values for the spirals.
            spiral_out_channels (Tuple[int, int, int, int]): Number of output channels at different spiral layers.
            imagenet_pretrain (bool): Whether to use ImageNet pretrained weights for the model.
            ckpt_pretrain (Optional[Path]): Path to the pretrained model checkpoint.
            variant (str): The variant of the model architecture to use.
            input_size (int): Size of the input images.
        """
        super().__init__()

        self.model = HandDGP(
            batch_size=batch_size,
            latent_size=latent_size,
            spiral_len=spiral_len,
            spiral_dilation=spiral_dilation,
            spiral_out_channels=spiral_out_channels,
            variant=variant,
            imagenet_pretrain=imagenet_pretrain,
            input_size=input_size,
        )

        if ckpt_pretrain:
            self.load_model_weights(ckpt_pretrain)

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predicts the outputs for a batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of input data.

        Returns:
            Dict[str, torch.Tensor]: The prediction results, including 2D keypoints.
        """
        image_size = batch["image"].shape[2]
        network_output = self.model(
            batch["image"], intrinsics=batch["K"].clone(), hand_scale=0.2
        )

        keypoints_2d_pred = network_output["keypoints2D"] * image_size

        outputs_dict = {
            "keypoints2D": keypoints_2d_pred,
        }

        # Add the remaining outputs to the dictionary
        outputs_dict.update(
            {k: v for k, v in network_output.items() if k not in outputs_dict}
        )

        return outputs_dict
