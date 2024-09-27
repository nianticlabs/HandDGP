import json
from pathlib import Path

import gin
import kornia
import numpy as np
import torch
from loguru import logger
from PIL import Image
from third_party.HandMesh.utils.preprocessing import generate_patch_image
from torch.utils.data import Dataset

from src.utils import ImageProcessor


@gin.configurable
class FreiHAND(Dataset):
    def __init__(
        self,
        input_size: int = 224,
        rectify_input_image: bool = True,
        canonical_focal: float = 1000.0,
        canonical_ppt: float = 112.0,
        data_dir: str = "data/freihand",
    ):
        """
        Initialize the FreiHAND dataset.

        Args:
            input_size (int): Size of input images after preprocessing.
            rectify_input_image (bool): Whether to rectify input images.
            canonical_focal (float): Canonical focal length.
            canonical_ppt (float): Canonical principal point (x and y).
            data_dir (str): Directory containing the dataset.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_name = "freihand"
        self.base_scale = 1.3
        self.input_size = input_size
        self.rectify_input_image = rectify_input_image
        self.canonical_focal = canonical_focal
        self.canonical_ppt = canonical_ppt
        self._load()
        logger.info(f"FreiHAND dataset loaded with {len(self.intrinsics)} samples.")

    def _load(self):
        data_dir_eval = self.data_dir / "FreiHAND_pub_v2_eval"
        intrinsics_file = data_dir_eval / "evaluation_K.json"
        scale_file = data_dir_eval / "evaluation_scale.json"
        keypoints3D_file = data_dir_eval / "evaluation_xyz.json"
        mesh_vertices_file = data_dir_eval / "evaluation_verts.json"

        with open(intrinsics_file, "r") as f:
            self.intrinsics = np.array(json.load(f), dtype=np.float32)
        with open(scale_file, "r") as f:
            self.hand_scale = np.array(json.load(f), dtype=np.float32)
        with open(keypoints3D_file, "r") as f:
            self.keypoints3D = np.array(json.load(f), dtype=np.float32)
        with open(mesh_vertices_file, "r") as f:
            self.mesh_vertices = np.array(json.load(f), dtype=np.float32)

        self.intrinsics.setflags(write=False)
        self.hand_scale.setflags(write=False)
        self.keypoints3D.setflags(write=False)
        self.mesh_vertices.setflags(write=False)

    def __len__(self) -> int:
        return len(self.intrinsics)

    def __getitem__(self, idx: int) -> dict:
        image_path = (
            self.data_dir
            / "FreiHAND_pub_v2_eval"
            / "evaluation"
            / "rgb"
            / f"{idx:08d}.jpg"
        )
        image = np.asarray(Image.open(image_path))
        original_image = image.copy()

        # Get camera intrinsics
        K = self.intrinsics[idx]
        principal_point = K[:2, 2]
        focal_length = np.array([K[0, 0], K[1, 1]], dtype=np.float32)

        # Define bounding box (centered, size 100x100)
        bbox = [image.shape[1] // 2 - 50, image.shape[0] // 2 - 50, 100, 100]
        w, h = bbox[2], bbox[3]
        center = [bbox[0] + w * 0.5, bbox[1] + h * 0.5]

        # Create square patch bbox
        max_wh = max(w, h)
        patch_bbox = (
            center[0] - 0.5 * max_wh,
            center[1] - 0.5 * max_wh,
            max_wh,
            max_wh,
        )

        # Generate patch image
        crop, img2bbox_transf, _, _, _ = generate_patch_image(
            cvimg=image,
            bbox=patch_bbox,
            scale=self.base_scale,
            rot=0.0,
            shift=(0.0, 0.0),
            do_flip=False,
            out_shape=(self.input_size, self.input_size),
            shift_wh=None,
            mask=None,
        )

        t_crop = ImageProcessor.preprocess(image=crop, input_size=self.input_size)

        t_K = torch.eye(4, dtype=torch.float32)
        focal_scaled = focal_length * self.input_size / (max_wh * self.base_scale)

        # Transform principal point
        principal_point_hom = np.array(
            [principal_point[0], principal_point[1], 1], dtype=np.float32
        )
        center_xy = img2bbox_transf @ principal_point_hom
        center_xy = center_xy[:2]

        t_K[0, 0] = float(focal_scaled[0])
        t_K[1, 1] = float(focal_scaled[1])
        t_K[:2, 2] = torch.from_numpy(center_xy)

        if self.rectify_input_image:
            # Rectify input image to canonical intrinsics
            k_normalized = np.array(
                [
                    [self.canonical_focal, 0.0, self.canonical_ppt, 0.0],
                    [0.0, self.canonical_focal, self.canonical_ppt, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            transformation = torch.from_numpy(
                k_normalized[:3, :3] @ np.linalg.inv(t_K.numpy()[:3, :3])
            )

            t_crop = kornia.geometry.transform.warp_affine(
                t_crop.unsqueeze(dim=0),
                transformation[:2, :3].unsqueeze(dim=0),
                dsize=(self.input_size, self.input_size),
            ).squeeze(dim=0)
            t_K = torch.from_numpy(k_normalized)
        else:
            transformation = torch.eye(3, dtype=torch.float32)

        t_keypoints3D = torch.tensor(self.keypoints3D[idx], dtype=torch.float32)
        t_mesh_vertices = torch.tensor(self.mesh_vertices[idx], dtype=torch.float32)
        hand_scale = torch.tensor(self.hand_scale[idx], dtype=torch.float32)

        return {
            "original_image": ImageProcessor.preprocess(original_image),
            "image": t_crop,
            "K": t_K,
            "keypoints3D": t_keypoints3D,
            "mesh_vertices3D": t_mesh_vertices,
            "hand_scale": hand_scale,
            "root": t_keypoints3D[0],
            "index": idx,
            "transformation": transformation,
        }
