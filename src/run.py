import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import gin
import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets import fetch_dataset
from src.runners import fetch_runner


@gin.configurable
def run_eval() -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs evaluation using the configured runner and dataset.

    Returns:
        Tuple containing the dataset name, predicted keypoints, vertices, and root positions.
    """
    runner = fetch_runner(gin.REQUIRED)
    dataset = fetch_dataset(gin.REQUIRED)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    predictions = runner.run(data_loader)

    keypoints, vertices, root_positions = [], [], []
    for pred in predictions:
        keypoints.append(pred["keypoints3D_cs"])
        vertices.append(pred["mesh_vertices3D_cs"])
        root_positions.append(pred["translation"])

    return (
        dataset.dataset_name,
        torch.cat(keypoints),
        torch.cat(vertices),
        torch.cat(root_positions),
    )


def save_predictions(
    output_dir: Path, keypoints: torch.Tensor, vertices: torch.Tensor
) -> None:
    """
    Saves the keypoints and vertices predictions to a JSON file in FreiHAND evaluation format.

    Args:
        output_dir (Path): The directory where the results should be saved.
        keypoints (torch.Tensor): The predicted keypoints.
        vertices (torch.Tensor): The predicted vertices.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    data = {
        "keypoints": [kp.tolist() for kp in keypoints],
        "vertices": [vert.tolist() for vert in vertices],
    }

    output_file = output_dir / "results.json"
    with output_file.open("w") as fo:
        json.dump(data, fo)

    logger.info(
        f"Dumped {len(keypoints)} joints and {len(vertices)} verts predictions to {output_file}"
    )


def main(hparams: Namespace) -> None:
    """
    Main function to run the evaluation and save the results.

    Args:
        hparams (Namespace): Parsed command line arguments.
    """
    gin.parse_config_file(hparams.config_file)
    dataset_name, kps, verts, root_positions = run_eval()

    output_dir = Path("outputs") / dataset_name
    save_predictions(output_dir, kps, verts)


if __name__ == "__main__":
    parser = ArgumentParser(description="Tester for HandDGP model.")
    parser.add_argument(
        "--config_file",
        type=Path,
        help="Path to the config file to run",
        default=Path("configs/test_freihand.gin"),
    )

    hparams = parser.parse_args()
    main(hparams)
