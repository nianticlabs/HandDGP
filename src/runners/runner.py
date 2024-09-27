from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger
from tqdm import tqdm


class Runner:
    def __init__(self, model: torch.nn.Module = None):
        """
        Initializes the Runner class.

        Args:
            model (torch.nn.Module, optional): The model to be used for prediction. Default is None.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

    def check_model_defined(self) -> None:
        """
        Checks if the model is defined.
        Raises:
            NotImplementedError: If the model is not defined.
        """
        if self.model is None:
            raise NotImplementedError("Model is not defined.")

    def load_model_weights(self, ckpt: Path) -> None:
        """
        Loads pretrained weights into the model from a checkpoint file.

        Args:
            ckpt (Path): The path to the checkpoint file.

        Raises:
            NotImplementedError: If the model is not defined.
        """
        self.check_model_defined()
        logger.info(f"Loading pretrained weights from {ckpt}")
        model_checkpoint = torch.load(ckpt, map_location=self.device)
        state_dict = {
            key.replace("model.", ""): val
            for key, val in model_checkpoint["state_dict"].items()
        }
        self.model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded pretrained weights.")

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Predicts outputs for a single batch of data.

        Args:
            batch (Dict[str, torch.Tensor]): The input batch of data.

        Returns:
            Dict[str, Any]: The prediction results for the batch.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("The `predict_batch` method needs to be implemented.")

    def put_on_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Transfers a batch of data to the appropriate device (CPU/GPU).

        Args:
            batch (Dict[str, torch.Tensor]): The input batch of data.

        Returns:
            Dict[str, torch.Tensor]: The batch of data on the correct device.
        """
        return {key: val.to(self.device) for key, val in batch.items()}

    def run(self, dataloader: torch.utils.data.DataLoader) -> List[Dict[str, Any]]:
        """
        Runs the model on the entire dataset provided by the dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing batches of data.

        Returns:
            List[Dict[str, Any]]: A list of prediction results for the entire dataset.
        """
        self.check_model_defined()

        self.model.eval()
        self.model.to(self.device)

        predictions = []
        with torch.no_grad():
            for sample in tqdm(dataloader, desc="Running inference"):
                sample = self.put_on_device(sample)
                outputs = self.predict_batch(sample)
                predictions.append(outputs)

        return predictions
