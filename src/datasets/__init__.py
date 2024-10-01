import gin
from torch.utils.data import Dataset

from src.datasets.freihand import FreiHAND


@gin.configurable
def fetch_dataset(dataset: gin.REQUIRED) -> Dataset:
    """Get the datamodule defined in the GIN config file"""
    return dataset
