import sys
from pathlib import Path

import gin
from torch.utils.data import Dataset

# Add the MobRecon package to the system path
handmesh_path = (
    Path(__file__).resolve().parent.parent.parent / "third_party" / "HandMesh"
)
print(handmesh_path)
if str(handmesh_path) not in sys.path:
    sys.path.insert(0, str(handmesh_path))

from src.datasets.freihand import FreiHAND


@gin.configurable
def fetch_dataset(dataset: gin.REQUIRED) -> Dataset:
    """Get the datamodule defined in the GIN config file"""
    return dataset
