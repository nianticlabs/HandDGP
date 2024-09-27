import gin

from src.runners.runner import Runner
from src.runners.runner_handdgp import HandDGPRunner


@gin.configurable
def fetch_runner(runner: gin.REQUIRED) -> Runner:
    """Get the runner defined in the GIN config file"""
    return runner
