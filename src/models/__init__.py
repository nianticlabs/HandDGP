import sys
from pathlib import Path

stubs_dir = Path(__file__).resolve().parent.parent / "stubs"

if stubs_dir.exists():
    sys.path.append(str(stubs_dir))
else:
    raise ImportError(f"Stubs directory not found: {stubs_dir}")

PARAMS = {
    "resnet50": {"num_ch": 2048, "model": "resnet50"},
    "resnet101": {"num_ch": 2048, "model": "resnet101"},
}
