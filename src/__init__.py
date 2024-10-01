import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

stubs_dir = Path(__file__).resolve().parent / "stubs"


class StubsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("utils"):
            stub_path = os.path.join(stubs_dir, *fullname.split(".")[1:]) + ".py"
            if os.path.exists(stub_path):
                return importlib.util.spec_from_file_location(fullname, stub_path)
        return None


sys.meta_path.insert(0, StubsFinder())

handmesh_path = Path(__file__).resolve().parent.parent / "third_party" / "HandMesh"

sys.path.insert(0, str(handmesh_path))
