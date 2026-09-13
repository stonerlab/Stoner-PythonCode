"""Keep the Stage 6 exit independent of the retired storage implementations."""

import ast
import importlib.util
from pathlib import Path

import numpy as np

import Stoner
from Stoner import Data, ImageFile
from Stoner.Image import ImageStack
from Stoner.Image.kerr import KerrImageFile


def test_retired_classes_are_absent_from_runtime_and_source():
    """A green functional suite must not depend on a retained array subclass."""
    import Stoner.core as core
    import Stoner.Image as images
    from Stoner.Image import core as image_core, kerr, stack

    removed = {"DataArray", "ImageArray", "KerrArray", "ImageStackMixin"}
    for module in (Stoner, core, images, image_core, kerr, stack):
        assert not removed.intersection(vars(module))
    assert importlib.util.find_spec("Stoner.core.array") is None
    for path in Path(Stoner.__file__).parent.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name not in removed, path
                for base in node.bases:
                    assert ast.unparse(base) not in {"np.ma.MaskedArray", "ma.MaskedArray", "np.ndarray"}, path
            elif isinstance(node, ast.ImportFrom):
                assert not removed.intersection(alias.name for alias in node.names), path


def test_numerical_exports_use_numpy_and_do_not_propagate_schema():
    """Array arithmetic stays native; committed state remains on the owner."""
    data = Data(np.arange(6.).reshape(3, 2), setas="xy")
    row = data[1]
    assert type(row) is np.ma.MaskedArray
    assert row.i == 1
    assert not hasattr(row.copy(), "i")
    assert type(data.data) is np.ma.MaskedArray
    assert "_data" not in data.__dict__
    for image_type in (ImageFile, KerrImageFile):
        image = image_type(np.arange(6.).reshape(2, 3))
        assert type(image.image) is np.ma.MaskedArray
        assert not hasattr(image.to_numpy() + 1, "metadata")
        assert "_image" not in image.__dict__
        assert not image.image.flags.writeable
    stack = ImageStack([image])
    assert type(stack.imarray) is np.ma.MaskedArray
    assert not hasattr(stack, "_stack")
