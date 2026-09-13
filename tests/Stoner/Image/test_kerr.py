# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

import os
import subprocess
import sys
from pathlib import Path
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import pytest

import Stoner
from Stoner import Data, __home__
from Stoner.Image.kerr import KerrImageFile, KerrStack
from Stoner.Image import kerr, kerrfuncs

Stoner.Options.multiprocessing = False

# data arrays for testing - some useful small images for tests

testdir = os.path.join(os.path.dirname(__file__), "kerr_testdata")
testdir2 = os.path.join(os.path.dirname(__file__), "coretestdata", "testims")
sample_data_dir = os.path.join(__home__, "../sample-data")


def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    # ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret


selfimage2 = KerrImageFile(os.path.join(testdir, "kermit3.png"))
selfimage3 = KerrImageFile(os.path.join(sample_data_dir, "testnormalsave.png"))
selfks = KerrStack(testdir2)


@pytest.mark.plotting
def test_kerr_ops():
    im = selfimage3.clone
    assert type(im.image) is np.ma.MaskedArray, "KerrImageFile not blessing the image property correctly"
    im1 = im.float_and_croptext()
    assert type(im1.image) is np.ma.MaskedArray, "Calling a crop routine without the _ argument returns a new KerrImageFile"
    im2 = im.float_and_croptext(_=True)
    assert im2 == im, "Calling crop method with _ argument changed the KerrImageFile"
    im = KerrImageFile(selfimage2.clone)
    im.float_and_croptext(_=True)
    mask = im.image.defect_mask_subtract_image()
    im[~mask] = np.mean(im.image[mask])
    _ = im
    assert mask.sum() == 343228, "Mask didn't work out right"
    im = KerrImageFile(selfimage2.clone)
    im.float_and_croptext(_=True)
    mask = im.image.defect_mask(radius=4)
    im[~mask] = np.mean(im.image[mask])
    selim2 = im
    assert mask.sum() == 342540, "Mask didn't work out right"
    selim2.level_image()
    selim2.remove_outliers()
    selim2.normalise()
    selim2.plot_histogram()
    selim2.imshow()
    assert len(plt.get_fignums()) == 2, "Didn't open the correct number of figures"
    plt.close("all")


@pytest.mark.ocr
def test_tesseract_ocr():
    if not kerr._tesseractable or which("tesseract") is None:
        pytest.skip("Optional pytesseract wrapper and Tesseract executable are required")
    image = KerrImageFile(os.path.join(testdir, "kermit3.png"), ocr_metadata=True)
    metadata = image.metadata
    assert metadata["ocr_scalebar_length_microns"] == pytest.approx(50.0)
    assert metadata["ocr_field"] == pytest.approx(-0.13, abs=0.01)
    assert metadata["ocr_scalebar_length_pixels"] == 189
    assert metadata["ocr_microns_per_pixel"] == pytest.approx(50.0 / 189)
    assert metadata["ocr_pixels_per_micron"] == pytest.approx(189 / 50.0)
    assert "ocr_field" not in selfimage2.metadata


@pytest.mark.parametrize("field_only", [False, True])
def test_ocr_uses_text_crops(monkeypatch, field_only):
    """Recognise each text region rather than the complete annotated image."""
    image = KerrImageFile(os.path.join(testdir, "kermit3.png"), asfloat=False, crop_text=False)
    calls = []

    def recognise(crop, key):
        assert crop.shape[0] in (13, 15)
        assert crop.shape[1] < 100
        calls.append(key)
        return {"ocr_field": -0.13, "ocr_scalebar_length_microns": 50.0}.get(key, "text")

    monkeypatch.setattr(KerrImageFile, "tesseractable", property(lambda self: True))
    monkeypatch.setattr(kerrfuncs, "_tesseract_image", recognise)
    image.ocr_metadata(field_only=field_only)
    assert image.metadata["ocr_field"] == -0.13
    assert len(calls) == (1 if field_only else 8)


@pytest.mark.parametrize("wrapper_available", [False, True])
def test_no_ocr_without_dependencies(monkeypatch, wrapper_available):
    """Keep ordinary image operations usable when either OCR dependency is absent."""
    monkeypatch.setattr(kerr, "_tesseractable", wrapper_available)
    monkeypatch.setattr(kerr, "which", lambda name: None, raising=False)
    image = KerrImageFile(os.path.join(testdir, "kermit3.png"))
    assert not image.tesseractable
    image.normalise()
    assert image.shape == (554, 672)
    image.ocr_metadata()
    assert np.isfinite(image).all()
    assert not any(key.startswith("ocr_") for key in image.metadata)


def test_import_without_ocr_wrapper(tmp_path):
    """Import and use Kerr images in a fresh process without the optional wrapper."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__home__).parent)
    filename = str(Path(testdir, "kermit3.png").resolve())
    code = (
        "import sys; sys.modules['pytesseract'] = None; from Stoner.Image.kerr import KerrImageFile; "
        f"import numpy as np; image = KerrImageFile({filename!r}); "
        "assert not image.tesseractable; image.normalise(); assert np.isfinite(image).all()"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_kerrstack():
    print("X" * 80 + "\n" + "Test Kerrstack")
    ks = selfks.clone
    ks.each.normalise(scale=(0, 1.0))
    assert np.min(ks.imarray) == 0.0 and np.max(ks.imarray) == 1.0, "KerrStack subtract failed min,max: {},{}".format(
        np.min(ks.imarray), np.max(ks.imarray)
    )
    d = ks.hysteresis()
    assert isinstance(d, Data), "hysteresis didn't return Data"
    assert d.data.shape == (len(ks), 2), "hysteresis didn't return correct shape"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
