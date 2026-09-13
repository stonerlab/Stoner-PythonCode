"""Smoke-test an installed distribution while using checkout files only as fixtures."""

from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
import sys
import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from Stoner import Data, ImageFile
import Stoner


def main():
    """Reject source imports, then exercise numerical, loader, plot and image paths."""
    checkout = Path(sys.argv[1]).resolve()
    installed = Path(Stoner.__file__).resolve()
    if installed.is_relative_to(checkout):
        raise RuntimeError(f'Imported checkout instead of installed distribution: {installed}')

    if version('Stoner') != Stoner.__version__:
        raise RuntimeError('Installed metadata and runtime versions differ')
    check_data_and_resources(checkout)
    print(f'Installed-package checks passed: {installed}')


def check_data_and_resources(checkout):
    """Exercise package APIs and resources; callable for a source-level preflight too."""
    data = Data(np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]), setas='xy')
    if data.shape != (3, 2):
        raise RuntimeError(f'Unexpected constructed data shape: {data.shape}')
    loaded = Data(checkout / 'sample-data' / 'TDI_Format_RT.txt')
    if loaded.shape[0] <= 0 or loaded.shape[1] < 2:
        raise RuntimeError(f'Unexpected sample data shape: {loaded.shape}')

    styles = files('Stoner.plot').joinpath('stylelib')
    for source in (checkout / 'Stoner' / 'plot' / 'stylelib').glob('*.mplstyle'):
        if not styles.joinpath(source.name).is_file():
            raise RuntimeError(f'Missing plot style: {source.name}')
    plt.style.use(str(styles.joinpath('default.mplstyle')))
    data.plot()
    plt.close('all')

    resources = files('Stoner.Image').joinpath('tessdata')
    for name in ('eng.traineddata', 'equ.traineddata', 'kerr-patterns.txt'):
        with resources.joinpath(name).open('rb') as stream:
            if not stream.read(1):
                raise RuntimeError(f'Empty OCR resource: {name}')

    image = ImageFile(np.arange(100, dtype=float).reshape(10, 10))
    with tempfile.TemporaryDirectory(prefix='stoner-installed-smoke-') as directory:
        target = Path(directory) / 'roundtrip.tiff'
        image.save(target)
        restored = ImageFile(target)
        if not np.allclose(restored.image, image.image):
            raise RuntimeError('Image pixels changed during the TIFF round trip')


if __name__ == '__main__':
    main()
