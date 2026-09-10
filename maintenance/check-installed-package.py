"""Smoke-test an installed distribution while using checkout files only as fixtures."""

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

    check_data_and_resources(checkout)
    print(f'Installed-package checks passed: {installed}')


def check_data_and_resources(checkout):
    """Exercise package APIs and resources; callable for a source-level preflight too."""
    data = Data(np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]]), setas='xy')
    assert data.shape == (3, 2)
    loaded = Data(checkout / 'sample-data' / 'TDI_Format_RT.txt')
    assert loaded.shape[0] > 0 and loaded.shape[1] >= 2

    styles = files('Stoner.plot').joinpath('stylelib')
    for source in (checkout / 'Stoner' / 'plot' / 'stylelib').glob('*.mplstyle'):
        assert styles.joinpath(source.name).is_file(), source.name
    plt.style.use(str(styles.joinpath('default.mplstyle')))
    data.plot()
    plt.close('all')

    resources = files('Stoner.Image').joinpath('tessdata')
    for name in ('eng.traineddata', 'equ.traineddata', 'kerr-patterns.txt'):
        with resources.joinpath(name).open('rb') as stream:
            assert stream.read(1), name

    image = ImageFile(np.arange(100, dtype=float).reshape(10, 10))
    with tempfile.TemporaryDirectory(prefix='stoner-installed-smoke-') as directory:
        target = Path(directory) / 'roundtrip.tiff'
        image.save(target)
        restored = ImageFile(target)
        assert np.allclose(restored.image, image.image)


if __name__ == '__main__':
    main()