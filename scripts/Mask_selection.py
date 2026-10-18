"""Demonstrate interactive image masking, normalisation and quantisation."""

from pathlib import Path

from Stoner import ImageFile


DEFAULT_FILE = Path(__file__).resolve().parents[1] / "sample-data" / "Sample_Image_2017-10-15_100.hdf5"


def demonstrate(filename=DEFAULT_FILE, select=True):
    """Return the normalised image and binary clone, optionally selecting a mask interactively."""
    image = ImageFile(filename)
    if select:
        image.mask.select()
    image.normalise(limits=(0.01, 0.99), scale_masked=True)
    image.imshow()
    binary = image.clone
    binary.quantize(output=[-1, 1], levels=[image.threshold_otsu()])
    binary.imshow()
    return image, binary


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    demonstrate()
    plt.show()
