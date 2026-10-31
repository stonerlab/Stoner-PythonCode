"""Reuse a calibrated background ROI for a reduction, outline and intensity histogram."""

import json

import numpy as np

from Stoner import ROI, ImageFile

x = np.linspace(-3, 3, 61)
y = np.linspace(-2, 2, 41)
xx, yy = np.meshgrid(x, y)
values = .1*xx + .05*yy + 2*np.exp(-((xx-.5)**2 + (yy+.3)**2)/.6)
image = ImageFile(values)
package = image.export_storage()
package.dataset = package.dataset.assign_coords(x=x, y=y)
for axis in ("x", "y"):
    package.dataset[axis].attrs["units"] = "um"
image = ImageFile.from_storage(package)
image.mask[3:6, 3:8] = True

reference = image.image.intensity
left = ROI.box(-3.1, -1.5, -2.1, 2.1, coordinates="native", reference=reference)
right = ROI.box(2, 3.1, -2.1, 2.1, coordinates="native", reference=reference)
background = left | right
record = json.dumps(background.to_dict())
image.roi.set(ROI.from_dict(json.loads(record)))
assert image.roi.active
assert image.roi.resolve().values.sum() < values.size
background_mean = image.mean()
whole_mean = image.mean(where=ROI.all())
assert background_mean < whole_mean

with image.plot.style("stoner", extra={"font.sans-serif": ["DejaVu Sans"]}):
    figure = image.plot.new_figure(figsize=(8.5, 3.5))
    with image.plot.MultiPanel((1, 2), figure=figure, adjust_figsize=False) as panels:
        ax = image.plot(ax=panels[0], cmap="viridis", mask_color="red", mask_alpha=.4)
        image.plot.roi(ax=ax, color="white", linewidths=1.5)
        ax.set_title("Reference patches in the whole image")
        figure.colorbar(ax.images[0], ax=ax, label="Intensity")
        image.plot.hist(ax=panels[1], where=ROI.all(), bins=35, histtype="step",
                        color="gray", label="Whole image")
        image.plot.hist(ax=panels[1], bins=35, histtype="stepfilled", alpha=.5,
                        color="steelblue", label="Current ROI")
        panels[1].set_title(f"ROI mean {background_mean:.3f}; whole mean {whole_mean:.3f}")
        panels[1].legend()
    figure.canvas.draw()

# Clearing selection never removes persistent exclusions.
image.roi.clear()
assert not image.roi.active and np.asarray(image.mask).sum() == 15
