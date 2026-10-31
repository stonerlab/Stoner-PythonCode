"""Fit plane/quadratic reference backgrounds and flatten the entire calibrated image."""

import numpy as np

from Stoner import ROI, ImageFile
from Stoner.enums import FitOutput

x, y = np.linspace(-3, 3, 61), np.linspace(-2, 2, 41)
xx, yy = np.meshgrid(x, y)
feature = 5*np.maximum(1-((xx-.3)/.7)**2-((yy+.2)/.6)**2, 0)**2
noise = np.random.default_rng(123).normal(0, .003, xx.shape)

with ImageFile().plot.style("stoner", extra={"font.sans-serif": ["DejaVu Sans"]}):
    figure = ImageFile().plot.new_figure(figsize=(8.5, 4.2))
    with ImageFile().plot.MultiPanel((2, 3), figure=figure, adjust_figsize=False) as panels:
        axes = panels.raveled_axes
        for row, model in enumerate(("plane", "quadratic_surface")):
            background = 3+.3*xx-.15*yy
            if model == "quadratic_surface":
                background += .08*xx**2+.06*xx*yy+.04*yy**2
            image = ImageFile(background+feature+noise)
            package = image.export_storage()
            package.dataset = package.dataset.assign_coords(x=x, y=y)
            for axis in ("x", "y"):
                package.dataset[axis].attrs["units"] = "um"
            package.dataset.intensity.attrs["units"] = "mV"
            image = ImageFile.from_storage(package)
            image.mask[2:5, 3:8] = True

            # Two disjoint reference patches, outside the feature support.
            reference = image.image.intensity
            left = ROI.box(-3.1, -1.7, -2.1, 2.1, coordinates="native", reference=reference)
            right = ROI.box(1.8, 3.1, -2.1, 2.1, coordinates="native", reference=reference)
            image.roi.set(left | right)
            report = image.fit(model, sigma=.003, absolute_sigma=True, output=FitOutput.REPORT)
            fitted = image.fit.evaluate(report)
            corrected = image.fit.flatten(report)
            assert np.max(np.abs(corrected.to_numpy()-feature)) < .02

            for column, (shown, title) in enumerate(((image, model.replace("_", " ")),
                                                     (fitted, "Fitted background"), (corrected, "Corrected"))):
                ax = shown.plot(ax=axes[3*row+column], cmap="viridis")
                ax.set_title(title)
                figure.colorbar(ax.images[0], ax=ax, label="mV", shrink=.65, fraction=.05, pad=.04)
            image.plot.roi(ax=axes[3*row], color="white", linewidths=1)
    figure.canvas.draw()
