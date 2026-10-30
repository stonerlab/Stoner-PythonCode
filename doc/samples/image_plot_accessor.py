"""Display calibrated images, non-uniform meshes, contours and intensity histograms."""

import numpy as np

from Stoner import ImageFile

x = np.linspace(-3, 3, 61)
y = np.linspace(-2, 2, 41)
xx, yy = np.meshgrid(x, y)
values = 2*np.exp(-((xx-.5)**2 + (yy+.3)**2)/.8) + .1*xx
image = ImageFile(values)
image.mask[5:12, 8:16] = True
package = image.export_storage()
package.dataset = package.dataset.assign_coords(x=x, y=y)
package.dataset.x.attrs = {"long_name": "Horizontal", "units": "um"}
package.dataset.y.attrs = {"long_name": "Vertical", "units": "um"}
image = ImageFile.from_storage(package)

# A second image demonstrates mesh rendering without resampling its pixels.
uneven = image.export_storage()
uneven.dataset.coords["x"] = x**3/9
uneven.dataset.x.attrs = {"long_name": "Horizontal", "units": "um"}
uneven = ImageFile.from_storage(uneven)

with image.plot.style("stoner", extra={"font.sans-serif": ["DejaVu Sans"]}):
    figure = image.plot.new_figure(figsize=(9, 7))
    with image.plot.MultiPanel((2, 2), figure=figure) as panels:
        panels = panels.raveled_axes
        ax = image.plot(ax=panels[0], cmap="viridis", mask_color="red", mask_alpha=.35)
        image.plot.title = "Calibrated image and exclusions"
        figure.colorbar(ax.images[0], ax=ax, label="Intensity")

        image.plot.contour(ax=panels[1], levels=8, cmap="viridis")
        image.plot.title = "Contours at pixel centres"

        uneven.plot(ax=panels[2], cmap="viridis")
        uneven.plot.title = "Non-uniform coordinates: mesh"

        image.plot.hist(ax=panels[3], bins=32, histtype="stepfilled", color="steelblue")
        image.plot.title = "Eligible raw intensities"
        panels[3].set_aspect("auto")
    figure.canvas.draw()
