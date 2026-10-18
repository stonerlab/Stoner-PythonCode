"""Exercise real-data plotting and the removal of implicit Matplotlib proxies."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Stoner import Data, __home__
from Stoner.plot.utils import hsl2rgb


@pytest.fixture(autouse=True)
def isolated_figures():
    """Keep test-owned figures and styles isolated."""
    with plt.rc_context():
        yield
    plt.close("all")


def test_real_xray_data_supports_explicit_axes_and_native_formatting():
    data = Data(Path(__home__).parent / "sample-data/New-XRay-Data.dql")
    ax = data.plot()
    ax.set(xlabel="Angle", title="Diffraction", yscale="log")
    data.plot.line(ax=ax, color="red")
    assert len(ax.lines) == 2 and ax.get_title() == "Diffraction"
    assert ax.get_xlabel() == "Angle" and ax.get_yscale() == "log"
    assert len(plt.get_fignums()) == 1


def test_hsl_mapping_remains_available_to_image_and_field_code():
    np.testing.assert_equal(hsl2rgb(0.5, 0.5, 0.5), [[63, 191, 191]])


@pytest.mark.parametrize("name", ["plot_xy", "plot_xyz", "plot_matrix", "plot_xyuv", "plot_xyuvw",
                                        "plot_xyzuvw", "plot_voxels", "quiver_plot", "colormap_xyz", "contour_xyz",
                                        "image_plot", "figure", "subplot", "subplot2grid", "x2", "y2", "fig",
                                        "fignum", "template", "showfig", "ax_lines", "plt_legend", "fig_dpi",
                                        "xlabel", "labels", "_span_slice", "_col_label"])
def test_removed_plotting_names_are_absent_and_do_not_create_figures(name):
    data = Data(np.arange(8.).reshape(4, 2))
    with pytest.raises(AttributeError):
        getattr(data, name)
    assert name not in dir(data) and not plt.get_fignums()
