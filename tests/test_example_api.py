"""Keep executable examples on the current public API, including dynamic calls."""

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = runpy.run_path(str(ROOT / "maintenance" / "check-example-api.py"))
pytestmark = pytest.mark.documentation


def test_all_documentation_and_scripts_use_current_api():
    """Audit every inventory entry, retaining explicit historical migration references."""
    report = CHECKER["audit"]()
    assert not report["findings"], report["findings"]


@pytest.mark.parametrize("code", [
    "from Stoner.compat import Hyperspy_ok",
    "from Stoner import ImageArray",
    "from Stoner.Image.core import ImageArray",
    "d.curve_fit(model)",
    "d.plot_xy(0, 1)",
    "image.imshow()",
    "d.fit.lmfit(model, xcol=0)",
    "d.plot.line(xcol=0, ycol=1)",
    "d.setas._get_cols()",
    'getattr(self, self.method.replace("-", "_"))',
])
def test_checker_detects_compatibility_calls(code):
    assert CHECKER["violations"](code)[0]


@pytest.mark.parametrize("code", [
    "from Stoner import Data, ImageFile",
    "d.fit.curve_fit(model, x=0, y=1)",
    "image.plot.imshow()",
    "ax.imshow(pixels)",
    "np.polyfit(x, y, 1)",
    "d.plot.annotate_fit(model)",
    "columns = d.setas.y",
])
def test_checker_accepts_current_api(code):
    assert not CHECKER["violations"](code)[0]


def test_dynamic_fitting_cannot_bypass_example_guard():
    """The source scan alone cannot infer a configuration-selected method name."""
    from Stoner import Data
    filename = str(ROOT / "scripts" / "audit_probe.py")
    with pytest.raises(AssertionError, match="compatibility API"):
        exec(compile("getattr(data, method)(None)", filename, "exec"),
             {"data": Data(), "method": "lmfit"})


def test_tex_and_rst_sources_are_both_checked(tmp_path):
    """Legacy calls in either source format fail the audit."""
    guide = tmp_path / "doc" / "UserGuide"
    guide.mkdir(parents=True)
    (guide / "guide.tex").write_text(
        "\\begin{lstlisting}\nStoner.DataFile()\n\\end{lstlisting}\n", encoding="utf-8")
    (guide / "current.rst").write_text("Example::\n\n    Stoner.DataFile()\n", encoding="utf-8")
    report = CHECKER["audit"](tmp_path)
    assert {finding["path"] for finding in report["findings"]} == {
        "doc/UserGuide/current.rst", "doc/UserGuide/guide.tex"}
