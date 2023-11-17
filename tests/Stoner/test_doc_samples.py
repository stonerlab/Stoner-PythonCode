import os.path as path
import os
import runpy
from Stoner.compat import listdir_recursive
import matplotlib.pyplot as plt
from traceback import format_exc
import pytest
import warnings

warnings.filterwarnings("ignore")

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
datadir = path.join(pth, "doc", "samples")


def get_scripts():
    skip_scipts = ["plot_folder_demo"]
    scripts = [path.realpath(x) for x in listdir_recursive(datadir, "*.py") if not x.endswith("__init__.py")]
    scripts = {
        path.splitext(path.basename(x))[0]: x
        for x in scripts
        if path.splitext(path.basename(x))[0].lower() not in skip_scipts
    }
    scripts = list(dict(sorted(scripts.items())).values())
    return scripts


@pytest.mark.parametrize("script", get_scripts())
@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_scripts(script, monkeypatch):
    """Import each of the sample scripts in turn and see if they ran without error"""
    print(f"Trying script {script}")
    try:
        os.chdir(datadir)
        runpy.run_path(script)
        fignum = len(plt.get_fignums())
        assert fignum >= 1, f"{script} Did not produce any figures !"
        print("Done")
        plt.close("all")
    except Exception:
        error = format_exc()
        print(f"Failed with\n{error}")
        assert False, f"Script {script} failed with {error}"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
