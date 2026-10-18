import os.path as path
import runpy

import matplotlib.pyplot as plt
import pytest

from Stoner.compat import listdir_recursive


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
def test_scripts(script, monkeypatch):
    """Run each example with an isolated working directory and figure lifecycle."""
    print(f"Trying script {script}")
    monkeypatch.chdir(datadir)
    plt.close("all")
    with plt.rc_context():
        try:
            runpy.run_path(script)
            figures = plt.get_fignums()
            assert figures, f"{script} Did not produce any figures !"
            for number in figures:
                plt.figure(number).canvas.draw()
            print("Done")
        finally:
            plt.close("all")


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
