import unittest
import sys
import os.path as path
import os
from Stoner.compat import listdir_recursive
from importlib import import_module
import matplotlib.pyplot as plt
from traceback import format_exc
import pytest
import warnings
warnings.filterwarnings("ignore")

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
datadir=path.join(pth,"doc","samples")

if pth not in sys.path:
    sys.path.insert(0,pth)
if datadir not in sys.path:
    sys.path.insert(0,datadir)

def get_scripts():
    skip_scipts=["plot-folder-test"]
    scripts=[path.relpath(x,datadir).replace(path.sep,".") for x in listdir_recursive(datadir,"*.py") if not x.endswith("__init__.py")]
    ret=[]
    for ix,filename in enumerate(scripts):
        script=filename[:-3]
        if script in skip_scipts:
            continue
        ret.append(script)
    return sorted(ret)

scripts=get_scripts()

@pytest.mark.parametrize("script",scripts)
@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
def test_scripts(script):
    """Import each of the sample scripts in turn and see if they ran without error"""
    print(f"Trying script {script}")
    try:
        os.chdir(datadir)
        code=import_module(script)
        fignum=len(plt.get_fignums())
        assert fignum>=1,f"{script} Did not produce any figures !"
        print("Done")
        plt.close("all")
    except Exception:
        error=format_exc()
        print(f"Failed with\n{error}")
        assert False,f"Script {script} failed with {error}"

if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
