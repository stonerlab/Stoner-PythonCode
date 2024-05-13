# -*- coding: utf-8 -*-
"""Test Util module"""


import sys
import os.path as path
import Stoner.Util as SU
from Stoner import Data
import pytest

pth = path.dirname(__file__)
pth = path.realpath(path.join(pth, "../../"))
sys.path.insert(0, pth)

def is_2tuple(x):
    """Return tru if x is a length two tuple of floats."""
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], float) and isinstance(x[1], float)


@pytest.mark.parametrize("meth", ["linear_intercept", "susceptibility", "delta_M"])
def test_hysteresis(meth):
    """Test the hysteresis analysis code."""
    x = SU.hysteresis_correct(
        path.join(pth, "./sample-data/QD-SQUID-VSM.dat"), setas="3.xy", h_sat_method=meth, saturated_fraction=0.25
    )
    assert (
        "Hc" in x and "Area" in x and "Hsat" in x and "BH_Max" in x and "BH_Max_H" in x
    ), "Hystersis loop analysis keys not present."

    assert is_2tuple(x["Hc"]) and x["Hc"][0] + 578 < 1.0, "Failed to find correct Hc in a SQUID loop"
    assert isinstance(x["Area"], float) and -0.0137 < x["Area"] < -0.0136, "Incorrect calculation of area under loop"


def test_failures():
    with pytest.raises(ValueError):
        _ = SU.hysteresis_correct(
            path.join(pth, "./sample-data/QD-SQUID-VSM.dat"),
            setas="3.xy",
            h_sat_method="bad",
            saturated_fraction=0.25,
        )


def test_split_up_down():
    testd = Data(path.join(pth, "./sample-data/QD-SQUID-VSM.dat"), setas="3.xy")
    fldr = SU.split_up_down(testd)
    x = SU.hysteresis_correct(fldr["falling"][0], saturated_fraction=0.25)
    assert (
        "Hc" in x and "Area" in x and "Hsat" in x and "BH_Max" in x and "BH_Max_H" in x
    ), "Hystersis loop analysis keys not present."

    assert x["Hc_mean"] - 570 < 1.0, "Failed to find correct Hc in a SQUID loop"
    assert isinstance(x["Area"], float) and -0.0055 < x["Area"] < -0.0054, "Incorrect calculation of area under loop"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
