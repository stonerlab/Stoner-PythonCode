"""Simple XMCD Data reduction example."""

# pylint: disable=invalid-name, redefined-outer-name,no-member
import re
from pathlib import Path

import numpy as np

# Normaliser with Stoner module
from Stoner import Data, DataFolder
from Stoner.analysis.fitting.models.generic import linear


def helicity(f):
    """Look for either pos or neg in the first column header."""
    if "pos" in f.column_headers[0]:
        return "Pos"
    if "neg" in f.column_headers[0]:
        return "Neg"
    return "Neither"


def position(f):
    """Look for meta data named magj1yins and us this rounded to 1dp as a key."""
    if "magj1yins" in f:
        if f["magj1yins"] == -8.5:
            return "Perp"
        return "Para"
    return "None"


def temp(f):
    """Get the temperature data in bins."""
    temperature = f.get("sensor_temp", f.get("itc2"))
    if temperature is not None:
        return str(np.round(temperature)) + "K"
    return "None"


def cycle(f):
    """If run >=100 then we're on the second cycle."""
    if f["run"] >= 49100:
        return "cycle2"
    return "cycle1"


def alt_norm(f, _, **kwargs):
    """Just do the normalisation per file and report run number and normalisation constants."""
    run = f["run"]
    signal = kwargs["signal"]
    lfit = kwargs["lfit"]
    rfit = kwargs["rfit"]
    ec = 0

    md = f.find_col(signal)
    parameters, _ = f.curve_fit(linear, ec, md, bounds=lambda x, row: lfit[0] <= x <= lfit[1])
    coeffs = parameters[::-1]
    linearfit = np.poly1d(coeffs)
    f.add_column(lambda r: r[md] - linearfit(r[ec]), "minus linear")
    highend = f.mean("minus", bounds=lambda r: rfit[0] <= r[ec] <= rfit[1])
    ret = [run]
    ret.extend(coeffs)
    ret.append(highend)
    return ret


def norm_group(pos, _, **kwargs):
    """Take the drain current for each file in group and builds an analysis file and works out the mean drain."""
    signal = kwargs.get("signal", "fluo")
    lfit = kwargs["lfit"]
    rfit = kwargs["rfit"]

    posfile = Data()
    posfile.metadata = pos[0].metadata
    posfile = posfile & pos[0].column(0)
    posfile.column_headers = ["Energy"]
    for f in pos:
        if not np.ma.allequal(f.column(0), pos[0].column(0)):
            raise ValueError("Grouped scans must use the same energy grid")
        print(str(f["run"]) + str(f.find_col(signal)))
        posfile = posfile & f.column(signal)
    posfile.add_column(lambda r: np.mean(r[1:]), "mean drain")
    ec = posfile.find_col("Energy")
    md = posfile.find_col("mean drain")
    parameters, _ = posfile.curve_fit(linear, ec, md, bounds=lambda x, row: lfit[0] <= x <= lfit[1])
    linearfit = np.poly1d(parameters[::-1])
    posfile.add_column(lambda r: r[md] - linearfit(r[ec]), "minus linear")
    highend = posfile.mean("minus", bounds=lambda r: rfit[0] <= r[ec] <= rfit[1])
    if not np.isfinite(highend) or highend == 0:
        raise ValueError("The normalisation window must contain a finite, non-zero edge step")
    ml = posfile.find_col("minus linear")
    posfile.add_column(lambda r: r[ml] / highend, "normalised")
    if "group_key" in kwargs:
        posfile[kwargs["group_key"]] = pos.key
    return posfile


def asym(grp, trail, **kwargs):
    """Take a group of two files for Pos and Neg helicity and calcualtes the Asymmetry ratio."""
    posfile = grp["Pos"]
    negfile = grp["Neg"]
    if not np.ma.allequal(posfile.column("Energy"), negfile.column("Energy")):
        raise ValueError("The two helicities must use the same energy grid")
    pdat = posfile.column("normalised")
    ndat = negfile.column("normalised")
    edat = posfile.column("Energy")
    ret = posfile.clone
    ret.del_column(list(range(ret.shape[1])))
    ret = ret & edat & pdat & ndat
    ret.column_headers = ["Energy", "I+", "I-"]
    ret.add_column(lambda r: (r[1] - r[2]), "Asymmetry")
    ret.filename = "-".join(trail)
    if "group_key" in kwargs:
        ret[kwargs["group_key"]] = grp.key
    ret["title"] = ret.filename
    if "save" in kwargs and kwargs["save"]:
        ret.save()
    return ret


def collate(grp, trail, **kwargs):
    """Gather all the data up again."""
    grp.sort()
    final = Data()
    final.add_column(grp[0].column("Energy"), "Energy")
    for g in grp:
        final.add_column(g.column("Asym"), g.get("title", g.filename))
    if "group_key" in kwargs:
        final[kwargs["group_key"]] = grp.key
    final["path"] = trail
    if "save" in kwargs and kwargs["save"]:
        final.save(kwargs["filename"])
    return final


def reduce(directory, startrun=49173, endrun=52000, signal="fluo", lfit=(615, 630), rfit=(660, 670),
           output_directory=None):
    """Reduce grouped helicity scans and optionally save the compiled position tables.

    Notes:
        The reported Asymmetry remains the normalised intensity difference I+ - I-.
        The two helicities must have the same energy grid. Output is written only
        when output_directory is supplied; input scans are not overwritten.
    """
    pattern = re.compile(r"i10-(?P<run>\d*)\.dat")
    folder = directory.clone if isinstance(directory, DataFolder) else DataFolder(directory, pattern=pattern)
    folder.sort("run")
    folder.filterout(lambda f: f["run"] > endrun or f["run"] < startrun)
    folder.group([position, temp, helicity])
    folder.walk_groups(norm_group, group=True, replace_terminal=True,
                       walker_args={"signal": signal, "lfit": lfit, "rfit": rfit})
    folder.walk_groups(asym, group=True, replace_terminal=True, walker_args={"group_key": "temperature"})
    folder.walk_groups(collate, group=True, replace_terminal=True, walker_args={"group_key": "Position"})
    if output_directory is not None:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
        for data in folder:
            data.save(output / f"Compiled Fluorescence Data {data['Position']}.txt")
    return folder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the i10 run files")
    parser.add_argument("--output-directory", type=Path, help="Save compiled tables in this directory")
    args = parser.parse_args()
    reduce(args.directory, output_directory=args.output_directory)
