"""Split Brookhaven scan files into individually loadable BNL files.

Existing matching scan files in the output directory are overwritten.
"""

from contextlib import ExitStack
from pathlib import Path

from Stoner import Data


def split_file(filename, output_directory=None):
    """Write and validate each scan, returning the paths of non-empty scans.

    Args:
        filename (str or pathlib.Path):
            Source file containing one or more numbered scans.
        output_directory (str or pathlib.Path or None):
            Destination directory. None uses BNLSplitFiles beside the source.

    Returns:
        list[pathlib.Path]:
            Validated scan paths in input order. Scan numbers need not start at one.

    Notes:
        Comment records are omitted, the preamble is retained as title.txt, and
        empty scans are removed. A one-point scan is valid two-dimensional data.
        The process working directory is unchanged.
    """
    filename = Path(filename)
    output = Path(output_directory) if output_directory is not None else filename.parent / "BNLSplitFiles"
    output.mkdir(parents=True, exist_ok=True)
    scans = []
    with filename.open(encoding="utf-8") as source, ExitStack() as stack:
        target = stack.enter_context((output / "title.txt").open("w", encoding="utf-8"))
        for line in source:
            if line.startswith("#S"):
                target.close()
                path = output / f"{int(line.split()[1])}.bnl"
                if path in scans:
                    raise ValueError(f"Duplicate scan number: {path.stem}")
                scans.append(path)
                target = stack.enter_context(path.open("w", encoding="utf-8"))
            if not line.startswith("#C"):
                target.write(line)
    valid = []
    for path in scans:
        # Aborted scans can have a header but no numerical rows.
        if not any(line.strip() and not line.lstrip().startswith("#")
                   for line in path.read_text(encoding="utf-8").splitlines()):
            path.unlink()
            continue
        data = Data(path, filetype="BNLFile")
        if len(data):
            valid.append(path)
    return valid


if __name__ == "__main__":
    directory = Path(input("Directory containing the data: ").strip())
    filename = input("Filename (including extension): ").strip()
    for scan in split_file(directory / filename):
        print(f"{scan.name} OK")
