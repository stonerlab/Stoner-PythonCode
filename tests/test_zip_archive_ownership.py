"""Preserve caller-owned ZIP handles while selecting a real TDI fixture member."""

from pathlib import Path

import numpy as np
import pytest

from Stoner import Data
from Stoner.formats.data import zip as zip_format

pytestmark = pytest.mark.formats


@pytest.mark.parametrize("handle_state", ["path", "open", "closed"])
def test_selected_member_preserves_archive_ownership(tmp_path, monkeypatch, handle_state):
    """Close newly opened archives, borrow live ones, and honour the selected member."""
    source = Path(__file__).parent / "Stoner" / "CoreTest.dat"
    filename = tmp_path / "selected.zip"
    original = zip_format.zf.ZipFile
    with original(filename, "w") as archive:
        archive.writestr("not-data.txt", "Not TDI text\n")
        archive.write(source, "data.txt")
    opened = []

    class TrackedArchive(original):
        """Record owned and borrowed archive handles without altering their behaviour."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(zip_format.zf, "ZipFile", TrackedArchive)
    handle = filename if handle_state == "path" else TrackedArchive(filename)
    if handle_state == "closed":
        handle.close()
    try:
        result = zip_format.load_zipfile(Data(), handle, member="data.txt")

        np.testing.assert_array_equal(result.to_numpy(), Data(source).to_numpy())
        assert Path(result.filename).name == "data.txt"
        assert all((archive.fp is not None) == (handle_state == "open") for archive in opened)
    finally:
        for archive in opened:
            archive.close()
