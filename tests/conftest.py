"""Categorise tests without changing the default collection."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_interactive_file_dialogs(monkeypatch, request):
    """Fail unexpected file dialogs instead of waiting for input on CI."""
    from Stoner.tools import widgets

    def unexpected_dialog(*args, **kwargs):
        pytest.fail(f"Unexpected file dialog in {request.node.nodeid}: {kwargs!r}")

    for mode in widgets.App.modes.values():
        monkeypatch.setitem(mode, "method", unexpected_dialog)


def pytest_collection_modifyitems(items):
    """Assign categories from the existing test layout and example names."""
    categories = {
        "core": "core", "analysis": "analysis", "folders": "folders",
        "image": "image", "plot": "plotting", "tools": "tools",
    }
    modules = {
        "test_core.py": "core", "test_analysis.py": "analysis",
        "test_util.py": "analysis", "test_fileformats.py": "formats",
        "test_hdf5.py": "formats", "test_doc_samples.py": "documentation",
    }
    for item in items:
        path = Path(str(item.path))
        category = next(
            (categories[part.lower()] for part in reversed(path.parts[:-1]) if part.lower() in categories),
            modules.get(path.name.lower()),
        )
        if category:
            item.add_marker(getattr(pytest.mark, category))
        if path.name == "test_widgets.py":
            item.add_marker(pytest.mark.gui)
            item.add_marker(pytest.mark.plotting)
        if category == "documentation":
            item.add_marker(pytest.mark.plotting)
            if Path(item.callspec.params["script"]).name == "lmfit_demo_url.py":
                item.add_marker(pytest.mark.network)
