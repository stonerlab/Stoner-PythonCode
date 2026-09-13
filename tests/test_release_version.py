"""Check release tag validation against an isolated version source."""

from pathlib import Path
import runpy

import pytest


check_release_version = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / 'maintenance' / 'check-release-version.py')
)['check_release_version']


@pytest.mark.parametrize('version, tag', [
    ('1.2.3', 'v1.2.3'), ('1.2.3', '1.2.3'), ('1.2.3rc1', 'v1.2.3rc1'),
])
def test_release_version_reads_source_without_import(tmp_path, version, tag):
    package = tmp_path / 'Stoner'
    package.mkdir()
    (package / '__init__.py').write_text(
        f'raise RuntimeError("Package must not be imported")\n__version__ = {version!r}\n',
        encoding='utf-8',
    )
    assert check_release_version(tag, tmp_path) == version


@pytest.mark.parametrize('tag', ['v1.2.4', 'stable', '', 'vv1.2.3'])
def test_release_version_rejects_mismatched_tag(tmp_path, tag):
    package = tmp_path / 'Stoner'
    package.mkdir()
    (package / '__init__.py').write_text('__version__ = "1.2.3"\n', encoding='utf-8')
    with pytest.raises(ValueError, match='does not match source version'):
        check_release_version(tag, tmp_path)
