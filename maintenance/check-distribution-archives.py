"""Validate the contents of CI-built wheel and source archives without extracting them."""

from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


def check(names, wheel):
    """Require runtime assets and licence files, and reject interpreter/build caches."""
    paths = [PurePosixPath(name) for name in names]
    assert not any('__pycache__' in path.parts for path in paths)
    assert not any(path.name.endswith(('.pyc', '.pyo')) or '.pyc.' in path.name or '.pyo.' in path.name for path in paths)
    assert not any('doc/_build/' in str(path) for path in paths)
    assert any(path.name == 'COPYING' for path in paths)
    assert any(path.name == 'LICENSE.md' for path in paths)
    repo = Path(__file__).resolve().parents[1]
    for pattern in ('Stoner/plot/stylelib/*.mplstyle', 'Stoner/Image/tessdata/*'):
        for source in repo.glob(pattern):
            if source.is_file():
                relative = source.relative_to(repo).as_posix()
                assert any(str(path) == relative or str(path).endswith('/' + relative) for path in paths), relative
    assert any(str(path).endswith('Stoner/tools/tests.py') for path in paths)
    if wheel:
        assert not any(path.parts[0] in ('tests', 'doc', 'sample-data', 'maintenance') for path in paths)


def main():
    output = Path(sys.argv[1])
    wheels, sources = list(output.glob('*.whl')), list(output.glob('*.tar.gz'))
    assert len(wheels) == len(sources) == 1, (wheels, sources)
    with zipfile.ZipFile(wheels[0]) as archive:
        check(archive.namelist(), wheel=True)
    with tarfile.open(sources[0], 'r:gz') as archive:
        check(archive.getnames(), wheel=False)
    print('Wheel and source archive content checks passed.')


if __name__ == '__main__':
    main()