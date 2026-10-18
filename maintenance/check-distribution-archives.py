"""Validate the contents of CI-built wheel and source archives without extracting them."""

from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


def check(names, wheel):
    """Require runtime assets and licence files, and reject interpreter/build caches."""
    paths = [PurePosixPath(name) for name in names]
    if any('__pycache__' in path.parts for path in paths):
        raise RuntimeError('Archive contains an interpreter cache directory')
    if any(path.name.endswith(('.pyc', '.pyo')) or '.pyc.' in path.name or '.pyo.' in path.name for path in paths):
        raise RuntimeError('Archive contains compiled Python bytecode')
    if any('doc/_build/' in str(path) for path in paths):
        raise RuntimeError('Archive contains generated documentation')
    for licence in ('COPYING', 'LICENSE.md'):
        if not any(path.name == licence for path in paths):
            raise RuntimeError(f'Archive is missing {licence}')
    repo = Path(__file__).resolve().parents[1]
    for pattern in ('Stoner/Image/tessdata/*',):
        for source in repo.glob(pattern):
            if source.is_file():
                relative = source.relative_to(repo).as_posix()
                if not any(str(path) == relative or str(path).endswith('/' + relative) for path in paths):
                    raise RuntimeError(f'Archive is missing runtime resource: {relative}')
    if not any(str(path).endswith('Stoner/tools/tests.py') for path in paths):
        raise RuntimeError('Archive is missing Stoner/tools/tests.py')
    if wheel:
        if any(path.parts[0] in ('tests', 'doc', 'sample-data', 'maintenance') for path in paths):
            raise RuntimeError('Wheel contains repository-only files')


def main():
    """Check exactly one wheel and one source archive from the CI build."""
    output = Path(sys.argv[1])
    wheels, sources = list(output.glob('*.whl')), list(output.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        raise RuntimeError(f'Expected one wheel and one source archive, found {wheels}, {sources}')
    with zipfile.ZipFile(wheels[0]) as archive:
        check(archive.namelist(), wheel=True)
        report_size(wheels[0], [(item.filename, item.file_size) for item in archive.infolist()])
    with tarfile.open(sources[0], 'r:gz') as archive:
        check(archive.getnames(), wheel=False)
        report_size(sources[0], [(item.name, item.size) for item in archive.getmembers() if item.isfile()])
    print('Wheel and source archive content checks passed.')


def report_size(path, entries):
    """Report actual CI archive sizes and their largest retained source assets."""
    print(f'{path.name}: {path.stat().st_size:,} compressed bytes; '
          f'{sum(size for _, size in entries):,} uncompressed bytes; {len(entries)} entries')
    for name, size in sorted(entries, key=lambda item: item[1], reverse=True)[:5]:
        print(f'  {size:>12,} bytes  {name}')


if __name__ == '__main__':
    main()
