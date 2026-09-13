"""Check resolved packaging inputs without building a distribution. Run from the repo root."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from setuptools import Distribution


def main():
    """Check package discovery and write a report without building distributions."""
    repo = Path.cwd()
    dist = Distribution({'script_name': 'setup.py'})
    dist.parse_config_files()
    build = dist.get_command_obj('build_py')
    build.ensure_finalized()
    resources = sorted(
        (Path(source) / filename).as_posix()
        for _, source, _, filenames in build.data_files for filename in filenames
    )
    expected = sorted(
        path.as_posix()
        for pattern in ('Stoner/plot/stylelib/*.mplstyle', 'Stoner/Image/tessdata/*')
        for path in Path('.').glob(pattern) if path.is_file()
    )
    if resources != expected:
        raise RuntimeError(f'Runtime resource discovery differs: {resources}, expected {expected}')
    modules = build.find_all_modules()
    if not any(package == 'Stoner.tools' and module == 'tests' for package, module, _ in modules):
        raise RuntimeError('Package discovery omitted Stoner.tools.tests')
    if any('__pycache__' in resource or resource.endswith(('.pyc', '.pyo')) for resource in resources):
        raise RuntimeError('Runtime resources include compiled Python bytecode')
    with TemporaryDirectory(prefix='stoner-package-metadata-') as temporary:
        metadata = dist.get_command_obj('egg_info')
        metadata.egg_base = temporary
        metadata.ensure_finalized()
        metadata.run()
        sources = (Path(metadata.egg_info) / 'SOURCES.txt').read_text().splitlines()
        if any('__pycache__' in path or path.endswith(('.pyc', '.pyo')) for path in sources):
            raise RuntimeError('Source manifest includes compiled Python bytecode')
        if any(path.startswith('doc/_build/') for path in sources):
            raise RuntimeError('Source manifest includes generated documentation')
        if not {'LICENSE.md', 'COPYING'}.issubset(sources):
            raise RuntimeError('Source manifest is missing licence files')
        if not all(resource in sources for resource in resources):
            raise RuntimeError('Source manifest is missing runtime resources')
        report = dict(
            packages=dist.packages, pythonModuleCount=len(modules), wheelResourceCandidates=resources,
            sourceManifestEntryCount=len(sources), sourceManifestBytecodeExcluded=True,
            sourceManifestDocBuildExcluded=True,
            scientificFixturesRetained=any(path.startswith('sample-data/') for path in sources),
            testsHelperRetained=True,
            validationBoundary=(
                'Setuptools discovery and fresh egg-info manifest only; no wheel/sdist was built or installed.'
            ),
        )
    output = repo / 'maintenance' / 'runs' / 'package-contents.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
