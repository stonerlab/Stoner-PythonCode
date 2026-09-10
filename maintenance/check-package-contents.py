"""Check resolved packaging inputs without building a distribution. Run from the repo root."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from setuptools import Distribution

repo = Path.cwd()
dist = Distribution({'script_name': 'setup.py'})
dist.parse_config_files()
build = dist.get_command_obj('build_py')
build.ensure_finalized()
resources = sorted(str(Path(source) / file).replace('\\', '/') for package, source, target, files in build.data_files for file in files)
expected = sorted(str(p).replace('\\', '/') for pattern in ('Stoner/plot/stylelib/*.mplstyle', 'Stoner/Image/tessdata/*') for p in Path('.').glob(pattern) if p.is_file())
assert resources == expected, (resources, expected)
modules = build.find_all_modules()
assert any(package == 'Stoner.tools' and module == 'tests' for package, module, source in modules)
assert not any('__pycache__' in resource or resource.endswith(('.pyc', '.pyo')) for resource in resources)
with TemporaryDirectory(prefix='stoner-phase2-metadata-') as temporary:
    metadata = dist.get_command_obj('egg_info')
    metadata.egg_base = temporary
    metadata.ensure_finalized()
    metadata.run()
    sources = (Path(metadata.egg_info) / 'SOURCES.txt').read_text().splitlines()
    assert not any('__pycache__' in path or path.endswith(('.pyc', '.pyo')) for path in sources)
    assert not any(path.startswith('doc/_build/') for path in sources)
    assert 'LICENSE.md' in sources and 'COPYING' in sources
    assert all(resource in sources for resource in resources)
    report = dict(packages=dist.packages, pythonModuleCount=len(modules), wheelResourceCandidates=resources,
                  sourceManifestEntryCount=len(sources), sourceManifestBytecodeExcluded=True,
                  sourceManifestDocBuildExcluded=True, scientificFixturesRetained=any(p.startswith('sample-data/') for p in sources),
                  testsHelperRetained=True, validationBoundary='Setuptools discovery and fresh egg-info manifest only; no wheel/sdist was built or installed.')
Path('maintenance/phase2/package-contents.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))