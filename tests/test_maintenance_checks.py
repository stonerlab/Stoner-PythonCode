"""Exercise archive rejection with interpreter assertions disabled."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('defect', [None, 'licence', 'resource', 'bytecode', 'documentation', 'repository'])
def test_archive_checks_under_optimisation(defect):
    """Accept required assets and reject broken manifests even under python -O."""
    resources = [
        path.relative_to(REPO).as_posix()
        for pattern in ('Stoner/plot/stylelib/*.mplstyle', 'Stoner/Image/tessdata/*')
        for path in REPO.glob(pattern) if path.is_file()
    ]
    names = ['COPYING', 'LICENSE.md', 'Stoner/tools/tests.py', *resources]
    if defect == 'licence':
        names.remove('LICENSE.md')
    elif defect == 'resource':
        names.remove(resources[0])
    elif defect == 'bytecode':
        names.append('Stoner/core/data.pyc')
    elif defect == 'documentation':
        names.append('doc/_build/index.html')
    elif defect == 'repository':
        names.append('tests/test_Core.py')
    code = (
        'import json, runpy, sys; '
        "check = runpy.run_path(sys.argv[1])['check']; "
        'check(json.loads(sys.argv[2]), wheel=True)'
    )
    result = subprocess.run(
        [sys.executable, '-O', '-c', code,
         str(REPO / 'maintenance' / 'check-distribution-archives.py'), json.dumps(names)],
        capture_output=True, text=True, check=False,
    )
    if defect is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert 'RuntimeError:' in result.stderr
