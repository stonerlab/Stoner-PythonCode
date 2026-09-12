"""Audit Data method binding without calling scientific methods or changing package code."""

import importlib
import inspect
import json
import os
from pathlib import Path
import pydoc
import sys
import typing

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Stoner import Data

MODULES = [
    'Stoner.plot.functions', 'Stoner.core.methods',
    'Stoner.analysis.fitting.functions', 'Stoner.analysis.columns',
    'Stoner.analysis.functions', 'Stoner.analysis.features', 'Stoner.analysis.filtering',
]
selected = {}
collisions = []
for module_name in MODULES:
    module = importlib.import_module(module_name)
    for name in dir(module):
        value = getattr(module, name)
        if name.startswith('_') or not callable(value) or not getattr(value, '__module__', '').startswith(module_name):
            continue
        if name in selected:
            collisions.append({'name': name, 'earlier': selected[name].__module__, 'later': value.__module__})
        selected[name] = value

instance = Data()
records = []
for name, source in sorted(selected.items()):
    actual = getattr(Data, name, None)
    bound = getattr(instance, name, None)
    signature = inspect.signature(source)
    expected_bound = signature.replace(parameters=list(signature.parameters.values())[1:])
    try:
        typing.get_type_hints(source)
        annotation_error = None
    except Exception as error:
        annotation_error = f'{type(error).__name__}: {error}'
    required_annotations = [p.name for p in list(signature.parameters.values())[1:]] + ['return']
    missing_annotations = [key for key in required_annotations if key not in source.__annotations__]
    help_text = pydoc.render_doc(bound, renderer=pydoc.plaintext)
    records.append({
        'name': name, 'source_module': source.__module__,
        'same_function': actual is source,
        'class_signature': str(inspect.signature(actual)),
        'bound_signature': str(inspect.signature(bound)),
        'correct_bound_signature': inspect.signature(bound) == expected_bound,
        'docstring_preserved': actual.__doc__ == source.__doc__,
        'annotations_preserved': actual.__annotations__ == source.__annotations__,
        'in_class_dir': name in dir(Data), 'in_instance_dir': name in dir(instance),
        'help_includes_doc': inspect.getdoc(source).splitlines()[0] in help_text if inspect.getdoc(source) else False,
        'missing_annotations': missing_annotations, 'annotation_resolution_error': annotation_error,
    })
result = {'python': sys.version, 'readthedocs': os.getenv('READTHEDOCS'),
          'count': len(records), 'collisions': collisions, 'methods': records}
output = Path(sys.argv[1])
output.write_text(json.dumps(result, indent=2), encoding='utf-8')
checks = ['same_function', 'correct_bound_signature', 'docstring_preserved', 'annotations_preserved',
          'in_class_dir', 'in_instance_dir', 'help_includes_doc']
print(json.dumps({'count': len(records), 'collisions': collisions,
                  'failures': {key: [r['name'] for r in records if not r[key]] for key in checks},
                  'fully_annotated': [r['name'] for r in records if not r['missing_annotations']],
                  'annotation_resolution_errors': {r['name']: r['annotation_resolution_error'] for r in records
                                                   if r['annotation_resolution_error']}}, indent=2))
