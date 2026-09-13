"""Reject release tags that disagree with the canonical package version."""

import argparse
import ast
from pathlib import Path


def check_release_version(tag, root):
    """Check a release tag without importing the scientific dependency stack.

    Args:
        tag (str):
            Release version, optionally prefixed with a lowercase v.
        root (pathlib.Path):
            Checkout containing the canonical Stoner/__init__.py file.

    Returns:
        str: The matching package version.

    Raises:
        ValueError: The source version is missing, ambiguous or differs from the tag.
    """
    tree = ast.parse((root / 'Stoner' / '__init__.py').read_text(encoding='utf-8'))
    versions = [ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__version__'
                        for target in node.targets)]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise ValueError('Expected one literal __version__ string in Stoner/__init__.py')
    version = versions[0]
    if tag.removeprefix('v') != version:
        raise ValueError(f'Release tag {tag!r} does not match source version {version!r}')
    return version


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    arguments = parser.parse_args()
    print(f'Release version verified: {check_release_version(arguments.tag, Path(__file__).resolve().parents[1])}')
