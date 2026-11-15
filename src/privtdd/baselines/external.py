"""Explicit, verified upstream source locations; no third-party source is shipped."""
import hashlib
import os
from pathlib import Path

COMMIT = 'b903ddc7d93aa85dae951c536de3eededad69c5d'
URL = f'https://github.com/AwfAwe/LDPGraph/tree/{COMMIT}'
# Git blob identities at the frozen commit. Normalize checkout CRLF to Git LF.
FILES = {'LDPGraph_main.py': '9f3634732ae05dce5efe710d033d4cf09d7aaa79',
         'utils.py': 'b8da05d34d50dee1fd073e9f3cbde400c86c5a6e'}
# The upstream utils.py blob itself contains CRLF; these are independently
# normalized identities of those same frozen blobs, not a permissive fallback.
NORMALIZED_FILES = {'LDPGraph_main.py': FILES['LDPGraph_main.py'],
                    'utils.py': 'd5b9c2e55cccf9741bd1c3b59e8d968e8bf19439'}


def source_directory(path=None):
    value = path or os.environ.get('PRIVTDD_LDPGRAPH_SOURCE')
    if not value:
        raise FileNotFoundError('LDPGraph is external. Set PRIVTDD_LDPGRAPH_SOURCE or --external-code-root '
                                f'to the permitted local checkout of {URL}.')
    return Path(value).expanduser().resolve()


def configure_source(path):
    checked = source_directory(path)
    validate_source(checked)
    os.environ['PRIVTDD_LDPGRAPH_SOURCE'] = str(checked)
    return checked


def validate_source(path=None, *, raise_on_error=True):
    issues, files = [], {}
    try:
        root = source_directory(path)
    except FileNotFoundError as exc:
        if raise_on_error:
            raise
        return dict(ok=False, issues=[str(exc)], commit=COMMIT, files={})
    for name, expected in FILES.items():
        item = root / name
        if not item.is_file():
            issues.append(f'Missing {name}')
            continue
        payload = item.read_bytes().replace(b'\r\n', b'\n')
        actual = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        files[name] = dict(git_blob=actual, sha256_lf=hashlib.sha256(payload).hexdigest())
        if actual != NORMALIZED_FILES[name]:
            issues.append(f'{name} differs from frozen upstream commit {COMMIT}')
    result = dict(ok=not issues, path=str(root), commit=COMMIT, issues=issues, files=files,
                  redistribution_status='not included; upstream license not found in audited commit')
    if issues and raise_on_error:
        raise ValueError('; '.join(issues))
    return result
