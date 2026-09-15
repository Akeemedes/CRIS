"""Resolve archived run metadata within the current repository copy.

Old manifests may embed a workstation prefix. Preserve those raw records but
resolve their explicit training subtree locally; never fall back to that machine.
"""
from pathlib import Path, PureWindowsPath

def training_run_path(root:Path,value:str)->Path:
    name=value.replace('\\','/')
    if name.startswith('/') or Path(name).is_absolute() or PureWindowsPath(name).is_absolute():
        prefix,separator,tail=name.partition('/training/')
        if not separator:raise ValueError('Absolute run path has no training subtree: '+value)
        name='training/'+tail
    if not name.startswith('training/') or '..' in Path(name).parts:
        raise ValueError('Run path must remain in training/: '+value)
    path=(root/name).resolve()
    if (root/'training').resolve() not in path.parents:
        raise ValueError('Run path escapes training/: '+value)
    return path
