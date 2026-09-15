"""Resolve repository-relative training paths used by the figure builders."""
from pathlib import Path, PureWindowsPath


def training_run_path(root: Path, value: str) -> Path:
    """Return a path beneath ``root/training``; reject absolute or escaping paths."""
    name = value.replace('\\', '/')
    if Path(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError('Expected a repository-relative training path: ' + value)
    if not name.startswith('training/') or '..' in Path(name).parts:
        raise ValueError('Run path must remain in training/: ' + value)
    path = (root / name).resolve()
    if (root / 'training').resolve() not in path.parents:
        raise ValueError('Run path escapes training/: ' + value)
    return path
