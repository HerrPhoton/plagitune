import re
from pathlib import Path


def normalize_filename(path: str | Path):

    path = Path(path)

    extension = path.suffix
    name = path.stem

    name = name.lower()
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)

    return path.parent / f"{name}{extension}"


def add_unique_suffix(path: str | Path, normalize: bool = False):

    path = Path(path)

    if normalize:
        path = normalize_filename(path)

    new_path = path
    index = 1

    while new_path.exists():
        new_name = f"{path.stem}_{index}{path.suffix}"
        new_path = path.parent / new_name

        index += 1

    return new_path


def add_dirname_prefix(path: str | Path, normalize: bool = False):

    path = Path(path)

    new_name = f"{path.parent.stem}_{path.name}"
    new_path = path.parent / Path(new_name)

    if normalize:
        return normalize_filename(new_path)

    return new_path
