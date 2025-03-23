import shutil
from typing import Literal
from pathlib import Path

from src.dataset_tools.renamer import add_unique_suffix, add_dirname_prefix


def collect_raw_data(
    source_dir: str,
    destination_dir: str,
    clear_source: bool = False,
    rename: Literal["parent_prefix", "unique_id"] | None = None
):
    AUDIO_DIR = Path("audio")
    LABELS_DIR = Path("labels")

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)

    audio_dir = destination_dir / AUDIO_DIR
    labels_dir = destination_dir / LABELS_DIR

    audio_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(list(source_dir.rglob("*.wav")))
    labels_files = sorted(list(source_dir.rglob("*.mid")))

    for audio_path, label_path in zip(audio_files, labels_files):

        new_audio_path = audio_dir / audio_path.name
        new_label_path = labels_dir / label_path.name

        if rename:

            match rename:

                case "parent_prefix":
                    new_audio_path = add_dirname_prefix(path=new_audio_path, normalize=True)
                    new_label_path = add_dirname_prefix(path=new_label_path, normalize=True)

                case "unique_id":
                    new_audio_path = add_unique_suffix(path=new_audio_path, normalize=True)
                    new_label_path = add_unique_suffix(path=new_label_path, normalize=True)

        if clear_source:
            audio_path.rename(new_audio_path)
            label_path.rename(new_label_path)

        else:
            shutil.copy(audio_path, new_audio_path)
            shutil.copy(label_path, new_label_path)
