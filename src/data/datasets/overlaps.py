from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.data.structures.note import Note
from src.data.structures.melody import Melody
from src.data.utils.levenshtein import calculate_levenshtein_distances
from src.data.matchers.smith_waterman import SmithWatermanMelodyMatcher


class OverlapsDataset(Dataset):

    def __init__(
        self,
        overlapping_pairs: list[tuple[Path, Path]],
        non_overlapping_pairs: list[tuple[Path, Path]],
        preprocess_data: bool = True
    ):
        """
        :param List[tuple[Path, Path]] overlapping_pairs: Пары перекрывающихся мелодий.
        :param List[tuple[Path, Path]] non_overlapping_pairs: Пары неперекрывающихся мелодий.
        :param bool preprocess_data: Флаг для предварительной обработки данных
        """
        super().__init__()

        self.overlapping_pairs = overlapping_pairs
        self.non_overlapping_pairs = non_overlapping_pairs

        if preprocess_data:
            self.features, self.targets = self._preprocess_data(self.overlapping_pairs, self.non_overlapping_pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Возвращает элемент датасета.

        :param int idx: Индекс элемента
        :return Tuple[Tensor, Tensor]: Признаки и целевые значения.
        """
        return Tensor(self.features[idx]), Tensor([self.targets[idx]]).float()

    def __len__(self) -> int:
        return len(self.overlapping_pairs) + len(self.non_overlapping_pairs)

    @classmethod
    def from_path(cls, dataset_path: str | Path, split: str | None = None, preprocess_data: bool = True) -> 'OverlapsDataset':
        """Загружает сэмплы из датасета.

        :param str | Path dataset_path: Путь к директории с сэмплами
        :param str split: Раздел датасета ('train', 'val', 'test'). Если None, используются все разделы
        :param bool preprocess_data: Флаг для предварительной обработки данных
        :return OverlapsDataset: Датасет
        """
        dataset_path = Path(dataset_path)

        overlaping_pairs = []
        non_overlaping_pairs = []

        melodies_by_dir = {}

        if split is not None:
            split_dirs = [dataset_path / split]

        else:
            split_dirs = [dataset_path / split_name for split_name in ['train', 'val', 'test'] if (dataset_path / split_name).exists()]

        for split_dir in split_dirs:
            if not split_dir.exists() or not split_dir.is_dir():
                continue

            for case_dir in split_dir.iterdir():
                if case_dir.is_dir():
                    midi_files = list(case_dir.glob("*.mid"))

                    melodies_by_dir[case_dir] = midi_files

                    for i, melody1 in enumerate(midi_files):
                        for melody2 in midi_files[i+1:]:
                            overlaping_pairs.append((melody1, melody2))

        dirs = list(melodies_by_dir.keys())

        for i, dir1 in enumerate(dirs):
            for dir2 in dirs[i+1:]:
                for melody1 in melodies_by_dir[dir1]:
                    for melody2 in melodies_by_dir[dir2]:
                        non_overlaping_pairs.append((melody1, melody2))

        return cls(overlaping_pairs, non_overlaping_pairs, preprocess_data)

    def _preprocess_data(self, overlapping_pairs: list[tuple[Path, Path]], non_overlapping_pairs: list[tuple[Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
        """Извлекает признаки из пар мелодий и предобрабатывает их.

        :param list[tuple[Path, Path]] overlapping_pairs: Пары перекрывающихся мелодий.
        :param list[tuple[Path, Path]] non_overlapping_pairs: Пары неперекрывающихся мелодий.
        :return tuple[np.ndarray, np.ndarray]: Признаки и целевые значения.
        """

        overlapping_distances = calculate_levenshtein_distances(overlapping_pairs, normalize=True)
        non_overlapping_distances = calculate_levenshtein_distances(non_overlapping_pairs, normalize=True)

        features_list = []

        for pair, distance in tqdm(zip(overlapping_pairs, overlapping_distances), total=len(overlapping_pairs), desc='Extracting overlapping features'):
            features = self._get_pair_features(pair, distance)
            features['is_overlapping'] = 1
            features_list.append(features)

        for pair, distance in tqdm(zip(non_overlapping_pairs, non_overlapping_distances), total=len(non_overlapping_pairs), desc='Extracting non-overlapping features'):
            features = self._get_pair_features(pair, distance)
            features['is_overlapping'] = 0
            features_list.append(features)

        df = pd.DataFrame(features_list)

        X = df.drop('is_overlapping', axis=1)
        y = df['is_overlapping']

        self.feature_names = X.columns.tolist()

        X_scaled = StandardScaler().fit_transform(X)

        return X_scaled, y.values

    def _get_pair_features(self, pair: tuple[Path, Path], distance: float) -> dict:
        """Получает признаки для пары мелодий.

        :param pair: Пара путей к MIDI файлам
        :param float distance: Нормализованное расстояние Левенштейна между мелодиями
        :return: Словарь признаков пары
        """
        melody1_path, melody2_path = pair

        melody1 = Melody.from_midi(melody1_path)
        melody2 = Melody.from_midi(melody2_path)

        pair_features = {
            'levenshtein_distance': distance,
        }

        matcher = SmithWatermanMelodyMatcher(melody1, melody2)
        matcher.find_patterns()

        matched_indices1 = set()
        matched_indices2 = set()

        for pattern in matcher.matched_patterns:
            for idx1, idx2 in pattern.notes_indices:
                matched_indices1.add(idx1)
                matched_indices2.add(idx2)

        # Извлечение признаков из целых мелодий
        total_notes1 = len([note for note in melody1.notes])
        total_notes2 = len([note for note in melody2.notes])

        match_ratio1 = len(matched_indices1) / total_notes1 if total_notes1 > 0 else 0
        match_ratio2 = len(matched_indices2) / total_notes2 if total_notes2 > 0 else 0

        # 1. Количество совпадающих нот
        pair_features['match_len1'] = len(matched_indices1)
        pair_features['match_len2'] = len(matched_indices2)

        # 2. Доля совпавших нот от общего количества нот в мелодии
        pair_features['match_ratio1'] = match_ratio1
        pair_features['match_ratio2'] = match_ratio2

        # 3. Разница распределений классов нот в целых мелодиях
        note_classes1 = melody1.get_classes()
        class_counts1 = Counter(note_classes1)

        note_classes2 = melody2.get_classes()
        class_counts2 = Counter(note_classes2)

        for i in range(len(Note.PITCH_LABELS)):
            count1 = class_counts1.get(i, 0)
            count2 = class_counts2.get(i, 0)
            pair_features[f'note_{Note.PITCH_LABELS[i]}'] = abs(count1 - count2)

        pair_features['note_rest'] = abs(class_counts1.get(12, 0) - class_counts2.get(12, 0))

        # 4. Разница распределений длительностей нот в целых мелодиях
        durations1 = melody1.get_durations()
        durations2 = melody2.get_durations()

        duration_values = sorted(np.arange(0.25, 4.25, 0.25).tolist())
        for d in duration_values:
            count1 = sum(1 for dur in durations1 if dur == d)
            count2 = sum(1 for dur in durations2 if dur == d)
            pair_features[f'dur_{d:.2f}'] = abs(count1 - count2)

        count1_gt = sum(1 for d in durations1 if d > 4.0)
        count2_gt = sum(1 for d in durations2 if d > 4.0)
        pair_features['dur_gt'] = abs(count1_gt - count2_gt)

        # 5. Разница распределений интервалов нот в целых мелодиях
        note_intervals1 = melody1.get_intervals()
        interval_counts1 = Counter(note_intervals1)

        note_intervals2 = melody2.get_intervals()
        interval_counts2 = Counter(note_intervals2)

        for interval in range(-10, 11):
            count1 = interval_counts1.get(interval, 0)
            count2 = interval_counts2.get(interval, 0)
            pair_features[f'interval_{interval}'] = abs(count1 - count2)

        count1_lt = sum(interval_counts1.get(i, 0) for i in interval_counts1 if isinstance(i, int) and i < -10)
        count2_lt = sum(interval_counts2.get(i, 0) for i in interval_counts2 if isinstance(i, int) and i < -10)
        pair_features['interval_lt'] = abs(count1_lt - count2_lt)

        count1_gt = sum(interval_counts1.get(i, 0) for i in interval_counts1 if isinstance(i, int) and i > 10)
        count2_gt = sum(interval_counts2.get(i, 0) for i in interval_counts2 if isinstance(i, int) and i > 10)
        pair_features['interval_gt'] = abs(count1_gt - count2_gt)

        count1_to_rest = interval_counts1.get(float('-inf'), 0)
        count2_to_rest = interval_counts2.get(float('-inf'), 0)
        pair_features['interval_to_rest'] = abs(count1_to_rest - count2_to_rest)

        count1_from_rest = interval_counts1.get(float('inf'), 0)
        count2_from_rest = interval_counts2.get(float('inf'), 0)
        pair_features['interval_from_rest'] = abs(count1_from_rest - count2_from_rest)

        return pair_features
