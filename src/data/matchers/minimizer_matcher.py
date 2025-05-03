from collections import defaultdict

import numpy as np

from src.data.structures.melody import Melody
from src.data.matchers.base_matcher import MatchedPattern, BaseMelodyMatcher


class MinimizerMelodyMatcher(BaseMelodyMatcher):

    def __init__(self, melody1: Melody, melody2: Melody):
        super().__init__(melody1, melody2)

        self.intervals1 = np.array(melody1.get_intervals())
        self.intervals2 = np.array(melody2.get_intervals())

    def find_patterns(
        self,
        min_length: int = 7,
        k: int = 5,
        l: int = 3,
        tolerance: float = 0.2,
        max_gap: int = 2
    ) -> list[MatchedPattern]:
        """Находит похожие паттерны в мелодиях используя алгоритм минимайзера.

        :param int min_length: Минимальная длина паттерна
        :param int k: Размер окна для извлечения k-меров (k > l)
        :param int l: Размер минимайзера (l < k)
        :param float tolerance: Допустимое отклонение при сравнении интервалов
        :param int max_gap: Максимальный разрыв между совпадениями для объединения
        :return List[MatchedPattern]: Список найденных паттернов
        """
        self.min_length = min_length

        minimizers1 = self._get_minimizers(self.intervals1, k, l)
        minimizers2 = self._get_minimizers(self.intervals2, k, l)

        matches = self._find_similar_minimizers(minimizers1, minimizers2, l, tolerance)

        patterns = self._merge_matches(matches, max_gap)

        self.matched_patterns = sorted(
            patterns,
            key=lambda x: x.length,
            reverse=True
        )

        return self.matched_patterns

    def _get_minimizers(self, sequence: np.ndarray, k: int, l: int) -> dict[tuple[float, ...], set[int]]:
        """Извлекает минимайзеры из последовательности.

        :param np.ndarray sequence: Последовательность интервалов
        :param int k: Размер окна для извлечения k-меров (k > l)
        :param int l: Размер минимайзера (l < k)
        :return dict[tuple[int, int], set[int]]: Словарь минимайзеров и их позиций
        """
        INF_VALUE = 10000

        if len(sequence) < k:
            return {}

        minimizers = defaultdict(set)

        for i in range(len(sequence) - k + 1):
            k_mer = sequence[i:i+k]

            min_l_mer = None
            min_pos = -1

            for j in range(k - l + 1):
                l_mer_values = []

                for val in k_mer[j:j+l]:
                    match val:
                        case float('inf'):
                            l_mer_values.append(INF_VALUE)
                        case float('-inf'):
                            l_mer_values.append(INF_VALUE)
                        case _:
                            l_mer_values.append(abs(val))

                l_mer = tuple(l_mer_values)

                if min_l_mer is None or l_mer < min_l_mer:
                    min_l_mer = l_mer
                    min_pos = i + j

            original_l_mer = tuple(
                float('inf') if val == INF_VALUE else float('-inf') if val == -INF_VALUE else val
                for val in min_l_mer
            )
            minimizers[original_l_mer].add(min_pos)

        return minimizers

    def _are_intervals_similar(self, interval1: float, interval2: float, tolerance: float = 0.5) -> bool:
        """Проверяет, являются ли два интервала похожими.

        :param float interval1: Первый интервал
        :param float interval2: Второй интервал
        :param float tolerance: Допустимое отклонение
        :return bool: True, если интервалы похожи, иначе False
        """
        if interval1 == float('inf') and interval2 == float('inf'):
            return True

        if interval1 == float('-inf') and interval2 == float('-inf'):
            return True

        if (abs(interval1) == float('inf') or abs(interval2) == float('inf')):
            return False

        return abs(interval1 - interval2) <= tolerance

    def _find_similar_minimizers(
        self,
        minimizers1: dict[tuple[int, int], set[int]],
        minimizers2: dict[tuple[int, int], set[int]],
        l: int,
        tolerance: float = 0.5
    ) -> list[tuple[int, int, int]]:
        """Находит похожие минимайзеры.

        :param Dict[Tuple, Set[int]] minimizers1: Минимайзеры первой мелодии
        :param Dict[Tuple, Set[int]] minimizers2: Минимайзеры второй мелодии
        :param int l: Размер минимайзера
        :param float tolerance: Допустимое отклонение при сравнении интервалов
        :return List[Tuple[int, int, int]]: Список совпадений (start1, start2, length)
        """
        matches = []

        for min1, positions1 in minimizers1.items():
            for min2, positions2 in minimizers2.items():

                if len(min1) == len(min2) == l:
                    similar = True

                    for i in range(l):
                        if not self._are_intervals_similar(min1[i], min2[i], tolerance):
                            similar = False
                            break

                    if similar:
                        for pos1 in positions1:
                            for pos2 in positions2:
                                matches.append((pos1, pos2, l))

        return matches

    def _merge_matches(
        self,
        matches: list[tuple[int, int, int]],
        max_gap: int = 2
    ) -> list[MatchedPattern]:
        """Объединяет близкие совпадения в более длинные паттерны.

        :param List[Tuple[int, int, int]] matches: Список совпадений
        :param int max_gap: Максимальный разрыв между совпадениями для объединения
        :return List[MatchedPattern]: Список объединенных паттернов
        """
        if not matches:
            return []

        matches.sort()

        patterns = []
        current_match = list(matches[0])
        matched_indices = []

        for i in range(current_match[0], current_match[0] + current_match[2]):
            j = i - current_match[0] + current_match[1]

            if (i < len(self.intervals1) + 1 and j < len(self.intervals2) + 1):
                matched_indices.append((i, j))

        for match in matches[1:]:
            gap1 = match[0] - (current_match[0] + current_match[2])
            gap2 = match[1] - (current_match[1] + current_match[2])

            if (gap1 <= max_gap and gap2 <= max_gap and
                match[1] - match[0] == current_match[1] - current_match[0]):

                new_end = match[0] + match[2]
                current_match[2] = new_end - current_match[0]

                for i in range(match[0], new_end):
                    j = i - match[0] + match[1]

                    if ((i, j) not in matched_indices and
                        i < len(self.intervals1) + 1 and j < len(self.intervals2) + 1):
                        matched_indices.append((i, j))
            else:
                if len(matched_indices) >= self.min_length:
                    patterns.append(MatchedPattern(
                        melody1_start=current_match[0],
                        melody2_start=current_match[1],
                        length=len(matched_indices),
                        notes_indices=matched_indices
                    ))

                current_match = list(match)
                matched_indices = []

                for i in range(match[0], match[0] + match[2]):
                    j = i - match[0] + match[1]
                    if (i < len(self.intervals1) + 1 and j < len(self.intervals2) + 1):
                        matched_indices.append((i, j))

        if len(matched_indices) >= self.min_length:
            patterns.append(MatchedPattern(
                melody1_start=current_match[0],
                melody2_start=current_match[1],
                length=len(matched_indices),
                notes_indices=matched_indices
            ))

        return patterns
