import numpy as np

from src.data.matchers.base import MatchedPattern, BaseMelodyMatcher
from src.data.structures.melody import Melody


class IntervalsMelodyMatcher(BaseMelodyMatcher):

    def __init__(self, melody1: Melody, melody2: Melody):
        super().__init__(melody1, melody2)

        self.intervals1 = np.array(melody1.get_intervals())
        self.intervals2 = np.array(melody2.get_intervals())

    def find_patterns(self, min_length: int = 7, tolerance: float = 1.0) -> list[MatchedPattern]:
        self.matched_patterns = []

        for i in range(len(self.intervals1) - min_length + 1):
            for j in range(len(self.intervals2) - min_length + 1):
                length = 0
                matched_indices = []

                while (
                    i + length < len(self.intervals1) and
                    j + length < len(self.intervals2) and
                    self._are_intervals_similar(self.intervals1[i + length], self.intervals2[j + length], tolerance)
                ):
                    matched_indices.append((i + length + 1, j + length + 1))
                    length += 1

                if length >= min_length:
                    matched_indices.insert(0, (i, j))

                    pattern = MatchedPattern(
                        melody1_start=i,
                        melody2_start=j,
                        length=length + 1,
                        notes_indices=matched_indices
                    )
                    self.matched_patterns.append(pattern)

        return self.matched_patterns

    def _are_intervals_similar(self, interval1: float, interval2: float, tolerance: float = 1.0) -> bool:
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
