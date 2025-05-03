import numpy as np

from src.data.matchers.base import MatchedPattern, BaseMelodyMatcher
from src.data.structures.melody import Melody


class OffsetsMelodyMatcher(BaseMelodyMatcher):

    def __init__(self, melody1: Melody, melody2: Melody):
        super().__init__(melody1, melody2)

        self.offsets1 = np.array(melody1.get_offsets())
        self.offsets2 = np.array(melody2.get_offsets())

    def find_patterns(self, min_length: int = 7, tolerance: float = 1.0) -> list[MatchedPattern]:
        self.matched_patterns = []

        for i in range(len(self.offsets1) - min_length + 1):
            for j in range(len(self.offsets2) - min_length + 1):
                length = 0
                matched_indices = []

                while (
                    i + length < len(self.offsets1) and
                    j + length < len(self.offsets2) and
                    self._are_offsets_similar(self.offsets1[i + length], self.offsets2[j + length], tolerance) and
                    self.offsets1[i + length] != 0
                ):
                    matched_indices.append((i + length, j + length))
                    length += 1

                if length >= min_length:
                    pattern = MatchedPattern(
                        melody1_start=i,
                        melody2_start=j,
                        length=length,
                        notes_indices=matched_indices
                    )
                    self.matched_patterns.append(pattern)

        return self.matched_patterns

    def _are_offsets_similar(self, offset1: float, offset2: float, tolerance: float = 1.0) -> bool:
        """Проверяет, являются ли два смещения похожими.

        :param float offset1: Первый интервал
        :param float offset2: Второй интервал
        :param float tolerance: Допустимое отклонение
        :return bool: True, если интервалы похожи, иначе False
        """
        return abs(offset1 - offset2) <= tolerance
