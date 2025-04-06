import numpy as np

from src.data.structures.melody import Melody
from src.data.matchers.base_matcher import MatchedPattern, BaseMelodyMatcher


class SmithWatermanMelodyMatcher(BaseMelodyMatcher):

    def __init__(self, melody1: Melody, melody2: Melody):
        super().__init__(melody1, melody2)

        self.intervals1 = np.array(melody1.get_intervals())
        self.intervals2 = np.array(melody2.get_intervals())

    def find_patterns(
        self,
        min_length: int = 7,
        match_score: int = 2,
        mismatch_score: int = -1,
        gap_penalty: int = -1,
        tolerance: float = 0.5
    ) -> list[MatchedPattern]:
        """Находит похожие паттерны в мелодиях используя алгоритм Смита-Ватермана.

        :param int min_length: Минимальная длина паттерна
        :param int match_score: Оценка за совпадение
        :param int mismatch_score: Оценка за несовпадение
        :param int gap_penalty: Штраф за разрыв
        :param float tolerance: Допустимое отклонение при сравнении интервалов
        :return list[MatchedPattern]: Список найденных паттернов
        """
        m, n = len(self.intervals1), len(self.intervals2)
        score_matrix = np.zeros((m + 1, n + 1), dtype=int)

        traceback_matrix = np.zeros((m + 1, n + 1), dtype=int)

        max_score = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):

                if self._are_intervals_similar(self.intervals1[i-1], self.intervals2[j-1], tolerance):
                    diag_score = score_matrix[i-1, j-1] + match_score

                else:
                    diag_score = score_matrix[i-1, j-1] + mismatch_score

                up_score = score_matrix[i-1, j] + gap_penalty
                left_score = score_matrix[i, j-1] + gap_penalty

                score_matrix[i, j] = max(0, diag_score, up_score, left_score)

                if score_matrix[i, j] == 0:
                    traceback_matrix[i, j] = 0

                elif score_matrix[i, j] == diag_score:
                    traceback_matrix[i, j] = 1

                elif score_matrix[i, j] == up_score:
                    traceback_matrix[i, j] = 2

                else:
                    traceback_matrix[i, j] = 3

                if score_matrix[i, j] > max_score:
                    max_score = score_matrix[i, j]

        self.matched_patterns = []
        threshold_score = match_score * min_length // 2

        high_scores = np.where(score_matrix >= threshold_score)
        positions = list(zip(high_scores[0], high_scores[1]))

        positions.sort(key=lambda pos: score_matrix[pos], reverse=True)

        used_positions = set()

        for i, j in positions:
            if score_matrix[i, j] == 0 or (i, j) in used_positions:
                continue

            align1 = []
            align2 = []

            current_i, current_j = i, j
            current_used = set()

            while current_i > 0 and current_j > 0 and traceback_matrix[current_i, current_j] != 0:
                current_used.add((current_i, current_j))

                if traceback_matrix[current_i, current_j] == 1:
                    align1.append(current_i - 1)
                    align2.append(current_j - 1)
                    current_i -= 1
                    current_j -= 1

                elif traceback_matrix[current_i, current_j] == 2:
                    current_i -= 1

                else:
                    current_j -= 1

            used_positions.update(current_used)

            if len(align1) >= min_length:

                align1.reverse()
                align2.reverse()

                notes_indices = []

                for a1, a2 in zip(align1, align2):
                    notes_indices.append((a1, a2))

                    if a1 + 1 < len(self.melody1.notes) and a2 + 1 < len(self.melody2.notes):
                        notes_indices.append((a1 + 1, a2 + 1))

                notes_indices = list(set(notes_indices))
                notes_indices.sort()

                pattern = MatchedPattern(
                    melody1_start=notes_indices[0][0],
                    melody2_start=notes_indices[0][1],
                    length=len(notes_indices),
                    notes_indices=notes_indices
                )

                self.matched_patterns.append(pattern)

        self.matched_patterns.sort(key=lambda x: x.length, reverse=True)

        return self.matched_patterns

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
