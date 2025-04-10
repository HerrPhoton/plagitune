from collections import defaultdict

from src.data.structures.melody import Melody
from src.data.matchers.base_matcher import MatchedPattern, BaseMelodyMatcher


class KMerMelodyMatcher(BaseMelodyMatcher):

    def __init__(self, melody1: Melody, melody2: Melody):
        super().__init__(melody1, melody2)

        self.intervals1 = melody1.get_intervals()
        self.intervals2 = melody2.get_intervals()

    def find_patterns(
        self,
        min_length: int = 7,
        k: int = 5,
        max_distance: float = 0.3,
        max_gap: int = 1
    ) -> list[MatchedPattern]:
        """Находит похожие паттерны в мелодиях используя k-mer алгоритм с расстоянием Хэмминга.

        :param int min_length: Минимальная длина паттерна
        :param int k: Длина k-мера
        :param float max_distance: Максимальное допустимое расстояние Хэмминга
        :param int max_gap: Максимальный разрыв между k-мерами для объединения
        :return list[MatchedPattern]: Список найденных паттернов
        """
        self.min_length = min_length

        interval_kmers1 = self._extract_kmers(self.intervals1, k)
        interval_kmers2 = self._extract_kmers(self.intervals2, k)

        interval_matches = self._find_similar_kmers(interval_kmers1, interval_kmers2, max_distance)

        interval_patterns = self._merge_matches(interval_matches, max_gap)

        self.matched_patterns = sorted(
            interval_patterns,
            key=lambda x: x.length,
            reverse=True
        )

        return self.matched_patterns

    def _extract_kmers(self, sequence: list[float], k: int) -> dict[str, set[int]]:
        """Извлекает k-меры из последовательности и их позиции.

        :param list[float] sequence: Последовательность элементов
        :param int k: Длина k-мера
        :return dict[str, set[int]]: Словарь k-меров и их позиций
        """
        kmers = defaultdict(set)

        for i in range(len(sequence) - k + 1):
            kmer = tuple(sequence[i:i + k])
            kmers[kmer].add(i)

        return kmers

    def _hamming_distance(self, kmer1: tuple, kmer2: tuple) -> float:
        """Вычисляет расстояние Хэмминга между двумя k-мерами.

        :param tuple kmer1: Первый k-мер
        :param tuple kmer2: Второй k-мер
        :return float: Расстояние Хэмминга (нормализованное)
        """
        differences = 0
        for a, b in zip(kmer1, kmer2):
            if abs(a - b) > 1:
                differences += 1

        return differences / len(kmer1)

    def _find_similar_kmers(
        self,
        kmers1: dict[tuple, set[int]],
        kmers2: dict[tuple, set[int]],
        max_distance: float = 0.3
    ) -> list[tuple[int, int, int]]:
        """Находит похожие k-меры с учетом расстояния Хэмминга.

        :param dict[tuple, set[int]] kmers1: K-меры первой мелодии
        :param dict[tuple, set[int]] kmers2: K-меры второй мелодии
        :param float max_distance: Максимальное допустимое расстояние Хэмминга
        :return list[tuple[int, int, int]]: Список совпадений (start1, start2, length)
        """
        matches = []

        for kmer1, positions1 in kmers1.items():
            for kmer2, positions2 in kmers2.items():
                distance = self._hamming_distance(kmer1, kmer2)

                if distance <= max_distance:
                    for pos1 in positions1:
                        for pos2 in positions2:
                            matches.append((pos1, pos2, len(kmer1)))

        return matches

    def _merge_matches(
        self,
        matches: list[tuple[int, int, int]],
        max_gap: int = 1
    ) -> list[MatchedPattern]:
        """Объединяет близкие совпадения в более длинные паттерны.

        :param list[tuple[int, int, int]] matches: Список совпадений
        :param int max_gap: Максимальный разрыв между совпадениями для объединения
        :return list[MatchedPattern]: Список объединенных паттернов
        """
        if not matches:
            return []

        matches.sort()

        patterns = []
        current_match = list(matches[0])
        matched_indices = [
            (i, i - current_match[0] + current_match[1])
            for i in range(current_match[0], current_match[0] + current_match[2])
        ]

        for match in matches[1:]:
            gap1 = match[0] - (current_match[0] + current_match[2])
            gap2 = match[1] - (current_match[1] + current_match[2])

            if (gap1 <= max_gap and gap2 <= max_gap and
                match[1] - match[0] == current_match[1] - current_match[0]):
                new_length = match[0] + match[2] - current_match[0]
                current_match[2] = new_length

                for i in range(current_match[0] + len(matched_indices), current_match[0] + new_length):
                    j = i - current_match[0] + current_match[1]
                    matched_indices.append((i, j))

            else:
                if current_match[2] >= self.min_length:
                    patterns.append(MatchedPattern(
                        melody1_start=current_match[0],
                        melody2_start=current_match[1],
                        length=current_match[2],
                        notes_indices=matched_indices
                    ))

                current_match = list(match)
                matched_indices = [
                    (i, i - current_match[0] + current_match[1])
                    for i in range(current_match[0], current_match[0] + current_match[2])
                ]

        if current_match[2] >= self.min_length:
            patterns.append(MatchedPattern(
                melody1_start=current_match[0],
                melody2_start=current_match[1],
                length=current_match[2],
                notes_indices=matched_indices
            ))

        return patterns
