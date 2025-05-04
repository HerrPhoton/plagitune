from pathlib import Path

from tqdm import tqdm
from joblib import Parallel, delayed

from src.data.structures.melody import Melody
from src.data.matchers.smith_waterman import SmithWatermanMelodyMatcher


def calculate_levenshtein_distances(pairs: list[tuple[Path, Path]], normalize: bool = False) -> list[int | float]:
    """Вычисляет расстояние Левенштейна для каждой пары мелодий.

    :param pairs: Список пар мелодий
    :return: Список расстояний Левенштейна
    """
    def process_pair(pair: tuple[Path, Path]) -> float:
        midi1_path, midi2_path = pair
        melody1 = Melody.from_midi(midi1_path)
        melody2 = Melody.from_midi(midi2_path)

        matcher = SmithWatermanMelodyMatcher(melody1, melody2)
        matcher.find_patterns()

        return matcher.overlaps_levenshtein_distance(normalize)

    distances = Parallel(n_jobs=-1, backend='loky')(
        delayed(process_pair)(pair) for pair in tqdm(pairs, desc="Calculating distances")
    )

    return distances
