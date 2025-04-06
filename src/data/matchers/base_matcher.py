from abc import ABC, abstractmethod
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data.structures.melody import Melody
from src.core.styles.matcher_style import MatcherStyle


@dataclass
class MatchedPattern:
    """Структура данных для хранения информации о найденном паттерне.

    :param int melody1_start: Индекс начала паттерна в первой мелодии
    :param int melody2_start: Индекс начала паттерна во второй мелодии
    :param int length: Длина паттерна
    :param list[tuple[int, int]] notes_indices: Список пар индексов совпадающих нот
    """
    melody1_start: int
    melody2_start: int
    length: int
    notes_indices: list[tuple[int, int]]


class BaseMelodyMatcher(ABC):
    """Базовый класс для поиска похожих паттернов в мелодиях."""

    def __init__(self, melody1: Melody, melody2: Melody):
        self.melody1 = melody1
        self.melody2 = melody2
        self.matched_patterns: list[MatchedPattern] = []

    @abstractmethod
    def find_patterns(self, min_length: int = 7) -> list[MatchedPattern]:
        """Находит все повторяющиеся паттерны минимальной длины.

        :param int min_length: Минимальная длина паттерна
        :return list[MatchedPattern]: Список найденных паттернов
        """
        pass

    def calculate_similarity(self) -> float:
        """Вычисляет схожесть между двумя мелодиями.

        :return float: Значение схожести от 0 до 1
        """
        if not self.matched_patterns:
            return 0.0

        matched_indices1 = set()
        matched_indices2 = set()

        for pattern in self.matched_patterns:
            for idx1, idx2 in pattern.notes_indices:
                matched_indices1.add(idx1)
                matched_indices2.add(idx2)

        total_notes1 = len([note for note in self.melody1.notes])
        total_notes2 = len([note for note in self.melody2.notes])

        match_ratio1 = len(matched_indices1) / total_notes1 if total_notes1 > 0 else 0
        match_ratio2 = len(matched_indices2) / total_notes2 if total_notes2 > 0 else 0

        return (match_ratio1 + match_ratio2) / 2

    def visualize_matches(self, pattern: MatchedPattern | None = None, **style_kwargs) -> None:
        """Визуализирует мелодии с подсветкой совпадающих паттернов.

        :param MatchedPattern | None pattern: Паттерн для визуализации
        :param style_kwargs: Параметры стиля
        """
        patterns = [pattern] if pattern else self.matched_patterns
        style = MatcherStyle(**style_kwargs)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=style.figsize)
        fig.patch.set_facecolor(style.background_color)

        self._visualize_melody_with_highlights(
            self.melody1,
            [p.notes_indices for p in patterns],
            ax1,
            style.subplot_titles[0] if style.subplot_titles else None,
            style
        )

        self._visualize_melody_with_highlights(
            self.melody2,
            [[(j, i) for i, j in p.notes_indices] for p in patterns],
            ax2,
            style.subplot_titles[1] if style.subplot_titles else None,
            style
        )

        if style.suptitle:
            plt.suptitle(
                style.suptitle,
                fontsize=style.suptitle_fontsize,
                color=style.text_color
            )

        plt.tight_layout()
        plt.show()

    def _visualize_melody_with_highlights(
        self,
        melody: Melody,
        patterns_indices: list[list[tuple[int, int]]],
        ax: plt.Axes,
        title: str,
        style: MatcherStyle
    ) -> None:
        """Визуализирует отдельную мелодию с подсветкой совпадающих нот.

        :param Melody melody: Мелодия для визуализации
        :param list[list[tuple[int, int]]] patterns_indices: Индексы совпадающих нот
        :param plt.Axes ax: Оси для визуализации
        :param str title: Заголовок для визуализации
        :param MatcherStyle style: Стиль для визуализации
        """
        midi_numbers = [note.midi_number for note in melody.notes if not note.is_rest]
        min_midi = min(midi_numbers) if midi_numbers else 60
        max_midi = max(midi_numbers) if midi_numbers else 71

        padding = 2
        min_midi = max(0, min_midi - padding)
        max_midi = min(127, max_midi + padding)

        pitches = []
        for midi_num in range(int(min_midi), int(max_midi) + 1):
            octave = (midi_num // 12) - 1
            note_idx = midi_num % 12
            pitches.append(f"{melody.notes[0].PITCH_LABELS[note_idx]}{octave}")

        ax.set_facecolor(style.background_color)

        for i in range(len(pitches)):
            ax.axhline(
                y=i,
                color=style.lines_color,
                linestyle=style.lines_linestyle,
                linewidth=style.lines_linewidth,
                alpha=style.lines_alpha
            )

        current_time = 0
        matched_indices = {idx for pattern in patterns_indices for idx, _ in pattern}

        for i, note in enumerate(melody.notes):
            if not note.is_rest:
                note_name = note.note_name
                if note_name in pitches:
                    pitch_idx = pitches.index(note_name)
                    duration = melody._beats_to_seconds(note.duration)

                    color = 'red' if i in matched_indices else style.note_gradient[note_name[:-1]]

                    rect = Rectangle(
                        (current_time, pitch_idx - 0.4),
                        duration,
                        0.8,
                        facecolor=color,
                        edgecolor=style.note_edge_color,
                        linewidth=style.note_linewidth,
                        alpha=style.note_alpha
                    )
                    ax.add_patch(rect)

            current_time += melody._beats_to_seconds(note.duration)

        ax.set_yticks(range(len(pitches)), pitches, fontsize=style.y_ticks_fontsize, color=style.text_color)
        ax.tick_params(axis='x', labelsize=style.x_ticks_fontsize, colors=style.text_color)
        ax.set_xlabel(style.x_label, color=style.text_color)

        ax.set_xlim(-0.1, melody.duration + 0.1)
        ax.set_ylim(-1, len(pitches))

        ax.grid(True, axis='x', linestyle=style.grid_linestyle, alpha=style.grid_alpha, color=style.grid_color)

        for spine in ax.spines.values():
            spine.set_color(style.grid_color)

        ax.set_title(title, fontsize=style.subplot_title_fontsize, color=style.text_color)
