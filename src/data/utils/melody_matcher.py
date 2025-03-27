from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data.structures.melody import Melody
from src.core.styles.matcher_style import MatcherStyle


@dataclass
class MatchedPattern:
    melody1_start: int
    melody2_start: int
    length: int
    notes_indices: list[tuple[int, int]]


class MelodyMatcher:

    def __init__(self, melody1: Melody, melody2: Melody):

        self.melody1 = melody1
        self.melody2 = melody2

        self.matched_patterns: list[MatchedPattern] = []

        self.offsets1 = np.array(melody1.get_offsets())
        self.offsets2 = np.array(melody2.get_offsets())

    def find_patterns(self, min_length: int = 7) -> list[MatchedPattern]:
        """Находит все повторяющиеся паттерны минимальной длины

        :param int min_length: Минимальная длина паттерна
        :return List[MatchedPattern]: Список найденных паттернов
        """
        self.matched_patterns = []

        for i in range(len(self.offsets1) - min_length + 1):
            for j in range(len(self.offsets2) - min_length + 1):
                length = 0
                matched_indices = []

                while (
                    i + length < len(self.offsets1) and
                    j + length < len(self.offsets2) and
                    self.offsets1[i + length] == self.offsets2[j + length] and
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

    def visualize_matches(self, pattern: MatchedPattern | None = None, **style_kwargs) -> None:
        """Визуализирует мелодии с подсветкой совпадающих паттернов

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
                color=style.text_color,
                pad=style.suptitle_pad
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
        """Визуализирует отдельную мелодию с подсветкой совпадающих нот

        :param Melody melody: Мелодия для визуализации
        :param List[List[Tuple[int, int]]] patterns_indices: Индексы совпадающих нот
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

        ax.tick_params(
            axis='x',
            labelsize=style.x_ticks_fontsize,
            colors=style.text_color
        )
        ax.set_xlabel(style.x_label, color=style.text_color)

        ax.set_xlim(-0.1, melody.duration + 0.1)
        ax.set_ylim(-1, len(pitches))

        ax.grid(True, axis='x', linestyle=style.grid_linestyle, alpha=style.grid_alpha, color=style.grid_color)

        for spine in ax.spines.values():
            spine.set_color(style.grid_color)

        ax.set_title(title, fontsize=style.subplot_title_fontsize, color=style.text_color, pad=style.title_pad)
