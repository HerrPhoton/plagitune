from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import pretty_midi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.data.structures.note import Note
from src.core.styles.melody_style import MelodyStyle


class Melody:

    def __init__(self, notes: List[Note], tempo: int):
        self._notes = notes
        self._tempo = tempo

    def to_wav(self, filename: str, sample_rate: int) -> None:
        """Сохраняет мелодию в WAV файл.

        :param filename (str): Путь к файлу
        :param sample_rate (int): Частота дискретизации
        """
        wave = self._get_wave(sample_rate)
        sf.write(filename, wave, sample_rate)

    def visualize(self, **style_kwargs) -> None:
        """Визуализация мелодии в виде пианоролла.

        :param style_kwargs: Дополнительные параметры визуализации
        """
        style = MelodyStyle(**style_kwargs)

        midi_numbers = [note.midi_number for note in self._notes if not note.is_rest]

        if not midi_numbers:
            min_midi = 60
            max_midi = 71
            
        else:
            min_midi = float('inf')
            max_midi = float('-inf')

        for note in self._notes:
            if not note.is_rest:
                midi_num = note.midi_number
                min_midi = min(min_midi, midi_num)
                max_midi = max(max_midi, midi_num)

        padding = 2
        min_midi = max(0, min_midi - padding)
        max_midi = min(127, max_midi + padding)

        pitches = []

        for midi_num in range(int(min_midi), int(max_midi) + 1):
            octave = (midi_num // 12) - 1
            note_idx = midi_num % 12
            pitches.append(f"{Note.PITCH_LABELS[note_idx]}{octave}")

        fig, ax = plt.subplots(figsize=style.figsize)
        ax.set_facecolor(style.background_color)
        fig.patch.set_facecolor(style.background_color)

        for i in range(len(pitches)):
            plt.axhline(
                y=i, 
                color=style.lines_color, 
                linestyle=style.lines_linestyle, 
                linewidth=style.lines_linewidth, 
                alpha=style.lines_alpha
            )

        current_time = 0

        for note in self._notes:
            if not note.is_rest:

                note_name = note.note_name

                if note_name in pitches:
                    pitch_idx = pitches.index(note_name)
                    duration = self._beats_to_seconds(note._duration)

                    note_color = style.note_gradient[note_name[:-1]]

                    rect = Rectangle((current_time, pitch_idx - 0.4),
                                     duration,
                                     0.8,
                                     facecolor=note_color,
                                     edgecolor=style.note_edge_color,
                                     linewidth=style.note_linewidth,
                                     alpha=style.note_alpha)
                    ax.add_patch(rect)

            current_time += self._beats_to_seconds(note._duration)

        plt.yticks(range(len(pitches)), pitches, fontsize=style.y_ticks_fontsize, color=style.text_color)
        plt.xticks(fontsize=style.x_ticks_fontsize, color=style.text_color)
        plt.xlabel(style.x_label, color=style.text_color)
        plt.xlim(-0.1, self.duration + 0.1)
        plt.ylim(-1, len(pitches))

        plt.grid(True, axis='x', linestyle=style.grid_linestyle, alpha=style.grid_alpha, color=style.grid_color)

        for spine in ax.spines.values():
            spine.set_color(style.grid_color)

        plt.title(style.title, fontsize=style.title_fontsize, color=style.text_color)
        plt.tight_layout()
        plt.show()

    @classmethod
    def from_midi(cls, midi_path: str | Path) -> 'Melody':
        """Создает экземпляр Melody из MIDI файла.

        :param midi_path (str | Path): Путь к MIDI файлу
        :return Melody: Новый экземпляр класса Melody
        """
        midi_path = str(midi_path)

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        tempo_value = int(midi_data.get_tempo_changes()[1])

        quarter_length = 60.0 / tempo_value

        notes = []
        if midi_data.instruments:
            midi_notes = midi_data.instruments[0].notes

            if midi_notes:
                current_time = midi_notes[0].start

                for midi_note in midi_notes:

                    if midi_note.start > current_time:
                        rest_duration = (midi_note.start - current_time) / quarter_length
                        notes.append(Note(None, rest_duration))

                    note_duration = (midi_note.end - midi_note.start) / quarter_length
                    note_name = pretty_midi.note_number_to_name(midi_note.pitch).replace('#', "♯")
                    notes.append(Note(note_name, note_duration))

                    current_time = midi_note.end

        return cls(notes, tempo_value)

    @property
    def duration(self) -> float:
        """Возвращает общую длительность мелодии в секундах.

        :return float: Общая длительность мелодии в секундах
        """
        return sum(self._beats_to_seconds(note._duration) for note in self._notes)

    def _get_wave(self, sample_rate: int) -> np.ndarray:
        """Возвращает мелодию в виде массива сэмплов.

        :param int sample_rate: Частота дискретизации
        :return np.ndarray: Массив сэмплов
        """
        wave = np.array([], dtype=np.float32)

        for note in self._notes:
            frequency = note.freq
            duration = self._beats_to_seconds(note._duration)
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

            note_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
            wave = np.concatenate((wave, note_wave))

        return wave

    def _beats_to_seconds(self, duration: float) -> float:
        """Преобразует длительность ноты в долях такта в длительность в секундах.

        :param float duration: Длительность ноты в долях такта
        :return float: Длительность ноты в секундах
        """
        beat_duration = 60 / self._tempo
        return beat_duration * duration

    def _seconds_to_beats(self, duration: float) -> float:
        """Преобразует длительность в секундах в длительность в долях такта.

        :param float duration: Длительность в секундах
        :return float: Длительность в долях такта
        """
        beat_duration = 60 / self._tempo
        return duration / beat_duration

