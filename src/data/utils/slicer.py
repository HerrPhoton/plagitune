from copy import deepcopy
from dataclasses import dataclass

import torch

from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.configs.slicer_config import SlicerConfig


@dataclass
class NotePosition:
    start: float
    duration: float
    note: str | None


class Slicer:

    def __init__(self, hop_beats: float = SlicerConfig.hop_beats):
        self.hop_beats = hop_beats
        self.window_size = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice
        self.hop_size = self.hop_beats * SlicerConfig.beats_per_measure

    def slice_audio_by_measure(self, audio: Audio, tempo: int, target_beats: float | None = None) -> list[Audio]:
        """Нарезка аудио на окна в соответствии с тактами.

        :param Audio audio: Аудиофайл
        :param int tempo: Темп мелодии
        :param float | None target_beats: Целевая длительность в битах (если None, используется длина аудио)
        :return List[Audio]: Список аудиофайлов, нарезанных на окна
        """
        samples_per_beat = int(audio.sample_rate * 60 / tempo)
        samples_per_measure = samples_per_beat * SlicerConfig.beats_per_measure
        samples_per_slice = samples_per_measure * SlicerConfig.measures_per_slice
        hop_beats = SlicerConfig.hop_beats * SlicerConfig.beats_per_measure

        total_samples = audio.waveform.shape[1]

        if target_beats is not None:
            total_beats = target_beats

        else:
            total_beats = total_samples / samples_per_beat
            total_beats = round(total_beats, 2)

        sliced_audio = []
        current_beat = 0.0

        while current_beat < total_beats:
            current_sample = int(current_beat * samples_per_beat)
            window_end_beat = current_beat + (SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice)

            window = torch.full(
                (1, samples_per_slice),
                SlicerConfig.audio_pad_value,
                dtype=audio.waveform.dtype
            )

            samples_to_copy = min(
                int((window_end_beat - current_beat) * samples_per_beat),
                total_samples - current_sample
            )

            if (samples_to_copy / samples_per_beat) < 0.25:
                break

            window[:, :samples_to_copy] = audio.waveform[:, current_sample:current_sample + samples_to_copy]

            window_audio = deepcopy(audio)
            window_audio.waveform = window
            sliced_audio.append(window_audio)

            if current_beat + hop_beats >= total_beats:
                break

            current_beat += hop_beats

        return sliced_audio

    def slice_melody_by_measure(self, melody: Melody) -> list[Melody]:
        """Нарезка мелодии на окна по 4 такта

        :param Melody melody: Мелодия
        :return List[Melody]: Список мелодий, нарезанных на окна
        """
        total_beats = sum(note.duration for note in melody.notes)
        total_beats = round(total_beats, 2)
        sliced_melody = []
        current_beat = 0.0

        while current_beat < total_beats:
            window_notes = []
            window_start = current_beat
            window_end = window_start + self.window_size
            current_pos = 0.0

            accumulated_time = 0.0
            for note in melody.notes:
                note_start = accumulated_time
                note_end = note_start + note.duration

                if note_start >= window_end:
                    break

                if note_end <= window_start:
                    accumulated_time = note_end
                    continue

                start_in_window = max(0.0, note_start - window_start)
                end_in_window = min(self.window_size, note_end - window_start)
                duration_in_window = end_in_window - start_in_window

                if start_in_window > current_pos:
                    pause_duration = start_in_window - current_pos
                    if pause_duration >= 0.25:
                        window_notes.append(Note(None, pause_duration))
                        current_pos += pause_duration

                if duration_in_window >= 0.25:
                    window_notes.append(Note(note.note, duration_in_window))
                    current_pos += duration_in_window

                accumulated_time = note_end

            if current_pos < self.window_size:
                final_pause = self.window_size - current_pos
                window_notes.append(Note(None, final_pause))
                current_pos += final_pause

            sliced_melody.append(Melody(window_notes, melody.tempo))

            if current_beat + self.hop_size >= total_beats:
                break

            current_beat += self.hop_size

        return sliced_melody

    def slice_data(self, audio: Audio, melody: Melody) -> tuple[list[Audio], list[Melody]]:
        """Нарезка аудио и мелодии на синхронизированные окна.

        :param Audio audio: Аудиофайл
        :param Melody melody: Мелодия
        :return Tuple[List[Audio], List[Melody]]: Нарезанные окна аудио и мелодии
        """
        melody_beats = sum(note.duration for note in melody.notes)
        melody_beats = round(melody_beats, 2)

        sliced_audio = self.slice_audio_by_measure(audio, melody.tempo, target_beats=melody_beats)
        sliced_melody = self.slice_melody_by_measure(melody)

        return sliced_audio, sliced_melody
