from copy import deepcopy

import torch

from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.configs.slicer_config import SlicerConfig


class Slicer:

    def __init__(self, hop_beats: float = SlicerConfig.hop_beats):
        self.hop_beats = hop_beats

    def slice_audio_by_measure(self, audio: Audio, tempo: int) -> list[Audio]:
        """Нарезка аудио на окна в соответствии с тактами.

        :param Audio audio: Аудиофайл
        :param int tempo: Темп мелодии
        :return List[Audio]: Список аудиофайлов, нарезанных на окна
        """
        samples_per_beat = int(audio.sample_rate * 60 / tempo)
        samples_per_measure = samples_per_beat * SlicerConfig.beats_per_measure
        samples_per_slice = samples_per_measure * SlicerConfig.measures_per_slice
        hop_beats = self.hop_beats * SlicerConfig.beats_per_measure

        total_samples = audio.waveform.shape[1]
        sliced_audio = []
        current_sample = 0
        current_beat = 0.0

        while current_sample < total_samples:
            window_end_beat = current_beat + (SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice)
            exact_samples_needed = int((window_end_beat - current_beat) * samples_per_beat)

            window = torch.full(
                (1, samples_per_slice),
                SlicerConfig.audio_pad_value,
                dtype=audio.waveform.dtype
            )

            samples_to_copy = min(exact_samples_needed, total_samples - current_sample)
            window[:, :samples_to_copy] = audio.waveform[:, current_sample:current_sample + samples_to_copy]

            window_audio = deepcopy(audio)
            window_audio.waveform = window
            sliced_audio.append(window_audio)

            next_beat = current_beat + hop_beats
            next_sample = int(next_beat * samples_per_beat)

            if next_sample >= total_samples:
                break

            current_beat = next_beat
            current_sample = next_sample

        return sliced_audio

    def slice_melody_by_measure(self, melody: Melody) -> list[Melody]:
        """Нарезка мелодии на окна по 4 такта.

        :param Melody melody: Объект мелодии
        :return List[Melody]: Список объектов мелодии для каждого окна
        """
        slice_beats = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice
        hop_beats = self.hop_beats * SlicerConfig.beats_per_measure

        note_positions = []
        current_beat = 0.0
        for note in melody.notes:
            note_positions.append({
                'start': current_beat,
                'end': current_beat + note.duration,
                'note': note
            })
            current_beat += note.duration

        total_beats = current_beat
        sliced_melody = []
        current_beat = 0.0

        while current_beat < total_beats:
            window_notes = []
            current_position = 0.0
            window_end = current_beat + slice_beats

            for note_info in note_positions:
                if note_info['end'] <= current_beat:
                    continue

                if note_info['start'] >= window_end:
                    break

                start_in_window = max(0, note_info['start'] - current_beat)
                end_in_window = min(slice_beats, note_info['end'] - current_beat)
                duration_in_window = end_in_window - start_in_window

                if duration_in_window < 0.25:
                    continue

                if duration_in_window > 0:
                    if start_in_window > current_position:
                        rest_duration = start_in_window - current_position
                        if rest_duration >= 0.25:
                            window_notes.append(Note(None, rest_duration))
                            current_position += rest_duration

                    window_notes.append(Note(note_info['note'].note, duration_in_window))
                    current_position = end_in_window

            remaining_duration = slice_beats - current_position
            if remaining_duration >= 0.25:
                window_notes.append(Note(None, remaining_duration))

            if window_notes:
                sliced_melody.append(Melody(window_notes, melody.tempo))

            if current_beat + hop_beats >= total_beats:
                break

            current_beat += hop_beats

        return sliced_melody

    def slice_data(self, audio: Audio, melody: Melody) -> tuple[list[Audio], list[Melody]]:
        """Нарезка аудио и мелодии на синхронизированные окна.

        :param Audio audio: Аудиофайл
        :param Melody melody: Мелодия
        :return: Кортеж из списков нарезанных аудио и мелодий
        """
        samples_per_beat = int(audio.sample_rate * 60 / melody.tempo)
        samples_per_measure = samples_per_beat * SlicerConfig.beats_per_measure
        samples_per_slice = samples_per_measure * SlicerConfig.measures_per_slice

        slice_beats = SlicerConfig.beats_per_measure * SlicerConfig.measures_per_slice
        hop_beats = self.hop_beats * SlicerConfig.beats_per_measure

        note_positions = []
        current_beat = 0.0
        for note in melody.notes:
            note_positions.append({
                'start': current_beat,
                'end': current_beat + note.duration,
                'note': note
            })
            current_beat += note.duration

        total_beats = current_beat
        total_samples = audio.waveform.shape[1]

        expected_samples = int(total_beats * samples_per_beat)
        if abs(total_samples - expected_samples) > samples_per_beat:
            print(f"Warning: Audio {audio.audio_path.name} ({total_samples/audio.sample_rate:.2f}s) and melody "
                  f"({total_beats * 60/melody.tempo:.2f}s) durations differ significantly")

        sliced_audio = []
        sliced_melody = []
        current_sample = 0
        current_beat = 0.0

        while current_sample < total_samples and current_beat < total_beats:

            window_end_beat = current_beat + slice_beats
            exact_samples_needed = int((window_end_beat - current_beat) * samples_per_beat)

            window = torch.full(
                (1, samples_per_slice),
                SlicerConfig.audio_pad_value,
                dtype=audio.waveform.dtype
            )

            samples_to_copy = min(exact_samples_needed, total_samples - current_sample)
            window[:, :samples_to_copy] = audio.waveform[:, current_sample:current_sample + samples_to_copy]

            window_audio = deepcopy(audio)
            window_audio.waveform = window
            sliced_audio.append(window_audio)

            window_notes = []
            current_position = 0.0
            window_end = current_beat + slice_beats

            for note_info in note_positions:
                if note_info['end'] <= current_beat:
                    continue

                if note_info['start'] >= window_end:
                    break

                start_in_window = max(0, note_info['start'] - current_beat)
                end_in_window = min(slice_beats, note_info['end'] - current_beat)
                duration_in_window = end_in_window - start_in_window

                if duration_in_window < 0.25:
                    continue

                if duration_in_window > 0:

                    if start_in_window > current_position:
                        rest_duration = start_in_window - current_position

                        if rest_duration >= 0.25:
                            window_notes.append(Note(None, rest_duration))
                            current_position += rest_duration

                    window_notes.append(Note(note_info['note'].note, duration_in_window))
                    current_position = end_in_window

            remaining_duration = slice_beats - current_position
            if remaining_duration >= 0.25:
                window_notes.append(Note(None, remaining_duration))

            if window_notes:
                sliced_melody.append(Melody(window_notes, melody.tempo))

            if current_beat + hop_beats >= total_beats:
                break

            current_beat += hop_beats
            current_sample = int(current_beat * samples_per_beat)

        return sliced_audio, sliced_melody
