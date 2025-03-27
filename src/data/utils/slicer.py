from copy import deepcopy

import torch

from src.data.structures.note import Note
from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.pipelines.configs.slice_config import SliceConfig
from src.data.pipelines.configs.spectrogram_config import SpectrogramConfig


class Slicer:

    def __init__(
        self,
        slice_config: SliceConfig,
        spectrogram_config: SpectrogramConfig,
    ):
        self.slice_config = slice_config
        self.spectrogram_config = spectrogram_config

        self.beats_per_measure = 4
        self.measures_per_slice = 4

    def slice_audio(self, audio: Audio) -> list[Audio]:
        """Нарезка аудио на окна фиксированного размера.

        :param Audio audio: Аудиофайл
        :return List[Audio]: Список аудиофайлов, нарезанных на окна
        """
        sliced_audio = []

        hop_length = self.spectrogram_config.hop_length

        slice_samples = self.slice_config.slice_size * hop_length
        hop_samples = self.slice_config.hop_size * hop_length
        total_samples = audio.waveform.shape[1]

        if total_samples <= slice_samples:

            padded_audio = deepcopy(audio)
            padded_window = torch.full((1, slice_samples), self.slice_config.audio_pad_value, dtype=audio.waveform.dtype)
            padded_window[:, :total_samples] = audio.waveform
            padded_audio.waveform = padded_window

            sliced_audio.append(padded_audio)
            return sliced_audio

        windows_to_process = []
        current_sample = 0

        while current_sample < total_samples:
            end_sample = min(current_sample + slice_samples, total_samples)
            windows_to_process.append((current_sample, end_sample))

            if end_sample == total_samples:
                break

            current_sample += hop_samples

        if windows_to_process[-1][1] < total_samples:
            last_window = (max(0, total_samples - slice_samples), total_samples)
            windows_to_process.append(last_window)

        windows_to_process = list(dict.fromkeys(windows_to_process))
        windows_to_process.sort()

        for start_sample, end_sample in windows_to_process:

            audio_slice = deepcopy(audio)
            samples_to_copy = end_sample - start_sample

            if samples_to_copy < slice_samples:
                padded_window = torch.full((1, slice_samples), self.slice_config.audio_pad_value, dtype=audio.waveform.dtype)
                padded_window[:, :samples_to_copy] = audio.waveform[:, start_sample:end_sample]
                audio_slice.waveform = padded_window

            else:
                audio_slice.waveform = audio.waveform[:, start_sample:end_sample]

            sliced_audio.append(audio_slice)

        return sliced_audio

    def slice_audio_by_measure(self, audio: Audio, tempo: int) -> list[Audio]:
        """Нарезка аудио на окна в соответствии с тактами.

        :param Audio audio: Аудиофайл
        :param int tempo: Темп мелодии
        :return List[Audio]: Список аудиофайлов, нарезанных на окна
        """
        sliced_audio = []

        samples_per_beat = int(audio.sample_rate * 60 / tempo)
        samples_per_measure = samples_per_beat * self.beats_per_measure
        samples_per_slice = samples_per_measure * self.measures_per_slice
        hop_samples = self.slice_config.hop_size * samples_per_measure

        total_samples = audio.waveform.shape[1]
        current_sample = 0

        while current_sample < total_samples:
            end_sample = min(current_sample + samples_per_slice, total_samples)

            window = torch.full(
                (1, samples_per_slice),
                self.slice_config.audio_pad_value,
                dtype=audio.waveform.dtype
            )

            samples_to_copy = end_sample - current_sample
            window[:, :samples_to_copy] = audio.waveform[:, current_sample:end_sample]

            window_audio = deepcopy(audio)
            window_audio.waveform = window

            sliced_audio.append(window_audio)
            current_sample += hop_samples

        return sliced_audio

    def slice_melody_by_measure(self, melody: Melody) -> list[Melody]:
        """Нарезка мелодии на окна по 4 такта.

        :param Melody melody: Объект мелодии
        :return List[Melody]: Список объектов мелодии для каждого окна
        """
        sliced_melody = []

        slice_beats = self.beats_per_measure * self.measures_per_slice
        hop_beats = self.slice_config.hop_size * self.beats_per_measure

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
        current_window_start = 0.0

        while current_window_start < total_beats:
            window_notes = []
            window_end = current_window_start + slice_beats
            current_position = 0.0

            for note_info in note_positions:
                if note_info['end'] <= current_window_start:
                    continue

                if note_info['start'] >= window_end:
                    break

                start_in_window = max(0, note_info['start'] - current_window_start)
                end_in_window = min(slice_beats, note_info['end'] - current_window_start)
                duration_in_window = end_in_window - start_in_window

                if duration_in_window > 0:
                    if start_in_window > current_position:
                        rest_duration = start_in_window - current_position
                        window_notes.append(Note(None, rest_duration))

                    window_notes.append(Note(note_info['note'].note, duration_in_window))
                    current_position = end_in_window

            if current_position < slice_beats:
                window_notes.append(Note(None, slice_beats - current_position))

            if window_notes:
                sliced_melody.append(Melody(window_notes, melody.tempo))

            current_window_start += hop_beats

        return sliced_melody

    def slice_data(self, audio: Audio, melody: Melody) -> tuple[list[Audio], list[Melody]]:
        """Нарезка аудио и мелодии на синхронизированные окна.

        :param Audio audio: Аудиофайл
        :param Melody melody: Мелодия
        :return: Кортеж из списков нарезанных аудио и мелодий
        """
        # Рассчитываем параметры нарезки
        samples_per_beat = int(audio.sample_rate * 60 / melody.tempo)
        samples_per_measure = samples_per_beat * self.beats_per_measure
        samples_per_slice = samples_per_measure * self.measures_per_slice
        hop_samples = self.slice_config.hop_size * samples_per_measure

        slice_beats = self.beats_per_measure * self.measures_per_slice
        hop_beats = self.slice_config.hop_size * self.beats_per_measure

        # Подготавливаем информацию о нотах
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

        # Проверяем, что длительности примерно совпадают
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

            # Вычисляем точное количество сэмплов для текущего окна мелодии
            exact_samples_needed = int((window_end_beat - current_beat) * samples_per_beat)
            end_sample = current_sample + exact_samples_needed

            # Создаем окно точного размера для текущего фрагмента мелодии
            window = torch.full(
                (1, samples_per_slice),
                self.slice_config.audio_pad_value,
                dtype=audio.waveform.dtype
            )

            # Копируем только нужное количество сэмплов
            samples_to_copy = min(exact_samples_needed, total_samples - current_sample)
            window[:, :samples_to_copy] = audio.waveform[:, current_sample:current_sample + samples_to_copy]

            window_audio = deepcopy(audio)
            window_audio.waveform = window
            sliced_audio.append(window_audio)

            # Нарезаем мелодию
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

                # Пропускаем слишком короткие ноты
                if duration_in_window < 0.25:
                    continue

                if duration_in_window > 0:
                    # Если есть промежуток между текущей позицией и началом ноты,
                    # добавляем паузу только если она не слишком короткая
                    if start_in_window > current_position:
                        rest_duration = start_in_window - current_position
                        if rest_duration >= 0.25:
                            window_notes.append(Note(None, rest_duration))
                            current_position += rest_duration

                    window_notes.append(Note(note_info['note'].note, duration_in_window))
                    current_position = end_in_window

            # Добавляем паузу в конец окна мелодии, если нужно
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
