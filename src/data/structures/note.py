import numpy as np
import librosa


class Note:

    PITCH_LABELS = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]

    def __init__(self, note: float | str | None, duration: float):
        self.note = note
        self.duration = Note.quantize_duration(duration)

    @staticmethod
    def quantize_duration(duration: float) -> float:
        """Возвращает ближайшую длительность ноты из допустимых длительностей.

        :return float: Длительность ноты
        """
        standard_durations = sorted(np.arange(0.25, 10.25, 0.25).tolist() + [0.33])
        return min(standard_durations, key=lambda x: abs(x - duration))

    @property
    def midi_number(self) -> int:
        """Преобразует ноту в MIDI-номер.

        :return int: MIDI-номер ноты
        """
        if self.note is None:
            return 0

        return round(librosa.hz_to_midi(self.freq))

    @property
    def octave(self) -> int | None:
        """Возвращает номер октавы ноты.

        :return int | None: Номер октавы ноты или None для паузы
        """
        if self.note is None:
            return None

        if isinstance(self.note, str):
            return int(self.note[-1])

        midi_num = librosa.hz_to_midi(self.freq)
        return (midi_num // 12) - 1

    @property
    def freq(self) -> float:
        """Возвращает абсолютную частоту ноты в Гц.

        :return float: Абсолютная частота ноты или 0 для паузы
        """
        if self.note is None:
            return 0.0

        if isinstance(self.note, (int, float)):
            return float(self.note)

        return librosa.note_to_hz(self.note)

    @property
    def note_name(self) -> str | None:
        """Возвращает ноту в буквенной нотации.

        :return str | None: Нота в буквенной нотации или None для паузы
        """
        if self.note is None:
            return None

        if isinstance(self.note, str):
            return self.note

        return librosa.hz_to_note(self.note)

    @property
    def is_rest(self) -> bool:
        """Проверяет, является ли нота паузой.

        :return bool: True, если нота пауза, иначе False
        """
        return self.note is None
