import librosa


class Note:

    PITCH_LABELS = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]

    def __init__(self, note: float | str | None, duration: float):
        self._note = note
        self._duration = duration
        self._validate_note()

    def _validate_note(self) -> None:
        """Проверяет корректность заданной ноты."""

        if self._note is None:
            return

        if isinstance(self._note, str):

            if len(self._note) < 2:
                raise ValueError(f"Invalid note format: {self._note}")

            pitch = self._note[:-1]

            try:
                octave = int(self._note[-1])

            except Exception:
                raise ValueError(f"Invalid octave in note: {self._note}")

            if pitch not in self.PITCH_LABELS:
                raise ValueError(f"Invalid pitch: {pitch}")

            if not (-1 <= octave <= 9):
                raise ValueError(f"Octave out of range: {octave}")

    @property
    def midi_number(self) -> int:
        """Преобразует ноту в MIDI-номер.

        :return int: MIDI-номер ноты
        """
        if self._note is None:
            return 0

        return round(librosa.hz_to_midi(self.freq))

    @property
    def octave(self) -> int | None:
        """Возвращает номер октавы ноты.

        :return int | None: Номер октавы ноты или None для паузы
        """
        if self._note is None:
            return None

        if isinstance(self._note, str):
            return int(self._note[-1])

        midi_num = librosa.hz_to_midi(self.freq)
        return (midi_num // 12) - 1

    @property
    def freq(self) -> float:
        """Возвращает абсолютную частоту ноты в Гц.

        :return float: Абсолютная частота ноты или 0 для паузы
        """
        if self._note is None:
            return 0.0

        if isinstance(self._note, (int, float)):
            return float(self._note)

        return librosa.note_to_hz(self._note)

    @property
    def note_name(self) -> str | None:
        """Возвращает ноту в буквенной нотации.

        :return str | None: Нота в буквенной нотации или None для паузы
        """
        if self._note is None:
            return None

        if isinstance(self._note, str):
            return self._note

        return librosa.hz_to_note(self._note)

    @property
    def is_rest(self) -> bool:
        """Проверяет, является ли нота паузой.

        :return bool: True, если нота пауза, иначе False
        """
        return self._note is None
