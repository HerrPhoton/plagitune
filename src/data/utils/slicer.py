import torch
import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy

from torch import Tensor

from src.data.structures.audio import Audio
from src.data.structures.melody import Melody
from src.data.structures.note import Note
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

        self.slice_size = self.slice_config.slice_size
        self.hop_size = self.slice_config.hop_size
    
    def slice_audio(self, audio: Audio) -> List[Audio]:
        """Нарезка аудио на окна.
        
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
    
    def slice_melody(self, melody: Melody, audio: Audio) -> List[Melody]:
        """Нарезка мелодии на окна в соответствии с окнами аудио.
        
        :param Melody melody: Объект мелодии
        :param Audio audio: Объект аудио (нужен для определения временных границ)
        :return List[Melody]: Список объектов мелодии для каждого окна
        """
        sliced_melody = []

        hop_length = self.spectrogram_config.hop_length
        
        slice_samples = self.slice_config.slice_size * hop_length
        hop_samples = self.slice_config.hop_size * hop_length
        total_samples = audio.waveform.shape[1]
        
        note_times = []
        current_time = 0.0
        
        for note in melody._notes:
            note_duration = melody._beats_to_seconds(note._duration)
            note_end = current_time + note_duration
            note_times.append((current_time, note_end, note))
            current_time = note_end
        
        if total_samples <= slice_samples:
      
            padding_samples = slice_samples - total_samples
            padding_duration = padding_samples / audio.sample_rate
            
            padded_melody = deepcopy(melody)
            
            if padding_duration > 0:
                pause_duration_beats = melody._seconds_to_beats(padding_duration)
                pause_note = Note(None, pause_duration_beats)
                padded_melody._notes.append(pause_note)
            
            sliced_melody.append(padded_melody)
        
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

            samples_to_copy = end_sample - start_sample
            
            if samples_to_copy < slice_samples:
                padded_window = torch.full((1, slice_samples), self.slice_config.audio_pad_value, dtype=audio.waveform.dtype)
                padded_window[:, :samples_to_copy] = audio.waveform[:, start_sample:end_sample]
                
                padding_samples = slice_samples - samples_to_copy
                padding_duration = padding_samples / audio.sample_rate
                
            else:
                padding_duration = 0
            
            window_start = start_sample / audio.sample_rate
            window_end = end_sample / audio.sample_rate
            
            window_notes = []
            notes_in_window = []

            for j, (note_start, note_end, note) in enumerate(note_times):

                if note_start < window_end and note_end > window_start:
                    intersection_start = max(note_start, window_start)
                    intersection_end = min(note_end, window_end)
                    
                    overlap_duration_sec = intersection_end - intersection_start
                    overlap_duration_beats = melody._seconds_to_beats(overlap_duration_sec)
                    
                    if note.is_rest:
                        window_notes.append(Note(None, overlap_duration_beats))

                    else:
                        window_notes.append(Note(note.note_name, overlap_duration_beats))
                    
                    notes_in_window.append(j+1)
            
            if not window_notes:
                pause_duration_beats = melody._seconds_to_beats(window_end - window_start)
                window_notes.append(Note(None, pause_duration_beats))
            
            if padding_duration > 0:
                pause_duration_beats = melody._seconds_to_beats(padding_duration)
                window_notes.append(Note(None, pause_duration_beats))
            
            window_melody = Melody(window_notes, melody._tempo)
            sliced_melody.append(window_melody)
    
        return sliced_melody
    
    
