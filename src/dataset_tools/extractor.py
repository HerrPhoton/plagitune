import os
from pathlib import Path

from tqdm import tqdm
from moviepy.editor import VideoFileClip


def video_to_audio(video_dir: str, audio_dir: str):

    for video_file in tqdm(os.listdir(video_dir)):

        video_path = os.path.join(video_dir, video_file)
        audio_path = os.path.join(audio_dir, Path(video_file).with_suffix(".mp3"))

        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
