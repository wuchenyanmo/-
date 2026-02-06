import os
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model("htdemucs")
model.to(device)
model.eval()

import librosa

def has_vocals_from_array(
    y,
    sr=44100,
    origin = None,
    rms_db_threshold=-30,
    min_duration_sec=5.0,
    min_relative_vocal_time=0.2
):
    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512
    )[0]

    if(origin is None):
        ref = 1.0
    else:
        ref = np.max(origin)
    rms_db = librosa.amplitude_to_db(rms, ref=ref)

    voiced = rms_db > rms_db_threshold
    voiced_time = voiced.sum() * 512 / sr
    song_time = rms_db.shape[0] * 512 / sr
    
    is_vocal = (voiced_time >= min_duration_sec) and (voiced_time >= min_relative_vocal_time * song_time)
    return is_vocal, voiced_time

def detect_vocals_m4a(m4a_path):
    waveform, sr = torchaudio.load(m4a_path)

    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    waveform = waveform.to(device)

    with torch.no_grad():
        sources = apply_model(
            model,
            waveform.unsqueeze(0),
            device=device,
            progress=False,
        )

    vocals = sources[0, model.sources.index("vocals")]
    vocals = vocals.mean(dim=0).cpu().numpy()

    return has_vocals_from_array(vocals, origin=waveform.mean(dim=0).cpu().numpy())

if __name__ == '__main__':
    # detect_vocals_m4a("music-source/ポケットをふくらませて.m4a")
    
    music_dir = "./music-source"
    for file in os.listdir(music_dir):
        if file.endswith('.m4a'):
            print(file, ':')
            is_vocal, vocal_time = detect_vocals_m4a(os.path.join(music_dir, file))
            print(f"is vocal: {is_vocal}, vocal time: {vocal_time}")
    