import os
import torch
import torchaudio
import numpy as np
import librosa

from demucs.pretrained import get_model
from demucs.apply import apply_model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model("htdemucs")
model.to(device)
model.eval()

def has_vocals_from_array(
    y: np.ndarray,
    sr=44100,
    origin = None,
    rms_db_threshold=-30,
    min_duration_sec=5.0,
    min_relative_vocal_time=0.2
) -> tuple[bool, float]:
    '''
    ndarray格式的输入音频，返回是否有人声以及人声时长（单位秒）
    '''
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

def detect_vocals_file(music_path) -> tuple[bool, float]:
    '''
    检测音频文件是否有人声，支持的格式为torchaudio.load支持的格式，返回是否有人声以及人声时长（单位秒）
    '''
    waveform, sr = torchaudio.load(music_path)

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