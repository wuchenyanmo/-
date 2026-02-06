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

# from qwen_asr import Qwen3ASRModel
# qwenModel = Qwen3ASRModel.from_pretrained(
#     "Qwen/Qwen3-ASR-1.7B",
#     dtype=torch.bfloat16,
#     device_map="cuda:0",
#     attn_implementation="flash_attention_2",
#     max_inference_batch_size=1, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
#     max_new_tokens=1024, # Maximum number of tokens to generate. Set a larger value for long audio input.
# )

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, __version__
print(__version__)
model_id = "whisper/whisper-large-v3"
whispermodel = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    use_safetensors=True,
)
model.to(device)
whisperprocessor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=whispermodel,
    tokenizer=whisperprocessor.tokenizer,
    feature_extractor=whisperprocessor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=1,
    return_timestamps=True,
    torch_dtype=torch.bfloat16,
    device=device,
)

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

def prepare_for_whisper(vocals, sr=44100):
    """
    vocals: torch.Tensor (2, T)
    """
    if(len(vocals.shape) > 1):
        y = vocals.mean(dim=0).cpu().numpy()
    else:
        y = vocals.cpu().numpy()
    y: np.ndarray

    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y

def transcribe_lyrics(music_path):
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
    vocals_mono = vocals.mean(dim=0).cpu().numpy()
    
    is_vocal, vocal_time = has_vocals_from_array(y=vocals_mono, origin=waveform.mean(dim=0).cpu().numpy())
    del waveform
    if(not is_vocal):
        return None
    else:
        y_16k = prepare_for_whisper(vocals)
        del vocals
        result = pipe(
                y_16k,
                generate_kwargs={"language": "japanese"}
            )
        return result

if __name__ == '__main__':
    # detect_vocals_m4a("music-source/ポケットをふくらませて.m4a")
    
    res = transcribe_lyrics("music-source/鳥の詩.m4a")
    # res = pipe(
    #             "separated/htdemucs/夜奏花/vocals.wav",
    #             generate_kwargs={"language": "japanese"}  # 或 None 自动检测
    #         )
    print(res['text'])
    for i in res['chunks']:
        print(res['text'])
        
    # music_dir = "./music-source"
    # for file in os.listdir(music_dir):
    #     if file.endswith('.m4a'):
    #         print(file, ':')
    #         is_vocal, vocal_time = detect_vocals_m4a(os.path.join(music_dir, file))
    #         print(f"is vocal: {is_vocal}, vocal time: {vocal_time}")
    