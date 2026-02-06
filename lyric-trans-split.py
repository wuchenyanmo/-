import torch
import torchaudio
import librosa
import numpy as np
from demucs import pretrained
from demucs.apply import apply_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gc

def extract_main_vocal(vocals_np):
    # vocals_np 形状为 (2, length) -> [左声道, 右声道]
    left = vocals_np[0]
    right = vocals_np[1]
    
    # 计算中置声道 (Mid)
    # 这能极大程度抵消掉左右声场差异较大的和声
    mid_channel = (left + right) / 2
    side =  (left - right) / 2
    
    # 如果想更极端一点，可以尝试：主唱 = Mid - abs(Side)
    # 但通常单声道化 (Mid) 已经能让和声的能量大幅下降
    return mid_channel - np.abs(side)

class LyricsTranscriber:
    def __init__(self, audio_path, device=None):
        self.audio_path = audio_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"正在使用设备: {self.device} (精度: {self.torch_dtype})")

    def load_and_separate(self):
        """[1/3] 加载音频并使用 Demucs 分离人声"""
        print(">>> [1/3] 正在加载音频并使用 Demucs 分离人声...")
        wav, sr = torchaudio.load(self.audio_path)
        if sr != 44100:
            wav = torchaudio.functional.resample(wav, sr, 44100)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        # 获取 htdemucs 模型
        model = pretrained.get_model('htdemucs')
        model.to(self.device)
        
        # 预处理
        wav_input = wav.unsqueeze(0).to(self.device)
        ref = wav_input.mean(0)
        wav_input = (wav_input - ref.mean()) / ref.std()
        
        # 推理
        sources = apply_model(model, wav_input, shifts=1, split=True, overlap=0.25)
        vocal_index = model.sources.index('vocals')
        vocals = sources[0, vocal_index]
        
        # 转移到 CPU 并转换为 numpy
        vocals_np = vocals.cpu().numpy()
        model_sr = 44100

        # === 关键：彻底清理 Demucs 显存 ===
        del model
        del sources
        del wav_input
        torch.cuda.empty_cache()
        gc.collect()
        print(">>> Demucs 显存已释放")

        return vocals_np, model_sr

    def smart_segment_vocals(self, vocals, sr, top_db=35, max_silence_break=3.0, padding=0.2,
                             min_vocal_duration = 1.0):
        """
        [2/3] 智能切分逻辑 (保持不变)
        合并短静音，仅在长静音(max_silence_break)处切断
        """
        print(f">>> [2/3] 正在智能切分 (保留 <{max_silence_break}s 的静音以维持上下文)...")

        # 转单声道
        if vocals.shape[0] > 1:
            vocals_mono = extract_main_vocal(vocals)
        else:
            vocals_mono = vocals[0]

        intervals = librosa.effects.split(
            vocals_mono, 
            top_db=top_db, 
            frame_length=2048, 
            hop_length=512
        )

        if len(intervals) == 0:
            return []

        merged_intervals = []
        current_start, current_end = intervals[0]
        
        for i in range(1, len(intervals)):
            next_start, next_end = intervals[i]
            silence_duration = (next_start - current_end) / sr
            
            if silence_duration < max_silence_break:
                # 合并
                current_end = next_end
            else:
                # 断开，只有大于阈值的片段才会被merge
                if((current_end - current_start) / sr > min_vocal_duration):
                    merged_intervals.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
        
        merged_intervals.append((current_start, current_end))

        segments = []
        total_samples = len(vocals_mono)
        pad_samples = int(padding * sr)

        for start_idx, end_idx in merged_intervals:
            adj_start = max(0, start_idx - pad_samples)
            adj_end = min(total_samples, end_idx + pad_samples)
            
            segments.append({
                "audio": vocals_mono[adj_start:adj_end],
                "start": adj_start / sr,
                "end": adj_end / sr
            })

        print(f"切分完成: 共 {len(segments)} 个片段")
        return segments

    def transcribe_segments(self, segments, input_sr):
        """
        [3/3] 使用 Transformers Pipeline 转写
        """
        print(">>> [3/3] 正在加载 Transformers Whisper 模型...")
        
        model_id = "whisper/whisper-large-v3"

        # 1. 加载模型
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(self.device)

        # 2. 加载处理器
        processor = AutoProcessor.from_pretrained(model_id)

        # 3. 创建 Pipeline
        # chunk_length_s=30 是 Whisper 的标准输入窗口
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            max_new_tokens=256,
            chunk_length_s=30,
        )

        print(">>> 模型加载完毕，开始转写...")
        results = []
        generate_kwargs = {
            "language": "japanese",
            "task": "transcribe",
            # "no_speech_threshold": 0.7,  # 判定为“无声”的阈值，默认通常是0.6。调高到0.8可强制其跳过模糊音频
            # "logprob_threshold": -0.8,    # 如果平均对数概率低于此值，则认为识别不可靠
        }

        for i, seg in enumerate(segments):
            audio = seg["audio"]
            
            # Transformers pipeline 接受 raw numpy array，但需要指定 sampling_rate
            # 必须重采样到 16000Hz
            if input_sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=input_sr, target_sr=16000)
            else:
                audio_16k = audio

            # 构造 Pipeline 输入格式
            input_features = {"array": audio_16k, "sampling_rate": 16000}

            # 执行转写
            # return_timestamps=True 可以获取单词级时间戳，这里我们只需要文本
            prediction = pipe(
                input_features, 
                generate_kwargs=generate_kwargs,
                return_timestamps=False 
            )
            
            text = prediction["text"].strip()
            
            if text:
                print(f"[{self.format_time(seg['start'])} -> {self.format_time(seg['end'])}] {text}")
                results.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text
                })
        
        # 清理显存
        del pipe, model, processor
        torch.cuda.empty_cache()
        
        return results

    @staticmethod
    def format_time(seconds):
        m, s = divmod(seconds, 60)
        return f"{int(m):02d}:{s:05.2f}"

    def run(self):
        # 1. Demucs 分离
        vocals, sr = self.load_and_separate()
        
        # 2. Librosa 智能切分 (保留上下文)
        segments = self.smart_segment_vocals(
            vocals, sr, 
            top_db=30, 
            max_silence_break=2.0,
            padding=0.8,
            min_vocal_duration=2.0
        )
        
        # 3. Transformers 转写
        lyrics_data = self.transcribe_segments(segments, input_sr=sr)
        return lyrics_data
    
if __name__ == "__main__":
    AUDIO_FILE = "music-source/鳥の詩.m4a" 
    
    import os
    if os.path.exists(AUDIO_FILE):
        transcriber = LyricsTranscriber(AUDIO_FILE)
        final_lyrics = transcriber.run()
        print(final_lyrics)
    else:
        print(f"未找到文件: {AUDIO_FILE}")