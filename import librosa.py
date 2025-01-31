import librosa
import numpy as np

def extract_mel_spectrogram(audio_path, n_mels=80, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    # 重新调整样本率（如果需要）
    if _ != sr:
        audio = librosa.resample(audio, _, sr)
    # 计算梅尔频谱
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    return mel_spectrogram

# 加载外文参考音并提取梅尔频谱
reference_audio_path = "path_to_foreign_audio.wav"
mel_reference = extract_mel_spectrogram(reference_audio_path)