import io
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULTChannels = 1

    @staticmethod
    def bytes_to_audio(
        audio_bytes: bytes,
        expected_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes))
            return data, sr
        except ImportError:
            from scipy.io import wavfile
            from scipy.signal import resample
            sr, data = wavfile.read(io.BytesIO(audio_bytes))
            if sr != expected_sr:
                num_samples = int(len(data) * expected_sr / sr)
                data = resample(data, num_samples)
            return data.astype(np.float32) / 32768.0, expected_sr

    @staticmethod
    def audio_to_bytes(
        audio: np.ndarray,
        sample_rate: int = 16000,
        format: str = "WAV"
    ) -> bytes:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if audio.max() <= 1.0:
            audio = audio * 32768.0

        audio = np.clip(audio, -32768, 32767).astype(np.int16)

        try:
            import soundfile as sf
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format=format)
            return buffer.getvalue()
        except ImportError:
            from scipy.io import wavfile
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, audio)
            return buffer.getvalue()

    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        target_level: float = -20.0
    ) -> np.ndarray:
        if audio.dtype == np.float32:
            audio = audio.copy()
        else:
            audio = audio.astype(np.float32)

        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
            gain = 10 ** ((target_level - db) / 20)
            audio = audio * gain

        return np.clip(audio, -1.0, 1.0)

    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        if orig_sr == target_sr:
            return audio

        from scipy.signal import resample
        num_samples = int(len(audio) * target_sr / orig_sr)
        return resample(audio, num_samples)

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        threshold: float = 0.01,
        frame_length: int = 1024
    ) -> np.ndarray:
        energy = np.array([
            np.sqrt(np.mean(audio[i:i+frame_length]**2))
            for i in range(0, len(audio) - frame_length, frame_length)
        ])

        if len(energy) == 0:
            return audio

        max_energy = np.max(energy)
        if max_energy > 0:
            energy = energy / max_energy

        non_silent = energy > threshold

        if not np.any(non_silent):
            return audio

        start_frame = np.argmax(non_silent)
        end_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1

        start_sample = start_frame * frame_length
        end_sample = min(end_frame * frame_length + frame_length, len(audio))

        return audio[start_sample:end_sample]

    @staticmethod
    def add_noise(
        audio: np.ndarray,
        noise_level: float = 0.005
    ) -> np.ndarray:
        noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
        return audio + noise

    @staticmethod
    def calculate_duration(audio: np.ndarray, sample_rate: int) -> float:
        return len(audio) / sample_rate

    @staticmethod
    def detect_language(audio: np.ndarray, sample_rate: int) -> str:
        return "zh"
