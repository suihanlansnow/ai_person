from typing import Dict, Any, Optional, Union, Tuple, List
import time
import logging
import numpy as np
from engines.base_engine import BaseASREngine

logger = logging.getLogger(__name__)


class FunASREngine(BaseASREngine):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        sample_rate: int = 16000,
        use_onnx: bool = True,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_path, device, batch_size, sample_rate, **kwargs)
        self.use_onnx = use_onnx
        self.vad_model = vad_model
        self.punc_model = punc_model
        self._model = None
        self._vad_model = None
        self._punc_model = None

    def load_model(self) -> None:
        if self._is_initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        start_time = time.time()
        logger.info(f"Loading FunASR model from {self.model_path}")

        try:
            from funasr import AutoModel
            import os

            if os.path.exists(self.model_path):
                model_name = self.model_path
                logger.info(f"Loading local model from {model_name}")
            elif "paraformer" in self.model_path.lower():
                model_name = "paraformer-zh"
                logger.info(f"Using pre-trained model: {model_name}")
            else:
                model_name = "paraformer-zh"
                logger.info(f"Defaulting to model: {model_name}")

            self._model = AutoModel(
                model=model_name,
                model_revision="v2.0.4",
                vad_model=self.vad_model,
                vad_model_revision="v2.0.4",
                punc_model=self.punc_model,
                punc_model_revision="v2.0.4",
                device=self.device,
                disable_update=False,
                ncpu=4,
            )

            self._is_initialized = True
            self._init_time = time.time() - start_time
            logger.info(
                f"FunASR model loaded successfully in {self._init_time:.2f}s"
            )

        except ImportError:
            logger.error(
                "funasr not installed. Install with: pip install funasr"
            )
            raise

    def unload_model(self) -> None:
        self._model = None
        self._vad_model = None
        self._punc_model = None
        self._is_initialized = False
        logger.info("FunASR model unloaded")

    def recognize(
        self,
        audio: Union[bytes, str, np.ndarray],
        language: str = "zh",
        **kwargs
    ) -> Dict[str, Any]:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        start_time = time.time()

        audio_data = self._preprocess_audio(audio)
        
        # 计算音频时长（秒）
        audio_duration = len(audio_data) / self.sample_rate if isinstance(audio_data, np.ndarray) else 0

        result = self._model.generate(
            input=audio_data,
            language=language,
            batch_size=self.batch_size,
            **kwargs
        )

        inference_time = time.time() - start_time
        
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        text = result.get("text", "")
        
        # 记录详细统计信息
        self._record_inference(
            inference_time=inference_time,
            audio_duration=audio_duration,
            text_length=len(text),
            language=language
        )

        response = {
            "text": text,
            "segments": result.get("segments", []),
            "inference_time": round(inference_time, 3),
            "audio_duration": round(audio_duration, 3),
            "real_time_factor": round(inference_time / max(audio_duration, 0.001), 3),
        }

        logger.debug(f"ASR recognized: '{text}' in {inference_time:.3f}s (RTF: {response['real_time_factor']:.3f})")
        return response

    async def arecognize(
        self,
        audio: Union[bytes, str, np.ndarray],
        language: str = "zh",
        **kwargs
    ) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(self.recognize, audio, language, **kwargs)

    def recognize_batch(
        self,
        audio_list: List[Union[bytes, str, np.ndarray]],
        language: str = "zh",
        **kwargs
    ) -> List[Dict[str, Any]]:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        start_time = time.time()

        results = []
        for audio in audio_list:
            audio_data = self._preprocess_audio(audio)
            result = self._model.generate(
                input=audio_data,
                language=language,
                **kwargs
            )
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            results.append({
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
            })

        inference_time = time.time() - start_time
        self._record_inference(inference_time)

        logger.debug(
            f"Batch ASR: {len(results)} audios recognized in {inference_time:.3f}s"
        )
        return results

    def _preprocess_audio(
        self,
        audio: Union[bytes, str, np.ndarray]
    ) -> np.ndarray:
        if isinstance(audio, str):
            audio = self._load_audio_file(audio)
        elif isinstance(audio, bytes):
            audio = self._bytes_to_audio(audio)

        if isinstance(audio, np.ndarray):
            audio = self._normalize_audio(audio)

        return audio

    def _load_audio_file(self, file_path: str) -> np.ndarray:
        try:
            import soundfile as sf
            data, sr = sf.read(file_path)
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)
            return data
        except ImportError:
            logger.warning("soundfile not installed, using scipy")
            from scipy.io import wavfile
            from scipy.signal import resample
            sr, data = wavfile.read(file_path)
            if sr != self.sample_rate:
                num_samples = int(len(data) * self.sample_rate / sr)
                data = resample(data, num_samples)
            return data.astype(np.float32) / 32768.0

    def _bytes_to_audio(self, audio_bytes: bytes) -> np.ndarray:
        import io
        
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            logger.warning(f"Failed to read with soundfile: {e}, trying torchaudio")
            try:
                import torchaudio
                waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), format="wav")
                data = waveform.numpy()
                if len(data.shape) > 1:
                    data = data.mean(axis=0)
            except Exception as e2:
                logger.warning(f"Failed to read with torchaudio wav: {e2}, trying raw stream")
                try:
                    import torchaudio
                    from torchaudio.io import StreamReader
                    streamer = StreamReader(io.BytesIO(audio_bytes))
                    streamer.add_basic_audio_stream(frames_per_chunk=-1)
                    for (chunk,) in streamer.stream():
                        data = chunk.numpy()
                        if len(data.shape) > 1:
                            data = data.mean(axis=0)
                        sr = streamer.get_src_stream_info(0).sample_rate
                        break
                except Exception as e3:
                    logger.warning(f"Failed to read with torchaudio stream: {e3}, trying pydub")
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                        sr = audio.frame_rate
                        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                        if audio.channels > 1:
                            data = data.reshape(-1, audio.channels).mean(axis=1)
                    except Exception as e4:
                        logger.warning(f"Failed to read with pydub: {e4}, trying scipy fallback")
                        from scipy.io import wavfile
                        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        
        if sr != self.sample_rate:
            from scipy.signal import resample
            num_samples = int(len(data) * self.sample_rate / sr)
            data = resample(data, num_samples)
        
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        
        if data.max() > 1.0 or data.min() < -1.0:
            data = data / 32768.0
        
        return data

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0
        return audio