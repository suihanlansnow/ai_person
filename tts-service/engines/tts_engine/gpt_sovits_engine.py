from typing import Dict, Any, Optional, Union
import time
import logging
import numpy as np
from .base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)


class GPTSovitsEngine(BaseTTSEngine):
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        sample_rate: int = 24000,
        use_trt: bool = False,
        **kwargs
    ):
        super().__init__(model_path, config_path, device, sample_rate, **kwargs)
        self.use_trt = use_trt
        self._model = None
        self._config = None
        self._hps = None
        self._available_voices = {}

    def load_model(self) -> None:
        if self._is_initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        start_time = time.time()
        logger.info(f"Loading GPT-SoVITS model from {self.model_path}")

        try:
            from GPT_SoVITS import GPT_SoVITS_Inference

            self._load_config()

            self._model = GPT_SoVITS_Inference(
                model_path=self.model_path,
                config_path=self.config_path,
                device=self.device,
                is_trt=self.use_trt,
            )

            self._init_default_voices()

            self._is_initialized = True
            self._init_time = time.time() - start_time
            logger.info(
                f"GPT-SoVITS model loaded successfully in {self._init_time:.2f}s"
            )

        except ImportError:
            logger.error(
                "GPT-SoVITS not installed. "
                "Install from: https://github.com/Soulghost/GPT-SoVITS"
            )
            raise

    def _load_config(self) -> None:
        import json
        from pathlib import Path

        config_file = self.config_path or Path(self.model_path) / "config.json"

        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
        else:
            logger.warning(f"Config file not found: {config_file}")
            self._config = {}

    def _init_default_voices(self) -> None:
        self._available_voices = {
            "default": {
                "id": "default",
                "name": "Default Voice",
                "path": None,
            }
        }

        voices_dir = Path(self.model_path).parent / "voices"
        if voices_dir.exists():
            for voice_file in voices_dir.glob("*.pth"):
                voice_id = voice_file.stem
                self._available_voices[voice_id] = {
                    "id": voice_id,
                    "name": voice_id.replace("_", " ").title(),
                    "path": str(voice_file),
                }

    def unload_model(self) -> None:
        self._model = None
        self._config = None
        self._is_initialized = False
        logger.info("GPT-SoVITS model unloaded")

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        start_time = time.time()

        voice_info = self._get_voice_info(voice_id)

        if emotion and hasattr(self._model, 'set_emotion'):
            self._model.set_emotion(emotion)

        audio = self._model.generate(
            text=text,
            ref_audio=voice_info.get("path"),
            ref_text=kwargs.get("ref_text"),
            prompt_text=kwargs.get("prompt_text"),
            prompt_audio=kwargs.get("prompt_audio"),
            top_k=kwargs.get("top_k", 5),
            top_p=kwargs.get("top_p", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            speed=kwargs.get("speed", 1.0),
        )

        inference_time = time.time() - start_time
        self._record_inference(inference_time)

        result = {
            "audio": audio,
            "sample_rate": self.sample_rate,
            "text": text,
            "voice_id": voice_id or "default",
            "emotion": emotion,
            "inference_time": round(inference_time, 3),
        }

        logger.debug(
            f"TTS synthesized: '{text[:50]}...' "
            f"with voice '{voice_id}' in {inference_time:.3f}s"
        )

        return result

    async def asynthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import asyncio
        return await asyncio.to_thread(
            self.synthesize, text, voice_id, emotion, **kwargs
        )

    def list_voices(self) -> list:
        return list(self._available_voices.values())

    def _get_voice_info(self, voice_id: Optional[str]) -> Dict[str, Any]:
        if voice_id is None:
            return self._available_voices.get("default", {})
        return self._available_voices.get(
            voice_id,
            self._available_voices.get("default", {})
        )

    def add_custom_voice(
        self,
        voice_id: str,
        voice_name: str,
        reference_audio: str,
        reference_text: Optional[str] = None
    ) -> None:
        self._available_voices[voice_id] = {
            "id": voice_id,
            "name": voice_name,
            "path": reference_audio,
            "ref_text": reference_text,
        }
        logger.info(f"Added custom voice: {voice_id}")