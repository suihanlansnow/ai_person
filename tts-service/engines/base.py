from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class EngineType(Enum):
    LLM = "llm"
    ASR = "asr"
    TTS = "tts"


class BaseEngine(ABC):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_trt: bool = False,
        **kwargs
    ):
        self.model_path = model_path
        self.device = device
        self.use_trt = use_trt
        self.kwargs = kwargs
        self._model = None
        self._is_initialized = False
        self._init_time: float = 0
        self._inference_count: int = 0
        self._total_inference_time: float = 0

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def engine_type(self) -> EngineType:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    def _record_inference(self, inference_time: float):
        self._inference_count += 1
        self._total_inference_time += inference_time

    def get_stats(self) -> Dict[str, Any]:
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0
        )
        return {
            "engine_type": self.engine_type.value,
            "is_initialized": self._is_initialized,
            "model_path": self.model_path,
            "device": self.device,
            "total_inferences": self._inference_count,
            "total_inference_time": round(self._total_inference_time, 3),
            "average_inference_time": round(avg_time, 3),
        }

    def reset_stats(self):
        self._inference_count = 0
        self._total_inference_time = 0