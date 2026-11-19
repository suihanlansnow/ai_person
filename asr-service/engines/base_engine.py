from abc import abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
from .base import BaseEngine, EngineType


class BaseASREngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.batch_size = batch_size
        self.sample_rate = sample_rate

    @property
    def engine_type(self) -> EngineType:
        return EngineType.ASR

    @abstractmethod
    def recognize(
        self,
        audio: Union[bytes, str, bytes],
        language: str = "zh",
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def arecognize(
        self,
        audio: Union[bytes, str, bytes],
        language: str = "zh",
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def recognize_batch(
        self,
        audio_list: list,
        language: str = "zh",
        **kwargs
    ) -> list:
        pass