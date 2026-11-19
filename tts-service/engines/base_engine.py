from abc import abstractmethod
from typing import Dict, Any, Optional, Union
from ..base import BaseEngine, EngineType


class BaseTTSEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
        sample_rate: int = 24000,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.config_path = config_path
        self.sample_rate = sample_rate

    @property
    def engine_type(self) -> EngineType:
        return EngineType.TTS

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def asynthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        emotion: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def list_voices(self) -> list:
        pass