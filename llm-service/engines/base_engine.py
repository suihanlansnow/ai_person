from abc import abstractmethod
from typing import List, Dict, Any, Optional, Union
from ..base import BaseEngine, EngineType


class BaseLLMEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        max_model_len: int = 4096,
        tensor_parallel: int = 1,
        **kwargs
    ):
        super().__init__(model_path, device, **kwargs)
        self.tokenizer_path = tokenizer_path or model_path
        self.max_model_len = max_model_len
        self.tensor_parallel = tensor_parallel

    @property
    def engine_type(self) -> EngineType:
        return EngineType.LLM

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        pass

    @abstractmethod
    async def agenerate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        pass