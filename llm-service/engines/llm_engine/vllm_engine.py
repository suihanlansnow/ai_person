from typing import List, Dict, Any, Optional, Union
import time
import logging
from .base_engine import BaseLLMEngine

logger = logging.getLogger(__name__)


class VLLMEngine(BaseLLMEngine):
    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        max_model_len: int = 4096,
        tensor_parallel: int = 1,
        quantization: str = "fp16",
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(
            model_path,
            tokenizer_path,
            device,
            max_model_len,
            tensor_parallel,
            **kwargs
        )
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._llm = None
        self._tokenizer = None

    def load_model(self) -> None:
        if self._is_initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        start_time = time.time()
        logger.info(f"Loading VLLM model from {self.model_path}")

        try:
            from vllm import LLM, SamplingParam

            self._llm = LLM(
                model=self.model_path,
                tokenizer=self.tokenizer_path,
                tensor_parallel_size=self.tensor_parallel,
                max_model_len=self.max_model_len,
                quantization=self.quantization,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
            self._is_initialized = True
            self._init_time = time.time() - start_time
            logger.info(
                f"VLLM model loaded successfully in {self._init_time:.2f}s"
            )
        except ImportError:
            logger.error(
                "vLLM not installed. Install with: pip install vllm"
            )
            raise

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._is_initialized = False
        logger.info("VLLM model unloaded")

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        from vllm import SamplingParams

        start_time = time.time()

        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )

        outputs = self._llm.generate(prompts, sampling_params)

        inference_time = time.time() - start_time
        self._record_inference(inference_time)

        results = [output.outputs[0].text for output in outputs]
        logger.debug(f"Generated {len(results)} outputs in {inference_time:.3f}s")

        return results

    async def agenerate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        import asyncio
        return await asyncio.to_thread(
            self.generate,
            prompts,
            max_tokens,
            temperature,
            top_p,
            stop,
            **kwargs
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")

        prompt = self._format_messages(messages)
        results = self.generate([prompt], max_tokens, temperature, **kwargs)
        return results[0] if results else ""

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        if not self._tokenizer:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True
                )
            except ImportError:
                logger.warning("transformers not installed, using raw message format")
                return self._raw_format(messages)

        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _raw_format(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role}: {content}\n"
        formatted += "assistant: "
        return formatted