from typing import Dict, List, Optional, Any, Callable
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResponseConfig:
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None


class ResponseGenerator:
    def __init__(
        self,
        llm_engine: Any,
        system_prompt: Optional[str] = None,
        default_config: Optional[ResponseConfig] = None,
    ):
        self.llm_engine = llm_engine
        self.system_prompt = system_prompt or ""
        self.default_config = default_config or ResponseConfig()

    def generate(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        config: Optional[ResponseConfig] = None,
        **kwargs
    ) -> str:
        config = config or self.default_config

        messages = self._build_messages(user_input, context)

        try:
            if hasattr(self.llm_engine, 'chat'):
                response = self.llm_engine.chat(
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    **kwargs
                )
            else:
                response = self.llm_engine.generate(
                    prompts=[messages],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop=config.stop,
                    **kwargs
                )
                response = response[0] if isinstance(response, list) else response

            return response

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "抱歉，我遇到了一些问题，请稍后再试。"

    async def agenerate(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        config: Optional[ResponseConfig] = None,
        **kwargs
    ) -> str:
        import asyncio
        return await asyncio.to_thread(
            self.generate, user_input, context, config, **kwargs
        )

    def _build_messages(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        if context:
            messages.extend(context)

        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages


class EmotionAwareGenerator(ResponseGenerator):
    def __init__(
        self,
        llm_engine: Any,
        emotion_detector: Any,
        system_prompt: Optional[str] = None,
        default_config: Optional[ResponseConfig] = None,
    ):
        super().__init__(llm_engine, system_prompt, default_config)
        self.emotion_detector = emotion_detector

    def generate(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        config: Optional[ResponseConfig] = None,
        emotion_hint: Optional[str] = None,
        **kwargs
    ) -> str:
        emotion_result = self.emotion_detector.detect(user_input, context)
        detected_emotion = emotion_hint or emotion_result.get("emotion", "neutral")

        emotion_prompt_suffix = self._get_emotion_prompt_suffix(detected_emotion)

        temp_system_prompt = self.system_prompt + emotion_prompt_suffix

        original_system = self.system_prompt
        self.system_prompt = temp_system_prompt

        try:
            response = super().generate(user_input, context, config, **kwargs)
        finally:
            self.system_prompt = original_system

        return response

    def _get_emotion_prompt_suffix(self, emotion: str) -> str:
        suffixes = {
            "happy": "\n\n请用温暖、愉快的语气回应，分享用户的快乐。",
            "sad": "\n\n请用温柔、同情的语气安慰用户，表达理解和关心。",
            "angry": "\n\n请用平和、理解的语气回应，不要争辩，先表达理解。",
            "fear": "\n\n请用镇定、安抚的语气帮助用户放松。",
            "surprise": "\n\n请用好奇、理解的语气回应。",
            "disgust": "\n\n请用中立、同理的语气回应。",
            "neutral": "",
        }
        return suffixes.get(emotion, "")


class StreamingGenerator(ResponseGenerator):
    def __init__(
        self,
        llm_engine: Any,
        system_prompt: Optional[str] = None,
        default_config: Optional[ResponseConfig] = None,
    ):
        super().__init__(llm_engine, system_prompt, default_config)

    def generate_stream(
        self,
        user_input: str,
        context: Optional[List[Dict[str, str]]] = None,
        config: Optional[ResponseConfig] = None,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        config = config or self.default_config
        messages = self._build_messages(user_input, context)

        try:
            full_response = ""

            for chunk in self._stream_generate(messages, config, **kwargs):
                full_response += chunk
                if callback:
                    callback(chunk)

            return full_response

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            return "抱歉，我遇到了一些问题，请稍后再试。"

    def _stream_generate(
        self,
        messages: List[Dict[str, str]],
        config: ResponseConfig,
        **kwargs
    ):
        if hasattr(self.llm_engine, 'generate_stream'):
            for chunk in self.llm_engine.generate_stream(
                prompts=[messages],
                **kwargs
            ):
                yield chunk
        else:
            response = self.llm_engine.generate(
                prompts=[messages],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                **kwargs
            )
            yield response[0] if isinstance(response, list) else response
