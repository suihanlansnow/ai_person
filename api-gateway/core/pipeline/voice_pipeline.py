from typing import Dict, Any, Optional, List
import time
import logging
import asyncio
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ..dialogue.context import DialogueContext
from ..dialogue.memory import DialogueMemory
from ..dialogue.emotion import EmotionDetector
from ..cache.cache_manager import CacheManager
from engines.asr_engine.funasr_engine import FunASREngine
from engines.llm_engine.vllm_engine import VLLMEngine
from engines.tts_engine.gpt_sovits_engine import GPTSovitsEngine

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    text: str
    audio: Optional[Any] = None
    sample_rate: int = 24000
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    asr_time: float = 0.0
    llm_time: float = 0.0
    tts_time: float = 0.0
    total_time: float = 0.0
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "sample_rate": self.sample_rate,
            "emotion": self.emotion,
            "emotion_confidence": self.emotion_confidence,
            "timings": {
                "asr": round(self.asr_time, 3),
                "llm": round(self.llm_time, 3),
                "tts": round(self.tts_time, 3),
                "total": round(self.total_time, 3),
            },
            "cached": self.cached,
        }


class VoicePipeline:
    def __init__(
        self,
        asr_engine: Optional[FunASREngine] = None,
        llm_engine: Optional[VLLMEngine] = None,
        tts_engine: Optional[GPTSovitsEngine] = None,
        cache_manager: Optional[CacheManager] = None,
        memory: Optional[DialogueMemory] = None,
        emotion_detector: Optional[EmotionDetector] = None,
        system_prompt: Optional[str] = None,
        enable_cache: bool = True,
        enable_parallel: bool = True,
    ):
        self.asr_engine = asr_engine
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.cache_manager = cache_manager or CacheManager()
        self.memory = memory
        self.emotion_detector = emotion_detector or EmotionDetector()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.enable_cache = enable_cache
        self.enable_parallel = enable_parallel

        self._executor = ThreadPoolExecutor(max_workers=4)

    def _default_system_prompt(self) -> str:
        return """你是一个有情感陪伴能力的AI助手。
你能够理解用户的情绪并给予适当的回应。
保持对话自然、亲切、有同理心。
记住重要的对话内容以便提供更好的个性化服务。"""

    async def process(
        self,
        audio_data: Any,
        session_id: str,
        voice_id: str = "default",
        language: str = "zh",
        return_audio: bool = True,
    ) -> PipelineResult:
        start_time = time.time()

        asr_result = await self._recognize(audio_data, language)
        asr_time = time.time() - start_time

        if not asr_result.get("text", "").strip():
            return PipelineResult(
                text="",
                total_time=time.time() - start_time,
                cached=False,
            )

        emotion_result = self.emotion_detector.detect(asr_result["text"])

        llm_result = await self._generate_response(
            prompt=asr_result["text"],
            session_id=session_id,
            emotion=emotion_result.get("emotion", "neutral"),
        )
        llm_time = time.time() - asr_time - start_time

        tts_result = None
        tts_time = 0.0

        if return_audio and self.tts_engine:
            cached_audio = None
            if self.enable_cache:
                cached_audio = self.cache_manager.tts_cache.get_tts_audio(
                    text=llm_result["text"],
                    voice_id=voice_id,
                    emotion=emotion_result.get("emotion"),
                )

            if cached_audio is not None:
                tts_result = {
                    "audio": cached_audio,
                    "cached": True,
                }
            else:
                tts_result = await self._synthesize(
                    text=llm_result["text"],
                    voice_id=voice_id,
                    emotion=emotion_result.get("emotion"),
                )
                if self.enable_cache and tts_result:
                    self.cache_manager.tts_cache.set_tts_audio(
                        text=llm_result["text"],
                        audio=tts_result.get("audio"),
                        voice_id=voice_id,
                        emotion=emotion_result.get("emotion"),
                    )

            if tts_result:
                tts_time = tts_result.get("inference_time", 0.0)

        total_time = time.time() - start_time

        return PipelineResult(
            text=llm_result["text"],
            audio=tts_result.get("audio") if tts_result else None,
            sample_rate=self.tts_engine.sample_rate if self.tts_engine else 24000,
            emotion=emotion_result.get("emotion", "neutral"),
            emotion_confidence=emotion_result.get("confidence", 0.0),
            asr_time=asr_time,
            llm_time=llm_time,
            tts_time=tts_time,
            total_time=total_time,
            cached=tts_result.get("cached", False) if tts_result else False,
        )

    async def _recognize(self, audio_data: Any, language: str) -> Dict[str, Any]:
        if self.enable_cache and hasattr(self.cache_manager, 'llm_cache'):
            pass

        if self.asr_engine:
            return await self.asr_engine.arecognize(audio_data, language)

        return {"text": "", "segments": []}

    async def _generate_response(
        self,
        prompt: str,
        session_id: str,
        emotion: str = "neutral",
    ) -> Dict[str, Any]:
        start_time = time.time()

        cached_response = None
        if self.enable_cache:
            cached_response = self.cache_manager.llm_cache.get_llm_response(
                prompt=prompt,
            )

        if cached_response:
            return {
                "text": cached_response,
                "cached": True,
                "inference_time": 0.0,
            }

        messages = []
        if self.system_prompt:
            emotion_prompt = f"{self.system_prompt}\n\n当前用户情绪状态: {emotion}"
            messages.append({"role": "system", "content": emotion_prompt})

        if self.memory:
            history = self.memory.get_messages_for_llm(
                session_id,
                max_turns=5
            )
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        if self.llm_engine:
            response = await self.llm_engine.agenerate(
                prompts=[messages],
                max_tokens=2048,
                temperature=0.7,
            )
            text = response[0] if isinstance(response, list) else response
        else:
            text = "抱歉，我现在无法回答。"

        inference_time = time.time() - start_time

        if self.memory:
            self.memory.add_turn(
                session_id=session_id,
                user_message=prompt,
                assistant_message=text,
            )

        if self.enable_cache:
            self.cache_manager.llm_cache.set_llm_response(
                prompt=prompt,
                response=text,
            )

        return {
            "text": text,
            "cached": False,
            "inference_time": inference_time,
        }

    async def _synthesize(
        self,
        text: str,
        voice_id: str = "default",
        emotion: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.tts_engine:
            return None

        emotion_params = self.emotion_detector.get_tts_emotion_params(emotion or "neutral")

        return await self.tts_engine.asynthesize(
            text=text,
            voice_id=voice_id,
            emotion=emotion,
            speed=emotion_params.get("speed", 1.0),
        )

    def get_context(self, session_id: str) -> Optional[DialogueContext]:
        return None

    def shutdown(self):
        self._executor.shutdown(wait=True)
        logger.info("VoicePipeline shutdown")
