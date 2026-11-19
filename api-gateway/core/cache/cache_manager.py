import hashlib
import json
import time
import logging
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import threading

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    def __init__(
        self,
        max_size_mb: int = 1024,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None
    ):
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _generate_key(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    def _serialize(self, value: Any) -> bytes:
        pass

    @abstractmethod
    def _deserialize(self, data: bytes) -> Any:
        pass

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None

            entry = self._cache[key]
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                return None

            logger.debug(f"Cache hit: {key[:20]}...")
            return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            ttl = ttl or self.ttl_seconds
            serialized = self._serialize(value)

            self._cache[key] = {
                "value": value,
                "serialized": serialized,
                "expires_at": time.time() + ttl,
                "size_mb": len(serialized) / (1024 * 1024),
            }

            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")

    def _evict_if_needed(self) -> None:
        total_size = sum(entry["size_mb"] for entry in self._cache.values())

        if total_size > self.max_size_mb:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1]["expires_at"]
            )

            while total_size > self.max_size_mb * 0.8 and sorted_entries:
                key, entry = sorted_entries.pop(0)
                total_size -= entry["size_mb"]
                del self._cache[key]

            logger.debug(f"Evicted entries, cache size: {total_size:.2f}MB")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total_size = sum(entry["size_mb"] for entry in self._cache.values())
            valid_entries = sum(
                1 for entry in self._cache.values()
                if time.time() <= entry["expires_at"]
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "total_size_mb": round(total_size, 2),
                "max_size_mb": self.max_size_mb,
            }


class LLMCache(BaseCache):
    def __init__(
        self,
        max_size_mb: int = 512,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None
    ):
        super().__init__(max_size_mb, ttl_seconds, cache_dir)

    def _generate_key(
        self,
        prompt: str,
        messages: Optional[list] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        key_data = {
            "prompt": prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _serialize(self, value: Any) -> bytes:
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    def get_llm_response(
        self,
        prompt: str,
        messages: Optional[list] = None,
        **kwargs
    ) -> Optional[str]:
        key = self._generate_key(prompt, messages, **kwargs)
        return self.get(key)

    def set_llm_response(
        self,
        prompt: str,
        response: str,
        messages: Optional[list] = None,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        key = self._generate_key(prompt, messages, **kwargs)
        self.set(key, response, ttl)


class TTSCache(BaseCache):
    def __init__(
        self,
        max_size_mb: int = 256,
        ttl_seconds: int = 7200,
        cache_dir: Optional[str] = None
    ):
        super().__init__(max_size_mb, ttl_seconds, cache_dir)

    def _generate_key(
        self,
        text: str,
        voice_id: str = "default",
        emotion: Optional[str] = None,
        **kwargs
    ) -> str:
        key_data = {
            "text": text,
            "voice_id": voice_id,
            "emotion": emotion,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _serialize(self, value: Any) -> bytes:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tobytes()
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        import numpy as np
        try:
            return np.frombuffer(data, dtype=np.float32)
        except Exception:
            return pickle.loads(data)

    def get_tts_audio(
        self,
        text: str,
        voice_id: str = "default",
        emotion: Optional[str] = None,
        **kwargs
    ) -> Optional[Any]:
        key = self._generate_key(text, voice_id, emotion, **kwargs)
        return self.get(key)

    def set_tts_audio(
        self,
        text: str,
        audio: Any,
        voice_id: str = "default",
        emotion: Optional[str] = None,
        ttl: Optional[int] = None,
        **kwargs
    ) -> None:
        key = self._generate_key(text, voice_id, emotion, **kwargs)
        self.set(key, audio, ttl)


class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        llm_cache_size_mb: int = 512,
        tts_cache_size_mb: int = 256,
        cache_dir: Optional[str] = None
    ):
        if self._initialized:
            return

        self.cache_dir = cache_dir or "./data/cache"
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.llm_cache = LLMCache(
            max_size_mb=llm_cache_size_mb,
            ttl_seconds=3600,
            cache_dir=f"{self.cache_dir}/llm"
        )

        self.tts_cache = TTSCache(
            max_size_mb=tts_cache_size_mb,
            ttl_seconds=7200,
            cache_dir=f"{self.cache_dir}/tts"
        )

        self._initialized = True
        logger.info("CacheManager initialized")

    def clear_all(self) -> None:
        self.llm_cache.clear()
        self.tts_cache.clear()
        logger.info("All caches cleared")

    def stats(self) -> Dict[str, Any]:
        return {
            "llm_cache": self.llm_cache.stats(),
            "tts_cache": self.tts_cache.stats(),
        }
