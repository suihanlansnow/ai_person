import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    user: str
    assistant: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class ShortTermMemory:
    def __init__(
        self,
        persist_path: str = "./data/memory/dialogue_memory.json",
        max_turns: int = 10,
        max_sessions: int = 1000,
        write_interval: float = 5.0,
        enable_redis: bool = False,
        redis_url: Optional[str] = None,
    ):
        self.persist_path = Path(persist_path)
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.write_interval = write_interval
        self._lock = threading.Lock()
        self._memory: Dict[str, List[Dict]] = {}
        
        # 异步写入相关
        self._write_queue: deque = deque()
        self._pending_writes = False
        self._write_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Redis支持
        self._enable_redis = enable_redis and redis_url is not None
        self._redis_client = None
        self._redis_cache: Dict[str, List[Dict]] = {}
        
        # 性能统计
        self._stats = {
            "total_add_turns": 0,
            "total_saves": 0,
            "total_loads": 0,
            "save_time_total": 0.0,
            "load_time_total": 0.0,
        }
        
        self._load()

    def _load(self):
        start_time = time.time()
        if self.persist_path.exists():
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    self._memory = json.load(f)
                self._stats["total_loads"] += 1
                self._stats["load_time_total"] += time.time() - start_time
                logger.info(f"Loaded {len(self._memory)} sessions from disk")
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
                self._memory = {}
        else:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory = {}
    
    def _save(self):
        start_time = time.time()
        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self._memory, f, ensure_ascii=False, indent=2)
            self._stats["total_saves"] += 1
            self._stats["save_time_total"] += time.time() - start_time
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _save_async(self):
        """标记需要异步写入"""
        self._pending_writes = True
        if self._loop and self._write_task is None:
            self._write_task = self._loop.create_task(self._background_writer())

    async def _background_writer(self):
        """后台异步写入任务"""
        while True:
            await asyncio.sleep(self.write_interval)
            if self._pending_writes:
                with self._lock:
                    self._save()
                    self._pending_writes = False
                self._write_task = None
                logger.debug("Background save completed")

    def initialize_async(self):
        """初始化异步写入循环（需要在事件循环中调用）"""
        try:
            self._loop = asyncio.get_event_loop()
            logger.info("Async writer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize async writer: {e}")

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None,
    ):
        with self._lock:
            if session_id not in self._memory:
                self._memory[session_id] = []

            turn = {
                "user": user_message,
                "assistant": assistant_message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            if metadata:
                turn["metadata"] = metadata

            self._memory[session_id].append(turn)
            self._stats["total_add_turns"] += 1

            if len(self._memory[session_id]) > self.max_turns:
                self._memory[session_id] = self._memory[session_id][-self.max_turns:]

            if len(self._memory) > self.max_sessions:
                oldest = min(
                    self._memory.keys(),
                    key=lambda k: self._memory[k][0]["timestamp"] if self._memory[k] else ""
                )
                del self._memory[oldest]

            # 异步写入（不阻塞主线程）
            self._save_async()

            # 同时写入Redis（如果启用）
            if self._enable_redis and self._redis_client:
                self._save_to_redis_async(session_id, turn)

    def _save_to_redis_async(self, session_id: str, turn: Dict):
        """异步写入Redis"""
        try:
            key = f"dialogue:session:{session_id}"
            pipe = self._redis_client.pipeline()
            pipe.rpush(key, json.dumps(turn, ensure_ascii=False))
            pipe.expire(key, 86400 * 7)
            pipe.execute()
            logger.debug(f"Saved turn to Redis: {session_id}")
        except Exception as e:
            logger.error(f"Redis write error: {e}")

    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> List[Dict]:
        with self._lock:
            history = self._memory.get(session_id, [])
            if max_turns:
                history = history[-max_turns:]
            return history

    def clear_session(self, session_id: str):
        with self._lock:
            if session_id in self._memory:
                del self._memory[session_id]
                self._save_async()
                
            if self._enable_redis and self._redis_client:
                try:
                    self._redis_client.delete(f"dialogue:session:{session_id}")
                except Exception as e:
                    logger.error(f"Redis delete error: {e}")

    def delete_session(self, session_id: str):
        self.clear_session(session_id)

    def get_all_sessions(self) -> List[str]:
        with self._lock:
            return list(self._memory.keys())

    def search(self, query: str, session_id: Optional[str] = None) -> List[Dict]:
        results = []
        with self._lock:
            sessions = [session_id] if session_id else self._memory.keys()

            for sid in sessions:
                for turn in self._memory.get(sid, []):
                    if query.lower() in turn["user"].lower() or query.lower() in turn["assistant"].lower():
                        results.append({
                            "session_id": sid,
                            **turn
                        })

            return results

    def get_stats(self) -> Dict:
        with self._lock:
            total_turns = sum(len(v) for v in self._memory.values())
            avg_save_time = (
                self._stats["save_time_total"] / self._stats["total_saves"]
                if self._stats["total_saves"] > 0 else 0
            )
            avg_load_time = (
                self._stats["load_time_total"] / self._stats["total_loads"]
                if self._stats["total_loads"] > 0 else 0
            )
            
            return {
                "total_sessions": len(self._memory),
                "total_turns": total_turns,
                "average_turns_per_session": total_turns / len(self._memory) if self._memory else 0,
                "persistence_stats": {
                    "total_loads": self._stats["total_loads"],
                    "total_saves": self._stats["total_saves"],
                    "average_save_time_ms": round(avg_save_time * 1000, 2),
                    "average_load_time_ms": round(avg_load_time * 1000, 2),
                    "pending_writes": self._pending_writes,
                },
                "redis_enabled": self._enable_redis,
            }

    def reset_stats(self):
        with self._lock:
            self._stats = {
                "total_add_turns": 0,
                "total_saves": 0,
                "total_loads": 0,
                "save_time_total": 0.0,
                "load_time_total": 0.0,
            }


class RedisMemory:
    """Redis-backed memory storage with memory cache"""
    
    def __init__(
        self,
        redis_url: str,
        memory_cache: Optional[ShortTermMemory] = None,
        session_ttl: int = 86400 * 7,
    ):
        import redis
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self._memory_cache = memory_cache
        self._session_ttl = session_ttl
        self._lock = threading.Lock()
        
    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict] = None,
    ):
        turn = {
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {}
        }
        
        key = f"dialogue:session:{session_id}"
        
        with self._lock:
            try:
                pipe = self.redis.pipeline()
                pipe.rpush(key, json.dumps(turn, ensure_ascii=False))
                pipe.expire(key, self._session_ttl)
                pipe.execute()
                
                # 同时更新内存缓存
                if self._memory_cache:
                    self._memory_cache.add_turn(session_id, user_message, assistant_message, metadata)
                    
            except Exception as e:
                logger.error(f"Redis add_turn error: {e}")
                # 回退到内存缓存
                if self._memory_cache:
                    self._memory_cache.add_turn(session_id, user_message, assistant_message, metadata)
    
    def get_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict]:
        # 优先从内存缓存获取
        if self._memory_cache:
            return self._memory_cache.get_history(session_id, max_turns)
        
        key = f"dialogue:session:{session_id}"
        try:
            turns = self.redis.lrange(key, 0, -1)
            history = [json.loads(turn) for turn in turns]
            if max_turns:
                history = history[-max_turns:]
            return history
        except Exception as e:
            logger.error(f"Redis get_history error: {e}")
            return []
    
    def delete_session(self, session_id: str):
        key = f"dialogue:session:{session_id}"
        try:
            self.redis.delete(key)
            if self._memory_cache:
                self._memory_cache.delete_session(session_id)
        except Exception as e:
            logger.error(f"Redis delete_session error: {e}")
    
    def get_all_sessions(self) -> List[str]:
        try:
            keys = self.redis.keys("dialogue:session:*")
            return [key.replace("dialogue:session:", "") for key in keys]
        except Exception as e:
            logger.error(f"Redis get_all_sessions error: {e}")
            return []
