from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import OrderedDict
import json
import threading


class DialogueMemory:
    def __init__(
        self,
        max_turns: int = 10,
        summary_mode: bool = False,
        persist_path: Optional[str] = None
    ):
        self.max_turns = max_turns
        self.summary_mode = summary_mode
        self.persist_path = persist_path
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()

        if persist_path:
            self._load()

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []

            turn = {
                "user": user_message,
                "assistant": assistant_message,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            self._sessions[session_id].append(turn)

            if len(self._sessions[session_id]) > self.max_turns:
                self._sessions[session_id].pop(0)

            if self.persist_path:
                self._save()

    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            history = self._sessions.get(session_id, [])
            if max_turns:
                history = history[-max_turns:]
            return history

    def get_messages_for_llm(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        history = self.get_history(session_id, max_turns)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        return messages

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].clear()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                if self.persist_path:
                    self._save()
                return True
            return False

    def get_all_sessions(self) -> List[str]:
        with self._lock:
            return list(self._sessions.keys())

    def _save(self) -> None:
        if not self.persist_path:
            return

        try:
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self._sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save memory: {e}")

    def _load(self) -> None:
        if not self.persist_path:
            return

        import os
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    self._sessions = json.load(f)
            except Exception as e:
                print(f"Failed to load memory: {e}")


class ShortTermMemory(DialogueMemory):
    def __init__(
        self,
        ttl_minutes: int = 30,
        cleanup_interval_minutes: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ttl_minutes = ttl_minutes
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._cleanup_if_needed()
        super().add_turn(session_id, user_message, assistant_message, metadata)

    def _cleanup_if_needed(self) -> None:
        now = datetime.now()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        cutoff = now - timedelta(minutes=self.ttl_minutes)
        cutoff_str = cutoff.isoformat()

        with self._lock:
            for session_id in list(self._sessions.keys()):
                self._sessions[session_id] = [
                    turn for turn in self._sessions[session_id]
                    if turn["timestamp"] > cutoff_str
                ]
                if not self._sessions[session_id]:
                    del self._sessions[session_id]

        self._last_cleanup = now


class LongTermMemory:
    def __init__(
        self,
        persist_path: str = "./data/memory/long_term.json",
        max_entries: int = 1000
    ):
        self.persist_path = persist_path
        self.max_entries = max_entries
        self._memories: List[Dict[str, Any]] = []
        self._user_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

        self._load()

    def add_memory(
        self,
        session_id: str,
        content: str,
        memory_type: str = "conversation",
        importance: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            memory = {
                "session_id": session_id,
                "content": content,
                "type": memory_type,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            self._memories.append(memory)

            if len(self._memories) > self.max_entries:
                self._evict_low_importance()

            self._save()

    def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            results = self._memories.copy()

            if memory_type:
                results = [m for m in results if m["type"] == memory_type]

            results.sort(key=lambda x: (x["importance"], x["timestamp"]), reverse=True)

            return results[:limit]

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._user_profiles.get(user_id, {})

    def update_user_profile(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> None:
        with self._lock:
            if user_id not in self._user_profiles:
                self._user_profiles[user_id] = {}
            self._user_profiles[user_id].update(updates)
            self._save()

    def _evict_low_importance(self) -> None:
        self._memories.sort(key=lambda x: (x["importance"], x["timestamp"]))
        self._memories = self._memories[-self.max_entries:]

    def _save(self) -> None:
        try:
            import os
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "memories": self._memories,
                    "user_profiles": self._user_profiles,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save long-term memory: {e}")

    def _load(self) -> None:
        import os
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._memories = data.get("memories", [])
                    self._user_profiles = data.get("user_profiles", {})
            except Exception as e:
                print(f"Failed to load long-term memory: {e}")
