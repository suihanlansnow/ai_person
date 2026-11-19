from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DialogueContext:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    user_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        message = Message(role="user", content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None) -> None:
        message = Message(role="assistant", content=content, metadata=metadata or {})
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

    def get_messages(self, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        messages = [msg.to_dict() for msg in self.messages]
        if max_turns:
            messages = messages[-max_turns * 2:]
        return messages

    def get_last_n_turns(self, n: int) -> List[Message]:
        return self.messages[-n * 2:] if len(self.messages) >= n * 2 else self.messages

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueContext':
        context = cls(
            session_id=data.get("session_id", ""),
            user_id=data.get("user_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )
        context.messages = [
            Message.from_dict(msg) for msg in data.get("messages", [])
        ]
        return context

    def __len__(self) -> int:
        return len(self.messages)
