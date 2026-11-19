from pydantic import BaseModel, Field
from typing import Optional, List


class DialogueMessage(BaseModel):
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class DialogueRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio: Optional[bytes] = Field(None, description="Audio data in bytes")
    text: Optional[str] = Field(None, description="Text input (alternative to audio)")
    voice_id: str = Field(default="default", description="Voice ID for TTS")
    language: str = Field(default="zh", description="Language code")
    return_audio: bool = Field(default=True, description="Whether to return audio")


class DialogueResponse(BaseModel):
    session_id: str
    text: str
    audio_url: Optional[str] = None
    emotion: str
    emotion_confidence: float
    timings: dict
    cached: bool = False


class VoiceTrainingRequest(BaseModel):
    voice_id: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    voice_name: str
