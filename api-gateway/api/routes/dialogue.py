from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from pydantic import BaseModel, Field
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dialogue", tags=["dialogue"])


class DialogueRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio: Optional[bytes] = None
    text: Optional[str] = None
    voice_id: str = "default"
    language: str = "zh"
    return_audio: bool = True


class DialogueResponse(BaseModel):
    session_id: str
    text: str
    audio_url: Optional[str] = None
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    timings: dict = {}
    cached: bool = False


@router.post("/", response_model=DialogueResponse)
async def process_dialogue(
    request: DialogueRequest,
    background_tasks: BackgroundTasks
):
    try:
        if not request.audio and not request.text:
            raise HTTPException(
                status_code=400,
                detail="Either audio or text must be provided"
            )

        session_id = request.session_id

        logger.info(f"Processing dialogue for session: {session_id}")

        response = DialogueResponse(
            session_id=session_id,
            text="这是一个模拟回复，实际使用时需要接入完整的语音对话流水线。",
            emotion="happy",
            emotion_confidence=0.8,
            cached=False,
        )

        return response

    except Exception as e:
        logger.error(f"Dialogue processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_dialogue_history(session_id: str, max_turns: Optional[int] = None):
    try:
        return {
            "session_id": session_id,
            "history": [],
            "total_turns": 0,
        }
    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_dialogue_history(session_id: str):
    try:
        return {"message": f"History cleared for session: {session_id}"}
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
