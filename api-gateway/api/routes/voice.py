from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/voice", tags=["voice"])


class VoiceInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    is_default: bool = False


class VoiceListResponse(BaseModel):
    voices: List[VoiceInfo]
    total: int


class VoiceTrainingRequest(BaseModel):
    voice_id: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    voice_name: str


@router.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    try:
        voices = [
            VoiceInfo(
                id="default",
                name="Default Voice",
                description="系统默认音色",
                is_default=True
            ),
        ]

        return VoiceListResponse(voices=voices, total=len(voices))

    except Exception as e:
        logger.error(f"List voices error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voices")
async def create_custom_voice(request: VoiceTrainingRequest):
    try:
        logger.info(f"Creating custom voice: {request.voice_id}")

        return {
            "message": "Voice creation started",
            "voice_id": request.voice_id,
            "status": "pending",
        }

    except Exception as e:
        logger.error(f"Create voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices/{voice_id}")
async def get_voice_info(voice_id: str):
    try:
        if voice_id == "default":
            return VoiceInfo(
                id="default",
                name="Default Voice",
                description="系统默认音色",
                is_default=True
            )

        raise HTTPException(status_code=404, detail="Voice not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    try:
        if voice_id == "default":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete default voice"
            )

        return {"message": f"Voice {voice_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voices/{voice_id}/train")
async def train_voice(
    voice_id: str,
    background_tasks=None
):
    try:
        logger.info(f"Training voice: {voice_id}")

        return {
            "message": "Voice training started",
            "voice_id": voice_id,
            "estimated_time": "30-60 minutes",
        }

    except Exception as e:
        logger.error(f"Train voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/voices/upload")
async def upload_reference_audio(
    file: UploadFile = File(...),
):
    try:
        logger.info(f"Uploading reference audio: {file.filename}")

        return {
            "message": "Audio uploaded successfully",
            "file_path": f"./data/training_data/{file.filename}",
        }

    except Exception as e:
        logger.error(f"Upload audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
