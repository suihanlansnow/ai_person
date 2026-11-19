from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import httpx
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Person API Gateway")

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = os.getenv("STATIC_PATH", "/app/static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
ASR_SERVICE_URL = os.getenv("ASR_SERVICE_URL", "http://asr-service:8002")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8003")
DIALOGUE_SERVICE_URL = os.getenv("DIALOGUE_SERVICE_URL", "http://dialogue-service:8004")


class DialogueRequest(BaseModel):
    session_id: str = ""
    text: str = ""
    voice_id: str = "default"
    language: str = "zh"
    return_audio: bool = True


class DialogueResponse(BaseModel):
    session_id: str
    text: str
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    timings: Dict[str, float] = {}
    cached: bool = False


async def call_asr_service(audio_bytes: bytes) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        response = await client.post(f"{ASR_SERVICE_URL}/recognize", files=files)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


async def call_llm_service(messages: list) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{LLM_SERVICE_URL}/chat",
            json={"messages": messages, "max_tokens": 2048, "temperature": 0.7},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


async def call_tts_service(text: str, voice_id: str = "default", emotion: str = "neutral") -> bytes:
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{TTS_SERVICE_URL}/synthesize/audio",
            json={"text": text, "voice_id": voice_id, "emotion": emotion},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.content


async def call_dialogue_service(session_id: str, user_message: str, assistant_message: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{DIALOGUE_SERVICE_URL}/turn",
            json={
                "session_id": session_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
            },
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


async def get_dialogue_history(session_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{DIALOGUE_SERVICE_URL}/history/{session_id}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


async def call_emotion_service(text: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{DIALOGUE_SERVICE_URL}/emotion",
            json={"text": text},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        return response.json()


@app.post("/api/dialogue", response_model=DialogueResponse)
async def process_dialogue(
    request: DialogueRequest,
):
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # 文本模式：直接使用文本
        user_text = request.text
        
        if not user_text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        emotion_result = await call_emotion_service(user_text)
        emotion = emotion_result.get("emotion", "neutral")

        messages = [
            {
                "role": "system",
                "content": f"你是一个有情感陪伴能力的AI助手。当前用户情绪: {emotion}",
            },
        ]

        try:
            history_result = await get_dialogue_history(session_id)
            for turn in history_result.get("history", []):
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            logger.info(f"Loaded {len(history_result.get('history', []))} turns from dialogue history")
        except Exception as e:
            logger.warning(f"Failed to load dialogue history: {e}")

        messages.append({"role": "user", "content": user_text})

        llm_result = await call_llm_service(messages)
        assistant_text = llm_result.get("response", "")

        await call_dialogue_service(session_id, user_text, assistant_text)

        timings = {
            "asr": 0,
            "emotion": 0,
            "llm": llm_result.get("inference_time", 0),
            "total": 0,
        }

        return DialogueResponse(
            session_id=session_id,
            text=assistant_text,
            emotion=emotion,
            emotion_confidence=emotion_result.get("confidence", 0.0),
            timings=timings,
        )

    except Exception as e:
        logger.error(f"Dialogue processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dialogue/audio", response_model=DialogueResponse)
async def process_dialogue_audio(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    try:
        session_id = session_id or str(uuid.uuid4())

        audio_bytes = await audio.read()

        asr_result = await call_asr_service(audio_bytes)
        user_text = asr_result.get("text", "")

        if not user_text.strip():
            return DialogueResponse(
                session_id=session_id,
                text="",
                timings={"asr": asr_result.get("inference_time", 0)},
            )

        emotion_result = await call_emotion_service(user_text)
        emotion = emotion_result.get("emotion", "neutral")

        messages = [
            {
                "role": "system",
                "content": f"你是一个有情感陪伴能力的AI助手。当前用户情绪: {emotion}",
            },
        ]

        try:
            history_result = await get_dialogue_history(session_id)
            for turn in history_result.get("history", []):
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            logger.info(f"Loaded {len(history_result.get('history', []))} turns from dialogue history")
        except Exception as e:
            logger.warning(f"Failed to load dialogue history: {e}")

        messages.append({"role": "user", "content": user_text})

        llm_result = await call_llm_service(messages)
        assistant_text = llm_result.get("response", "")

        await call_dialogue_service(session_id, user_text, assistant_text)

        timings = {
            "asr": asr_result.get("inference_time", 0),
            "emotion": 0,
            "llm": llm_result.get("inference_time", 0),
            "total": 0,
        }

        return DialogueResponse(
            session_id=session_id,
            text=assistant_text,
            emotion=emotion,
            emotion_confidence=emotion_result.get("confidence", 0.0),
            timings=timings,
        )

    except Exception as e:
        logger.error(f"Dialogue audio processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}


@app.get("/api/services")
async def get_service_status():
    services = {
        "llm": {"url": LLM_SERVICE_URL, "status": "unknown"},
        "asr": {"url": ASR_SERVICE_URL, "status": "unknown"},
        "tts": {"url": TTS_SERVICE_URL, "status": "unknown"},
        "dialogue": {"url": DIALOGUE_SERVICE_URL, "status": "unknown"},
    }

    async with httpx.AsyncClient(timeout=5) as client:
        for name, service in services.items():
            try:
                response = await client.get(f"{service['url']}/health")
                services[name]["status"] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception:
                services[name]["status"] = "unhealthy"

    return services


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)