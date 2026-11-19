from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Dict, Optional, List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Service")

tts_engine = None


class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"
    emotion: Optional[str] = None
    speed: float = 1.0


class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    inference_time: float
    cached: bool = False


class VoiceInfo(BaseModel):
    id: str
    name: str


@app.on_event("startup")
async def startup_event():
    global tts_engine
    try:
        from engines.tts_engine.gpt_sovits_engine import GPTSovitsEngine

        model_path = os.getenv("MODEL_PATH", "/models/tts/gpt_sovits_v2")
        device = os.getenv("DEVICE", "cuda")

        tts_engine = GPTSovitsEngine(
            model_path=model_path,
            device=device,
        )
        tts_engine.load_model()
        logger.info("TTS engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TTS engine: {e}")
        logger.info("TTS service will use mock mode")


@app.post("/synthesize", response_model=TTSResponse)
async def synthesize(request: TTSRequest):
    try:
        import time
        start_time = time.time()

        if tts_engine:
            result = tts_engine.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                emotion=request.emotion,
                speed=request.speed,
            )
        else:
            result = {"audio": None, "sample_rate": 24000}

        inference_time = time.time() - start_time

        return TTSResponse(
            inference_time=round(inference_time, 3),
            cached=False,
        )

    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize/audio")
async def synthesize_audio(request: TTSRequest):
    try:
        import time
        import io
        import wave

        start_time = time.time()

        if tts_engine:
            result = tts_engine.synthesize(
                text=request.text,
                voice_id=request.voice_id,
                emotion=request.emotion,
                speed=request.speed,
            )
            audio = result.get("audio")
            sample_rate = result.get("sample_rate", 24000)

            if audio is not None:
                audio_bytes = io.BytesIO()
                with wave.open(audio_bytes, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes((audio * 32767).astype('<h').tobytes())
                audio_bytes = audio_bytes.getvalue()
            else:
                audio_bytes = b""
        else:
            audio_bytes = b""

        inference_time = time.time() - start_time

        logger.info(f"Synthesized {len(request.text)} chars in {inference_time:.3f}s")

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Inference-Time": str(round(inference_time, 3)),
            },
        )

    except Exception as e:
        logger.error(f"TTS audio synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    if tts_engine:
        voices = tts_engine.list_voices()
        return [VoiceInfo(id=v["id"], name=v["name"]) for v in voices]
    return [
        VoiceInfo(id="default", name="默认女声"),
        VoiceInfo(id="male", name="男声"),
    ]


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "TTS Service"}


@app.get("/stats")
async def get_stats():
    if tts_engine:
        return tts_engine.get_stats()
    return {"status": "mock", "total_inferences": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)