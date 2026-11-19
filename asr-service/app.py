from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Optional
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Service")

asr_engine = None


class ASRResponse(BaseModel):
    text: str
    segments: list = []
    inference_time: float


@app.on_event("startup")
async def startup_event():
    global asr_engine
    try:
        from engines.asr_engine.funasr_engine import FunASREngine

        model_path = os.getenv("MODEL_PATH", "/models/asr/speech_paraformer-large")
        use_onnx = os.getenv("USE_ONNX", "true").lower() == "true"
        device = os.getenv("DEVICE", "cuda")

        asr_engine = FunASREngine(
            model_path=model_path,
            use_onnx=use_onnx,
            device=device,
        )
        asr_engine.load_model()
        logger.info("ASR engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR engine: {e}")
        logger.info("ASR service will use mock mode")


@app.post("/recognize", response_model=ASRResponse)
async def recognize_audio(file: UploadFile = File(...), language: Optional[str] = "zh"):
    try:
        import time
        start_time = time.time()

        audio_bytes = await file.read()

        if asr_engine:
            result = asr_engine.recognize(audio_bytes, language=language)
        else:
            result = {"text": "这是Mock识别结果。", "segments": []}

        inference_time = time.time() - start_time

        return ASRResponse(
            text=result.get("text", ""),
            segments=result.get("segments", []),
            inference_time=round(inference_time, 3),
        )

    except Exception as e:
        logger.error(f"ASR recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ASR Service"}


@app.get("/stats")
async def get_stats():
    if asr_engine:
        return asr_engine.get_stats()
    return {"status": "mock", "total_inferences": 0}


@app.get("/stats/recent")
async def get_recent_inferences(limit: int = 20):
    if asr_engine:
        return {
            "recent_inferences": asr_engine.get_recent_inferences(limit=limit)
        }
    return {"status": "mock", "recent_inferences": []}


@app.post("/stats/reset")
async def reset_stats():
    if asr_engine:
        asr_engine.reset_stats()
        return {"status": "success", "message": "Statistics reset successfully"}
    return {"status": "mock", "message": "No engine to reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)