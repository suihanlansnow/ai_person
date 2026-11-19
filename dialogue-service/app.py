from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dialogue Service")

memory = None
emotion_detector = None


class DialogueTurn(BaseModel):
    user: str
    assistant: str
    timestamp: Optional[str] = None


class DialogueHistory(BaseModel):
    session_id: str
    history: List[DialogueTurn]
    total_turns: int


class AddTurnRequest(BaseModel):
    session_id: str
    user_message: str
    assistant_message: str
    metadata: Optional[Dict] = None


class EmotionRequest(BaseModel):
    text: str
    context: Optional[List[Dict[str, str]]] = None


class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    intensity: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[Dict]
    total: int


@app.on_event("startup")
async def startup_event():
    global memory, emotion_detector
    try:
        from core.dialogue.memory import ShortTermMemory

        memory_path = os.getenv("MEMORY_PATH", "/data/memory")
        os.makedirs(memory_path, exist_ok=True)
        
        redis_url = os.getenv("REDIS_URL", None)
        enable_redis = redis_url is not None
        
        memory = ShortTermMemory(
            persist_path=f"{memory_path}/dialogue_memory.json",
            max_turns=10,
            max_sessions=1000,
            write_interval=5.0,
            enable_redis=enable_redis,
            redis_url=redis_url,
        )
        
        if enable_redis:
            logger.info(f"Dialogue memory initialized with Redis: {redis_url}")
        else:
            logger.info("Dialogue memory initialized (Redis disabled)")
        
        memory.initialize_async()
        
    except Exception as e:
        logger.error(f"Failed to load dialogue memory: {e}")
        memory = None
    
    try:
        from core.dialogue.emotion import EmotionDetector
        emotion_detector = EmotionDetector()
        logger.info("Emotion detector initialized")
    except Exception as e:
        logger.error(f"Failed to initialize emotion detector: {e}")


@app.post("/turn", response_model=Dict)
async def add_turn(request: AddTurnRequest):
    try:
        if memory:
            memory.add_turn(
                session_id=request.session_id,
                user_message=request.user_message,
                assistant_message=request.assistant_message,
                metadata=request.metadata,
            )
        return {"status": "success", "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Add turn error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}", response_model=DialogueHistory)
async def get_history(session_id: str, max_turns: Optional[int] = None):
    try:
        if memory:
            history = memory.get_history(session_id, max_turns)
        else:
            history = []

        turns = [
            DialogueTurn(
                user=h["user"],
                assistant=h["assistant"],
                timestamp=h["timestamp"],
            )
            for h in history
        ]

        return DialogueHistory(
            session_id=session_id,
            history=turns,
            total_turns=len(turns),
        )

    except Exception as e:
        logger.error(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    try:
        if memory:
            memory.delete_session(session_id)
        return {"status": "success", "message": f"History cleared for {session_id}"}
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    try:
        if memory:
            sessions = memory.get_all_sessions()
        else:
            sessions = []
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotion", response_model=EmotionResponse)
async def detect_emotion(request: EmotionRequest):
    try:
        global emotion_detector
        if emotion_detector is None:
            from core.dialogue.emotion import EmotionDetector
            emotion_detector = EmotionDetector()
        
        result = emotion_detector.detect(request.text, request.context)

        return EmotionResponse(
            emotion=result["emotion"],
            confidence=result["confidence"],
            all_emotions={
                k: v.get("confidence", 0)
                for k, v in result.get("all_emotions", {}).items()
            },
            intensity=result.get("intensity"),
        )
    except Exception as e:
        logger.error(f"Emotion detection error: {e}")
        return EmotionResponse(
            emotion="neutral",
            confidence=0.5,
            all_emotions={"neutral": 0.5},
        )


@app.post("/emotion/batch")
async def detect_emotion_batch(texts: List[str]) -> List[EmotionResponse]:
    """批量情感检测"""
    try:
        global emotion_detector
        if emotion_detector is None:
            from core.dialogue.emotion import EmotionDetector
            emotion_detector = EmotionDetector()
        
        results = emotion_detector.detect_batch(texts)
        
        return [
            EmotionResponse(
                emotion=r["emotion"],
                confidence=r["confidence"],
                all_emotions={
                    k: v.get("confidence", 0)
                    for k, v in r.get("all_emotions", {}).items()
                },
                intensity=r.get("intensity"),
            )
            for r in results
        ]
    except Exception as e:
        logger.error(f"Batch emotion detection error: {e}")
        return [
            EmotionResponse(
                emotion="neutral",
                confidence=0.5,
                all_emotions={"neutral": 0.5},
            )
            for _ in texts
        ]


@app.post("/search", response_model=SearchResponse)
async def search_dialogue(request: SearchRequest):
    """搜索对话历史"""
    try:
        if memory:
            results = memory.search(request.query, request.session_id)
        else:
            results = []
        
        return SearchResponse(
            results=results,
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/memory")
async def get_memory_stats():
    """获取对话记忆统计"""
    if memory:
        return memory.get_stats()
    return {"error": "Memory not initialized"}


@app.get("/stats/emotion")
async def get_emotion_stats():
    """获取情感检测统计"""
    if emotion_detector:
        return emotion_detector.get_stats()
    return {"error": "Emotion detector not initialized"}


@app.post("/stats/reset")
async def reset_stats():
    """重置所有统计"""
    try:
        if memory:
            memory.reset_stats()
        if emotion_detector:
            emotion_detector.reset_stats()
        return {"status": "success", "message": "Statistics reset successfully"}
    except Exception as e:
        logger.error(f"Reset stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/emotion/clear")
async def clear_emotion_cache():
    """清空情感检测缓存"""
    if emotion_detector:
        emotion_detector.clear_cache()
        return {"status": "success", "message": "Emotion cache cleared"}
    return {"error": "Emotion detector not initialized"}


@app.get("/health")
async def health_check():
    memory_status = "healthy" if memory is not None else "unavailable"
    emotion_status = "healthy" if emotion_detector is not None else "unavailable"
    
    return {
        "status": "healthy",
        "service": "Dialogue Service",
        "components": {
            "memory": memory_status,
            "emotion_detector": emotion_status
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
