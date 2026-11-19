from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Service")

llm_engine = None


class LLMRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class LLMResponse(BaseModel):
    responses: List[str]
    inference_time: float
    cached: bool = False


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 2048
    temperature: float = 0.7


class ChatResponse(BaseModel):
    response: str
    inference_time: float


@app.on_event("startup")
async def startup_event():
    global llm_engine
    backend = os.getenv("LLM_BACKEND", "mock")

    if backend == "mock":
        logger.info("Using mock LLM backend")
        return

    try:
        if backend == "vllm":
            from engines.llm_engine.vllm_engine import VLLMEngine
            model_path = os.getenv("MODEL_PATH", "/models/llm/qwen2-7b")
            quantization = os.getenv("QUANTIZATION", "fp16")
            tensor_parallel = int(os.getenv("TENSOR_PARALLEL", "1"))
            max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))

            llm_engine = VLLMEngine(
                model_path=model_path,
                quantization=quantization,
                tensor_parallel=tensor_parallel,
                max_model_len=max_model_len,
            )
            llm_engine.load_model()
            logger.info("VLLM engine loaded successfully")
        elif backend == "ollama":
            from core.llm_connector import OllamaConnector
            base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            model = os.getenv("OLLAMA_MODEL", "qwen2:7b")
            llm_engine = OllamaConnector({"base_url": base_url, "model": model})
            logger.info(f"Ollama connector initialized: {base_url}, model={model}")
        else:
            logger.warning(f"Backend {backend} not fully implemented, using mock")
    except Exception as e:
        logger.error(f"Failed to load LLM engine: {e}")
        logger.info("Falling back to mock mode")


@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    try:
        import time
        start_time = time.time()

        if llm_engine:
            responses = llm_engine.generate(
                prompts=request.prompts,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )
        else:
            responses = [f"Mock response for: {p[:50]}..." for p in request.prompts]

        inference_time = time.time() - start_time

        return LLMResponse(
            responses=responses,
            inference_time=round(inference_time, 3),
        )

    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        import time
        start_time = time.time()

        backend = os.getenv("LLM_BACKEND", "mock")
        logger.info(f"Chat request: backend={backend}, llm_engine={llm_engine}, messages={request.messages}")

        if backend == "mock" or not llm_engine:
            from core.llm_connector import MockLLMConnector
            connector = MockLLMConnector()
            result = connector.chat(request.messages, max_tokens=request.max_tokens, temperature=request.temperature)
            response_text = result["response"]
            inference_time = result.get("inference_time", 0)
        else:
            result = llm_engine.chat(
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if isinstance(result, dict):
                response_text = result["response"]
                inference_time = result.get("inference_time", time.time() - start_time)
            else:
                response_text = str(result)
                inference_time = time.time() - start_time

        return ChatResponse(
            response=response_text,
            inference_time=round(inference_time, 3),
        )

    except Exception as e:
        logger.error(f"LLM chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LLM Service"}


@app.get("/stats")
async def get_stats():
    if llm_engine:
        return llm_engine.get_stats()
    return {"backend": os.getenv("LLM_BACKEND", "mock"), "total_inferences": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)