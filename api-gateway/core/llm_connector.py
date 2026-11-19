"""
LLM连接器 - 支持多种后端
- OpenAI API (GPT系列)
- Ollama API (本地模型)
- vLLM Server (远程服务)
- 模拟模式 (开发测试)
"""

import os
import time
import logging
from typing import List, Dict, Optional, Union

logger = logging.getLogger(__name__)


class LLMConnector:
    """LLM连接器基类"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.inference_count = 0

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """对话接口"""
        raise NotImplementedError

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> List[str]:
        """生成接口"""
        raise NotImplementedError


class MockLLMConnector(LLMConnector):
    """模拟LLM连接器 - 用于开发测试"""

    MOCK_RESPONSES = {
        "greeting": [
            "你好！很高兴见到你。今天有什么我可以帮助你的吗？",
            "嗨！你好呀。我是AI语音助手，有什么想聊的吗？",
            "你好！欢迎来到AI对话世界。让我们开始聊天吧！"
        ],
        "sad": [
            "我理解你现在的感受。愿意和我聊聊是什么让你感到不开心吗？",
            "难过的时候，有人倾听会让事情变得好一些。我在这里陪着你。",
            "生活中总会有低谷，但这也意味着高峰即将到来。你想聊聊吗？"
        ],
        "happy": [
            "太棒了！听到你开心我也很高兴！有什么特别的原因吗？",
            "哇！看起来你今天心情很好！有什么好事发生了吗？",
            "开心是最美好的事情！继续保持这份快乐吧！"
        ],
        "default": [
            "这是一个很有趣的话题。能告诉我更多关于这方面的想法吗？",
            "我理解你的意思。让我想想如何更好地回答你。",
            "感谢你的分享。你希望我在这件事上给你什么建议呢？",
        ]
    }

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        self.inference_count += 1

        last_message = messages[-1]["content"] if messages else ""
        emotion = kwargs.get("emotion", "neutral")

        import random
        response_key = emotion if emotion in self.MOCK_RESPONSES else "default"
        response = random.choice(self.MOCK_RESPONSES.get(response_key, self.MOCK_RESPONSES["default"]))

        return {
            "response": response,
            "inference_time": round(random.uniform(0.3, 0.8), 3),
            "usage": {"prompt_tokens": len(last_message), "completion_tokens": len(response)}
        }

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        import random
        responses = [random.choice(self.MOCK_RESPONSES["default"]) for _ in prompts]
        return responses


class OpenAIConnector(LLMConnector):
    """OpenAI API连接器"""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.api_key = config.get("api_key", os.getenv("OPENAI_API_KEY", ""))
        self.model = config.get("model", "gpt-3.5-turbo")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        import urllib.request
        import json

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)

        start_time = time.time()
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read())

        inference_time = time.time() - start_time

        return {
            "response": result["choices"][0]["message"]["content"],
            "inference_time": round(inference_time, 3),
            "usage": result.get("usage", {})
        }

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        messages = [{"role": "user", "content": p} for p in prompts]
        results = []
        for msg in messages:
            result = self.chat([msg], **kwargs)
            results.append(result["response"])
        return results


class OllamaConnector(LLMConnector):
    """Ollama本地模型连接器"""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2:7b")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        import urllib.request
        import json

        url = f"{self.base_url}/api/chat"
        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})

        start_time = time.time()
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())

        inference_time = time.time() - start_time

        return {
            "response": result["message"]["content"],
            "inference_time": round(inference_time, 3),
            "usage": result.get("usage", {})
        }

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        url = f"{self.base_url}/api/generate"
        results = []

        for prompt in prompts:
            data = {
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "stream": False
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read())
                results.append(result["response"])

        return results


class VLLMServerConnector(LLMConnector):
    """vLLM Server连接器"""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.model = config.get("model", "qwen2-7b")

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        import urllib.request
        import json

        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}

        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
        }

        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)

        start_time = time.time()
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())

        inference_time = time.time() - start_time

        return {
            "response": result["choices"][0]["message"]["content"],
            "inference_time": round(inference_time, 3),
            "usage": result.get("usage", {})
        }

    def generate(self, prompts: Union[str, List[str]], **kwargs) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]

        url = f"{self.base_url}/completions"
        headers = {"Content-Type": "application/json"}
        results = []

        for prompt in prompts:
            data = {
                "model": kwargs.get("model", self.model),
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.7),
            }
            req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers)

            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read())
                results.append(result["choices"][0]["text"])

        return results


class LLMConnectorFactory:
    """LLM连接器工厂"""

    CONNECTORS = {
        "mock": MockLLMConnector,
        "openai": OpenAIConnector,
        "ollama": OllamaConnector,
        "vllm": VLLMServerConnector,
    }

    @classmethod
    def create(cls, backend: str = "mock", config: dict = None) -> LLMConnector:
        connector_class = cls.CONNECTORS.get(backend, MockLLMConnector)
        return connector_class(config)


def create_llm_connector(backend: str = None) -> LLMConnector:
    """创建LLM连接器"""
    backend = backend or os.getenv("LLM_BACKEND", "mock")

    configs = {
        "openai": {"api_key": os.getenv("OPENAI_API_KEY", ""), "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")},
        "ollama": {"base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), "model": os.getenv("OLLAMA_MODEL", "qwen2:7b")},
        "vllm": {"base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"), "model": os.getenv("VLLM_MODEL", "qwen2-7b")},
    }

    config = configs.get(backend, {})
    return LLMConnectorFactory.create(backend, config)


if __name__ == "__main__":
    print("=== LLM连接器测试 ===\n")

    print("1. 模拟模式:")
    mock_llm = create_llm_connector("mock")
    result = mock_llm.chat([{"role": "user", "content": "你好"}])
    print(f"   回复: {result['response']}")
    print(f"   耗时: {result['inference_time']}s\n")

    print("2. 支持的后端:")
    for backend in LLMConnectorFactory.CONNECTORS.keys():
        print(f"   - {backend}")
