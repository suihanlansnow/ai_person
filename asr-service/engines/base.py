from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EngineType(Enum):
    LLM = "llm"
    ASR = "asr"
    TTS = "tts"


class BaseEngine(ABC):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_trt: bool = False,
        **kwargs
    ):
        self.model_path = model_path
        self.device = device
        self.use_trt = use_trt
        self.kwargs = kwargs
        self._model = None
        self._is_initialized = False
        self._init_time: float = 0
        
        # 基础统计
        self._inference_count: int = 0
        self._total_inference_time: float = 0
        
        # 增强统计指标
        self._min_inference_time: float = float('inf')
        self._max_inference_time: float = 0
        self._total_audio_duration: float = 0
        self._total_text_length: int = 0
        
        # 语言分布统计
        self._language_stats: Dict[str, Dict[str, Any]] = {}
        
        # 最近N次推理记录（用于趋势分析）
        self._recent_inferences = deque(maxlen=100)
        
        # 错误统计
        self._error_count: int = 0
        self._error_types: Dict[str, int] = {}

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def engine_type(self) -> EngineType:
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def unload_model(self) -> None:
        pass

    def _record_inference(
        self, 
        inference_time: float, 
        audio_duration: float = 0, 
        text_length: int = 0, 
        language: str = "zh"
    ):
        """记录推理统计信息"""
        self._inference_count += 1
        self._total_inference_time += inference_time
        self._total_audio_duration += audio_duration
        self._total_text_length += text_length
        
        # 更新最小/最大推理时间
        if inference_time < self._min_inference_time:
            self._min_inference_time = inference_time
        if inference_time > self._max_inference_time:
            self._max_inference_time = inference_time
        
        # 更新语言统计
        if language not in self._language_stats:
            self._language_stats[language] = {
                "count": 0,
                "total_time": 0,
                "total_audio_duration": 0,
                "total_text_length": 0
            }
        self._language_stats[language]["count"] += 1
        self._language_stats[language]["total_time"] += inference_time
        self._language_stats[language]["total_audio_duration"] += audio_duration
        self._language_stats[language]["total_text_length"] += text_length
        
        # 记录最近推理时间（用于趋势分析）
        self._recent_inferences.append({
            "timestamp": time.time(),
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "text_length": text_length,
            "language": language
        })

    def _record_error(self, error_type: str):
        """记录错误类型统计"""
        self._error_count += 1
        if error_type not in self._error_types:
            self._error_types[error_type] = 0
        self._error_types[error_type] += 1

    def get_stats(self) -> Dict[str, Any]:
        avg_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0
        )
        
        # 计算实时率 (Real-Time Factor) - 推理时间/音频时长
        real_time_factor = (
            self._total_inference_time / self._total_audio_duration
            if self._total_audio_duration > 0
            else 0
        )
        
        # 计算字符处理效率 (字符/秒)
        char_per_second = (
            self._total_text_length / self._total_inference_time
            if self._total_inference_time > 0
            else 0
        )
        
        # 计算平均音频时长
        avg_audio_duration = (
            self._total_audio_duration / self._inference_count
            if self._inference_count > 0
            else 0
        )
        
        # 计算平均文本长度
        avg_text_length = (
            self._total_text_length / self._inference_count
            if self._inference_count > 0
            else 0
        )
        
        # 计算最近N次的平均推理时间（趋势分析）
        recent_avg_time = 0
        if self._recent_inferences:
            recent_avg_time = sum(inf["inference_time"] for inf in self._recent_inferences) / len(self._recent_inferences)

        stats = {
            # 基础信息
            "engine_type": self.engine_type.value,
            "is_initialized": self._is_initialized,
            "model_path": self.model_path,
            "device": self.device,
            "init_time": round(self._init_time, 3),
            
            # 核心统计
            "total_inferences": self._inference_count,
            "total_inference_time": round(self._total_inference_time, 3),
            "average_inference_time": round(avg_time, 3),
            "min_inference_time": round(self._min_inference_time, 3) if self._min_inference_time != float('inf') else 0,
            "max_inference_time": round(self._max_inference_time, 3),
            
            # 语句解析效率指标
            "total_audio_duration": round(self._total_audio_duration, 3),
            "average_audio_duration": round(avg_audio_duration, 3),
            "total_text_length": self._total_text_length,
            "average_text_length": round(avg_text_length, 1),
            "real_time_factor": round(real_time_factor, 3),  # 实时率：<1表示实时
            "char_per_second": round(char_per_second, 1),     # 字符处理效率
            
            # 趋势分析
            "recent_inference_count": len(self._recent_inferences),
            "recent_average_time": round(recent_avg_time, 3),
            
            # 错误统计
            "error_count": self._error_count,
            "error_types": self._error_types,
        }
        
        # 添加语言分布统计
        if self._language_stats:
            lang_stats = {}
            for lang, data in self._language_stats.items():
                lang_avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
                lang_avg_audio = data["total_audio_duration"] / data["count"] if data["count"] > 0 else 0
                lang_avg_text = data["total_text_length"] / data["count"] if data["count"] > 0 else 0
                lang_stats[lang] = {
                    "count": data["count"],
                    "total_time": round(data["total_time"], 3),
                    "average_time": round(lang_avg_time, 3),
                    "average_audio_duration": round(lang_avg_audio, 3),
                    "average_text_length": round(lang_avg_text, 1)
                }
            stats["language_distribution"] = lang_stats
        
        return stats

    def reset_stats(self):
        """重置所有统计数据"""
        self._inference_count = 0
        self._total_inference_time = 0
        self._min_inference_time = float('inf')
        self._max_inference_time = 0
        self._total_audio_duration = 0
        self._total_text_length = 0
        self._language_stats = {}
        self._recent_inferences.clear()
        self._error_count = 0
        self._error_types = {}

    def get_recent_inferences(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近N次推理记录"""
        recent = list(self._recent_inferences)[-limit:]
        return [{
            "timestamp": inf["timestamp"],
            "inference_time": round(inf["inference_time"], 3),
            "audio_duration": round(inf["audio_duration"], 3),
            "text_length": inf["text_length"],
            "language": inf["language"],
            "real_time_factor": round(inf["inference_time"] / max(inf["audio_duration"], 0.001), 3)
        } for inf in recent]