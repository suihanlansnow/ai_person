from typing import Dict, List, Optional, Tuple
import re
import hashlib
import threading
from collections import OrderedDict


class EmotionDetector:
    _instance = None
    _lock = threading.Lock()
    
    EMOTION_KEYWORDS = {
        "happy": ["开心", "高兴", "快乐", "愉快", "棒", "太好了", "喜欢", "爱了", "完美", " excellent", "happy", "joy", "great", "wonderful", "amazing"],
        "sad": ["难过", "伤心", "痛苦", "悲伤", "沮丧", "郁闷", "可怜", "伤心", " tears", "sad", "unhappy", "depressed", "crying", "miserable"],
        "angry": ["生气", "愤怒", "恼火", "气死了", "讨厌", "可恶", "该死", "讨厌", "angry", "mad", "furious", "hate", "annoyed"],
        "fear": ["害怕", "恐惧", "担心", "紧张", "怕", "不安", "焦虑", " fear", "scared", "afraid", "anxious", "worried", "nervous"],
        "surprise": ["惊讶", "吃惊", "意外", "震惊", "吓人", "没想到", "surprise", "surprised", "shocked", "wow", "unexpected"],
        "disgust": ["讨厌", "恶心", "厌恶", "反感", "真差", "无语", "disgust", "disgusted", "gross", "nasty"],
    }

    EMOTION_VALUES = {
        "happy": 1.0,
        "sad": -0.8,
        "angry": -0.9,
        "fear": -0.7,
        "surprise": 0.5,
        "disgust": -0.6,
        "neutral": 0.0,
    }
    
    EMOTION_INTENSITY_MODIFIERS = {
        "very": 1.5,
        "非常": 1.5,
        "超级": 1.5,
        "极其": 1.5,
        "有点": 0.7,
        "稍微": 0.7,
        "有点": 0.7,
        "不太": 0.5,
        "not": -0.8,
        "不": -0.6,
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # LRU缓存
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000
        
        # 预编译正则表达式
        self._compile_patterns()
        
        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_detections": 0,
        }
    
    def _compile_patterns(self):
        """预编译常用模式"""
        self._intensity_pattern = re.compile(
            r'(非常|超级|极其|very|超级|有点|稍微|不太|not|不)',
            re.IGNORECASE
        )
        
        # 编译情感关键词模式
        self._emotion_patterns = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            pattern = '|'.join(re.escape(k) for k in keywords)
            self._emotion_patterns[emotion] = re.compile(pattern, re.IGNORECASE)
    
    def _get_cache_key(self, text: str, context: Optional[List[Dict[str, str]]]) -> str:
        """生成缓存键"""
        context_str = str(context) if context else ""
        combined = f"{text}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取"""
        with self._cache_lock:
            if cache_key in self._cache:
                # 移到末尾（LRU）
                self._cache.move_to_end(cache_key)
                self._stats["cache_hits"] += 1
                return self._cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, result: Dict):
        """添加到缓存"""
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
            else:
                self._cache[cache_key] = result
                if len(self._cache) > self._max_cache_size:
                    # 删除最旧的项
                    self._cache.popitem(last=False)
            self._stats["cache_misses"] += 1
    
    def _calculate_intensity(self, text: str) -> float:
        """计算情感强度"""
        intensity = 1.0
        matches = self._intensity_pattern.findall(text)
        for match in matches:
            if match in self.EMOTION_INTENSITY_MODIFIERS:
                intensity *= self.EMOTION_INTENSITY_MODIFIERS[match]
        return min(max(intensity, 0.3), 2.0)  # 限制在0.3-2.0范围
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """检测情感得分"""
        emotion_scores = {}
        text_lower = text.lower()
        
        for emotion, pattern in self._emotion_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                base_score = len(matches) * 1.0
                emotion_scores[emotion] = base_score * self.EMOTION_VALUES.get(emotion, 0)
        
        return emotion_scores
    
    def _analyze_context(self, context: Optional[List[Dict[str, str]]]) -> float:
        """分析上下文调整情感"""
        if not context:
            return 1.0
        
        context_modifier = 1.0
        for msg in context[-3:]:  # 只看最近3条
            content = msg.get("content", "").lower()
            if any(word in content for word in ["但是", "然而", "不过", "可是", "however", "but"]):
                context_modifier *= 0.8
        
        return context_modifier
    
    def detect(self, text: str, context: Optional[List[Dict[str, str]]] = None) -> Dict:
        """检测文本情感"""
        self._stats["total_detections"] += 1
        
        # 检查缓存
        cache_key = self._get_cache_key(text, context)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            # 返回缓存结果（浅拷贝避免修改）
            return cached_result.copy()
        
        # 检测情感
        emotion_scores = self._detect_emotions(text)
        
        if not emotion_scores:
            result = {
                "emotion": "neutral",
                "confidence": 0.7,
                "all_emotions": {"neutral": {"confidence": 0.7}},
                "intensity": 1.0,
            }
        else:
            # 计算情感强度
            intensity = self._calculate_intensity(text)
            
            # 应用上下文调整
            context_modifier = self._analyze_context(context)
            
            # 调整得分
            for emotion in emotion_scores:
                emotion_scores[emotion] *= intensity * context_modifier
            
            # 归一化
            total_score = sum(abs(s) for s in emotion_scores.values())
            if total_score > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= total_score
            
            max_emotion = max(emotion_scores.keys(), key=lambda k: abs(emotion_scores[k]))
            confidence = min(abs(emotion_scores[max_emotion]), 1.0)
            
            all_emotions = {
                emotion: {"confidence": min(abs(score), 1.0)}
                for emotion, score in emotion_scores.items()
            }
            
            result = {
                "emotion": max_emotion,
                "confidence": round(confidence, 3),
                "all_emotions": all_emotions,
                "intensity": round(intensity, 2),
            }
        
        # 缓存结果
        self._add_to_cache(cache_key, result)
        
        return result
    
    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """批量检测"""
        return [self.detect(text) for text in texts]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        cache_total = self._stats["cache_hits"] + self._stats["cache_misses"]
        cache_hit_rate = (
            self._stats["cache_hits"] / cache_total if cache_total > 0 else 0
        )
        
        return {
            "total_detections": self._stats["total_detections"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
        }
    
    def clear_cache(self):
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
        self._stats["cache_hits"] = 0
        self._stats["cache_misses"] = 0
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_detections": 0,
        }
