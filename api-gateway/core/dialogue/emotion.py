from typing import Dict, List, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class EmotionDetector:
    EMOTION_KEYWORDS = {
        "happy": ["开心", "高兴", "快乐", "幸福", "愉快", "太好了", "真棒", "喜欢", "兴奋", "happy", "joy", "great"],
        "sad": ["难过", "伤心", "痛苦", "悲伤", "失落", "沮丧", "郁闷", "哭", "sad", "unhappy", "depressed"],
        "angry": ["生气", "愤怒", "恼火", "气愤", "讨厌", "恨", "烦躁", "angry", "mad", "hate", "furious"],
        "fear": ["害怕", "恐惧", "担心", "紧张", "焦虑", "不安", "怕", "fear", "scared", "afraid", "worried"],
        "surprise": ["惊讶", "吃惊", "意外", "震惊", "没想到", "surprise", "shocked", "amazed", "unexpected"],
        "disgust": ["恶心", "厌恶", "反感", "讨厌", "嫌弃", "disgust", "dislike", "gross"],
    }

    EMOTION_INTENSITY_PATTERNS = [
        (r"非常|特别|极其|十分|超|巨|太.*了", 1.5),
        (r"有点|稍微|略微|一点点|一点", 0.7),
        (r"非常|特别|极其", 1.3),
        (r"有点|稍微", 0.6),
    ]

    def __init__(self, enable_llm_detection: bool = True):
        self.enable_llm_detection = enable_llm_detection
        self._emotion_weights = {
            "happy": 1.0,
            "sad": -1.0,
            "angry": -0.8,
            "fear": -0.6,
            "surprise": 0.3,
            "disgust": -0.5,
        }

    def detect(
        self,
        text: str,
        context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        if not text:
            return self._empty_result()

        keyword_result = self._detect_by_keywords(text)
        intensity = self._detect_intensity(text)

        primary_emotion = keyword_result.get("primary_emotion", "neutral")
        confidence = keyword_result.get("confidence", 0.5)

        adjusted_confidence = min(1.0, confidence * intensity)

        if self.enable_llm_detection and context:
            llm_emotion = self._detect_by_llm(text, context)
            if llm_emotion:
                primary_emotion = llm_emotion.get("emotion", primary_emotion)
                confidence = (confidence + llm_emotion.get("confidence", 0.5)) / 2

        return {
            "emotion": primary_emotion,
            "confidence": round(adjusted_confidence, 2),
            "all_emotions": keyword_result.get("all_emotions", {}),
            "intensity": round(intensity, 2),
        }

    def _detect_by_keywords(self, text: str) -> Dict[str, any]:
        emotion_scores = {}

        text_lower = text.lower()

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)

            if score > 0:
                confidence = min(0.9, 0.3 + score * 0.15)
                emotion_scores[emotion] = {
                    "score": score,
                    "confidence": confidence,
                    "keywords": matched_keywords,
                }

        if not emotion_scores:
            return self._empty_result()

        primary = max(emotion_scores.items(), key=lambda x: x[1]["score"])

        return {
            "primary_emotion": primary[0],
            "confidence": primary[1]["confidence"],
            "all_emotions": emotion_scores,
        }

    def _detect_intensity(self, text: str) -> float:
        intensity = 1.0

        for pattern, weight in self.EMOTION_INTENSITY_PATTERNS:
            if re.search(pattern, text):
                intensity = max(intensity, weight)

        exclamation_count = text.count("!")
        question_count = text.count("?")

        if exclamation_count >= 3:
            intensity *= 1.2
        if question_count >= 2:
            intensity *= 0.9

        return min(2.0, max(0.5, intensity))

    def _detect_by_llm(
        self,
        text: str,
        context: List[Dict[str, str]]
    ) -> Optional[Dict[str, any]]:
        try:
            prompt = f"""分析以下对话中用户的情绪状态。
只返回一个JSON格式的回复：
{{"emotion": "情绪类型", "confidence": 0.0-1.0}}

对话内容：{text}
上下文：{context[-3:] if context else []}

情绪类型可选：happy, sad, angry, fear, surprise, disgust, neutral
"""
            return {
                "emotion": "neutral",
                "confidence": 0.5,
            }
        except Exception as e:
            logger.warning(f"LLM emotion detection failed: {e}")
            return None

    def _empty_result(self) -> Dict[str, any]:
        return {
            "emotion": "neutral",
            "confidence": 0.0,
            "all_emotions": {},
            "intensity": 1.0,
        }

    def get_tts_emotion_params(self, emotion: str) -> Dict[str, any]:
        emotion_params = {
            "happy": {"speed": 1.1, "pitch": 1.05, "energy": 1.2},
            "sad": {"speed": 0.9, "pitch": 0.95, "energy": 0.8},
            "angry": {"speed": 1.15, "pitch": 0.9, "energy": 1.3},
            "fear": {"speed": 1.1, "pitch": 1.1, "energy": 1.1},
            "surprise": {"speed": 1.05, "pitch": 1.1, "energy": 1.15},
            "disgust": {"speed": 0.95, "pitch": 0.9, "energy": 0.9},
            "neutral": {"speed": 1.0, "pitch": 1.0, "energy": 1.0},
        }
        return emotion_params.get(emotion, emotion_params["neutral"])


class SentimentAnalyzer:
    def __init__(self):
        self._positive_words = set()
        self._negative_words = set()
        self._load_sentiment_lexicon()

    def _load_sentiment_lexicon(self):
        self._positive_words = {
            "好", "棒", "不错", "优秀", "出色", "完美", "赞", "喜欢",
            "开心", "高兴", "快乐", "幸福", "满意", "感谢", "谢谢",
            "哈哈", "嘿嘿", "嘻嘻", "good", "great", "excellent"
        }
        self._negative_words = {
            "差", "糟糕", "不好", "烂", "讨厌", "恨", "不喜欢",
            "难过", "伤心", "失望", "生气", "愤怒", "bad", "terrible"
        }

    def analyze(self, text: str) -> Dict[str, any]:
        positive_count = sum(1 for word in self._positive_words if word in text)
        negative_count = sum(1 for word in self._negative_words if word in text)

        total = positive_count + negative_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total

        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": round(sentiment_score, 3),
            "positive_count": positive_count,
            "negative_count": negative_count,
        }
