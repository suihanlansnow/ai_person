class DialogueFeedbackSystem:
    """对话反馈系统 - 管理机器人与用户的交互反馈"""

    class FeedbackType:
        INPUT_RECEIVED = "input_received"
        ASR_PROCESSING = "asr_processing"
        ASR_COMPLETE = "asr_complete"
        ASR_FAILED = "asr_failed"
        LLM_PROCESSING = "llm_processing"
        LLM_COMPLETE = "llm_complete"
        LLM_FAILED = "llm_failed"
        TTS_PROCESSING = "tts_processing"
        TTS_COMPLETE = "tts_complete"
        NETWORK_ERROR = "network_error"
        TIMEOUT = "timeout"

    class FeedbackPriority:
        CRITICAL = "critical"    # 必须显示
        HIGH = "high"          # 重要反馈
        NORMAL = "normal"       # 一般反馈
        LOW = "low"            # 可忽略

    def __init__(self):
        self.feedback_callbacks = []
        self.current_state = "idle"
        self.feedback_history = []

    def on_feedback(self, feedback_type, data=None, priority=NORMAL):
        """触发反馈"""
        feedback = {
            "type": feedback_type,
            "data": data,
            "priority": priority,
            "timestamp": time.time(),
            "state": self.current_state
        }
        self.feedback_history.append(feedback)

        for callback in self.feedback_callbacks:
            callback(feedback)

        return feedback

    def register_callback(self, callback):
        """注册反馈回调"""
        self.feedback_callbacks.append(callback)

    def set_state(self, state):
        """设置当前状态"""
        self.current_state = state

class FeedbackUI:
    """反馈UI控制器"""

    def __init__(self, feedback_system):
        self.fs = feedback_system
        self.fs.register_callback(self.handle_feedback)
        self.active_indicators = []

    def handle_feedback(self, feedback):
        """处理各类反馈"""
        handlers = {
            FeedbackSystem.FeedbackType.INPUT_RECEIVED: self.show_input_received,
            FeedbackSystem.FeedbackType.ASR_PROCESSING: self.show_asr_processing,
            FeedbackSystem.FeedbackType.ASR_COMPLETE: self.show_asr_complete,
            FeedbackSystem.FeedbackType.LLM_PROCESSING: self.show_llm_processing,
            FeedbackSystem.FeedbackType.LLM_COMPLETE: self.show_llm_complete,
            FeedbackSystem.FeedbackType.NETWORK_ERROR: self.show_network_error,
            FeedbackSystem.FeedbackType.TIMEOUT: self.show_timeout,
        }

        handler = handlers.get(feedback['type'])
        if handler:
            handler(feedback)

    def show_input_received(self, feedback):
        """显示输入已收到"""
        print(f"✅ 收到输入: {feedback['data']}")

    def show_asr_processing(self, feedback):
        """显示语音识别中"""
        print("🎤 正在识别语音...")

    def show_asr_complete(self, feedback):
        """显示语音识别完成"""
        print(f"📝 已识别: {feedback['data']['text']}")

    def show_llm_processing(self, feedback):
        """显示LLM处理中"""
        print("🤖 AI正在思考...")

    def show_llm_complete(self, feedback):
        """显示LLM回复完成"""
        print(f"💬 AI回复: {feedback['data']['response']}")

    def show_network_error(self, feedback):
        """显示网络错误"""
        print(f"❌ 网络错误: {feedback['data']}")

    def show_timeout(self, feedback):
        """显示超时"""
        print(f"⏰ 请求超时: {feedback['data']}")


class DialoguePipeline:
    """对话处理流水线 - 协调各服务并产生反馈"""

    def __init__(self, feedback_system):
        self.fs = feedback_system

    async def process_audio_input(self, audio_data):
        """处理音频输入的完整流程"""

        # 1. 输入接收反馈
        self.fs.on_feedback(
            FeedbackSystem.FeedbackType.INPUT_RECEIVED,
            data={"audio_size": len(audio_data)},
            priority=FeedbackSystem.FeedbackPriority.CRITICAL
        )

        try:
            # 2. ASR处理
            self.fs.set_state("asr")
            self.fs.on_feedback(
                FeedbackSystem.FeedbackType.ASR_PROCESSING,
                priority=FeedbackSystem.FeedbackPriority.HIGH
            )

            asr_result = await self.asr_service.recognize(audio_data)

            if asr_result.success:
                # 3. ASR完成反馈
                self.fs.on_feedback(
                    FeedbackSystem.FeedbackType.ASR_COMPLETE,
                    data={"text": asr_result.text, "confidence": asr_result.confidence},
                    priority=FeedbackSystem.FeedbackPriority.HIGH
                )

                # 4. LLM处理
                self.fs.set_state("llm")
                self.fs.on_feedback(
                    FeedbackSystem.FeedbackType.LLM_PROCESSING,
                    priority=FeedbackSystem.FeedbackPriority.HIGH
                )

                llm_result = await self.llm_service.chat([
                    {"role": "user", "content": asr_result.text}
                ])

                if llm_result.success:
                    # 5. LLM完成反馈
                    self.fs.on_feedback(
                        FeedbackSystem.FeedbackType.LLM_COMPLETE,
                        data={"response": llm_result.response},
                        priority=FeedbackSystem.FeedbackPriority.HIGH
                    )

                    # 6. TTS处理（可选）
                    if self.tts_service:
                        self.fs.set_state("tts")
                        self.fs.on_feedback(
                            FeedbackSystem.FeedbackType.TTS_PROCESSING
                        )
                        await self.tts_service.synthesize(llm_result.response)

                    self.fs.set_state("idle")
                    return llm_result.response
                else:
                    self.fs.on_feedback(
                        FeedbackSystem.FeedbackType.LLM_FAILED,
                        data={"error": llm_result.error}
                    )
            else:
                self.fs.on_feedback(
                    FeedbackSystem.FeedbackType.ASR_FAILED,
                    data={"error": asr_result.error}
                )

        except NetworkError as e:
            self.fs.on_feedback(
                FeedbackSystem.FeedbackType.NETWORK_ERROR,
                data=str(e),
                priority=FeedbackSystem.FeedbackPriority.CRITICAL
            )
        except TimeoutError as e:
            self.fs.on_feedback(
                FeedbackSystem.FeedbackType.TIMEOUT,
                data=str(e),
                priority=FeedbackSystem.FeedbackPriority.CRITICAL
            )
        finally:
            self.fs.set_state("idle")


class EmotionAwareResponse:
    """情感感知反馈系统"""

    EMOTION_FEEDBACK = {
        "happy": {
            "emoji": "😊",
            "message": "很高兴帮到你！",
            "animation": "bounce"
        },
        "sad": {
            "emoji": "😢",
            "message": "我理解你的感受。",
            "animation": "fade"
        },
        "angry": {
            "emoji": "😠",
            "message": "请冷静，我会尽力帮助你。",
            "animation": "shake"
        },
        "confused": {
            "emoji": "🤔",
            "message": "抱歉，我不太明白，能再说一次吗？",
            "animation": "pulse"
        },
        "neutral": {
            "emoji": "😐",
            "message": "有什么我可以帮助你的？",
            "animation": "none"
        }
    }

    def __init__(self):
        self.current_emotion = "neutral"

    def detect_emotion(self, text):
        """检测用户情感"""
        # 简化版情感检测
        if any(word in text for word in ["开心", "高兴", "快乐"]):
            return "happy"
        elif any(word in text for word in ["难过", "伤心", "沮丧"]):
            return "sad"
        elif any(word in text for word in ["生气", "愤怒", "烦"]):
            return "angry"
        return "neutral"

    def get_feedback_for_emotion(self, emotion):
        """获取情感反馈"""
        return self.EMOTION_FEEDBACK.get(emotion, self.EMOTION_FEEDBACK["neutral"])


class UserFeedbackCollector:
    """用户反馈收集器 - 收集用户对回复的满意度"""

    class FeedbackType:
        THUMBS_UP = "thumbs_up"
        THUMBS_DOWN = "thumbs_down"
        REGENERATE = "regenerate"
        RATE_STARS = "rate_stars"

    def __init__(self):
        self.feedback_data = []

    def collect(self, response_id, feedback_type, rating=None, comment=None):
        """收集用户反馈"""
        feedback = {
            "response_id": response_id,
            "type": feedback_type,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time()
        }
        self.feedback_data.append(feedback)
        return feedback

    def get_average_rating(self):
        """获取平均评分"""
        rated = [f for f in self.feedback_data if f.get('rating')]
        if not rated:
            return None
        return sum(f['rating'] for f in rated) / len(rated)
