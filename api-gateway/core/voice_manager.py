class VoiceConfig:
    """音色配置"""

    def __init__(self, voice_id, name, gender, age_range, characteristics):
        self.voice_id = voice_id
        self.name = name
        self.gender = gender
        self.age_range = age_range
        self.characteristics = characteristics

    def __repr__(self):
        return f"Voice({self.voice_id}, {self.name}, {self.gender})"


class VoiceManager:
    """音色管理器 - 管理所有可用音色"""

    PRESET_VOICES = {
        "default": VoiceConfig(
            voice_id="default",
            name="默认女声",
            gender="female",
            age_range="young_adult",
            characteristics={"pitch": 1.0, "speed": 1.0, "emotion": "neutral"}
        ),
        "female_young": VoiceConfig(
            voice_id="female_young",
            name="活泼女声",
            gender="female",
            age_range="young",
            characteristics={"pitch": 1.1, "speed": 1.05, "emotion": "bright"}
        ),
        "female_mature": VoiceConfig(
            voice_id="female_mature",
            name="知性女声",
            gender="female",
            age_range="adult",
            characteristics={"pitch": 0.95, "speed": 0.95, "emotion": "calm"}
        ),
        "male_young": VoiceConfig(
            voice_id="male_young",
            name="活力男声",
            gender="male",
            age_range="young_adult",
            characteristics={"pitch": 0.9, "speed": 1.0, "emotion": "energetic"}
        ),
        "male_mature": VoiceConfig(
            voice_id="male_mature",
            name="成熟男声",
            gender="male",
            age_range="adult",
            characteristics={"pitch": 0.85, "speed": 0.9, "emotion": "steady"}
        ),
        "child": VoiceConfig(
            voice_id="child",
            name="童声",
            gender="neutral",
            age_range="child",
            characteristics={"pitch": 1.3, "speed": 1.1, "emotion": "playful"}
        ),
    }

    EMOTION_VOICE_MAP = {
        "happy": "female_young",
        "sad": "female_mature",
        "angry": "male_mature",
        "fear": "male_young",
        "surprise": "female_young",
        "neutral": "default",
    }

    def __init__(self):
        self.current_voice_id = "default"
        self.user_preference = None

    def get_voice(self, voice_id=None):
        """获取音色"""
        voice_id = voice_id or self.current_voice_id
        return self.PRESET_VOICES.get(voice_id, self.PRESET_VOICES["default"])

    def set_voice(self, voice_id):
        """设置当前音色"""
        if voice_id in self.PRESET_VOICES:
            self.current_voice_id = voice_id
            return True
        return False

    def get_voice_for_emotion(self, emotion):
        """根据情感获取对应音色"""
        voice_id = self.EMOTION_VOICE_MAP.get(emotion, "default")
        return self.get_voice(voice_id)

    def set_user_preference(self, voice_id):
        """设置用户偏好"""
        self.user_preference = voice_id
        self.current_voice_id = voice_id

    def list_voices(self):
        """列出所有可用音色"""
        return [
            {"voice_id": v.voice_id, "name": v.name, "gender": v.gender}
            for v in self.PRESET_VOICES.values()
        ]


class TTSService:
    """TTS服务 - 支持多音色"""

    def __init__(self, voice_manager):
        self.voice_manager = voice_manager
        self.current_params = {
            "pitch": 1.0,
            "speed": 1.0,
            "volume": 1.0,
        }

    def synthesize(self, text, voice_id=None, emotion=None):
        """合成语音"""
        voice = self.voice_manager.get_voice(voice_id)

        if emotion:
            emotion_voice = self.voice_manager.get_voice_for_emotion(emotion)
            voice = emotion_voice

        params = self.current_params.copy()
        params.update(voice.characteristics)

        return {
            "audio_data": self._generate_audio(text, params),
            "voice_id": voice.voice_id,
            "voice_name": voice.name,
            "params": params
        }

    def _generate_audio(self, text, params):
        """生成音频（实际实现需调用TTS引擎）"""
        pass


class DynamicVoiceSelector:
    """动态音色选择器 - 根据上下文选择最佳音色"""

    def __init__(self, voice_manager):
        self.voice_manager = voice_manager

    def select_voice(self, context):
        """根据上下文选择音色"""
        voice_id = "default"

        if context.get("emotion"):
            voice_id = self.voice_manager.EMOTION_VOICE_MAP.get(
                context["emotion"], "default"
            )

        if context.get("scene"):
            voice_id = self._select_by_scene(context["scene"])

        if context.get("user_preference"):
            voice_id = context["user_preference"]

        return voice_id

    def _select_by_scene(self, scene):
        """根据场景选择音色"""
        scene_map = {
            "greeting": "female_young",
            "serious": "male_mature",
            "casual": "default",
            "story": "female_mature",
            "joke": "child",
        }
        return scene_map.get(scene, "default")


class VoicePresets:
    """预设音色参数"""

    PARAM_PRESETS = {
        " narration": {
            "pitch": 0.9,
            "speed": 0.95,
            "volume": 1.0,
            "description": "旁白/讲故事"
        },
        "dialogue": {
            "pitch": 1.0,
            "speed": 1.0,
            "volume": 1.0,
            "description": "日常对话"
        },
        "whisper": {
            "pitch": 1.2,
            "speed": 0.8,
            "volume": 0.7,
            "description": "轻声细语"
        },
        "excited": {
            "pitch": 1.1,
            "speed": 1.15,
            "volume": 1.0,
            "description": "兴奋激动"
        },
        "calm": {
            "pitch": 0.95,
            "speed": 0.85,
            "volume": 0.9,
            "description": "平静舒缓"
        },
    }


def main():
    """使用示例"""
    vm = VoiceManager()

    print("=== 可用音色 ===")
    for v in vm.list_voices():
        print(f"  {v['voice_id']}: {v['name']} ({v['gender']})")

    print("\n=== 情感音色映射 ===")
    for emotion, voice_id in vm.EMOTION_VOICE_MAP.items():
        print(f"  {emotion} → {voice_id}")

    print("\n=== 音色参数示例 ===")
    voice = vm.get_voice("female_young")
    print(f"  {voice.name}: pitch={voice.characteristics['pitch']}, speed={voice.characteristics['speed']}")

    print("\n=== 根据情感选择音色 ===")
    for emotion in ["happy", "sad", "angry", "neutral"]:
        voice = vm.get_voice_for_emotion(emotion)
        print(f"  {emotion} → {voice.name}")


if __name__ == "__main__":
    main()
