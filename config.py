"""项目配置管理"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置"""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ASR
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
    SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # TTS
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-JennyNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "+0%")
    TTS_PITCH: str = os.getenv("TTS_PITCH", "+0Hz")

    # 服务
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # VAD
    VAD_AGGRESSIVENESS: int = int(os.getenv("VAD_AGGRESSIVENESS", "2"))

    # Web 服务 — 对话历史
    MAX_HISTORY_ROUNDS: int = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))

    # Web 服务 — WebSocket
    WS_HEARTBEAT_INTERVAL: int = int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))

    # Evaluation — 纠错评测
    GRAMMAR_CHECK_MODEL: str = os.getenv(
        "GRAMMAR_CHECK_MODEL",
        os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    PRONUNCIATION_THRESHOLD: int = int(os.getenv("PRONUNCIATION_THRESHOLD", "60"))

    # Report — 课后报告
    REPORT_USE_AI: bool = os.getenv("REPORT_USE_AI", "true").lower() == "true"

    # 项目路径
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    SCENES_DIR: str = os.path.join(PROJECT_ROOT, "scenes")


config = Config()
