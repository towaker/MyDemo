"""
ASR 模块：语音识别

提供 Whisper 语音识别、麦克风音频采集、语音活动检测（VAD）功能。
"""

from .recognizer import WhisperRecognizer
from .audio_capture import AudioCapture
from .vad import VAD

__all__ = ["WhisperRecognizer", "AudioCapture", "VAD"]
