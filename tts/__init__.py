"""
TTS 模块：语音合成

基于 edge-tts 提供文本转语音功能，支持文件合成和流式合成。
"""

from .synthesizer import Synthesizer, text_to_speech

__all__ = ["Synthesizer", "text_to_speech"]
