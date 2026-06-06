"""
Web 服务模块 — FastAPI + WebSocket 服务串联。

将 coach（对话引擎）、asr（语音识别）、tts（语音合成）三大模块
通过 HTTP / WebSocket 接口串联为完整的英语口语陪练服务。
"""

from .handler import MessageHandler

__all__ = ["MessageHandler"]