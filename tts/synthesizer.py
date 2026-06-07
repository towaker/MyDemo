"""
TTS 语音合成模块 — 基于 edge-tts 的文本转语音。

提供：
- 文本合成 MP3 文件
- 流式音频生成（供 WebSocket 实时推流）
- 可用语音角色列表查询
"""

import asyncio
import logging
import re
import unicodedata
from typing import AsyncGenerator

import edge_tts

from config import config

logger = logging.getLogger(__name__)

# edge-tts 可能无法正确处理的控制字符
_TTS_UNSAFE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class Synthesizer:
    """基于 edge-tts 的文本转语音合成器。

    使用微软 Edge TTS 引擎，提供文件合成和流式合成两种模式。
    语音角色、语速、音调均可配置。

    Attributes:
        voice: 语音角色标识符（如 "en-US-JennyNeural"）。
        rate: 语速偏移（如 "+0%", "-10%"）。
        pitch: 音调偏移（如 "+0Hz", "-10Hz"）。
    """

    def __init__(
        self,
        voice: str | None = None,
        rate: str | None = None,
        pitch: str | None = None,
    ):
        """初始化合成器。

        Args:
            voice: 语音角色，默认从 config.TTS_VOICE 读取。
            rate: 语速偏移，默认从 config.TTS_RATE 读取（如 "+0%"）。
            pitch: 音调偏移，默认从 config.TTS_PITCH 读取（如 "+0Hz"）。
        """
        self.voice: str = voice or config.TTS_VOICE
        self.rate: str = rate or config.TTS_RATE
        self.pitch: str = pitch or config.TTS_PITCH

    async def synthesize(self, text: str, output_path: str) -> str:
        """将文本合成为 MP3 音频文件并保存到指定路径。

        Args:
            text: 要合成的文本内容。
            output_path: 输出 MP3 文件路径。

        Returns:
            output_path，即合成后的音频文件路径。

        Raises:
            ValueError: 文本为空时抛出。
            ConnectionError: 网络不可用或 edge-tts 服务不可达时抛出。
            RuntimeError: 合成过程失败时抛出。
        """
        # 文本清理：移除控制字符、规范化 Unicode
        cleaned = self._sanitize_text(text)

        if not cleaned or not cleaned.strip():
            raise ValueError("合成文本不能为空")

        logger.info("开始合成语音，文本长度=%d，输出路径=%s", len(cleaned), output_path)

        try:
            communicate = edge_tts.Communicate(
                text=cleaned,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            await communicate.save(output_path)
        except edge_tts.exceptions.NoAudioReceived as exc:
            raise RuntimeError(
                "edge-tts 未返回音频数据，请检查语音角色是否有效"
            ) from exc
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(
                kw in error_msg
                for kw in ["connect", "dns", "resolve", "timeout", "network"]
            ):
                raise ConnectionError(
                    f"无法连接 edge-tts 服务，请检查网络连接: {exc}"
                ) from exc
            raise RuntimeError(f"语音合成失败: {exc}") from exc

        logger.info("语音合成完成: %s", output_path)
        return output_path

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """清理文本中可能导致 edge-tts 合成失败的特殊字符。

        处理策略：
        1. 移除 ASCII 控制字符（U+0000-U+001F，保留 \\t \\n \\r）
        2. 移除 Unicode 不可见字符（U+007F-U+009F）
        3. Unicode 规范化（NFC），将全角/组合字符转为标准形式
        4. 若清理后仅剩空白/标点（无可朗读内容），返回空字符串

        Args:
            text: 原始文本。

        Returns:
            清理后的安全文本。
        """
        if not text:
            return ""

        # Step 1: Unicode NFC 规范化（统一组合字符、全角转半角等）
        text = unicodedata.normalize("NFC", text)

        # Step 2: 移除控制字符（保留 \\n, \\r, \\t 作为韵律边界）
        text = _TTS_UNSAFE_PATTERN.sub(" ", text)

        # Step 3: 将连续的空白合并为单个空格
        text = re.sub(r"\s+", " ", text)

        cleaned = text.strip()

        # Step 4: 检查是否只剩标点/符号（无可朗读内容）
        if cleaned and re.match(r"^[\s\W_]+$", cleaned):
            logger.warning("清理后文本仅含标点/符号，无可朗读内容: %r", cleaned[:50])
            return ""

        return cleaned

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成语音，逐块 yield 音频数据。

        适用于 WebSocket 实时推流场景，无需等待完整文件生成。

        Args:
            text: 要合成的文本内容。

        Yields:
            bytes: MP3 音频数据块。

        Raises:
            ValueError: 文本为空时抛出。
            ConnectionError: 网络不可用时抛出。
            RuntimeError: 合成失败时抛出。
        """
        if not text or not text.strip():
            raise ValueError("合成文本不能为空")

        cleaned = self._sanitize_text(text)
        if not cleaned:
            raise ValueError("合成文本不能为空")

        logger.info("开始流式合成语音，文本长度=%d", len(cleaned))

        try:
            communicate = edge_tts.Communicate(
                text=cleaned,
                voice=self.voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(kw in error_msg for kw in ["connect", "dns", "resolve", "timeout", "network"]):
                raise ConnectionError(
                    f"无法连接 edge-tts 服务，请检查网络连接: {exc}"
                ) from exc
            raise RuntimeError(f"流式语音合成失败: {exc}") from exc

        logger.info("流式语音合成完成")

    @staticmethod
    async def list_voices() -> list[dict]:
        """列出所有可用的语音角色。

        调用 edge_tts 的 list_voices API 获取完整语音列表。

        Returns:
            list[dict]: 语音角色信息列表，每个 dict 包含 Name, ShortName,
                Gender, Locale 等字段。

        Raises:
            ConnectionError: 网络不可用或 edge-tts 服务不可达时抛出。
            RuntimeError: 获取语音列表失败时抛出。
        """
        logger.info("获取可用语音角色列表")

        try:
            voices = await edge_tts.list_voices()
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(kw in error_msg for kw in ["connect", "dns", "resolve", "timeout", "network"]):
                raise ConnectionError(
                    f"无法连接 edge-tts 服务，请检查网络连接: {exc}"
                ) from exc
            raise RuntimeError(f"获取语音列表失败: {exc}") from exc

        voice_list = []
        for voice in voices:
            voice_list.append({
                "Name": voice.get("Name", ""),
                "ShortName": voice.get("ShortName", ""),
                "Gender": voice.get("Gender", ""),
                "Locale": voice.get("Locale", ""),
                "FriendlyName": voice.get("FriendlyName", ""),
                "Status": voice.get("Status", ""),
            })

        logger.info("获取到 %d 个语音角色", len(voice_list))
        return voice_list


# 便捷异步函数：快速合成并保存

async def text_to_speech(
    text: str,
    output_path: str,
    voice: str | None = None,
    rate: str | None = None,
    pitch: str | None = None,
) -> str:
    """便捷函数：快速将文本合成为 MP3 文件。

    Args:
        text: 要合成的文本。
        output_path: 输出 MP3 文件路径。
        voice: 语音角色，默认使用 config.TTS_VOICE。
        rate: 语速，默认使用 config.TTS_RATE。
        pitch: 音调，默认使用 config.TTS_PITCH。

    Returns:
        output_path。
    """
    synth = Synthesizer(voice=voice, rate=rate, pitch=pitch)
    return await synth.synthesize(text, output_path)
