"""对话引擎 — 集成 DeepSeek Chat API，支持场景化多轮对话与流式生成。"""

import asyncio
import logging
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

from config import config
from coach.scene_manager import SceneManager

logger = logging.getLogger(__name__)


class CoachEngine:
    """对话引擎核心类。

    整合 DeepSeek Chat API，支持：
    - 场景化 System Prompt 注入
    - 多轮对话历史管理
    - SSE 兼容的流式响应
    - 异常重试
    """

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0  # 秒，指数退避起点

    def __init__(self, scene_manager: SceneManager):
        self._scene_manager = scene_manager
        self._client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._model = config.DEEPSEEK_MODEL
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset_history(self) -> None:
        """清空对话历史。"""
        self._history.clear()

    def append_system(self, content: str) -> None:
        """向历史追加一条 system 消息。"""
        self._history.append({"role": "system", "content": content})

    def append_user(self, content: str) -> None:
        """向历史追加一条 user 消息。"""
        self._history.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        """向历史追加一条 assistant 消息。"""
        self._history.append({"role": "assistant", "content": content})

    def get_history(self) -> list[dict[str, str]]:
        """返回当前对话历史副本。"""
        return list(self._history)

    def build_messages(
        self, user_text: str, scene_name: Optional[str] = None
    ) -> list[dict[str, str]]:
        """构建发送给 API 的完整消息列表。

        如果 history 中没有 system 消息，自动注入当前场景的 system_prompt。
        """
        # 确保 system prompt 在最前面
        if not any(m["role"] == "system" for m in self._history):
            prompt = self._scene_manager.get_system_prompt(scene_name)
            self._history.insert(0, {"role": "system", "content": prompt})

        return self._history + [{"role": "user", "content": user_text}]

    async def generate_reply(
        self,
        user_text: str,
        scene_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """生成完整 AI 回复（非流式）。"""
        messages = self.build_messages(user_text, scene_name)
        self.append_user(user_text)

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                reply = response.choices[0].message.content or ""
                self.append_assistant(reply)
                return reply

            except Exception as exc:
                logger.warning(
                    "DeepSeek API error (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, exc,
                )
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"DeepSeek API failed after {self.MAX_RETRIES} attempts"
                    ) from exc
                await asyncio.sleep(self.RETRY_DELAY_BASE * (2 ** (attempt - 1)))

        # 理论上不会走到这里，但保持类型安全
        raise RuntimeError("Unreachable")

    async def generate_reply_stream(
        self,
        user_text: str,
        scene_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """流式生成 AI 回复（SSE 兼容）。

        每次 yield 一个文本片段（delta），适合通过 WebSocket / SSE 推送给前端。
        """
        messages = self.build_messages(user_text, scene_name)
        self.append_user(user_text)

        full_reply = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                stream = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_reply += delta.content
                        yield delta.content

                self.append_assistant(full_reply)
                return  # 成功，退出

            except Exception as exc:
                logger.warning(
                    "DeepSeek API stream error (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, exc,
                )
                if attempt == self.MAX_RETRIES:
                    raise RuntimeError(
                        f"DeepSeek API streaming failed after "
                        f"{self.MAX_RETRIES} attempts"
                    ) from exc
                await asyncio.sleep(self.RETRY_DELAY_BASE * (2 ** (attempt - 1)))
                # 重试时重置已收集的片段
                full_reply = ""
