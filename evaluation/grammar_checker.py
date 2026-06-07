"""
语法纠错模块 — 基于 DeepSeek API 对用户输入进行语法/词汇/风格纠错。

GrammarChecker 封装异步 API 调用，通过专用 system prompt 引导
大模型输出结构化 JSON 纠错结果，覆盖语法错误、词汇误用和风格问题。
"""

import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from config import config

logger = logging.getLogger(__name__)

# 引导 DeepSeek 输出结构化 JSON 的 system prompt
_GRAMMAR_SYSTEM_PROMPT = """You are an expert English language tutor. Analyze the user's English sentence and provide corrections for grammar, vocabulary, and style issues.

Return ONLY a valid JSON object with the following structure (no extra text, no markdown fences):
{
  "original": "the original sentence exactly as provided",
  "corrected": "the fully corrected sentence",
  "errors": [
    {
      "type": "grammar|vocabulary|style",
      "original": "the problematic word or phrase",
      "suggestion": "the suggested correction",
      "explanation": "brief explanation in the same language as the user's input"
    }
  ]
}

Rules:
1. If the sentence is already correct, return an empty errors array and set corrected equal to original.
2. Be precise: only flag genuine errors, do not over-correct stylistic preferences.
3. For each error, specify the exact type: "grammar" (tense, agreement, word order, article), "vocabulary" (wrong word choice), or "style" (awkward phrasing, formality).
4. The explanation should be concise and helpful, written in the same language as the user's input.
5. Output ONLY valid JSON — no markdown code fences, no trailing commas, no extra commentary."""


class GrammarChecker:
    """语法纠错器，调用 DeepSeek API 对英文句子进行语法/词汇/风格纠错。

    用法::

        checker = GrammarChecker()
        result = await checker.check("He go to school yesterday")
        # {
        #     "original": "He go to school yesterday",
        #     "corrected": "He went to school yesterday",
        #     "errors": [{"type": "grammar", "original": "go", ...}]
        # }
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
        )
        self._model: str = getattr(config, "GRAMMAR_CHECK_MODEL", config.DEEPSEEK_MODEL)

    async def check(self, text: str, scene_name: Optional[str] = None) -> dict:
        """对输入文本进行语法纠错检查。

        Args:
            text: 待检查的英文句子或段落。
            scene_name: 可选，场景名称（当前未使用，保留用于未来按场景调整纠错策略）。

        Returns:
            dict: {
                "original": "原句",
                "corrected": "纠错后完整句子",
                "errors": [
                    {
                        "type": "grammar|vocabulary|style",
                        "original": "错误片段",
                        "suggestion": "建议修改",
                        "explanation": "解释说明"
                    },
                    ...
                ]
            }
            当 API 不可用或解析失败时，errors 为空列表，corrected 等于 original。

        Raises:
            不会抛出异常，所有异常均在内部处理并以空 errors 返回。
        """
        if not text or not text.strip():
            return {"original": text or "", "corrected": text or "", "errors": []}

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _GRAMMAR_SYSTEM_PROMPT},
                    {"role": "user", "content": text.strip()},
                ],
                temperature=0.1,  # 低温度以保证结构化输出稳定
                max_tokens=1024,
            )

            raw = response.choices[0].message.content or ""
            result = self._parse_response(raw, text)

        except Exception as exc:
            logger.warning("Grammar check API call failed: %s", exc)
            result = {"original": text.strip(), "corrected": text.strip(), "errors": []}

        return result

    def _parse_response(self, raw: str, original_text: str) -> dict:
        """解析 DeepSeek 返回的原始文本为结构化 dict。

        尝试多种解析策略以增强鲁棒性：
        1. 直接 JSON 解析
        2. 去除 markdown 代码块标记后 JSON 解析
        3. 正则提取 JSON 对象

        Args:
            raw: API 返回的原始文本。
            original_text: 用户原始输入，用于兜底。

        Returns:
            结构化的纠错结果 dict。
        """
        fallback = {"original": original_text.strip(), "corrected": original_text.strip(), "errors": []}

        # 策略1：直接解析
        try:
            data = json.loads(raw)
            return self._validate_result(data, original_text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 策略2：去除 ```json ... ``` 包裹
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        try:
            data = json.loads(cleaned)
            return self._validate_result(data, original_text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 策略3：正则提取第一个 JSON 对象
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._validate_result(data, original_text)
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("Failed to parse grammar check response: %s", raw[:200])
        return fallback

    @staticmethod
    def _validate_result(data: dict, original_text: str) -> dict:
        """校验并规范化解析结果。

        Args:
            data: 解析出的原始 dict。
            original_text: 用户原始输入。

        Returns:
            结构完整的纠错结果。
        """
        result = {
            "original": data.get("original", original_text.strip()),
            "corrected": data.get("corrected", original_text.strip()),
            "errors": [],
        }

        raw_errors = data.get("errors")
        if isinstance(raw_errors, list):
            for err in raw_errors:
                if isinstance(err, dict):
                    result["errors"].append({
                        "type": err.get("type", "grammar"),
                        "original": err.get("original", ""),
                        "suggestion": err.get("suggestion", ""),
                        "explanation": err.get("explanation", ""),
                    })

        return result
