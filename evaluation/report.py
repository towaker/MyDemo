"""
课后报告生成器 — 基于会话评测数据生成结构化学习报告。

ReportGenerator 接收 SessionTracker 的会话汇总数据，
通过 DeepSeek API 进行综合分析，生成多维度评分和改进建议。
同时支持纯本地统计模式（API 不可用时降级）。
"""

import json
import logging
import re
from typing import Optional

from openai import AsyncOpenAI

from config import config

logger = logging.getLogger(__name__)

_REPORT_SYSTEM_PROMPT = """You are an expert English language learning coach. Based on the student's practice session data, generate a comprehensive learning report.

The session data includes:
- Scene practiced
- Dialogue turn count and duration
- Grammar errors found (with error details)
- Pronunciation evaluation scores

Return ONLY a valid JSON object with the following structure (no markdown fences, no extra text):
{
  "overall_score": <int 0-100, weighted composite score>,
  "summary": "<2-3 sentence overall assessment in Chinese, encouraging and constructive>",
  "strengths": ["<strength 1>", "<strength 2>", "..."],
  "weaknesses": ["<weakness 1>", "<weakness 2>", "..."],
  "grammar_summary": "<1-2 sentence grammar assessment in Chinese>",
  "pronunciation_summary": "<1-2 sentence pronunciation assessment in Chinese>",
  "vocabulary_tip": "<1 vocabulary improvement tip in Chinese>",
  "practice_suggestions": ["<suggestion 1>", "<suggestion 2>", "..."]
}

Rules:
1. overall_score: weight grammar 40%, pronunciation 30%, dialogue engagement 30%
2. If no grammar/pronunciation data available, note that in summaries but still provide constructive feedback
3. Be specific and actionable in suggestions — mention actual error patterns if available
4. Summaries must be in Chinese
5. strengths and weaknesses at least 2 each, at most 5 each
6. Output ONLY valid JSON"""


class ReportGenerator:
    """课后报告生成器。

    使用两种策略：
    1. AI 增强模式（默认）：调用 DeepSeek API 分析会话数据，生成
       个性化评语和改进建议。
    2. 本地统计模式（降级）：仅基于统计数据生成基础评分。

    用法::

        generator = ReportGenerator()
        summary = tracker.get_session_summary()
        report = await generator.generate(summary)
    """

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None
        try:
            self._client = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL,
            )
        except Exception as exc:
            logger.warning("ReportGenerator: DeepSeek 客户端初始化失败: %s", exc)

    async def generate(
        self,
        session_summary: dict,
        use_ai: bool = True,
    ) -> dict:
        """生成课后学习报告。

        Args:
            session_summary: SessionTracker.get_session_summary() 返回的会话汇总数据。
            use_ai: 是否使用 AI 增强分析。设为 False 则使用纯统计模式。

        Returns:
            dict: {
                "session_id": str,
                "scene_name": str,
                "duration_seconds": float,
                "dialogue_count": int,
                "overall_score": int,
                "summary": str,
                "strengths": [str, ...],
                "weaknesses": [str, ...],
                "grammar": {
                    "total_errors": int,
                    "errors_by_type": dict,
                    "checks": int,
                    "summary": str,
                },
                "pronunciation": {
                    "avg_score": float,
                    "avg_fluency": float,
                    "avg_accuracy": float,
                    "checks": int,
                    "summary": str,
                },
                "vocabulary_tip": str,
                "practice_suggestions": [str, ...],
                "detailed_errors": list,
                "generated_by": "ai" | "local",
            }
        """
        if use_ai and self._client is not None:
            try:
                return await self._generate_with_ai(session_summary)
            except Exception as exc:
                logger.warning("AI 报告生成失败，降级到本地统计: %s", exc)

        return self._generate_local(session_summary)

    async def _generate_with_ai(self, session_summary: dict) -> dict:
        """使用 DeepSeek API 生成个性化报告。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            包含 AI 评语的完整报告 dict。
        """
        # 构建精简的会话数据摘要送给 AI
        ai_input = self._build_ai_input(session_summary)
        response = await self._client.chat.completions.create(
            model=getattr(config, "DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": ai_input},
            ],
            temperature=0.5,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content or ""
        ai_result = self._parse_json_response(raw)

        return self._merge_results(session_summary, ai_result, generated_by="ai")

    def _generate_local(self, session_summary: dict) -> dict:
        """纯本地统计模式生成基础报告。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            结构化报告 dict。
        """
        # 计算综合评分
        grammar_score = self._calc_grammar_score(session_summary)
        pron_score = int(session_summary.get("pronunciation_avg_score", 0))
        dialogue_score = min(100, session_summary.get("dialogue_count", 0) * 10)
        overall = int(grammar_score * 0.4 + pron_score * 0.3 + dialogue_score * 0.3)

        scene = session_summary.get("scene_name", "")

        # 默认评语
        if session_summary.get("dialogue_count", 0) > 0:
            summary = f"本次练习共进行了 {session_summary['dialogue_count']} 轮对话，场景为 {scene}。"
            if overall >= 80:
                summary += "表现优秀，继续保持！"
            elif overall >= 60:
                summary += "表现良好，还有提升空间。"
            else:
                summary += "建议多加练习，关注语法和发音的准确性。"
        else:
            summary = "本次会话对话较少，建议增加练习量以获取更全面的评估。"

        strengths, weaknesses = self._default_strengths_weaknesses(session_summary)

        grammar_summary = self._default_grammar_summary(session_summary)
        pron_summary = self._default_pron_summary(session_summary)

        return {
            "session_id": session_summary.get("session_id", ""),
            "scene_name": scene,
            "duration_seconds": session_summary.get("duration_seconds", 0),
            "dialogue_count": session_summary.get("dialogue_count", 0),
            "overall_score": overall,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "grammar": {
                "total_errors": session_summary.get("total_grammar_errors", 0),
                "errors_by_type": session_summary.get("grammar_errors_by_type", {}),
                "checks": session_summary.get("grammar_checks", 0),
                "summary": grammar_summary,
            },
            "pronunciation": {
                "avg_score": session_summary.get("pronunciation_avg_score", 0),
                "avg_fluency": session_summary.get("pronunciation_avg_fluency", 0),
                "avg_accuracy": session_summary.get("pronunciation_avg_accuracy", 0),
                "checks": session_summary.get("pronunciation_checks", 0),
                "summary": pron_summary,
            },
            "vocabulary_tip": "可以尝试在对话中使用更多样化的词汇，例如用 'excellent' 替代 'good'。",
            "practice_suggestions": [
                "每次练习前先浏览目标场景的常用表达",
                "关注每次发音评测中标红的单词，重点练习",
                "尝试用更长的句子表达观点，提升流利度",
            ],
            "detailed_errors": session_summary.get("grammar_records", []),
            "generated_by": "local",
        }

    # ------------------------------------------------------------------ #
    #  内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_ai_input(session_summary: dict) -> str:
        """构建发送给 AI 的精简会话数据摘要。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            JSON 格式的会话摘要字符串。
        """
        # 只传递 AI 需要的核心统计
        grammar_errors = []
        for rec in session_summary.get("grammar_records", []):
            for err in rec.get("errors", []):
                grammar_errors.append({
                    "type": err.get("type", ""),
                    "original": err.get("original", ""),
                    "suggestion": err.get("suggestion", ""),
                })

        input_data = {
            "scene": session_summary.get("scene_name", ""),
            "dialogue_count": session_summary.get("dialogue_count", 0),
            "duration_minutes": round(session_summary.get("duration_seconds", 0) / 60, 1),
            "grammar": {
                "checks": session_summary.get("grammar_checks", 0),
                "total_errors": session_summary.get("total_grammar_errors", 0),
                "errors_by_type": session_summary.get("grammar_errors_by_type", {}),
                "sample_errors": grammar_errors[:10],
            },
            "pronunciation": {
                "checks": session_summary.get("pronunciation_checks", 0),
                "avg_score": session_summary.get("pronunciation_avg_score", 0),
                "avg_fluency": session_summary.get("pronunciation_avg_fluency", 0),
                "avg_accuracy": session_summary.get("pronunciation_avg_accuracy", 0),
            },
        }
        return json.dumps(input_data, ensure_ascii=False)

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """解析 AI 返回的 JSON（兼容 markdown 包裹）。

        Args:
            raw: API 返回的原始文本。

        Returns:
            解析后的 dict，失败时返回空 dict。
        """
        # 去除可能的 markdown 包裹
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        # 正则提取
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("ReportGenerator: 无法解析 AI 返回的 JSON: %s", raw[:200])
        return {}

    @staticmethod
    def _calc_grammar_score(session_summary: dict) -> int:
        """根据语法错误数量计算语法评分。

        无错误 → 100，每增加一个错误扣 5 分，最低 20。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            语法评分 (0-100)。
        """
        total_errors = session_summary.get("total_grammar_errors", 0)
        if total_errors == 0:
            return 100
        score = 100 - total_errors * 5
        return max(20, min(100, score))

    @staticmethod
    def _merge_results(
        session_summary: dict,
        ai_result: dict,
        generated_by: str = "ai",
    ) -> dict:
        """将 AI 分析结果与会话统计数据合并为完整报告。

        Args:
            session_summary: 会话汇总数据。
            ai_result: AI 返回的分析结果 dict。
            generated_by: 标记报告生成方式。

        Returns:
            合并后的完整报告 dict。
        """
        return {
            "session_id": session_summary.get("session_id", ""),
            "scene_name": session_summary.get("scene_name", ""),
            "duration_seconds": session_summary.get("duration_seconds", 0),
            "dialogue_count": session_summary.get("dialogue_count", 0),
            "overall_score": ai_result.get("overall_score", 0),
            "summary": ai_result.get("summary", ""),
            "strengths": ai_result.get("strengths", []),
            "weaknesses": ai_result.get("weaknesses", []),
            "grammar": {
                "total_errors": session_summary.get("total_grammar_errors", 0),
                "errors_by_type": session_summary.get("grammar_errors_by_type", {}),
                "checks": session_summary.get("grammar_checks", 0),
                "summary": ai_result.get("grammar_summary", ""),
            },
            "pronunciation": {
                "avg_score": session_summary.get("pronunciation_avg_score", 0),
                "avg_fluency": session_summary.get("pronunciation_avg_fluency", 0),
                "avg_accuracy": session_summary.get("pronunciation_avg_accuracy", 0),
                "checks": session_summary.get("pronunciation_checks", 0),
                "summary": ai_result.get("pronunciation_summary", ""),
            },
            "vocabulary_tip": ai_result.get("vocabulary_tip", ""),
            "practice_suggestions": ai_result.get("practice_suggestions", []),
            "detailed_errors": session_summary.get("grammar_records", []),
            "generated_by": generated_by,
        }

    @staticmethod
    def _default_strengths_weaknesses(session_summary: dict) -> tuple:
        """根据统计数据生成默认的优缺点。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            (strengths: list, weaknesses: list)
        """
        strengths = []
        weaknesses = []

        pron_score = session_summary.get("pronunciation_avg_score", 0)
        if pron_score >= 80:
            strengths.append("发音表现优秀，准确度高")
        elif pron_score > 0:
            weaknesses.append("发音准确度有提升空间")

        grammar_errors = session_summary.get("total_grammar_errors", 0)
        if grammar_errors <= 2 and session_summary.get("grammar_checks", 0) > 0:
            strengths.append("语法基本功扎实，错误较少")
        elif grammar_errors > 0:
            weaknesses.append(f"存在 {grammar_errors} 处语法错误，建议针对性练习")

        dialogue_count = session_summary.get("dialogue_count", 0)
        if dialogue_count >= 5:
            strengths.append("对话参与度高，练习量充足")
        elif dialogue_count > 0:
            weaknesses.append("对话轮次较少，建议增加练习时长")

        if not strengths:
            strengths.append("开始了英语口语练习，迈出重要一步")
        if not weaknesses:
            weaknesses.append("可以尝试更多不同场景的对话练习")

        return strengths, weaknesses

    @staticmethod
    def _default_grammar_summary(session_summary: dict) -> str:
        """生成默认语法总结文本。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            语法总结字符串。
        """
        total = session_summary.get("total_grammar_errors", 0)
        checks = session_summary.get("grammar_checks", 0)
        if checks == 0:
            return "本次会话未进行语法检查。"
        if total == 0:
            return f"共进行 {checks} 次语法检查，未发现错误，表现优秀！"
        by_type = session_summary.get("grammar_errors_by_type", {})
        parts = [f"共进行 {checks} 次语法检查，发现 {total} 处错误"]
        if by_type.get("grammar", 0) > 0:
            parts.append(f"语法错误 {by_type['grammar']} 处")
        if by_type.get("vocabulary", 0) > 0:
            parts.append(f"词汇错误 {by_type['vocabulary']} 处")
        if by_type.get("style", 0) > 0:
            parts.append(f"风格问题 {by_type['style']} 处")
        return "，".join(parts) + "。"

    @staticmethod
    def _default_pron_summary(session_summary: dict) -> str:
        """生成默认发音总结文本。

        Args:
            session_summary: 会话汇总数据。

        Returns:
            发音总结字符串。
        """
        checks = session_summary.get("pronunciation_checks", 0)
        if checks == 0:
            return "本次会话未进行发音评测。"
        avg = session_summary.get("pronunciation_avg_score", 0)
        avg_acc = session_summary.get("pronunciation_avg_accuracy", 0)
        avg_flu = session_summary.get("pronunciation_avg_fluency", 0)
        return (
            f"共进行 {checks} 次发音评测，平均分 {avg}/100。"
            f"准确度 {avg_acc}/100，流利度 {avg_flu}/100。"
        )
