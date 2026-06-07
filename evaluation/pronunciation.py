"""
发音评测模块 — 评估用户发音准确度。

PronunciationEvaluator 通过 Whisper 转写用户音频，将转写文本与
参考文本进行对比（词错率 + 文本相似度），综合给出评分。

设计上保留接口扩展空间，未来可接入专业发音评测 API
（如 Azure Cognitive Services Speech — PronunciationAssessment）。
"""

import logging
import os
from difflib import SequenceMatcher
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class PronunciationEvaluator:
    """发音评测器，对比用户朗读音频的转写文本与参考文本给出评分。

    评分策略：
        - 用 WhisperRecognizer 转写音频 → 得到用户实际发音文本
        - 计算词错率（Word Error Rate, WER）
        - 结合 SequenceMatcher 文本相似度
        - 综合映射为 0-100 的 overall_score

    用法::

        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello world", "/tmp/audio.wav")
        # {"overall_score": 85, "fluency": 88, "accuracy": 82, "words": [...]}

    扩展接口：
        子类可覆盖 _call_pronunciation_api() 方法接入专业评测 API，
        返回统一格式的 dict 即可。
    """

    def __init__(self):
        """初始化发音评测器，创建 Whisper 识别器实例。"""
        self._recognizer = None  # 延迟导入，避免循环依赖
        self._threshold: int = getattr(config, "PRONUNCIATION_THRESHOLD", 60)

    @property
    def recognizer(self):
        """延迟加载 WhisperRecognizer 实例。"""
        if self._recognizer is None:
            # 延迟导入，上层 web/handler 会组合使用，避免循环依赖
            from asr.recognizer import WhisperRecognizer
            self._recognizer = WhisperRecognizer()
        return self._recognizer

    async def evaluate(
        self,
        reference_text: str,
        audio_path: str,
        *,
        use_professional_api: bool = False,
    ) -> dict:
        """评估用户对 reference_text 的发音准确度。

        Args:
            reference_text: 用户应当朗读的参考文本。
            audio_path: 用户朗读录音的 WAV 文件路径（16kHz 单声道）。
            use_professional_api: 是否尝试调用专业发音评测 API（当前未实现）。

        Returns:
            dict: {
                "overall_score": int (0-100),
                "fluency": int (0-100),
                "accuracy": int (0-100),
                "words": [
                    {
                        "word": "hello",
                        "score": int (0-100),
                        "phoneme_errors": list[str]
                    },
                    ...
                ]
            }

        Raises:
            不会抛出异常：音频文件不存在等问题均以 score=0 形式返回。
        """
        # 参数校验
        if not reference_text or not reference_text.strip():
            logger.warning("参考文本为空")
            return self._empty_result()

        if not audio_path or not os.path.exists(audio_path):
            logger.warning("音频文件不存在: %s", audio_path)
            return self._empty_result()

        # 尝试专业 API（子类/未来扩展）
        if use_professional_api:
            try:
                result = await self._call_pronunciation_api(reference_text, audio_path)
                if result:
                    return result
            except Exception as exc:
                logger.warning("专业发音评测 API 不可用，回退到本地评测: %s", exc)

        # 本地评测路径：Whisper 转写 + 文本对比
        return await self._evaluate_local(reference_text, audio_path)

    async def _evaluate_local(self, reference_text: str, audio_path: str) -> dict:
        """本地评测：转写 + 文本相似度。

        Args:
            reference_text: 参考文本。
            audio_path: 音频文件路径。

        Returns:
            评测结果 dict。
        """
        import asyncio

        try:
            # 用 Whisper 转写用户音频
            user_text = await asyncio.to_thread(
                self.recognizer.transcribe, audio_path
            )
            user_text = user_text.strip()
        except Exception as exc:
            logger.warning("音频转写失败: %s", exc)
            return self._empty_result()

        if not user_text:
            logger.info("转写结果为空，可能为静音或音频质量问题")
            return self._empty_result()

        # 文本归一化：转小写、去除多余空格和标点差异
        ref_norm = self._normalize_text(reference_text)
        user_norm = self._normalize_text(user_text)

        # 词错率 (WER) 计算
        ref_words = ref_norm.split()
        user_words = user_norm.split()
        wer = self._calculate_wer(ref_words, user_words)

        # 文本相似度（基于字符序列）
        similarity = SequenceMatcher(None, ref_norm, user_norm).ratio()

        # 综合评分
        # WER 越低越好 → accuracy，similarity 越高越好
        accuracy = max(0, int((1.0 - wer) * 100))
        fluency = max(0, int(similarity * 100))
        overall = int(accuracy * 0.6 + fluency * 0.4)

        # 逐词评分（简化版：对比词级别的匹配）
        words_detail = self._build_word_details(ref_words, user_words)

        return {
            "overall_score": overall,
            "fluency": fluency,
            "accuracy": accuracy,
            "words": words_detail,
        }

    async def _call_pronunciation_api(
        self, _reference_text: str, _audio_path: str
    ) -> Optional[dict]:
        """预留接口：调用专业发音评测 API。

        子类或未来版本覆盖此方法，接入如 Azure PronunciationAssessmentConfig。
        当前默认返回 None，触发本地评测回退。

        Returns:
            评测结果 dict 或 None（回退本地评测）。
        """
        return None

    # ------------------------------------------------------------------ #
    #  静态工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_text(text: str) -> str:
        """文本归一化：转小写、去除多余标点和空格。"""
        import re
        text = text.lower().strip()
        # 移除标点符号（保留字母数字和空格）
        text = re.sub(r'[^\w\s]', '', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def _calculate_wer(reference_words: list[str], hypothesis_words: list[str]) -> float:
        """计算词错率 (Word Error Rate)。

        使用 Levenshtein 距离计算最小编辑操作数（替换/插入/删除），
        WER = 编辑距离 / 参考词数。

        Args:
            reference_words: 参考文本分词列表。
            hypothesis_words: 转写文本分词列表。

        Returns:
            WER 值，范围 [0.0, ∞)，0 表示完全匹配。
        """
        ref_len = len(reference_words)
        if ref_len == 0:
            return float(len(hypothesis_words)) if hypothesis_words else 0.0

        # Levenshtein 距离 DP
        dp = [[0] * (len(hypothesis_words) + 1) for _ in range(ref_len + 1)]
        for i in range(ref_len + 1):
            dp[i][0] = i
        for j in range(len(hypothesis_words) + 1):
            dp[0][j] = j

        for i in range(1, ref_len + 1):
            for j in range(1, len(hypothesis_words) + 1):
                cost = 0 if reference_words[i - 1] == hypothesis_words[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + cost,  # 替换
                )

        edit_distance = dp[ref_len][len(hypothesis_words)]
        return edit_distance / ref_len

    @staticmethod
    def _build_word_details(
        ref_words: list[str], user_words: list[str]
    ) -> list[dict]:
        """构建逐词评分详情。

        通过对齐参考词与用户词，为每个参考词给出评分。
        使用简单的序列对齐策略。

        Args:
            ref_words: 参考词列表。
            user_words: 用户词列表。

        Returns:
            [{"word": "...", "score": int, "phoneme_errors": []}, ...]
        """
        details = []
        # 简单对齐：按位置匹配，多余的忽略或标记
        for i, ref_word in enumerate(ref_words):
            if i < len(user_words) and ref_word == user_words[i]:
                score = 100
                phoneme_errors = []
            elif i < len(user_words):
                # 词不匹配，给低分
                score = 50
                phoneme_errors = []
            else:
                # 用户漏读
                score = 0
                phoneme_errors = []
            details.append({
                "word": ref_word,
                "score": score,
                "phoneme_errors": phoneme_errors,
            })

        return details

    @staticmethod
    def _empty_result() -> dict:
        """返回空评测结果（评分均为 0）。"""
        return {
            "overall_score": 0,
            "fluency": 0,
            "accuracy": 0,
            "words": [],
        }
