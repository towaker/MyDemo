"""
评测数据追踪器 — 在对话过程中收集语法纠错和发音评测记录。

SessionTracker 在 MessageHandler 中作为单例使用，每次对话开始
时自动创建新会话，记录评测结果供课后报告使用。
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GrammarRecord:
    """单次语法纠错记录。"""
    timestamp: float = field(default_factory=time.time)
    scene: str = ""
    user_text: str = ""
    corrected_text: str = ""
    error_count: int = 0
    errors: list = field(default_factory=list)  # list[dict] 语法错误详情


@dataclass
class PronunciationRecord:
    """单次发音评测记录。"""
    timestamp: float = field(default_factory=time.time)
    scene: str = ""
    reference_text: str = ""
    overall_score: int = 0
    fluency: int = 0
    accuracy: int = 0
    words: list = field(default_factory=list)


@dataclass
class DialogueTurn:
    """单轮对话记录。"""
    timestamp: float = field(default_factory=time.time)
    scene: str = ""
    user_text: str = ""
    ai_text: str = ""


class SessionTracker:
    """会话级别评测数据追踪器。

    追踪一个完整练习会话中的所有评测记录和对话轮次，
    提供生成课后报告所需的原始数据。

    用法:
        tracker = SessionTracker()
        tracker.record_dialogue("free_talk", "Hello", "Hi there!")
        tracker.record_grammar(scene, original, corrected, errors)
        tracker.record_pronunciation(scene, ref_text, result_dict)
        report_data = tracker.get_session_summary()
    """

    def __init__(self, session_id: Optional[str] = None):
        """初始化会话追踪器。

        Args:
            session_id: 会话标识，默认使用时间戳生成。
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.scene_name: str = "free_talk"

        # 记录集合
        self.dialogue_turns: list[DialogueTurn] = []
        self.grammar_records: list[GrammarRecord] = []
        self.pronunciation_records: list[PronunciationRecord] = []

        logger.info("SessionTracker 已创建, session_id=%s", self.session_id)

    # ------------------------------------------------------------------ #
    #  记录方法
    # ------------------------------------------------------------------ #

    def set_scene(self, scene_name: str) -> None:
        """设置当前场景名。

        Args:
            scene_name: 场景名称。
        """
        self.scene_name = scene_name

    def record_dialogue(self, user_text: str, ai_text: str) -> None:
        """记录一轮对话。

        Args:
            user_text: 用户输入文本。
            ai_text: AI 回复文本。
        """
        turn = DialogueTurn(
            scene=self.scene_name,
            user_text=user_text,
            ai_text=ai_text,
        )
        self.dialogue_turns.append(turn)

    def record_grammar(
        self,
        user_text: str,
        corrected_text: str = "",
        errors: Optional[list] = None,
    ) -> None:
        """记录一次语法纠错结果。

        Args:
            user_text: 用户原文。
            corrected_text: 纠错后文本。
            errors: 语法错误列表 [{"type": ..., "original": ..., ...}, ...]。
        """
        error_list = list(errors) if errors else []
        record = GrammarRecord(
            scene=self.scene_name,
            user_text=user_text,
            corrected_text=corrected_text or user_text,
            error_count=len(error_list),
            errors=error_list,
        )
        self.grammar_records.append(record)

    def record_pronunciation(
        self,
        reference_text: str,
        result: dict,
    ) -> None:
        """记录一次发音评测结果。

        Args:
            reference_text: 参考文本。
            result: 发音评测结果 dict，包含 overall_score/fluency/accuracy/words。
        """
        record = PronunciationRecord(
            scene=self.scene_name,
            reference_text=reference_text,
            overall_score=result.get("overall_score", 0),
            fluency=result.get("fluency", 0),
            accuracy=result.get("accuracy", 0),
            words=list(result.get("words", [])),
        )
        self.pronunciation_records.append(record)

    # ------------------------------------------------------------------ #
    #  统计与汇总
    # ------------------------------------------------------------------ #

    def get_session_summary(self) -> dict:
        """获取会话级别的汇总数据，供报告生成器使用。

        Returns:
            dict: {
                "session_id": str,
                "scene_name": str,
                "duration_seconds": float,
                "dialogue_count": int,
                "grammar_checks": int,
                "total_grammar_errors": int,
                "grammar_errors_by_type": {"grammar": n, "vocabulary": n, "style": n},
                "pronunciation_checks": int,
                "pronunciation_avg_score": float,
                "pronunciation_avg_fluency": float,
                "pronunciation_avg_accuracy": float,
                "grammar_records": list,
                "pronunciation_records": list,
                "dialogue_turns": list,
            }
        """
        duration = time.time() - self.start_time

        # 语法汇总
        total_errors = 0
        errors_by_type = {"grammar": 0, "vocabulary": 0, "style": 0}
        for rec in self.grammar_records:
            total_errors += rec.error_count
            for err in rec.errors:
                err_type = err.get("type", "grammar")
                if err_type in errors_by_type:
                    errors_by_type[err_type] += 1

        # 发音汇总
        pron_scores = [r.overall_score for r in self.pronunciation_records]
        pron_fluency = [r.fluency for r in self.pronunciation_records]
        pron_accuracy = [r.accuracy for r in self.pronunciation_records]

        return {
            "session_id": self.session_id,
            "scene_name": self.scene_name,
            "duration_seconds": round(duration, 1),
            "dialogue_count": len(self.dialogue_turns),
            "grammar_checks": len(self.grammar_records),
            "total_grammar_errors": total_errors,
            "grammar_errors_by_type": errors_by_type,
            "pronunciation_checks": len(self.pronunciation_records),
            "pronunciation_avg_score": round(_safe_avg(pron_scores), 1),
            "pronunciation_avg_fluency": round(_safe_avg(pron_fluency), 1),
            "pronunciation_avg_accuracy": round(_safe_avg(pron_accuracy), 1),
            "grammar_records": [
                {
                    "scene": r.scene,
                    "user_text": r.user_text,
                    "corrected_text": r.corrected_text,
                    "error_count": r.error_count,
                    "errors": r.errors,
                }
                for r in self.grammar_records
            ],
            "pronunciation_records": [
                {
                    "scene": r.scene,
                    "reference_text": r.reference_text,
                    "overall_score": r.overall_score,
                    "fluency": r.fluency,
                    "accuracy": r.accuracy,
                    "words": r.words,
                }
                for r in self.pronunciation_records
            ],
            "dialogue_turns": [
                {
                    "scene": t.scene,
                    "user_text": t.user_text,
                    "ai_text": t.ai_text,
                }
                for t in self.dialogue_turns
            ],
        }

    def reset(self) -> None:
        """重置当前会话追踪数据，开始新会话。"""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.dialogue_turns.clear()
        self.grammar_records.clear()
        self.pronunciation_records.clear()
        logger.info("SessionTracker 已重置, 新 session_id=%s", self.session_id)


def _safe_avg(values: list) -> float:
    """安全计算平均值，空列表返回 0.0。"""
    if not values:
        return 0.0
    return sum(values) / len(values)
