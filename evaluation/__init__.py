"""Evaluation 评测与纠错模块。

包含：
    - GrammarChecker: 基于 DeepSeek API 的语法纠错
    - PronunciationEvaluator: 基于 Whisper 转写对比的发音评测
    - SessionTracker: 会话评测数据追踪
    - ReportGenerator: 课后报告生成
"""

from .grammar_checker import GrammarChecker
from .pronunciation import PronunciationEvaluator
from .tracker import SessionTracker
from .report import ReportGenerator

__all__ = [
    "GrammarChecker",
    "PronunciationEvaluator",
    "SessionTracker",
    "ReportGenerator",
]