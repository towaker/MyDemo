"""
PronunciationEvaluator 单元测试 — mock WhisperRecognizer 转写，
测试评分逻辑的正确性与边界情况处理。
"""

import asyncio
import os
import tempfile
import wave

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.pronunciation import PronunciationEvaluator


# ------------------------------------------------------------------ #
#  辅助工具
# ------------------------------------------------------------------ #

def _create_wav_file(text: str = "placeholder") -> str:
    """创建一个有效的 WAV 文件供测试使用。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 32000)  # 1 秒静音
    return tmp.name


# ------------------------------------------------------------------ #
#  evaluate() — 正常评分逻辑
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_evaluate_perfect_match():
    """用户发音与参考文本完全一致时获得高分。"""
    wav = _create_wav_file()

    with patch("asyncio.to_thread", AsyncMock(return_value="Hello world how are you")):
        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello world how are you", wav)

    os.unlink(wav)

    assert result["overall_score"] == 100
    assert result["fluency"] == 100
    assert result["accuracy"] == 100
    assert len(result["words"]) == 5
    for w in result["words"]:
        assert w["score"] == 100


@pytest.mark.asyncio
async def test_evaluate_partial_match():
    """部分词匹配时给出中等评分。"""
    wav = _create_wav_file()

    with patch("asyncio.to_thread", AsyncMock(return_value="Hello world wrong")):
        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello world right", wav)

    os.unlink(wav)

    assert 50 < result["overall_score"] < 90
    assert 3 <= len(result["words"]) <= 4


@pytest.mark.asyncio
async def test_evaluate_no_match():
    """转写文本与参考文本完全不同时获得低分。"""
    wav = _create_wav_file()

    with patch("asyncio.to_thread", AsyncMock(return_value="completely different words here")):
        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello world", wav)

    os.unlink(wav)

    assert result["overall_score"] < 40
    assert result["fluency"] < 50


# ------------------------------------------------------------------ #
#  边界情况
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_evaluate_empty_reference():
    """参考文本为空字符串时返回全零结果。"""
    evaluator = PronunciationEvaluator()
    result = await evaluator.evaluate("", "/nonexistent.wav")

    assert result["overall_score"] == 0
    assert result["fluency"] == 0
    assert result["accuracy"] == 0
    assert result["words"] == []


@pytest.mark.asyncio
async def test_evaluate_none_reference():
    """参考文本为 None 的等价情况（空白）。"""
    evaluator = PronunciationEvaluator()
    result = await evaluator.evaluate("   ", "/nonexistent.wav")

    assert result["overall_score"] == 0


@pytest.mark.asyncio
async def test_evaluate_missing_audio_file():
    """音频文件不存在时返回全零结果。"""
    evaluator = PronunciationEvaluator()
    result = await evaluator.evaluate("Hello world", "C:\\nonexistent\\file.wav")

    assert result["overall_score"] == 0
    assert result["fluency"] == 0
    assert result["accuracy"] == 0


@pytest.mark.asyncio
async def test_evaluate_empty_audio_path():
    """audio_path 为空字符串时返回全零结果。"""
    evaluator = PronunciationEvaluator()
    result = await evaluator.evaluate("Hello", "")

    assert result["overall_score"] == 0


@pytest.mark.asyncio
async def test_evaluate_empty_transcription():
    """转写返回空文本（静音）时返回全零结果。"""
    wav = _create_wav_file()

    with patch("asyncio.to_thread", AsyncMock(return_value="")):
        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello world", wav)

    os.unlink(wav)

    assert result["overall_score"] == 0
    assert result["words"] == []


@pytest.mark.asyncio
async def test_evaluate_transcription_exception():
    """转写抛出异常时返回全零结果，不传播异常。"""
    wav = _create_wav_file()

    with patch("asyncio.to_thread", AsyncMock(side_effect=RuntimeError("Model load failed"))):
        evaluator = PronunciationEvaluator()
        result = await evaluator.evaluate("Hello", wav)

    os.unlink(wav)

    assert result["overall_score"] == 0
    assert result["fluency"] == 0


# ------------------------------------------------------------------ #
#  WER 计算逻辑
# ------------------------------------------------------------------ #

def test_calculate_wer_identical():
    """相同文本 WER 为 0。"""
    assert PronunciationEvaluator._calculate_wer(
        ["hello", "world"], ["hello", "world"]
    ) == 0.0


def test_calculate_wer_insertion():
    """多插入一个词导致 WER > 0。"""
    wer = PronunciationEvaluator._calculate_wer(
        ["hello"], ["hello", "there"]
    )
    assert wer == 1.0  # 1 次插入 / 1 个参考词


def test_calculate_wer_deletion():
    """漏读一个词（2个参考词→1个转写词），WER=0.5。"""
    wer = PronunciationEvaluator._calculate_wer(
        ["hello", "world"], ["hello"]
    )
    assert wer == 0.5


def test_calculate_wer_empty_reference():
    """参考文本为空时 WER 为 0（len(hypothesis)）。"""
    wer = PronunciationEvaluator._calculate_wer([], ["hello"])
    assert wer == 1.0


def test_calculate_wer_both_empty():
    """两端均为空时 WER 为 0。"""
    wer = PronunciationEvaluator._calculate_wer([], [])
    assert wer == 0.0


# ------------------------------------------------------------------ #
#  文本归一化
# ------------------------------------------------------------------ #

def test_normalize_text():
    """归一化去除标点和多余空格，统一小写。"""
    result = PronunciationEvaluator._normalize_text("Hello, World!   How are you?")
    assert result == "hello world how are you"


def test_normalize_text_empty():
    """空文本归一化返回空字符串。"""
    assert PronunciationEvaluator._normalize_text("") == ""
