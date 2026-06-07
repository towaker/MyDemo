"""
GrammarChecker 单元测试 — mock openai 响应，测试纠错结果的
结构正确性与异常兜底。
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 确保项目根在 sys.path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.grammar_checker import GrammarChecker


# ------------------------------------------------------------------ #
#  辅助工具
# ------------------------------------------------------------------ #

def _mock_chat_response(content: str) -> MagicMock:
    """构造一个模拟的 openai chat completion 响应对象。"""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message = MagicMock()
    mock.choices[0].message.content = content
    return mock


# ------------------------------------------------------------------ #
#  check() 正常流程
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_check_with_grammar_error():
    """有语法错误时返回完整的错误列表与纠错后文本。"""
    fake_json = json.dumps({
        "original": "He go to school yesterday",
        "corrected": "He went to school yesterday",
        "errors": [
            {
                "type": "grammar",
                "original": "go",
                "suggestion": "went",
                "explanation": "应使用过去时 went 而非 go",
            }
        ],
    })

    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(fake_json)
        )
        checker = GrammarChecker()
        result = await checker.check("He go to school yesterday")

    assert result["original"] == "He go to school yesterday"
    assert result["corrected"] == "He went to school yesterday"
    assert len(result["errors"]) == 1
    assert result["errors"][0]["type"] == "grammar"
    assert result["errors"][0]["suggestion"] == "went"


@pytest.mark.asyncio
async def test_check_no_error():
    """无语法错误时 errors 为空，corrected 等于 original。"""
    fake_json = json.dumps({
        "original": "I like reading books",
        "corrected": "I like reading books",
        "errors": [],
    })

    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(fake_json)
        )
        checker = GrammarChecker()
        result = await checker.check("I like reading books")

    assert result["original"] == "I like reading books"
    assert result["corrected"] == "I like reading books"
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_check_multiple_errors():
    """多个不同类型错误全部返回。"""
    fake_json = json.dumps({
        "original": "She don't have any ideas about that",
        "corrected": "She doesn't have any ideas about that",
        "errors": [
            {
                "type": "grammar",
                "original": "don't",
                "suggestion": "doesn't",
                "explanation": "主语是第三人称单数，应使用 doesn't",
            },
            {
                "type": "vocabulary",
                "original": "ideas about",
                "suggestion": "ideas regarding",
                "explanation": "about 可替换为更正式的 regarding",
            },
        ],
    })

    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(fake_json)
        )
        checker = GrammarChecker()
        result = await checker.check("She don't have any ideas about that")

    assert len(result["errors"]) == 2
    assert result["errors"][0]["type"] == "grammar"
    assert result["errors"][1]["type"] == "vocabulary"


# ------------------------------------------------------------------ #
#  异常兜底
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_check_api_exception():
    """API 调用抛出异常时返回空 errors，不传播异常。"""
    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("Connection timeout")
        )
        checker = GrammarChecker()
        result = await checker.check("Some sentence")

    assert result["original"] == "Some sentence"
    assert result["corrected"] == "Some sentence"
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_check_empty_input():
    """空文本输入直接返回空结果，不调用 API。"""
    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock()
        checker = GrammarChecker()
        result = await checker.check("")

    assert result["original"] == ""
    assert result["corrected"] == ""
    assert result["errors"] == []
    # 确认没有调用 API
    MockClient.return_value.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_check_whitespace_only():
    """仅空白字符输入直接返回空结果。"""
    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        checker = GrammarChecker()
        result = await checker.check("   \n  ")

    assert result["original"] == "   \n  "
    assert result["corrected"] == "   \n  "
    assert result["errors"] == []


# ------------------------------------------------------------------ #
#  _parse_response 鲁棒性
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_parse_markdown_code_fence_response():
    """API 返回带 ```json 包裹的响应也能正确解析。"""
    raw = '```json\n{"original": "I is", "corrected": "I am", "errors": [{"type": "grammar", "original": "is", "suggestion": "am", "explanation": ""}]}\n```'

    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(raw)
        )
        checker = GrammarChecker()
        result = await checker.check("I is")

    assert result["corrected"] == "I am"
    assert len(result["errors"]) == 1


@pytest.mark.asyncio
async def test_parse_malformed_json():
    """API 返回不可解析的文本时回退到兜底结果。"""
    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response("Sorry, I cannot parse this.")
        )
        checker = GrammarChecker()
        result = await checker.check("Hello")

    assert result["original"] == "Hello"
    assert result["corrected"] == "Hello"
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_parse_empty_response():
    """API 返回空内容时回退到兜底结果。"""
    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response("")
        )
        checker = GrammarChecker()
        result = await checker.check("Hello")

    assert result["original"] == "Hello"
    assert result["corrected"] == "Hello"
    assert result["errors"] == []


# ------------------------------------------------------------------ #
#  字段缺失容错
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_validate_result_missing_fields():
    """API 返回的 JSON 缺少字段时使用默认值填充。"""
    fake_json = json.dumps({"original": "Test"})  # 缺少 corrected/errors

    with patch("evaluation.grammar_checker.AsyncOpenAI") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=_mock_chat_response(fake_json)
        )
        checker = GrammarChecker()
        result = await checker.check("Test")

    assert result["original"] == "Test"
    assert result["corrected"] == "Test"  # 回退为原句
    assert result["errors"] == []
