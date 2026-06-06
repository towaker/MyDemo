"""CoachEngine 纯逻辑单元测试 — 不涉及真实 API 调用。

仅测试：对话历史管理、消息构建（Prompt 注入逻辑）。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coach.engine import CoachEngine
from coach.scene_manager import Scene, SceneManager


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_scene_mgr():
    """返回一个带有预设场景的 SceneManager mock。"""
    mgr = MagicMock(spec=SceneManager)
    mgr.get_system_prompt.return_value = "You are an interviewer."
    mgr.get_first_message.return_value = "Hello!"
    return mgr


@pytest.fixture
def engine(mock_scene_mgr):
    """返回一个已初始化的 CoachEngine。"""
    with patch("coach.engine.AsyncOpenAI"):
        return CoachEngine(mock_scene_mgr)


# ------------------------------------------------------------------ #
#  History management
# ------------------------------------------------------------------ #

class TestHistoryManagement:
    """对话历史管理测试。"""

    def test_initial_history_is_empty(self, engine):
        assert engine.get_history() == []

    def test_append_user(self, engine):
        engine.append_user("Hello")
        history = engine.get_history()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}

    def test_append_assistant(self, engine):
        engine.append_assistant("Hi there")
        history = engine.get_history()
        assert len(history) == 1
        assert history[0] == {"role": "assistant", "content": "Hi there"}

    def test_append_system(self, engine):
        engine.append_system("System prompt")
        history = engine.get_history()
        assert history[0] == {"role": "system", "content": "System prompt"}

    def test_reset_history(self, engine):
        engine.append_user("msg1")
        engine.append_assistant("msg2")
        assert len(engine.get_history()) == 2
        engine.reset_history()
        assert engine.get_history() == []

    def test_history_is_independent_copy(self, engine):
        engine.append_user("original")
        hist = engine.get_history()
        hist.append({"role": "assistant", "content": "extra"})
        # 不影响原始历史
        assert len(engine.get_history()) == 1


# ------------------------------------------------------------------ #
#  Message building
# ------------------------------------------------------------------ #

class TestMessageBuilding:
    """消息构建测试。"""

    def test_build_injects_system_prompt_when_empty(self, engine, mock_scene_mgr):
        """history 为空时，自动注入 system prompt。"""
        msgs = engine.build_messages("Hi", scene_name="interview")

        # system prompt 应被注入到 history
        assert engine.get_history()[0]["role"] == "system"

        # 返回的消息列表包含 system + user
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are an interviewer."
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Hi"

    def test_build_does_not_duplicate_system(self, engine, mock_scene_mgr):
        """已有 system 消息时不重复注入。"""
        engine.append_system("Existing system")
        msgs = engine.build_messages("Hi again")
        # 应该只有 1 条 system
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_build_preserves_history_order(self, engine, mock_scene_mgr):
        """多次对话后消息顺序正确：system → user → assistant → user → ..."""
        engine.append_user("Q1")
        engine.append_assistant("A1")
        engine.append_user("Q2")
        engine.append_assistant("A2")

        msgs = engine.build_messages("Q3")

        expected_roles = ["system", "user", "assistant", "user", "assistant", "user"]
        assert [m["role"] for m in msgs] == expected_roles
        assert msgs[-1]["content"] == "Q3"

    def test_build_does_not_mutate_history_with_incoming_message(self, engine, mock_scene_mgr):
        """build_messages 不应将 user_text 写入 history（由 generate_* 负责）。"""
        msgs_before = engine.get_history()
        engine.build_messages("test message")
        msgs_after = engine.get_history()
        # user_text 不写入，只有 system prompt 可能被注入
        non_system_before = [m for m in msgs_before if m["role"] != "system"]
        non_system_after = [m for m in msgs_after if m["role"] != "system"]
        assert non_system_after == non_system_before
