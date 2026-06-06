"""
MessageHandler 单元测试。

测试 MessageHandler 的初始化、场景管理、对话历史等功能。
使用 mock 对象替代真实 Engine / Recognizer / Synthesizer，避免依赖外部服务。
"""

import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def mock_scene_manager():
    """返回 mock SceneManager，包含 5 个场景。"""
    mgr = MagicMock()
    mgr.list_scenes.return_value = [
        "free_talk", "interview", "meeting", "ordering", "travel"
    ]
    mgr.switch_scene.return_value = MagicMock()
    mgr.get_system_prompt.return_value = "You are a helpful assistant."
    mgr.get_first_message.return_value = "Hello!"
    return mgr


@pytest.fixture
def handler_with_mocks(mock_scene_manager):
    """创建 MessageHandler，用 mock 替换各外部模块。"""
    with patch("web.handler.SceneManager", return_value=mock_scene_manager), \
         patch("web.handler.CoachEngine") as mock_engine_cls, \
         patch("web.handler.WhisperRecognizer") as mock_recognizer_cls, \
         patch("web.handler.Synthesizer") as mock_synthesizer_cls, \
         patch("web.handler.AudioCapture") as mock_capture_cls:

        mock_engine = MagicMock()
        mock_engine.generate_reply = AsyncMock(return_value="Mock AI reply.")
        mock_engine.get_history.return_value = []
        mock_engine_cls.return_value = mock_engine

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = "Hello, how are you?"
        mock_recognizer_cls.return_value = mock_recognizer

        mock_synthesizer = MagicMock()
        mock_synthesizer.synthesize = AsyncMock(return_value="/tmp/output.mp3")
        mock_synthesizer_cls.return_value = mock_synthesizer

        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture

        from web.handler import MessageHandler
        handler = MessageHandler()
        yield handler


# ------------------------------------------------------------------ #
#  初始化测试
# ------------------------------------------------------------------ #

class TestMessageHandlerInit:
    """测试 MessageHandler 初始化。"""

    def test_creates_scene_manager(self, handler_with_mocks):
        """初始化时应创建 SceneManager 实例。"""
        assert handler_with_mocks.scene_manager is not None

    def test_creates_engine(self, handler_with_mocks):
        """初始化时应创建 CoachEngine 实例。"""
        assert handler_with_mocks.engine is not None

    def test_creates_recognizer(self, handler_with_mocks):
        """初始化时应创建 WhisperRecognizer 实例。"""
        assert handler_with_mocks.recognizer is not None

    def test_creates_synthesizer(self, handler_with_mocks):
        """初始化时应创建 Synthesizer 实例。"""
        assert handler_with_mocks.synthesizer is not None

    def test_audio_capture_is_none_by_default(self, handler_with_mocks):
        """默认情况下 audio_capture 应为 None。"""
        assert handler_with_mocks.audio_capture is None

    def test_default_scene_is_free_talk(self, handler_with_mocks):
        """默认场景应为 free_talk。"""
        assert handler_with_mocks.current_scene == "free_talk"


# ------------------------------------------------------------------ #
#  场景切换测试
# ------------------------------------------------------------------ #

class TestSceneSwitch:
    """测试场景切换功能。"""

    def test_list_scenes_returns_all(self, handler_with_mocks):
        """应返回所有可用场景名列表。"""
        scenes = handler_with_mocks.list_scenes()
        assert "free_talk" in scenes
        assert "interview" in scenes
        assert len(scenes) == 5

    def test_switch_scene_success(self, handler_with_mocks):
        """切换存在场景应返回 ok。"""
        result = handler_with_mocks.switch_scene("interview")
        assert result["status"] == "ok"
        assert result["scene"] == "interview"
        assert handler_with_mocks.current_scene == "interview"

    def test_switch_scene_not_found(self, handler_with_mocks):
        """切换不存在场景时不应抛异常，返回错误信息。"""
        handler_with_mocks.scene_manager.switch_scene.side_effect = ValueError("not found")
        result = handler_with_mocks.switch_scene("nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    def test_switch_scene_resets_history(self, handler_with_mocks):
        """切换场景时应清除对话历史。"""
        handler_with_mocks.switch_scene("meeting")
        handler_with_mocks.engine.reset_history.assert_called()


# ------------------------------------------------------------------ #
#  对话历史测试
# ------------------------------------------------------------------ #

class TestHistory:
    """测试对话历史管理。"""

    def test_get_history_delegates_to_engine(self, handler_with_mocks):
        """get_history 应委托给 engine。"""
        handler_with_mocks.engine.get_history.return_value = [
            {"role": "user", "content": "Hi"}
        ]
        hist = handler_with_mocks.get_history()
        assert len(hist) == 1
        assert hist[0]["role"] == "user"

    def test_reset_history_delegates_to_engine(self, handler_with_mocks):
        """reset_history 应委托给 engine。"""
        handler_with_mocks.reset_history()
        handler_with_mocks.engine.reset_history.assert_called_once()


# ------------------------------------------------------------------ #
#  纯文本处理测试 (process_text)
# ------------------------------------------------------------------ #

class TestProcessText:
    """测试纯文本对话处理。"""

    @pytest.mark.asyncio
    async def test_process_text_returns_fields(self, handler_with_mocks):
        """正常流程应返回 user_text / ai_text / ai_audio 三个字段。"""
        handler_with_mocks.engine.generate_reply = AsyncMock(
            return_value="That's great!"
        )
        handler_with_mocks.synthesizer.synthesize = AsyncMock(
            return_value="/tmp/test.mp3"
        )
        # mock file read for base64
        with patch("builtins.open", MagicMock()) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake_mp3"
            result = await handler_with_mocks.process_text("Hello")

        assert result["user_text"] == "Hello"
        assert result["ai_text"] == "That's great!"
        assert result["ai_audio"] != ""

    @pytest.mark.asyncio
    async def test_process_text_empty_input(self, handler_with_mocks):
        """空输入应返回空字段。"""
        result = await handler_with_mocks.process_text("")
        assert result["user_text"] == ""
        assert result["ai_text"] == ""

    @pytest.mark.asyncio
    async def test_process_text_whitespace_only(self, handler_with_mocks):
        """仅空格的输入应返回空字段。"""
        result = await handler_with_mocks.process_text("   ")
        assert result["user_text"] == "   "
        assert result["ai_text"] == ""

    @pytest.mark.asyncio
    async def test_process_text_engine_error(self, handler_with_mocks):
        """Engine 失败时 ai_text 应为空。"""
        handler_with_mocks.engine.generate_reply = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        result = await handler_with_mocks.process_text("test")
        assert result["ai_text"] == ""
        assert result["ai_audio"] == ""

    @pytest.mark.asyncio
    async def test_process_text_empty_reply_skips_synthesis(self, handler_with_mocks):
        """AI 回复为空时不应调用语音合成。"""
        handler_with_mocks.engine.generate_reply = AsyncMock(return_value="")
        result = await handler_with_mocks.process_text("Hi")
        assert result["ai_text"] == ""
        assert result["ai_audio"] == ""


# ------------------------------------------------------------------ #
#  录音控制测试
# ------------------------------------------------------------------ #

class TestRecording:
    """测试麦克风采集控制。"""

    def test_start_recording_creates_capture(self, handler_with_mocks):
        """start_recording 应创建 AudioCapture 并启动。"""
        handler_with_mocks.start_recording()
        assert handler_with_mocks.audio_capture is not None
        handler_with_mocks.audio_capture.start.assert_called_once()

    def test_start_recording_twice_raises(self, handler_with_mocks):
        """重复调用 start_recording 应抛出 RuntimeError。"""
        handler_with_mocks.start_recording()
        with pytest.raises(RuntimeError, match="已在采集"):
            handler_with_mocks.start_recording()

    def test_stop_recording_cleans_up(self, handler_with_mocks):
        """stop_recording 应停止采集并清理引用。"""
        handler_with_mocks.start_recording()
        capture_ref = handler_with_mocks.audio_capture  # save ref before cleanup
        handler_with_mocks.stop_recording()
        capture_ref.stop.assert_called_once()
        assert handler_with_mocks.audio_capture is None

    def test_stop_recording_idempotent(self, handler_with_mocks):
        """多次调用 stop_recording 不应崩溃。"""
        handler_with_mocks.stop_recording()
        handler_with_mocks.stop_recording()
        # 不应抛出异常

    def test_start_recording_with_capture_error(self, handler_with_mocks):
        """AudioCapture 启动失败时应传递错误。"""
        with patch("web.handler.AudioCapture") as mock_cls:
            mock_capture = MagicMock()
            mock_capture.start.side_effect = RuntimeError("Mic not found")
            mock_cls.return_value = mock_capture

            # 重建 handler 以使用新的 mock
            with patch("web.handler.SceneManager"), \
                 patch("web.handler.CoachEngine"), \
                 patch("web.handler.WhisperRecognizer"), \
                 patch("web.handler.Synthesizer"):
                from web.handler import MessageHandler
                h = MessageHandler()

            with pytest.raises(RuntimeError, match="启动麦克风采集失败"):
                h.start_recording()
            assert h.audio_capture is None
