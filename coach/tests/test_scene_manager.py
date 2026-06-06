"""SceneManager 单元测试 — 场景加载、切换、查询。"""

import tempfile
import os
from pathlib import Path

import pytest

from coach.scene_manager import Scene, SceneManager


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def scenes_dir():
    """创建包含测试场景 YAML 的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        (base / "interview.yaml").write_text(
            "name: interview\n"
            "system_prompt: You are an interviewer.\n"
            "first_message: Tell me about yourself.\n",
            encoding="utf-8",
        )
        (base / "ordering.yaml").write_text(
            "name: ordering\n"
            "system_prompt: You are a waiter.\n"
            "first_message: What would you like to order?\n",
            encoding="utf-8",
        )
        (base / "no_first.yaml").write_text(
            "name: no_first\n"
            "system_prompt: You are helpful.\n",
            encoding="utf-8",
        )
        yield str(base)


@pytest.fixture
def scene_manager(scenes_dir):
    return SceneManager(scenes_dir)


# ------------------------------------------------------------------ #
#  Loading
# ------------------------------------------------------------------ #

class TestSceneLoading:
    """场景加载测试。"""

    def test_loads_all_scenes(self, scene_manager):
        names = scene_manager.list_scenes()
        assert "interview" in names
        assert "ordering" in names
        assert "no_first" in names
        assert len(names) == 3

    def test_scene_has_system_prompt(self, scene_manager):
        scene = scene_manager.get_scene("interview")
        assert scene is not None
        assert scene.system_prompt == "You are an interviewer."
        assert scene.first_message == "Tell me about yourself."

    def test_scene_without_first_message(self, scene_manager):
        scene = scene_manager.get_scene("no_first")
        assert scene is not None
        assert scene.first_message == ""

    def test_missing_system_prompt_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "bad.yaml").write_text("name: bad\n", encoding="utf-8")
            with pytest.raises(ValueError, match="system_prompt"):
                SceneManager(str(base))

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            SceneManager("/nonexistent/dir/12345")

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="No YAML scene files"):
                SceneManager(str(tmpdir))


# ------------------------------------------------------------------ #
#  Scene switching
# ------------------------------------------------------------------ #

class TestSceneSwitching:
    """场景切换测试。"""

    def test_switch_scene_returns_scene(self, scene_manager):
        scene = scene_manager.switch_scene("interview")
        assert scene.name == "interview"

    def test_switch_sets_current(self, scene_manager):
        scene_manager.switch_scene("ordering")
        assert scene_manager.current_name == "ordering"

    def test_switch_nonexistent_raises(self, scene_manager):
        with pytest.raises(ValueError, match="not found"):
            scene_manager.switch_scene("does_not_exist")

    def test_get_prompt_without_selection_raises(self, scene_manager):
        with pytest.raises(RuntimeError, match="No scene selected"):
            scene_manager.get_system_prompt()

    def test_get_prompt_by_explicit_name(self, scene_manager):
        prompt = scene_manager.get_system_prompt("ordering")
        assert prompt == "You are a waiter."

    def test_get_first_message_by_explicit_name(self, scene_manager):
        msg = scene_manager.get_first_message("interview")
        assert msg == "Tell me about yourself."


# ------------------------------------------------------------------ #
#  Reload
# ------------------------------------------------------------------ #

class TestReload:
    """热重载测试。"""

    def test_reload_picks_up_new_scene(self, scenes_dir):
        sm = SceneManager(scenes_dir)
        initial_count = len(sm.list_scenes())

        # 写新文件
        (Path(scenes_dir) / "meeting.yaml").write_text(
            "name: meeting\n"
            "system_prompt: You are a meeting host.\n"
            "first_message: Let's start.\n",
            encoding="utf-8",
        )
        sm.reload()
        assert len(sm.list_scenes()) == initial_count + 1
        assert sm.get_scene("meeting") is not None

    def test_reload_updates_modified_scene(self, scenes_dir):
        sm = SceneManager(scenes_dir)
        scene_before = sm.get_scene("interview")

        (Path(scenes_dir) / "interview.yaml").write_text(
            "name: interview\n"
            "system_prompt: You are a strict interviewer.\n"
            "first_message: Why should we hire you?\n",
            encoding="utf-8",
        )
        sm.reload()
        scene_after = sm.get_scene("interview")
        assert scene_after.system_prompt == "You are a strict interviewer."
        assert scene_after.first_message == "Why should we hire you?"
