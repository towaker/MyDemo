"""场景管理器 — 加载和管理 YAML 格式的场景 Prompt 模板。"""

import os
from pathlib import Path
from typing import Optional

import yaml


class Scene:
    """单个场景的数据模型。"""

    def __init__(self, name: str, system_prompt: str, first_message: str):
        self.name = name
        self.system_prompt = system_prompt
        self.first_message = first_message

    def __repr__(self) -> str:
        return f"<Scene name={self.name!r}>"


class SceneManager:
    """场景管理器：加载、查询、切换场景 Prompt 模板。"""

    def __init__(self, scenes_dir: str):
        self._scenes_dir = Path(scenes_dir)
        self._scenes: dict[str, Scene] = {}
        self._current: Optional[str] = None
        self._load_all()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def current_scene(self) -> Optional[Scene]:
        """返回当前激活的场景对象，未设置时返回 None。"""
        if self._current is None:
            return None
        return self._scenes.get(self._current)

    @property
    def current_name(self) -> Optional[str]:
        """返回当前激活场景的名称。"""
        return self._current

    def list_scenes(self) -> list[str]:
        """返回所有可用场景名称列表，free_talk 固定排在第一位。"""
        scenes = sorted(self._scenes.keys())
        if "free_talk" in scenes:
            scenes.remove("free_talk")
            scenes.insert(0, "free_talk")
        return scenes

    def get_scene(self, name: str) -> Optional[Scene]:
        """根据名称获取场景对象。"""
        return self._scenes.get(name)

    def switch_scene(self, name: str) -> Scene:
        """切换到指定场景并返回该场景对象。"""
        scene = self._scenes.get(name)
        if scene is None:
            available = ", ".join(self.list_scenes())
            raise ValueError(
                f"Scene '{name}' not found. Available: {available}"
            )
        self._current = name
        return scene

    def get_system_prompt(self, name: Optional[str] = None) -> str:
        """获取场景的系统 Prompt。name 为 None 时使用当前场景。"""
        scene = self._resolve(name)
        return scene.system_prompt

    def get_first_message(self, name: Optional[str] = None) -> str:
        """获取场景的首次开场消息。name 为 None 时使用当前场景。"""
        scene = self._resolve(name)
        return scene.first_message

    def reload(self) -> None:
        """重新加载所有场景文件（热更新）。"""
        self._scenes.clear()
        self._load_all()

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _resolve(self, name: Optional[str]) -> Scene:
        """解析场景名称 → 场景对象。"""
        target = name or self._current
        if target is None:
            available = ", ".join(self.list_scenes())
            raise RuntimeError(
                "No scene selected. Call switch_scene() first, "
                f"or pass name explicitly. Available: {available}"
            )
        scene = self._scenes.get(target)
        if scene is None:
            available = ", ".join(self.list_scenes())
            raise ValueError(
                f"Scene '{target}' not found. Available: {available}"
            )
        return scene

    def _load_all(self) -> None:
        """扫描 scenes_dir 目录，加载所有 .yaml / .yml 文件。"""
        if not self._scenes_dir.exists():
            raise FileNotFoundError(
                f"Scenes directory not found: {self._scenes_dir}"
            )

        yaml_files = sorted(self._scenes_dir.glob("*.yaml")) + sorted(
            self._scenes_dir.glob("*.yml")
        )
        if not yaml_files:
            raise RuntimeError(
                f"No YAML scene files found in {self._scenes_dir}"
            )

        for yaml_file in yaml_files:
            scene = self._load_file(yaml_file)
            self._scenes[scene.name] = scene

    @staticmethod
    def _load_file(file_path: Path) -> Scene:
        """加载单个 YAML 文件并返回 Scene 对象。"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid scene file {file_path}: expected a YAML mapping"
            )

        name = data.get("name", file_path.stem)
        system_prompt = data.get("system_prompt", "")
        first_message = data.get("first_message", "")

        if not system_prompt:
            raise ValueError(
                f"Scene file {file_path} is missing 'system_prompt' field"
            )

        return Scene(
            name=name,
            system_prompt=system_prompt.strip(),
            first_message=first_message.strip(),
        )
