"""Coach 对话引擎模块。

提供：
- CoachEngine：对话引擎核心类
- SceneManager：场景管理器
- Scene：场景数据模型
"""

from coach.engine import CoachEngine
from coach.scene_manager import Scene, SceneManager

__all__ = ["CoachEngine", "SceneManager", "Scene"]