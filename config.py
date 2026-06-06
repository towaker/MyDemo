"""项目配置管理"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """全局配置 — 各模块配置项将在对应开发阶段逐步追加"""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 服务
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # 项目路径
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    SCENES_DIR: str = os.path.join(PROJECT_ROOT, "scenes")


config = Config()