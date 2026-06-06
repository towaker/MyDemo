"""AI 英语口语陪练 — 入口文件"""

import uvicorn
from config import config


def main():
    """启动 FastAPI 服务"""
    uvicorn.run(
        "web.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
