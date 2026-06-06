"""
FastAPI + WebSocket 服务 — AI 英语口语陪练后端入口。

提供 REST API 和 WebSocket 端点，串联 coach / asr / tts 模块，
并通过 StaticFiles 挂载 frontend 静态页面。
"""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import config
from web.handler import MessageHandler

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  全局状态
# ------------------------------------------------------------------ #

handler: Optional[MessageHandler] = None


# ------------------------------------------------------------------ #
#  应用生命周期
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时初始化 MessageHandler。"""
    global handler
    logger.info("正在初始化 MessageHandler ...")
    handler = MessageHandler()
    logger.info("MessageHandler 初始化完成，当前场景=%s", handler.current_scene)
    yield
    # 关闭时清理
    if handler is not None:
        handler.stop_recording()
    logger.info("服务已关闭")


# ------------------------------------------------------------------ #
#  创建应用
# ------------------------------------------------------------------ #

app = FastAPI(
    title="AI English Speaking Coach",
    description="AI 英语口语陪练 — Coach + ASR + TTS 服务",
    version="0.1.0",
    lifespan=lifespan,
)

# 静态文件挂载
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ------------------------------------------------------------------ #
#  REST API 路由
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端对话界面。"""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>Frontend not found</h1><p>Please create frontend/index.html</p>",
        status_code=404,
    )


@app.get("/api/scenes")
async def get_scenes():
    """获取所有可用场景列表。"""
    if handler is None:
        return JSONResponse(
            content={"error": "Service not initialized"}, status_code=503
        )
    return {"scenes": handler.list_scenes()}


@app.post("/api/scenes/{scene_name}")
async def switch_scene(scene_name: str):
    """切换到指定场景。

    Args:
        scene_name: 场景名称，如 interview、free_talk。
    """
    if handler is None:
        return JSONResponse(
            content={"error": "Service not initialized"}, status_code=503
        )
    result = handler.switch_scene(scene_name)
    status_code = 200 if result.get("status") == "ok" else 400
    return JSONResponse(content=result, status_code=status_code)


# ------------------------------------------------------------------ #
#  WebSocket 端点
# ------------------------------------------------------------------ #

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket 端点，处理实时对话消息。

    支持的消息类型：
        - {"type": "audio", "data": "<base64>"}
            客户端上传录音，返回 {"type": "reply", ...}
        - {"type": "text", "content": "..."}
            纯文本对话（降级模式），返回 {"type": "reply", ...}
        - {"type": "switch_scene", "scene": "..."}
            切换场景
        - {"type": "check_grammar", "text": "..."}
            语法纠错，返回 {"type": "grammar_result", ...}
        - {"type": "evaluate_pronunciation", "reference_text": "...", "audio_path": "..."}
            发音评测，返回 {"type": "pronunciation_result", ...}

    返回的消息格式：
        {"type": "reply", "user_text": "...", "ai_text": "...",
         "ai_audio": "<base64>"}
    """
    await ws.accept()
    logger.info("WebSocket 客户端已连接")

    if handler is None:
        await ws.send_json({"type": "error", "message": "Service not initialized"})
        await ws.close()
        return

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({
                    "type": "error",
                    "message": "Invalid JSON",
                })
                continue

            msg_type = msg.get("type", "")

            # --- 音频消息（语音输入 / 流式）---
            if msg_type == "audio":
                audio_b64 = msg.get("data", "")
                if not audio_b64:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing audio data",
                    })
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid base64 audio data",
                    })
                    continue

                try:
                    async for event in handler.process_audio_stream(audio_bytes):
                        await ws.send_json(event)
                except asyncio.CancelledError:
                    logger.info("音频流水线被打断")
                    await ws.send_json({
                        "type": "stream_cancelled",
                        "status": "interrupted",
                    })
                except WebSocketDisconnect:
                    logger.info("客户端在音频流式处理期间断开")
                except Exception as exc:
                    logger.error("音频流水线异常: %s", exc)
                    await ws.send_json({
                        "type": "error",
                        "message": f"Audio processing failed: {exc}",
                    })

            # --- 文本消息（降级模式 / 流式）---
            elif msg_type == "text":
                content = msg.get("content", "")
                if not content:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing text content",
                    })
                    continue

                try:
                    async for event in handler.process_text_stream(content):
                        await ws.send_json(event)
                except asyncio.CancelledError:
                    logger.info("文本流水线被打断")
                    await ws.send_json({
                        "type": "stream_cancelled",
                        "status": "interrupted",
                    })
                except WebSocketDisconnect:
                    logger.info("客户端在文本流式处理期间断开")
                except Exception as exc:
                    logger.error("文本流水线异常: %s", exc)
                    await ws.send_json({
                        "type": "error",
                        "message": f"Text processing failed: {exc}",
                    })

            # --- 场景切换 ---
            elif msg_type == "switch_scene":
                scene_name = msg.get("scene", "")
                if not scene_name:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing scene name",
                    })
                    continue

                switch_result = handler.switch_scene(scene_name)
                await ws.send_json({
                    "type": "scene_switched",
                    **switch_result,
                })

            # --- 语法纠错 ---
            elif msg_type == "check_grammar":
                text = msg.get("text", "")
                if not text:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing text for grammar check",
                    })
                    continue

                result = await handler.check_grammar(text)
                await ws.send_json({
                    "type": "grammar_result",
                    **result,
                })

            # --- 发音评测 ---
            elif msg_type == "evaluate_pronunciation":
                reference_text = msg.get("reference_text", "")
                audio_path = msg.get("audio_path", "")
                if not reference_text or not audio_path:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing reference_text or audio_path",
                    })
                    continue

                result = await handler.evaluate_pronunciation(reference_text, audio_path)
                await ws.send_json({
                    "type": "pronunciation_result",
                    **result,
                })

            # --- 跟读：合成参考音频 ---
            elif msg_type == "synthesize_text":
                text = msg.get("text", "")
                if not text:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing text for TTS synthesis",
                    })
                    continue

                try:
                    audio_b64 = await handler.synthesize_text(text)
                    await ws.send_json({
                        "type": "tts_audio",
                        "audio": audio_b64,
                    })
                except Exception as e:
                    logger.error("TTS 合成失败: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": f"TTS synthesis failed: {e}",
                    })

            # --- 跟读：发音评测 ---
            elif msg_type == "practice_pronounce":
                reference_text = msg.get("reference_text", "")
                audio_b64 = msg.get("audio", "")
                if not reference_text or not audio_b64:
                    await ws.send_json({
                        "type": "error",
                        "message": "Missing reference_text or audio",
                    })
                    continue

                try:
                    result = await handler.practice_pronounce(
                        reference_text, audio_b64
                    )
                    await ws.send_json({
                        "type": "pronunciation_result",
                        "reference_text": reference_text,
                        **result,
                    })
                except Exception as e:
                    logger.error("跟读评测失败: %s", e)
                    await ws.send_json({
                        "type": "error",
                        "message": f"Pronunciation evaluation failed: {e}",
                    })

            # --- 生成课后报告 ---
            elif msg_type == "generate_report":
                result = await handler.generate_report()
                await ws.send_json({
                    "type": "report_result",
                    **result,
                })

            # --- 重置会话追踪 ---
            elif msg_type == "reset_tracker":
                handler.tracker.reset()
                await ws.send_json({
                    "type": "tracker_reset",
                    "status": "ok",
                })

            # --- 打断 AI 讲话 ---
            elif msg_type == "barge_in":
                await handler.cancel_current()
                await ws.send_json({
                    "type": "barge_in_ack",
                    "status": "ok",
                })

            # --- 未知类型 ---
            else:
                await ws.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端已断开")
    except Exception as e:
        logger.error("WebSocket 处理异常: %s", e)
        try:
            await ws.send_json({
                "type": "error",
                "message": f"Internal error: {e}",
            })
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  启动入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("启动 AI English Speaking Coach 服务 ...")
    logger.info("地址: http://%s:%d", config.HOST, config.PORT)

    uvicorn.run(
        "web.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=False,
        log_level="info",
    )
