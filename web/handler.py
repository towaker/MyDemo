"""
消息处理中枢 — 协调 ASR → Coach → TTS 全链路。

MessageHandler 是 web 层的核心调度类，负责：
- 管理场景切换与查询
- 处理音频输入（Base64 → WAV → 转写 → AI 回复 → 语音合成）
- 控制麦克风采集的启动/停止
- 维护多轮对话历史
"""

import asyncio
import base64
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import AsyncIterator, Optional

from config import config
from coach.scene_manager import SceneManager
from coach.engine import CoachEngine
from asr.recognizer import WhisperRecognizer
from asr.audio_capture import AudioCapture
from tts.synthesizer import Synthesizer
from evaluation.grammar_checker import GrammarChecker
from evaluation.pronunciation import PronunciationEvaluator
from evaluation.tracker import SessionTracker
from evaluation.report import ReportGenerator
from config import config as app_config

logger = logging.getLogger(__name__)


class MessageHandler:
    """消息处理中枢，串联 ASR → Coach → TTS 全链路。

    对外暴露：
        - 场景管理：switch_scene / list_scenes
        - 音频处理：process_audio（完整的语音 → 对话 → 语音流水线）
        - 麦克风控制：start_recording / stop_recording
        - 对话历史：get_history / reset_history
    """

    def __init__(self):
        """初始化 MessageHandler，创建各模块实例。

        各组件在构造时即初始化（SceneManager 加载 YAML、Engine 准备
        API 客户端），Whisper 模型采用延迟加载（首次调用时才加载）。
        """
        self.scene_manager = SceneManager(config.SCENES_DIR)
        self.engine = CoachEngine(self.scene_manager)
        self.recognizer = WhisperRecognizer()
        self.synthesizer = Synthesizer()
        self.grammar_checker = GrammarChecker()
        self.pronunciation_evaluator = PronunciationEvaluator()
        self.tracker = SessionTracker()
        self.report_generator = ReportGenerator()
        # 打断机制：跟踪当前正在执行的流水线任务
        self._current_task: Optional[asyncio.Task] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.current_scene: str = "free_talk"

        # 初始化当前场景的系统 Prompt 到对话引擎
        try:
            self.scene_manager.switch_scene(self.current_scene)
        except ValueError:
            logger.warning("默认场景 '%s' 不存在，使用第一个可用场景", self.current_scene)
            available = self.scene_manager.list_scenes()
            if available:
                self.current_scene = available[0]
                self.scene_manager.switch_scene(self.current_scene)

    # ------------------------------------------------------------------ #
    #  场景管理
    # ------------------------------------------------------------------ #

    def list_scenes(self) -> list[str]:
        """列出所有可用场景名称。

        Returns:
            list[str]: 场景名称列表，按字母排序。
        """
        return self.scene_manager.list_scenes()

    def switch_scene(self, scene_name: str) -> dict:
        """切换到指定场景。

        切换时会清空对话历史并注入新场景的 system prompt，
        同时更新 current_scene 属性。

        Args:
            scene_name: 目标场景名称（如 "interview", "free_talk"）。

        Returns:
            dict: {"scene": scene_name, "status": "ok"} 或包含错误信息。

        Raises:
            不会抛出异常，错误信息通过返回 dict 传递。
        """
        try:
            self.scene_manager.switch_scene(scene_name)
            self.current_scene = scene_name
            self.engine.reset_history()
            self.tracker.set_scene(scene_name)
            logger.info("已切换到场景: %s", scene_name)
            return {"scene": scene_name, "status": "ok"}
        except ValueError as e:
            logger.warning("场景切换失败: %s", e)
            return {"scene": scene_name, "status": "error", "message": str(e)}

    # ------------------------------------------------------------------ #
    #  对话历史
    # ------------------------------------------------------------------ #

    def get_history(self) -> list[dict[str, str]]:
        """获取当前对话历史。

        Returns:
            list[dict]: 消息列表，每条含 role 和 content。
        """
        return self.engine.get_history()

    def reset_history(self) -> None:
        """清空对话历史。"""
        self.engine.reset_history()

    # ------------------------------------------------------------------ #
    #  纠错评测
    # ------------------------------------------------------------------ #

    async def check_grammar(self, text: str) -> dict:
        """语法纠错检查。

        调用 GrammarChecker 对用户输入文本进行语法/词汇/风格分析，
        返回结构化纠错结果。

        Args:
            text: 待检查的英文文本。

        Returns:
            dict: {"original": str, "corrected": str, "errors": list[dict]}
        """
        result = await self.grammar_checker.check(text)
        self.tracker.record_grammar(
            user_text=text,
            corrected_text=result.get("corrected", text),
            errors=result.get("errors", []),
        )
        return result

    async def evaluate_pronunciation(
        self, reference_text: str, audio_path: str
    ) -> dict:
        """发音评测。

        调用 PronunciationEvaluator 对用户朗读音频进行发音准确度评估。

        Args:
            reference_text: 参考文本。
            audio_path: 用户朗读录音的 WAV 文件路径。

        Returns:
            dict: {"overall_score": int, "fluency": int, "accuracy": int, "words": list}
        """
        result = await self.pronunciation_evaluator.evaluate(reference_text, audio_path)
        self.tracker.record_pronunciation(reference_text, result)
        return result

    # ------------------------------------------------------------------ #
    #  跟读评测
    # ------------------------------------------------------------------ #

    async def synthesize_text(self, text: str) -> str:
        """将文本合成为语音并返回 Base64 编码的 MP3 音频。

        用于跟读评测流程中合成参考音频（用户消息的 TTS 示范发音）。

        Args:
            text: 待合成的英文文本。

        Returns:
            str: Base64 编码的 MP3 音频字符串。

        Raises:
            Exception: TTS 合成失败时抛出，由调用方处理。
        """
        tmp_mp3_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as tmp:
                tmp_mp3_path = tmp.name
            await self.synthesizer.synthesize(text, tmp_mp3_path)
            with open(tmp_mp3_path, "rb") as f:
                audio_bytes = f.read()
            logger.info("TTS 合成成功，文本: %s", text[:50])
            return base64.b64encode(audio_bytes).decode("ascii")
        finally:
            self._cleanup_temp(tmp_mp3_path)

    async def practice_pronounce(
        self, reference_text: str, audio_base64: str
    ) -> dict:
        """跟读评测：将前端录制的 Base64 音频转为 WAV 后评测发音。

        流程：
            1. 将 Base64 音频解码为字节
            2. 通过 ffmpeg 转为 16kHz 单声道 WAV
            3. 调用发音评测引擎
            4. 清理临时文件

        Args:
            reference_text: 参考文本（用户应跟读的内容）。
            audio_base64: 前端录制的 Base64 编码音频（webm 格式）。

        Returns:
            dict: {"overall_score": int, "fluency": int,
                   "accuracy": int, "words": list[dict]}
        """
        tmp_wav_path = None
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_wav_path = tmp.name
            self._convert_to_wav(audio_bytes, tmp_wav_path)
            result = await self.pronunciation_evaluator.evaluate(
                reference_text, tmp_wav_path
            )
            self.tracker.record_pronunciation(reference_text, result)
            logger.info(
                "发音评测完成，总分=%s, 准确度=%s, 流利度=%s",
                result.get("overall_score"),
                result.get("accuracy"),
                result.get("fluency"),
            )
            return result
        finally:
            self._cleanup_temp(tmp_wav_path)

    # ------------------------------------------------------------------ #
    #  翻译
    # ------------------------------------------------------------------ #

    async def translate_text(self, text: str) -> dict:
        """翻译英文文本为中文。

        调用 CoachEngine.translate_text，不写入对话历史。

        Args:
            text: 待翻译的英文文本。

        Returns:
            dict: {"original": text, "translation": result}
        """
        result = await self.engine.translate_text(text)
        logger.info("翻译完成，原文长度=%d，译文长度=%d", len(text), len(result))
        return {"original": text, "translation": result}

    # ------------------------------------------------------------------ #
    #  音频处理核心流水线
    # ------------------------------------------------------------------ #

    async def process_audio(self, audio_data: bytes) -> dict:
        """音频 → 文本 → AI 回复 → 语音 的完整处理链路。

        流程：
            1. 将原始 PCM 字节写入临时 WAV 文件
            2. WhisperRecognizer.transcribe() 转写为用户文本
            3. CoachEngine.generate_reply() 获取 AI 回复
            4. Synthesizer.synthesize() 合成 AI 语音 MP3
            5. 将 MP3 读回并 Base64 编码，返回完整结果

        Args:
            audio_data: 原始 PCM 音频字节（16kHz, 16bit, 单声道）。

        Returns:
            dict: {
                "user_text": "用户说话内容",
                "ai_text": "AI 回复内容",
                "ai_audio": "<base64 编码的 MP3>",
            }
            如任一步骤失败，对应字段为空字符串。

        Raises:
            不会抛出异常，错误通过空字段体现。
        """
        self._current_task = asyncio.current_task()
        try:
            result = {"user_text": "", "ai_text": "", "ai_audio": ""}

            # Step 1: 保存原始 PCM 为临时 WAV 文件
            tmp_wav_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_wav_path = tmp.name
                self._convert_to_wav(audio_data, tmp_wav_path)
            except Exception as e:
                logger.error("写入临时 WAV 文件失败: %s", e)
                return result

            # Step 2: 语音转写
            try:
                user_text = await asyncio.to_thread(self.recognizer.transcribe, tmp_wav_path)
                result["user_text"] = user_text.strip()
                logger.info("转写结果: %s", result["user_text"])
            except Exception as e:
                logger.error("语音转写失败: %s", e)
                self._cleanup_temp(tmp_wav_path)
                return result

            # 转写为空时跳过后续步骤
            if not result["user_text"]:
                logger.info("转写结果为空，跳过 AI 回复与合成")
                self._cleanup_temp(tmp_wav_path)
                return result

            # 清理临时 WAV
            self._cleanup_temp(tmp_wav_path)

            # Step 3: AI 对话生成
            try:
                ai_text = await self.engine.generate_reply(
                    result["user_text"],
                    scene_name=self.current_scene,
                )
                result["ai_text"] = ai_text.strip()
                self.tracker.record_dialogue(result["user_text"], ai_text)
                logger.info("AI 回复: %s", result["ai_text"])
            except Exception as e:
                logger.error("AI 对话生成失败: %s", e)
                return result

            if not result["ai_text"]:
                logger.info("AI 回复为空，跳过语音合成")
                return result

            # Step 4: 语音合成
            tmp_mp3_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as tmp:
                    tmp_mp3_path = tmp.name
                await self.synthesizer.synthesize(
                    result["ai_text"], tmp_mp3_path
                )
                # 读回并 Base64 编码
                with open(tmp_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                result["ai_audio"] = base64.b64encode(audio_bytes).decode("ascii")
            except Exception as e:
                logger.error("语音合成失败: %s", e)
            finally:
                self._cleanup_temp(tmp_mp3_path)

            return result
        finally:
            self._current_task = None

    async def process_text(self, text: str) -> dict:
        """纯文本对话（降级模式，不依赖麦克风）。

        Args:
            text: 用户输入的文本。

        Returns:
            dict: {
                "ai_text": "AI 回复内容",
                "ai_audio": "<base64 编码的 MP3>",
                "user_text": text,
            }
        """
        self._current_task = asyncio.current_task()
        try:
            result = {"user_text": text, "ai_text": "", "ai_audio": ""}

            if not text or not text.strip():
                return result

            # Step 1: AI 对话生成
            try:
                ai_text = await self.engine.generate_reply(
                    text.strip(),
                    scene_name=self.current_scene,
                )
                result["ai_text"] = ai_text.strip()
                self.tracker.record_dialogue(text.strip(), ai_text)
            except Exception as e:
                logger.error("AI 对话生成失败: %s", e)
                return result

            if not result["ai_text"]:
                return result

            # Step 2: 语音合成
            tmp_mp3_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as tmp:
                    tmp_mp3_path = tmp.name
                await self.synthesizer.synthesize(
                    result["ai_text"], tmp_mp3_path
                )
                with open(tmp_mp3_path, "rb") as f:
                    audio_bytes = f.read()
                result["ai_audio"] = base64.b64encode(audio_bytes).decode("ascii")
            except Exception as e:
                logger.error("语音合成失败: %s", e)
            finally:
                self._cleanup_temp(tmp_mp3_path)

            return result
        finally:
            self._current_task = None

    # ------------------------------------------------------------------ #
    #  流式处理方法（P7b 延迟优化）
    # ------------------------------------------------------------------ #

    async def process_audio_stream(self, audio_data: bytes):
        """流式处理音频输入（ASR → Coach 流 → 分片 TTS）。

        与 process_audio 的区别：
        - Coach 使用流式生成，边生成边 yield 事件
        - 每个完整句子立即合成 TTS，无需等待整段回复完成
        - 通过 AsyncIterator 逐条产出事件，由 server.py 推送给前端

        产出的事件类型：
            stream_start  {"user_text": str}
            text_chunk   {"text": str}
            audio_chunk  {"audio": str (base64)}
            stream_end   {"user_text": str, "ai_text": str}

        Yields:
            dict: 可直接 JSON 序列化的事件。
        """
        # 打断机制：注册当前任务
        self._current_task = asyncio.current_task()
        try:
            user_text = ""

            # Step 1: WAV 转换 + 转写
            tmp_wav_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_wav_path = tmp.name
                self._convert_to_wav(audio_data, tmp_wav_path)
            except Exception as exc:
                logger.error("写入临时 WAV 文件失败: %s", exc)
                yield {
                    "type": "stream_error",
                    "message": f"音频转换失败: {exc}",
                }
                return

            try:
                user_text = (
                    await asyncio.to_thread(
                        self.recognizer.transcribe, tmp_wav_path
                    )
                ).strip()
                logger.info("转写结果: %s", user_text)
            except Exception as exc:
                logger.error("语音转写失败: %s", exc)
                yield {
                    "type": "stream_error",
                    "message": f"语音识别失败: {exc}",
                }
                return
            finally:
                self._cleanup_temp(tmp_wav_path)

            if not user_text:
                logger.info("转写结果为空，跳过对话生成")
                yield {
                    "type": "stream_error",
                    "message": "未检测到语音内容，请重试",
                }
                return

            # Step 2: 发送 stream_start
            yield {
                "type": "stream_start",
                "user_text": user_text,
            }

            # Step 3: 流式 Coach + 分片 TTS
            buffer = ""
            full_ai_text = ""
            try:
                async for chunk in self.engine.generate_reply_stream(
                    user_text,
                    scene_name=self.current_scene,
                ):
                    buffer += chunk
                    full_ai_text += chunk

                    # 检测句子边界：. ! ? 或换行
                    if buffer.rstrip().endswith(('.', '!', '?', '\n')):
                        sentence = buffer.strip()
                        buffer = ""
                        if len(sentence) < 5:
                            continue

                        # 发送文本块
                        yield {"type": "text_chunk", "text": sentence}

                        # 合成并发送音频块
                        audio_b64 = await self._synthesize_chunk(sentence)
                        if audio_b64:
                            yield {"type": "audio_chunk", "audio": audio_b64}

                # 处理残留 buffer
                if buffer.strip():
                    sentence = buffer.strip()
                    full_ai_text = full_ai_text.rstrip()
                    yield {"type": "text_chunk", "text": sentence}
                    audio_b64 = await self._synthesize_chunk(sentence)
                    if audio_b64:
                        yield {"type": "audio_chunk", "audio": audio_b64}
            except asyncio.CancelledError:
                raise  # 重新抛出，让 server.py 处理打断
            except Exception as exc:
                logger.error("AI 流式生成失败: %s", exc)
                yield {
                    "type": "stream_error",
                    "message": f"AI 生成失败: {exc}",
                }
                return

            # Step 4: 记录对话 + 发送结束事件
            if full_ai_text:
                self.tracker.record_dialogue(user_text, full_ai_text)
                logger.info("AI 流式回复完成, 总长度=%d", len(full_ai_text))

            yield {
                "type": "stream_end",
                "user_text": user_text,
                "ai_text": full_ai_text,
            }

        finally:
            self._current_task = None

    async def process_text_stream(self, text: str):
        """流式处理文本输入（Coach 流 → 分片 TTS）。

        与 process_text 的区别：
        - Coach 使用流式生成，边生成边 yield 事件
        - 每个完整句子立即合成 TTS

        产出的事件类型同 process_audio_stream。

        Yields:
            dict: 可直接 JSON 序列化的事件。
        """
        self._current_task = asyncio.current_task()
        try:
            content = text.strip()
            if not content:
                return

            # 发送 stream_start
            yield {
                "type": "stream_start",
                "user_text": content,
            }

            # 流式 Coach + 分片 TTS
            buffer = ""
            full_ai_text = ""
            try:
                async for chunk in self.engine.generate_reply_stream(
                    content,
                    scene_name=self.current_scene,
                ):
                    buffer += chunk
                    full_ai_text += chunk

                    if buffer.rstrip().endswith(('.', '!', '?', '\n')):
                        sentence = buffer.strip()
                        buffer = ""
                        if len(sentence) < 5:
                            continue

                        yield {"type": "text_chunk", "text": sentence}
                        audio_b64 = await self._synthesize_chunk(sentence)
                        if audio_b64:
                            yield {"type": "audio_chunk", "audio": audio_b64}

                if buffer.strip():
                    sentence = buffer.strip()
                    full_ai_text = full_ai_text.rstrip()
                    yield {"type": "text_chunk", "text": sentence}
                    audio_b64 = await self._synthesize_chunk(sentence)
                    if audio_b64:
                        yield {"type": "audio_chunk", "audio": audio_b64}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("AI 流式生成失败: %s", exc)
                yield {
                    "type": "stream_error",
                    "message": f"AI 生成失败: {exc}",
                }
                return

            if full_ai_text:
                self.tracker.record_dialogue(content, full_ai_text)

            yield {
                "type": "stream_end",
                "user_text": content,
                "ai_text": full_ai_text,
            }

        finally:
            self._current_task = None

    # ------------------------------------------------------------------ #
    #  麦克风采集控制
    # ------------------------------------------------------------------ #

    def start_recording(self) -> None:
        """启动麦克风音频采集。

        创建 AudioCapture 实例并开始采集，音频数据将流入
        audio_capture.queue 供 stream_transcribe 消费。

        Raises:
            RuntimeError: 麦克风不可用或已在采集时抛出。
        """
        if self.audio_capture is not None:
            raise RuntimeError("麦克风已在采集中，请先调用 stop_recording()")

        try:
            self.audio_capture = AudioCapture()
            self.audio_capture.start()
            logger.info("麦克风采集已启动")
        except Exception as e:
            self.audio_capture = None
            raise RuntimeError(f"启动麦克风采集失败: {e}") from e

    def stop_recording(self) -> None:
        """停止麦克风音频采集。

        安全地停止并清理 AudioCapture 实例。如果未在采集中，调用无副作用。
        """
        if self.audio_capture is not None:
            try:
                self.audio_capture.stop()
                logger.info("麦克风采集已停止")
            except Exception as e:
                logger.warning("停止麦克风采集时出错: %s", e)
            finally:
                self.audio_capture = None

    # ------------------------------------------------------------------ #
    #  内部工具方法
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_wav(path: str, pcm_data: bytes) -> None:
        """将原始 PCM 数据写入 WAV 文件。

        采用 16kHz、单声道、16bit 格式，与 AudioCapture 默认配置一致。

        Args:
            path: 目标 WAV 文件路径。
            pcm_data: 原始 PCM 音频字节。
        """
        sample_rate = getattr(config, "SAMPLE_RATE", 16000)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

    @staticmethod
    def _cleanup_temp(file_path: Optional[str]) -> None:
        """安全清理临时文件。

        Args:
            file_path: 要删除的文件路径，为 None 时跳过。
        """
        if file_path is None:
            return
        try:
            os.unlink(file_path)
        except OSError:
            pass

    @staticmethod
    def _convert_to_wav(audio_bytes: bytes, output_path: str) -> None:
        """将 webm 等格式转为 16kHz 16bit mono WAV（通过 ffmpeg stdin）。

        前端 MediaRecorder 输出 webm 容器格式，Whisper 需要 16kHz
        16bit 单声道 WAV。本方法通过 subprocess 调用 ffmpeg，
        将音频字节经 stdin 管道传入，转码后写入 output_path。

        Args:
            audio_bytes: 输入音频字节（通常为 webm 格式）。
            output_path: 输出 WAV 文件路径。

        Raises:
            RuntimeError: ffmpeg 转换失败或 ffmpeg 未安装。
        """
        import subprocess

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "webm",
                    "-i", "pipe:0",
                    "-ar", "16000",
                    "-ac", "1",
                    "-sample_fmt", "s16",
                    output_path,
                ],
                input=audio_bytes,
                capture_output=True,
                timeout=10,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"音频格式转换失败: {e.stderr.decode()}"
            ) from e
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg 未安装。请执行: winget install ffmpeg "
                "或从 https://ffmpeg.org 下载"
            ) from None

    # ------------------------------------------------------------------ #
    #  流式 TTS 辅助
    # ------------------------------------------------------------------ #

    async def _synthesize_chunk(self, text: str) -> str:
        """将短文本片段合成为 Base64 MP3。

        用于流式 TTS，每个文本分片独立合成。失败时返回空字符串。

        Args:
            text: 待合成的短文本（一句话）。

        Returns:
            Base64 编码的 MP3 音频字符串，失败返回空字符串。
        """
        if not text or not text.strip():
            return ""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ) as tmp:
                tmp_path = tmp.name
            await self.synthesizer.synthesize(text.strip(), tmp_path)
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            return base64.b64encode(audio_bytes).decode("ascii")
        except Exception as exc:
            logger.warning("分片 TTS 合成失败: %s", exc)
            return ""
        finally:
            self._cleanup_temp(tmp_path)

    # ------------------------------------------------------------------ #
    #  课后报告
    # ------------------------------------------------------------------ #

    async def generate_report(self) -> dict:
        """生成当前会话的课后学习报告。

        基于 SessionTracker 收集的评测数据，调用 ReportGenerator
        生成包含多维度评分、改进建议和错误详情的学习报告。

        Returns:
            dict: 结构化的课后报告，包含 overall_score / summary /
                  grammar / pronunciation / strengths / weaknesses /
                  suggestions / detailed_errors 等字段。
        """
        try:
            summary = self.tracker.get_session_summary()
            use_ai = getattr(app_config, "REPORT_USE_AI", True)
            report = await self.report_generator.generate(summary, use_ai=use_ai)
            logger.info(
                "报告生成完成, overall_score=%s, generated_by=%s",
                report.get("overall_score"),
                report.get("generated_by"),
            )
            return report
        except Exception as exc:
            logger.error("报告生成失败: %s", exc)
            return {
                "session_id": "",
                "overall_score": 0,
                "summary": f"报告生成失败: {exc}",
                "generated_by": "error",
            }

    # ------------------------------------------------------------------ #
    #  打断机制
    # ------------------------------------------------------------------ #

    async def cancel_current(self) -> None:
        """取消当前正在执行的流水线任务（打断 AI 讲话）。

        用户按下录音键时调用，终止正在执行的 ASR→Coach→TTS 流水线。
        取消后 _current_task 会被重置为 None。
        """
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.info("已发送取消信号到当前流水线任务")
        self._current_task = None
