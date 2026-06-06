"""
Whisper 语音识别模块

基于 faster-whisper 实现本地 GPU 推理，支持整段音频转写和流式转写。
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Generator, Optional

from config import config
from .vad import VAD

logger = logging.getLogger(__name__)


class WhisperRecognizer:
    """Whisper 语音识别器，基于 faster-whisper GPU 推理。

    特性：
        - 模型延迟加载，首次调用时才加载，避免启动时占用显存
        - 支持整段音频文件转写
        - 支持从 AudioCapture 队列中实时消费音频并流式转写
    """

    # faster-whisper model_size 到 huggingface 模型 ID 的映射
    _MODEL_MAP = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large",
        "large-v2": "large-v2",
        "large-v3": "large-v3",
        "tiny.en": "tiny.en",
        "base.en": "base.en",
        "small.en": "small.en",
        "medium.en": "medium.en",
    }

    def __init__(self):
        """初始化识别器，不加载模型。"""
        self._model = None
        self._model_lock = threading.Lock()
        self._model_size: str = getattr(config, "WHISPER_MODEL_SIZE", "small")
        self._device: str = getattr(config, "WHISPER_DEVICE", "cuda")
        self._compute_type: str = getattr(config, "WHISPER_COMPUTE_TYPE", "float16")
        self._vad: Optional[VAD] = None

    @property
    def model(self):
        """延迟加载的 faster-whisper 模型实例。"""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._load_model()
        return self._model

    def _load_model(self):
        """加载 faster-whisper 模型到 GPU。

        Raises:
            ImportError: faster-whisper 未安装
            RuntimeError: 模型加载失败（如显存不足、CUDA 不可用）
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper 未安装。请执行: pip install faster-whisper"
            )

        model_id = self._MODEL_MAP.get(self._model_size, self._model_size)
        logger.info(
            "正在加载 Whisper 模型: %s (device=%s, compute_type=%s)",
            model_id, self._device, self._compute_type,
        )

        try:
            self._model = WhisperModel(
                model_id,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper 模型加载成功: %s", model_id)
        except Exception as e:
            raise RuntimeError(
                f"Whisper 模型加载失败 (model={model_id}, device={self._device}): {e}"
            ) from e

    def transcribe(self, audio_path: str) -> str:
        """转写整段音频文件。

        Args:
            audio_path: WAV 文件绝对路径，要求 16kHz 单声道 PCM 格式。

        Returns:
            转写后的完整文本，各片段以空格拼接。

        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 转写过程出错
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        try:
            segments, _info = self.model.transcribe(str(audio_file))
            texts = [seg.text.strip() for seg in segments]
            return " ".join(texts)
        except Exception as e:
            raise RuntimeError(f"音频转写失败 ({audio_path}): {e}") from e

    def stream_transcribe(
        self,
        audio_queue: queue.Queue,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[str, None, None]:
        """从音频队列中实时消费并流式转写。

        从 AudioCapture 提供的 queue.Queue 中读取音频块，
        用 VAD 检测语音段，积累足够语音后调用 Whisper 转写，
        逐段 yield 转写文本。

        Args:
            audio_queue: 包含 PCM 音频块（bytes）的队列。
            stop_event: 用于外部通知停止的事件，为 None 时通过队列终止哨兵 None 停止。

        Yields:
            转写出的文本片段字符串。
        """
        # 延迟初始化 VAD
        if self._vad is None:
            self._vad = VAD()

        audio_buffer = bytearray()
        silence_duration = 0.0
        sample_rate = getattr(config, "SAMPLE_RATE", 16000)
        bytes_per_ms = sample_rate * 2 // 1000  # 16bit mono = 2 bytes/sample
        # 静音阈值：连续静音超过此时间（秒）视为语音段结束
        silence_threshold = 0.8
        # 最小语音段长度（秒），避免碎片化
        min_speech_duration = 0.3

        def _flush_buffer():
            """将累积的音频缓冲区转写为文本。"""
            nonlocal audio_buffer
            if len(audio_buffer) < min_speech_duration * sample_rate * 2:
                audio_buffer = bytearray()
                return ""
            import tempfile
            import wave
            # 写入临时 WAV 文件供 faster-whisper 转写
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(bytes(audio_buffer))
                result = self.transcribe(tmp_path)
                return result
            except Exception as e:
                logger.warning("流式转写片段失败: %s", e)
                return ""
            finally:
                import os
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        try:
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # 超时后检查是否需要 flush 缓冲区
                    if len(audio_buffer) > 0 and silence_duration >= silence_threshold:
                        text = _flush_buffer()
                        audio_buffer = bytearray()
                        silence_duration = 0.0
                        if text:
                            yield text
                    if stop_event and stop_event.is_set():
                        break
                    continue

                # 哨兵值，停止采集
                if chunk is None:
                    break

                if not isinstance(chunk, bytes):
                    continue

                # VAD 逐帧检测
                frame_duration_ms = self._vad.frame_duration_ms
                frame_size = sample_rate * 2 * frame_duration_ms // 1000

                offset = 0
                while offset + frame_size <= len(chunk):
                    frame = chunk[offset : offset + frame_size]
                    is_speech = self._vad.is_speech(frame)
                    offset += frame_size

                    if is_speech:
                        audio_buffer.extend(frame)
                        silence_duration = 0.0
                    elif len(audio_buffer) > 0:
                        # 语音段结束后开始积累静音
                        audio_buffer.extend(frame)
                        silence_duration += frame_duration_ms / 1000.0
                        if silence_duration >= silence_threshold:
                            text = _flush_buffer()
                            audio_buffer = bytearray()
                            silence_duration = 0.0
                            if text:
                                yield text
                    # 无声且无累积缓冲，丢弃该帧

                # 把不足一帧的尾部追加到缓冲
                remainder = chunk[offset:]
                if remainder:
                    audio_buffer.extend(remainder)

            # 退出循环后 flush 剩余缓冲
            if len(audio_buffer) > 0:
                text = _flush_buffer()
                if text:
                    yield text

        finally:
            # 确保 stop_event 被设置时释放资源
            pass
