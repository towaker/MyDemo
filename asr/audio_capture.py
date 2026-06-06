"""
麦克风音频采集模块

基于 pyaudio 实现实时麦克风音频采集，通过回调模式将音频数据
放入 queue.Queue 供下游消费。
"""

import logging
import queue
import threading
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class AudioCapture:
    """麦克风音频采集器。

    使用 pyaudio 库以回调模式从默认麦克风实时采集音频，
    音频块通过 queue.Queue 传递给消费者（如 WhisperRecognizer）。

    支持上下文管理器协议（with 语句），进入上下文时自动启动，
    退出时自动停止。
    """

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        chunk_size: Optional[int] = None,
        channels: int = 1,
        sample_width: int = 2,  # 16bit
    ):
        """初始化音频采集器。

        Args:
            sample_rate: 采样率，默认从 config.SAMPLE_RATE 读取（16000）。
            chunk_size: 每次回调采集的帧数，默认 sample_rate // 10（100ms）。
            channels: 声道数，默认 1（单声道）。
            sample_width: 采样位宽（字节），默认 2（16bit）。
        """
        self.sample_rate = sample_rate or getattr(config, "SAMPLE_RATE", 16000)
        self.chunk_size = chunk_size or self.sample_rate // 10  # 100ms per chunk
        self.channels = channels
        self.sample_width = sample_width

        self._audio: Optional["pyaudio.PyAudio"] = None
        self._stream: Optional["pyaudio.Stream"] = None
        self._queue: queue.Queue = queue.Queue()
        self._is_running = threading.Event()
        self._is_paused = threading.Event()
        self._lock = threading.Lock()

    @property
    def queue(self) -> queue.Queue:
        """获取音频数据队列，供消费者读取。"""
        return self._queue

    @property
    def is_running(self) -> bool:
        """采集是否正在进行中。"""
        return self._is_running.is_set()

    @property
    def is_paused(self) -> bool:
        """采集是否已暂停。"""
        return self._is_paused.is_set()

    def start(self):
        """开始音频采集。

        Raises:
            ImportError: pyaudio 未安装。
            RuntimeError: 麦克风不可用或已被占用。
        """
        if self._is_running.is_set():
            logger.warning("音频采集已在运行")
            return

        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio 未安装。请执行: pip install pyaudio\n"
                "如果安装失败，请访问 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio "
                "下载对应 Python 版本的 .whl 文件手动安装。"
            )

        with self._lock:
            self._audio = pyaudio.PyAudio()

            # 检查默认输入设备
            try:
                device_info = self._audio.get_default_input_device_info()
                logger.info("使用麦克风: %s", device_info.get("name", "Unknown"))
            except OSError:
                self._audio.terminate()
                self._audio = None
                raise RuntimeError(
                    "未检测到可用麦克风，请检查麦克风是否已连接并开启权限。"
                )

            try:
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback,
                )
            except OSError as e:
                self._audio.terminate()
                self._audio = None
                raise RuntimeError(f"无法打开麦克风音频流: {e}") from e

            self._is_running.set()
            self._is_paused.clear()
            logger.info(
                "音频采集已启动 (sample_rate=%d, chunk_size=%d)",
                self.sample_rate, self.chunk_size,
            )

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """pyaudio 回调函数，将音频数据放入队列。

        Args:
            in_data: 采集到的音频字节数据。
            frame_count: 帧数。
            time_info: 时间信息字典。
            status: pyaudio 状态标志。

        Returns:
            (None, pyaudio.paContinue) 元组。
        """
        import pyaudio
        if status:
            logger.debug("音频回调状态异常: %s", status)
        if not self._is_paused.is_set():
            self._queue.put(in_data)
        return (None, pyaudio.paContinue)

    def pause(self):
        """暂停音频采集（停止将数据放入队列，但保持流打开）。"""
        if not self._is_running.is_set():
            logger.warning("音频采集未运行，无法暂停")
            return
        self._is_paused.set()
        logger.info("音频采集已暂停")

    def resume(self):
        """恢复音频采集。"""
        if not self._is_running.is_set():
            logger.warning("音频采集未运行，无法恢复")
            return
        self._is_paused.clear()
        logger.info("音频采集已恢复")

    def stop(self):
        """停止音频采集并释放资源。"""
        if not self._is_running.is_set():
            return

        with self._lock:
            self._is_running.clear()
            self._is_paused.clear()

            if self._stream is not None:
                try:
                    self._stream.stop_stream()
                    self._stream.close()
                except Exception as e:
                    logger.warning("关闭音频流时出错: %s", e)
                finally:
                    self._stream = None

            if self._audio is not None:
                try:
                    self._audio.terminate()
                except Exception as e:
                    logger.warning("终止 pyaudio 时出错: %s", e)
                finally:
                    self._audio = None

            # 放入哨兵通知消费者结束
            self._queue.put(None)

        logger.info("音频采集已停止")

    def __enter__(self):
        """上下文管理器入口，自动开始采集。"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动停止采集。"""
        self.stop()
        return False
