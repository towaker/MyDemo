"""
语音活动检测（VAD）模块

基于 webrtcvad 实现语音/静音帧检测和语音段切分。
"""

import logging
from typing import List, Tuple

from config import config

logger = logging.getLogger(__name__)


class VAD:
    """语音活动检测器。

    基于 webrtcvad 库，支持 16kHz 单声道 16bit PCM 音频的语音帧检测，
    以及从完整音频中切分语音段。

    帧长支持 10ms (160 采样点)、20ms (320 采样点)、30ms (480 采样点)。
    """

    # 帧长到采样点的映射（16kHz）
    _FRAME_SIZE_MAP = {
        10: 160,
        20: 320,
        30: 480,
    }

    def __init__(
        self,
        aggressiveness: int = None,
        frame_duration_ms: int = 30,
    ):
        """初始化 VAD 检测器。

        Args:
            aggressiveness: VAD 激进度 0-3，0 最不敏感，3 最敏感。
                            默认从 config.VAD_AGGRESSIVENESS 读取。
            frame_duration_ms: 帧长（毫秒），支持 10/20/30。
        """
        if aggressiveness is None:
            aggressiveness = getattr(config, "VAD_AGGRESSIVENESS", 2)
        if aggressiveness not in (0, 1, 2, 3):
            raise ValueError(f"VAD 激进度必须在 0-3 之间，实际传入: {aggressiveness}")
        if frame_duration_ms not in self._FRAME_SIZE_MAP:
            raise ValueError(
                f"不支持的帧长 {frame_duration_ms}ms，"
                f"支持: {list(self._FRAME_SIZE_MAP.keys())}"
            )

        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = self._FRAME_SIZE_MAP[frame_duration_ms]
        self.sample_rate = 16000  # webrtcvad 固定要求

        try:
            import webrtcvad
        except ImportError:
            raise ImportError("webrtcvad 未安装。请执行: pip install webrtcvad")

        self._vad = webrtcvad.Vad()
        self._vad.set_mode(aggressiveness)
        logger.debug(
            "VAD 初始化完成: aggressiveness=%d, frame_duration=%dms",
            aggressiveness, frame_duration_ms,
        )

    def is_speech(self, audio_frame: bytes) -> bool:
        """检测单个音频帧是否为语音。

        Args:
            audio_frame: 16kHz 单声道 16bit PCM 音频字节数据，
                         长度必须等于 frame_size 字节。

        Returns:
            True 如果该帧包含语音，否则 False。

        Raises:
            ValueError: 音频帧长度不匹配。
        """
        expected_len = self.frame_size * 2  # 16bit = 2 bytes per sample
        actual_len = len(audio_frame)
        if actual_len != expected_len:
            raise ValueError(
                f"音频帧长度不匹配: 期望 {expected_len} 字节 "
                f"({self.frame_duration_ms}ms @ 16kHz 16bit)，实际 {actual_len} 字节"
            )
        return self._vad.is_speech(audio_frame, self.sample_rate)

    def segment(self, audio_bytes: bytes) -> List[Tuple[float, float, bool]]:
        """将完整音频切分为语音段列表。

        将输入的 PCM 音频按帧扫描，标记每帧是否为语音，
        然后合并相邻的同类帧形成段。

        Args:
            audio_bytes: 16kHz 单声道 16bit PCM 音频数据。

        Returns:
            [(start_ms, end_ms, is_speech), ...] 语音段列表。
            is_speech 为 True 表示语音段，False 表示静音段。

        Raises:
            ValueError: 音频为空或帧对齐失败。
        """
        if not audio_bytes:
            raise ValueError("音频数据为空")

        frame_byte_size = self.frame_size * 2
        total_frames = len(audio_bytes) // frame_byte_size

        if total_frames == 0:
            raise ValueError(
                f"音频数据不足一帧: {len(audio_bytes)} 字节，"
                f"需要至少 {frame_byte_size} 字节"
            )

        # 逐帧检测语音
        frame_labels = []
        for i in range(total_frames):
            start_byte = i * frame_byte_size
            end_byte = start_byte + frame_byte_size
            frame = audio_bytes[start_byte:end_byte]
            frame_labels.append(self.is_speech(frame))

        # 合并连续同类帧为段
        segments = []
        if not frame_labels:
            return segments

        seg_start_frame = 0
        current_label = frame_labels[0]

        for i in range(1, total_frames):
            if frame_labels[i] != current_label:
                start_ms = seg_start_frame * self.frame_duration_ms
                end_ms = i * self.frame_duration_ms
                segments.append((start_ms, end_ms, current_label))
                seg_start_frame = i
                current_label = frame_labels[i]

        # 最后一个段
        start_ms = seg_start_frame * self.frame_duration_ms
        end_ms = total_frames * self.frame_duration_ms
        segments.append((start_ms, end_ms, current_label))

        return segments
