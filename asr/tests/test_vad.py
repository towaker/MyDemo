"""Tests for VAD module - voice activity detection core logic."""

import struct
import pytest
from asr.vad import VAD


class TestVAD:
    """VAD unit tests."""

    @pytest.fixture
    def vad(self):
        """Create default VAD instance (aggressiveness=2, 30ms frames)."""
        return VAD(aggressiveness=2, frame_duration_ms=30)

    @pytest.fixture
    def vad_10ms(self):
        """10ms frame VAD instance."""
        return VAD(aggressiveness=2, frame_duration_ms=10)

    @pytest.fixture
    def vad_20ms(self):
        """20ms frame VAD instance."""
        return VAD(aggressiveness=2, frame_duration_ms=20)

    def _make_frame(self, frame_duration_ms: int, amplitude: int = 0) -> bytes:
        """Generate a PCM audio frame of given duration and amplitude.

        Args:
            frame_duration_ms: Frame duration in ms (10, 20, or 30).
            amplitude: Sample amplitude, 0 = silence.

        Returns:
            Raw PCM bytes.
        """
        size_map = {10: 160, 20: 320, 30: 480}
        num_samples = size_map[frame_duration_ms]
        samples = [amplitude] * num_samples
        return struct.pack(f"<{num_samples}h", *samples)

    def _make_silence(self, frame_duration_ms: int) -> bytes:
        """Generate a silent audio frame."""
        return self._make_frame(frame_duration_ms, amplitude=0)

    def _make_speech_like(self, frame_duration_ms: int) -> bytes:
        """Generate a non-silent audio frame (simulated speech)."""
        return self._make_frame(frame_duration_ms, amplitude=500)

    # ---- Initialization tests ----

    def test_default_aggressiveness(self):
        """Default aggressiveness should be 2."""
        v = VAD()
        assert v.aggressiveness == 2

    def test_custom_aggressiveness(self):
        """Custom aggressiveness should be accepted."""
        for agg in [0, 1, 2, 3]:
            v = VAD(aggressiveness=agg)
            assert v.aggressiveness == agg

    def test_invalid_aggressiveness(self):
        """Aggressiveness outside 0-3 should raise ValueError."""
        with pytest.raises(ValueError, match="0-3"):
            VAD(aggressiveness=4)
        with pytest.raises(ValueError, match="0-3"):
            VAD(aggressiveness=-1)

    def test_invalid_frame_duration(self):
        """Unsupported frame duration should raise ValueError."""
        with pytest.raises(ValueError, match="不支持的帧长"):
            VAD(frame_duration_ms=15)

    def test_frame_size_mapping(self, vad, vad_10ms, vad_20ms):
        """Frame size should match frame duration."""
        assert vad.frame_size == 480       # 30ms
        assert vad_10ms.frame_size == 160  # 10ms
        assert vad_20ms.frame_size == 320  # 20ms

    # ---- is_speech tests ----

    def test_is_speech_silence(self, vad):
        """Silence frame should return False."""
        frame = self._make_silence(30)
        assert vad.is_speech(frame) is False

    def test_is_speech_nonzero(self, vad):
        """Non-zero amplitude frame may or may not be speech (depends on webrtcvad)."""
        frame = self._make_speech_like(30)
        result = vad.is_speech(frame)
        assert isinstance(result, bool)

    def test_is_speech_wrong_frame_size(self, vad):
        """Wrong frame size should raise ValueError."""
        wrong_frame = b'\x00' * 100  # Not 960 bytes
        with pytest.raises(ValueError, match="音频帧长度不匹配"):
            vad.is_speech(wrong_frame)

    def test_is_speech_10ms_frame(self, vad_10ms):
        """10ms frame is_speech should work."""
        frame = self._make_frame(10, amplitude=0)
        result = vad_10ms.is_speech(frame)
        assert isinstance(result, bool)

    def test_is_speech_20ms_frame(self, vad_20ms):
        """20ms frame is_speech should work."""
        frame = self._make_frame(20, amplitude=0)
        result = vad_20ms.is_speech(frame)
        assert isinstance(result, bool)

    def test_is_speech_30ms_frame(self, vad):
        """30ms frame is_speech should work."""
        frame = self._make_frame(30, amplitude=0)
        result = vad.is_speech(frame)
        assert isinstance(result, bool)

    # ---- segment tests ----

    def test_segment_empty_data(self, vad):
        """Empty audio data should raise ValueError."""
        with pytest.raises(ValueError, match="音频数据为空"):
            vad.segment(b"")

    def test_segment_too_short(self, vad):
        """Audio data shorter than one frame should raise ValueError."""
        with pytest.raises(ValueError, match="不足一帧"):
            vad.segment(b"\x00" * 100)

    def test_segment_single_frame(self, vad):
        """Single frame should produce one segment."""
        frame = self._make_silence(30)
        segments = vad.segment(frame)
        assert len(segments) == 1
        start_ms, end_ms, is_speech = segments[0]
        assert start_ms == 0.0
        assert end_ms == 30.0

    def test_segment_multiple_frames(self, vad):
        """Multiple frames should produce segments with correct timing."""
        # 10 frames of 30ms each = 300ms total
        frame = self._make_silence(30)
        audio = frame * 10
        segments = vad.segment(audio)
        assert len(segments) >= 1
        # Last segment should end at 300ms
        assert segments[-1][1] == 300.0

    def test_segment_continuity(self, vad):
        """Segments should cover the entire audio without gaps."""
        frame = self._make_silence(30)
        audio = frame * 5
        segments = vad.segment(audio)
        assert segments[0][0] == 0.0
        assert segments[-1][1] == 150.0
        # Check continuity between segments
        for i in range(len(segments) - 1):
            assert segments[i][1] == segments[i + 1][0], (
                f"Segment discontinuity at index {i}"
            )
