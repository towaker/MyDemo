"""Tests for AudioCapture module - config and queue logic (no real mic)."""

import queue
import pytest
from asr.audio_capture import AudioCapture


class TestAudioCaptureConfig:
    """AudioCapture configuration and initialization tests."""

    def test_default_init(self):
        """Default init should use config sample rate."""
        cap = AudioCapture()
        assert cap.sample_rate == 16000
        assert cap.channels == 1
        assert cap.sample_width == 2
        assert cap.chunk_size == 1600  # 16000 // 10

    def test_custom_sample_rate(self):
        """Custom sample rate should be honored."""
        cap = AudioCapture(sample_rate=8000)
        assert cap.sample_rate == 8000
        assert cap.chunk_size == 800  # 8000 // 10

    def test_custom_chunk_size(self):
        """Custom chunk size should be honored."""
        cap = AudioCapture(sample_rate=16000, chunk_size=800)
        assert cap.chunk_size == 800

    def test_custom_channels(self):
        """Custom channels should be stored."""
        cap = AudioCapture(channels=2)
        assert cap.channels == 2

    def test_custom_sample_width(self):
        """Custom sample width should be stored."""
        cap = AudioCapture(sample_width=4)
        assert cap.sample_width == 4


class TestAudioCaptureState:
    """AudioCapture state management tests."""

    def test_initial_state(self):
        """Initially, capture should not be running or paused."""
        cap = AudioCapture()
        assert cap.is_running is False
        assert cap.is_paused is False

    def test_queue_empty_initially(self):
        """Queue should be empty on init."""
        cap = AudioCapture()
        assert cap.queue.empty()


class TestAudioCaptureContextManager:
    """AudioCapture context manager tests."""

    def test_has_context_manager_methods(self):
        """__enter__ and __exit__ should exist."""
        cap = AudioCapture()
        assert hasattr(cap, "__enter__")
        assert hasattr(cap, "__exit__")
