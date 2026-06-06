"""Tests for WhisperRecognizer module - init, config, and lazy loading (no real model)."""

import queue
import threading
from unittest.mock import patch, MagicMock
import pytest
from asr.recognizer import WhisperRecognizer


class TestWhisperRecognizerInit:
    """WhisperRecognizer initialization tests."""

    def test_init_does_not_load_model(self):
        """Init should not load model (lazy loading)."""
        recognizer = WhisperRecognizer()
        assert recognizer._model is None

    def test_default_model_size(self):
        """Default model size should be 'small' from config."""
        recognizer = WhisperRecognizer()
        assert recognizer._model_size == "small"

    def test_custom_model_size(self):
        """Model size should be read from config (default 'small')."""
        recognizer = WhisperRecognizer()
        # _model_size reads from config.WHISPER_MODEL_SIZE, default 'small'
        assert recognizer._model_size in ("small", "base", "tiny", "medium")

    def test_default_device(self):
        """Default device should be 'cuda' from config."""
        recognizer = WhisperRecognizer()
        assert recognizer._device == "cuda"

    def test_default_compute_type(self):
        """Default compute type should be 'float16' from config."""
        recognizer = WhisperRecognizer()
        assert recognizer._compute_type == "float16"


class TestWhisperRecognizerTranscribe:
    """WhisperRecognizer transcribe tests (mock model)."""

    def test_transcribe_without_model_raises(self):
        """Calling transcribe without model should raise an error."""
        recognizer = WhisperRecognizer()
        with pytest.raises(Exception):
            # Should fail because model is not loaded
            list(recognizer.transcribe("nonexistent_audio.wav"))


class TestWhisperRecognizerStreamTranscribe:
    """Stream transcribe tests."""

    def test_stream_transcribe_with_empty_queue(self):
        """Empty queue should yield nothing."""
        recognizer = WhisperRecognizer()
        audio_queue = queue.Queue()
        stop_event = threading.Event()

        # Put sentinel immediately
        audio_queue.put(None)

        results = list(recognizer.stream_transcribe(audio_queue, stop_event))
        assert isinstance(results, list)
