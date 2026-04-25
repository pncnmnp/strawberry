import os

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from lm import _suppress_stderr, _restore_stderr

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
THRESHOLD = 0.5

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "smart-turn-v3.2-cpu.onnx")


def _build_session(path: str) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # one time cost, graph optimizations at load time
    return ort.InferenceSession(path, sess_options=so)


saved = _suppress_stderr()
try:
    _features = WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)
    _session = _build_session(_MODEL_PATH)
finally:
    _restore_stderr(saved)


def is_turn_complete(audio: np.ndarray) -> tuple[bool, float]:
    """Return (complete, probability). Pass the audio of the user's current turn (16kHz mono)."""
    if len(audio) > WINDOW_SAMPLES:
        audio = audio[-WINDOW_SAMPLES:] # most recent 8 seconds
    elif len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (WINDOW_SAMPLES - len(audio), 0))

    inputs = _features(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    feats = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
    output = np.asarray(_session.run(None, {"input_features": feats})[0])
    prob = float(output.flatten()[0])
    return prob > THRESHOLD, prob
