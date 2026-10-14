"""Pure NumPy/SciPy inference for the exported no-E5 sequence-fusion model.

This module is intentionally independent of PyTorch.  It implements the
specific one-layer bidirectional GRU and MLP used by
``refit_sequence_fusion_noe5_numpy.py``.  It is an experiment artifact until
the main submission has separately accepted it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.special import erf


ArrayMap = Mapping[str, np.ndarray]


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: archive[name].astype(np.float32, copy=False) for name in archive.files}


def _gelu(value: np.ndarray) -> np.ndarray:
    # This is PyTorch nn.GELU(approximate="none"), the default in the model.
    return (0.5 * value * (1.0 + erf(value / np.float32(np.sqrt(2.0))))).astype(np.float32, copy=False)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))).astype(np.float32, copy=False)


def _linear(value: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (value @ weight.T + bias).astype(np.float32, copy=False)


def _layer_norm(value: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = value.mean(axis=-1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((value - mean) / np.sqrt(variance + eps) * weight + bias).astype(np.float32, copy=False)


def _gru_direction(sequence: np.ndarray, weights: ArrayMap, suffix: str, reverse: bool) -> np.ndarray:
    """One PyTorch GRU direction, batch-first and one layer only."""
    w_ih = weights[f"gru.weight_ih_l0{suffix}"]
    w_hh = weights[f"gru.weight_hh_l0{suffix}"]
    b_ih = weights[f"gru.bias_ih_l0{suffix}"]
    b_hh = weights[f"gru.bias_hh_l0{suffix}"]
    rows, steps, _ = sequence.shape
    hidden_size = w_hh.shape[1]
    hidden = np.zeros((rows, hidden_size), dtype=np.float32)
    outputs = np.zeros((rows, steps, hidden_size), dtype=np.float32)
    indices = range(steps - 1, -1, -1) if reverse else range(steps)
    for index in indices:
        input_gates = _linear(sequence[:, index], w_ih, b_ih)
        hidden_gates = _linear(hidden, w_hh, b_hh)
        in_reset, in_update, in_new = np.split(input_gates, 3, axis=1)
        hid_reset, hid_update, hid_new = np.split(hidden_gates, 3, axis=1)
        reset = _sigmoid(in_reset + hid_reset)
        update = _sigmoid(in_update + hid_update)
        new = np.tanh(in_new + reset * hid_new).astype(np.float32, copy=False)
        hidden = ((1.0 - update) * new + update * hidden).astype(np.float32, copy=False)
        outputs[:, index] = hidden
    return outputs


def logits(
    action: np.ndarray,
    status: np.ndarray,
    arg: np.ndarray,
    base: np.ndarray,
    meta: np.ndarray,
    weights: ArrayMap,
) -> np.ndarray:
    """Compute logits for arrays built by ``sequence_tensors``.

    The exported candidate is deliberately no-E5; its E5 branch sees a fixed
    zero scalar and therefore remains a learned constant contribution.
    """
    action = np.asarray(action, dtype=np.int64)
    status = np.asarray(status, dtype=np.int64)
    arg = np.asarray(arg, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    meta = np.asarray(meta, dtype=np.float32)
    positions = weights["position_embedding.weight"][np.arange(action.shape[1])][None, :, :]
    sequence = np.concatenate((
        weights["action_embedding.weight"][action],
        weights["status_embedding.weight"][status],
        np.broadcast_to(positions, (len(action),) + positions.shape[1:]),
        _gelu(_linear(arg, weights["arg_projection.0.weight"], weights["arg_projection.0.bias"])),
    ), axis=2).astype(np.float32, copy=False)
    forward = _gru_direction(sequence, weights, "", reverse=False)
    backward = _gru_direction(sequence, weights, "_reverse", reverse=True)
    encoded = np.concatenate((forward, backward), axis=2)
    last = np.clip((action != 0).sum(axis=1), 1, None) - 1
    history = encoded[np.arange(len(action)), last]

    base_block = _gelu(_linear(
        _layer_norm(base, weights["base.0.weight"], weights["base.0.bias"]),
        weights["base.1.weight"], weights["base.1.bias"],
    ))
    # LayerNorm over a one-dimensional zero E5 input yields its learned bias.
    e5_input = np.zeros((len(action), 1), dtype=np.float32)
    e5_block = _gelu(_linear(
        _layer_norm(e5_input, weights["e5.0.weight"], weights["e5.0.bias"]),
        weights["e5.1.weight"], weights["e5.1.bias"],
    ))
    meta_block = _gelu(_linear(meta, weights["meta.0.weight"], weights["meta.0.bias"]))
    hidden = _gelu(_linear(
        np.concatenate((history, base_block, e5_block, meta_block), axis=1),
        weights["head.0.weight"], weights["head.0.bias"],
    ))
    return _linear(hidden, weights["head.3.weight"], weights["head.3.bias"])


def probabilities(*args: object, **kwargs: object) -> np.ndarray:
    value = logits(*args, **kwargs)
    value = value - value.max(axis=1, keepdims=True)
    value = np.exp(value).astype(np.float32, copy=False)
    return (value / value.sum(axis=1, keepdims=True)).astype(np.float32, copy=False)
