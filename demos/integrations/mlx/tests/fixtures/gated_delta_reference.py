"""Independent array reference for sequential gated-delta forward and backward."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    batch: int
    key_heads: int
    value_heads: int
    checkpoint: int
    length: int
    dtype: str = "float32"
    uniform: bool = False

    @property
    def entry(self):
        scalar = "float" if self.dtype == "float32" else "float16_t"
        return f"seq_gated_delta_vjp_{scalar}_128_128_{self.key_heads}_{self.value_heads}_{self.checkpoint}"


def forward(parameters, checkpoint):
    q, k, v, g, beta, initial = (value.astype("float64") for value in parameters)
    head_map = np.arange(v.shape[2]) // (v.shape[2] // q.shape[2])
    q, k = q[:, :, head_map], k[:, :, head_map]
    state = initial.copy()
    states, checkpoints, outputs = [], [], []
    for t in range(q.shape[1]):
        states.append(state.copy())
        if t % checkpoint == 0:
            checkpoints.append(state.astype("<f4"))
        state = state * g[:, t, :, None, None]
        prediction = (state * k[:, t, :, None, :]).sum(axis=-1)
        update = beta[:, t, :, None] * (v[:, t] - prediction)
        state += update[:, :, :, None] * k[:, t, :, None, :]
        outputs.append((state * q[:, t, :, None, :]).sum(axis=-1))
    cache = np.stack(checkpoints, axis=2)
    return np.stack(outputs, axis=1), state, states, cache


def backward(parameters, cotangents, states):
    q, k, v, g, beta, _initial = (value.astype("float64") for value in parameters)
    co, ch = (value.astype("float64") for value in cotangents)
    repeats = v.shape[2] // q.shape[2]
    head_map = np.arange(v.shape[2]) // repeats
    dq, dk = np.zeros_like(q), np.zeros_like(k)
    dv, dg, db = np.zeros_like(v), np.zeros_like(g), np.zeros_like(beta)
    q, k = q[:, :, head_map], k[:, :, head_map]
    adjoint = ch.copy()
    for t in reversed(range(q.shape[1])):
        previous = states[t]
        decayed = previous * g[:, t, :, None, None]
        prediction = (decayed * k[:, t, :, None, :]).sum(axis=-1)
        adjoint += co[:, t, :, :, None] * q[:, t, :, None, :]
        weight = (adjoint * k[:, t, :, None, :]).sum(axis=-1)
        residual = v[:, t] - prediction
        update = beta[:, t, :, None] * residual
        state = decayed + update[:, :, :, None] * k[:, t, :, None, :]
        query_gradient = (co[:, t, :, :, None] * state).sum(axis=2)
        key_gradient = (
            beta[:, t, :, None, None]
            * (residual[:, :, :, None] * adjoint - weight[:, :, :, None] * decayed)
        ).sum(axis=2)
        dq[:, t] = query_gradient.reshape(q.shape[0], dq.shape[2], repeats, 128).sum(
            axis=2
        )
        dk[:, t] = key_gradient.reshape(q.shape[0], dk.shape[2], repeats, 128).sum(
            axis=2
        )
        dv[:, t] = beta[:, t, :, None] * weight
        db[:, t] = (weight * residual).sum(axis=-1)
        adjoint -= (
            beta[:, t, :, None, None] * weight[:, :, :, None] * k[:, t, :, None, :]
        )
        dg[:, t] = (adjoint * previous).sum(axis=(2, 3))
        adjoint *= g[:, t, :, None, None]
    return dq, dk, dv, dg, db, adjoint


def dataset(case):
    rng = np.random.default_rng(9471 + case.length + case.value_heads)
    qshape = (case.batch, case.length, case.key_heads, 128)
    vshape = (case.batch, case.length, case.value_heads, 128)
    hshape = (case.batch, case.value_heads, 128, 128)

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(case.dtype)

    q, k, v = normal(qshape, 0.1), normal(qshape, 0.1), normal(vshape, 0.1)
    k = (k / np.linalg.norm(k.astype("float64"), axis=-1, keepdims=True)).astype(
        case.dtype
    )
    g, beta = (rng.uniform(0.1, 0.9, vshape[:-1]).astype(case.dtype) for _ in range(2))
    initial = normal(hshape, 0.1).astype("<f4")
    co, ch = normal(vshape, 0.01), normal(hshape, 0.01).astype("<f4")
    if case.uniform:
        for value, constant in zip(
            (q, k, v, g, beta, initial, co, ch),
            (1 / 16, 1 / 32, 1 / 8, 1 / 2, 1 / 4, 1 / 64, 1 / 32, 1 / 128),
        ):
            value.fill(constant)
    parameters, cotangents = (q, k, v, g, beta, initial), (co, ch)
    output, final, states, cache = forward(parameters, case.checkpoint)
    gradients = backward(parameters, cotangents, states)
    buffers = [q, k, v, g, beta, co, ch, cache, np.array([case.length], dtype="<i4")]
    buffers += [np.zeros(value.shape, dtype="<f4") for value in gradients[:-1]]
    buffers.append(np.full(gradients[-1].shape, -999, dtype="<f4"))
    return parameters, cotangents, buffers, gradients, output, final
