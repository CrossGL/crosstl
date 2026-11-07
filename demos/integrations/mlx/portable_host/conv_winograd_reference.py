"""A certified reference for conv.cpp's Winograd route.

``winograd_conv_2D_gpu`` stores each stage in the element type: the weight
transform ``G g G^T`` with ``G``'s non-dyadic float32 constants, the input
transform ``B^T d B``, the 64 batched GEMMs that accumulate in float32, and
the output transform ``A^T m A``. Their sums round in float32 and every
stored stage rounds again, so no exact reference exists. This module follows
the stages instead, each from the previous stage's rounded values:

* ``S`` is the stage's exact value (float64 here, whose error is far below
  the margins) from the rounded inputs ``r`` of the stage;
* ``delta`` bounds how far the kernel's float32 value can be from ``S``:
  ``gamma_n * |stage|(|r| + eps)``, the forward error of an n-term dot
  product in any order, fused or not, plus the input uncertainty ``eps``
  carried through the stage's absolute values and, for ``G``, the float32
  rounding of its constants;
* the kernel then stores a value in ``[round(S - delta), round(S + delta)]``,
  since rounding is monotone; ``r = round(S)`` and the half-width of that
  interval, ``eps``, carry to the next stage.

The input and output tables are dyadic. Where every term of a product is a
multiple of a quantum and their magnitudes sum below 2^24 quanta, float32
computes it exactly in any order, so the stage carries no margin there; this
matters at storage ties, which coarse dyadic inputs reach often. The host's
output must lie in the last stage's interval. For float16 and bfloat16,
rounding boundaries rarely fall inside ``delta``, so most stored values are
certain. float32 storage keeps every arithmetic margin, which the transforms
amplify: half the outputs are certain to about 2 x 10^-3 of their magnitude.
A misplaced tile, channel block or batch row exceeds either by far.
"""

from fractions import Fraction

U32 = 2.0**-24
# Unit roundoff of each element type's storage.
UNIT = {"float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-8}
# Each transform is two 8-term products of 8 x 8 SIMD-group matrices.
TERMS = 8
# float64 evaluation of S: far below every float32 margin, bounded generously.
FLOAT64_MARGIN = 2.0**-40
# conv.metal's float tables, as exact rationals; the kernels zero-fill the
# columns a row leaves out.
WT_TRANSFORM = (
    (1, 0, 0),
    ("-2/9", "-2/9", "-2/9"),
    ("-2/9", "2/9", "-2/9"),
    ("1/90", "1/45", "2/45"),
    ("1/90", "-1/45", "2/45"),
    ("32/45", "16/45", "8/45"),
    ("32/45", "-16/45", "8/45"),
    (0, 0, 1),
)
IN_TRANSFORM = (
    (1, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, -1, "1/2", "-1/2", 2, -2, -1),
    ("-21/4", 1, 1, "1/4", "1/4", 4, 4, 0),
    (0, "-17/4", "17/4", "-5/2", "5/2", "-5/2", "5/2", "21/4"),
    ("21/4", "-17/4", "-17/4", "-5/4", "-5/4", -5, -5, 0),
    (0, 1, -1, 2, -2, "1/2", "-1/2", "-21/4"),
    (-1, 1, 1, 1, 1, 1, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 1),
)
OUT_TRANSFORM = (
    (1, 0, 0, 0, 0, 0),
    (1, 1, 1, 1, 1, 1),
    (1, -1, 1, -1, 1, -1),
    (1, 2, 4, 8, 16, 32),
    (1, -2, 4, -8, 16, -32),
    (1, "1/2", "1/4", "1/8", "1/16", "1/32"),
    (1, "-1/2", "1/4", "-1/8", "1/16", "-1/32"),
    (0, 0, 0, 0, 0, 1),
)
# The finite maxima of each storage type.
MAXIMA = {"float32": 3.4028234663852886e38, "float16": 65504.0, "bfloat16": 3.3895e38}


def gamma(n):
    """Higham's gamma_n for float32: the relative bound of n roundings."""
    return n * U32 / (1 - n * U32)


def _table(np, rows):
    return np.array([[float(Fraction(value)) for value in row] for row in rows])


def round_to(np, values, dtype):
    """Round float64 ``values`` to ``dtype`` with ties to even, as float64."""
    single = values.astype(np.float32)
    if dtype == "float32":
        return single.astype(np.float64)
    if dtype == "float16":
        return values.astype(np.float16).astype(np.float64)
    bits = single.view(np.uint32).astype(np.uint64)
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return bits.astype(np.uint32).view(np.float32).astype(np.float64)


def _store(np, exact, delta, dtype):
    """The stored value and its uncertainty from an exact value and margin."""
    if np.abs(exact).max(initial=0.0) + delta.max(initial=0.0) >= MAXIMA[dtype] / 2:
        raise ValueError("Winograd reference stage exceeds its storage range")
    if dtype == "float32":
        # float32 storage rounds the float32 sum it already holds: no flip
        # beyond the arithmetic margin, which the bound already covers.
        stored = round_to(np, exact, dtype)
        return stored, delta + np.abs(stored - exact)
    stored = round_to(np, exact, dtype)
    low = round_to(np, exact - delta, dtype)
    high = round_to(np, exact + delta, dtype)
    return stored, np.maximum(stored - low, high - stored)


# The quantum exponent of a zero, which constrains no sum.
NO_QUANTUM = 1 << 20
SIGNIFICAND = 24


def quantum_exponents(np, values):
    """The exponent of each value's lowest set bit; NO_QUANTUM for zeros."""
    fraction, exponent = np.frexp(values)
    significand = np.ldexp(np.abs(fraction), 53).astype(np.int64)
    lowest = significand & -significand
    _, low = np.frexp(lowest.astype(np.float64))
    return np.where(values == 0, NO_QUANTUM, exponent - 53 + low - 1)


def _exact_sums(np, quanta, bound):
    """Whether sums of terms with these minimum quanta stay exact in float32.

    Every term and partial sum is a multiple of ``2**quanta`` at most
    ``bound`` in magnitude, so below ``2**(quanta + 24)`` each is a float32
    in any order, fused or not.
    """
    free = quanta >= NO_QUANTUM // 2
    scaled = np.ldexp(bound, -np.where(free, 0, quanta + SIGNIFICAND))
    return free | (scaled < 1)


def _exact_product(np, left, values, right, right_first):
    """Where ``left @ values @ right`` is exact in float32 in the kernel's order.

    The tables are dyadic, and the kernel keeps the first product in float32
    SIMD-group matrices before the second.
    """
    q_left, q_right = quantum_exponents(np, left), quantum_exponents(np, right)
    q_values = quantum_exponents(np, values)
    if right_first:
        # values[..., a, b, c] * right[b, s]: (..., a, s, c).
        terms = q_values[..., :, :, None, :] + q_right[:, :, None]
        first_q = terms.min(axis=-3)
        first = np.einsum("...abc,bs->...asc", values, right)
        first_bound = np.einsum("...abc,bs->...asc", np.abs(values), np.abs(right))
        first_exact = _exact_sums(np, first_q, first_bound)
        # left[r, a] * first[..., a, s, c]: (..., r, s, c).
        terms = q_left[:, :, None, None] + first_q[..., None, :, :, :]
        second_q = terms.min(axis=-3)
        bound = np.einsum("ra,...asc->...rsc", np.abs(left), np.abs(first))
        exact = first_exact.all(axis=-3)[..., None, :, :]
    else:
        # left[r, a] * values[..., a, b, c]: (..., r, b, c).
        terms = q_left[:, :, None, None] + q_values[..., None, :, :, :]
        first_q = terms.min(axis=-3)
        first = np.einsum("ra,...abc->...rbc", left, values)
        first_bound = np.einsum("ra,...abc->...rbc", np.abs(left), np.abs(values))
        first_exact = _exact_sums(np, first_q, first_bound)
        # first[..., r, b, c] * right[b, s]: (..., r, s, c).
        terms = first_q[..., :, :, None, :] + q_right[:, :, None]
        second_q = terms.min(axis=-3)
        bound = np.einsum("...rbc,bs->...rsc", np.abs(first), np.abs(right))
        exact = first_exact.all(axis=-2)[..., :, None, :]
    return exact & _exact_sums(np, second_q, bound)


def _transform(
    np, left, values, right, eps, dtype, constant_error=0.0, right_first=None
):
    """``left @ values @ right`` over the last two tile axes, with its bound.

    ``values`` holds the rounded inputs in ``(..., 8, 8, channels)`` and ``eps``
    their uncertainty. For dyadic tables ``right_first`` names the kernel's
    product order, and outputs exact in float32 from certain inputs carry no
    margin. Returns the stored values and their uncertainty.
    """
    exact = np.einsum("ra,...abc,bs->...rsc", left, values, right)
    magnitude = np.einsum(
        "ra,...abc,bs->...rsc", np.abs(left), np.abs(values) + eps, np.abs(right)
    )
    carried = np.einsum("ra,...abc,bs->...rsc", np.abs(left), eps, np.abs(right))
    arithmetic = 2 * gamma(TERMS) + gamma(TERMS) ** 2 + constant_error
    delta = carried + (arithmetic + FLOAT64_MARGIN) * magnitude
    if right_first is not None:
        certain = (carried == 0) & _exact_product(np, left, values, right, right_first)
        delta = np.where(certain, 0.0, delta)
    return _store(np, exact, delta, dtype)


def stages(np, x, w, pad, dtype):
    """The certified output interval of the Winograd route.

    ``x`` holds the (N, H, W, C) input and ``w`` the (O, 3, 3, C) weights as
    float64 values of ``dtype``; ``pad`` is the symmetric padding. Returns
    ``(value, uncertainty)``, each (N, oH, oW, O).
    """
    n, h, wd, c = x.shape
    o = w.shape[0]
    out_h, out_w = h + 2 * pad[0] - 2, wd + 2 * pad[1] - 2
    tiles_h, tiles_w = -(-out_h // 6), -(-out_w // 6)
    g = _table(np, WT_TRANSFORM)
    b = _table(np, IN_TRANSFORM)
    a = _table(np, OUT_TRANSFORM)
    # G's constants are float32 roundings of the rationals, which float64
    # holds within its own rounding: two factors, each within u32 + 2 u64.
    constant_error = 2.02 * U32
    # The weight transform reads each filter's taps as g[kh][kw] per channel:
    # (C, 3, 3, O) puts them on the tile axes, and the result moves to
    # filt_wg's (8, 8, C, O).
    weights = w.transpose(3, 1, 2, 0)
    filters, filter_eps = (
        np.moveaxis(value, 0, 2)
        for value in _transform(
            np, g, weights, g.T, np.zeros_like(weights), dtype, constant_error
        )
    )
    padded = np.zeros((n, 6 * tiles_h + 2, 6 * tiles_w + 2, c))
    padded[:, pad[0] : pad[0] + h, pad[1] : pad[1] + wd] = x
    windows = np.lib.stride_tricks.sliding_window_view(padded, (8, 8), axis=(1, 2))
    # (N, tiles_h, tiles_w, 8, 8, C)
    tiles = np.moveaxis(windows[:, ::6, ::6], 3, -1)
    # (Bt * I) * B.
    inputs, input_eps = _transform(
        np, b.T, tiles, b, np.zeros_like(tiles), dtype, right_first=False
    )

    # The 64 batched GEMMs: each domain position sums C products in float32.
    def gemm(left, right):
        # (N, th, tw, 8, 8, C) x (8, 8, C, O) as 64 batched matrix products.
        batch = left.shape[:3]
        rows = np.moveaxis(left.reshape(-1, 64, c), 1, 0)
        product = np.matmul(rows, right.reshape(64, c, o))
        return np.moveaxis(product, 0, 1).reshape(*batch, 8, 8, o)

    exact = gemm(inputs, filters)
    rounded_magnitude = gemm(np.abs(inputs), np.abs(filters))
    carried = (
        gemm(input_eps, np.abs(filters))
        + gemm(np.abs(inputs), filter_eps)
        + gemm(input_eps, filter_eps)
    )
    magnitude = rounded_magnitude + carried
    products, product_eps = _store(
        np,
        exact,
        carried + (gamma(c) + FLOAT64_MARGIN) * magnitude,
        dtype,
    )
    # Bt * (O * B), whose B is the A above.
    outputs, output_eps = _transform(
        np, a.T, products, a, product_eps, dtype, right_first=True
    )

    # (N, tiles_h, tiles_w, 6, 6, O) -> (N, oH, oW, O)
    def assemble(values):
        values = values.transpose(0, 1, 3, 2, 4, 5)
        values = values.reshape(n, tiles_h * 6, tiles_w * 6, o)
        return values[:, :out_h, :out_w]

    return assemble(outputs), assemble(output_eps)


def contains(np, actual, value, uncertainty):
    """Whether every output lies within its certified interval."""
    return bool(np.all(np.abs(actual.astype(np.float64) - value) <= uncertainty))
