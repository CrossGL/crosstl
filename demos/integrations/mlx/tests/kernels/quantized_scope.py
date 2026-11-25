"""How the quantized.metal gates divide its 2,052 entries.

Every matrix gate's contract records its own entries, each other matrix
gate's entry count and the kernels no gate covers yet. A new gate adds
itself here and regenerates the other gates' contract scopes.
"""

UNIT_ENTRY_COUNT = 2052

# Entries per gate, keyed by the name its siblings' scopes give it.
GATE_ENTRY_COUNTS = {
    # test_current_quantized.py: affine_quantize and affine_dequantize.
    "quantize": 108,
    # test_current_quantized_qmv.py: affine_gather_qmv_fast and
    # affine_qmv_fast.
    "qmvFast": 162,
    # test_current_quantized_qmv_unaligned.py: affine_qmv and
    # affine_gather_qmv.
    "qmvUnaligned": 162,
    # test_current_quantized_qmv_wide.py: affine_qmv_wide.
    "qmvWide": 432,
    # test_current_quantized_qvm.py: affine_qvm and affine_gather_qvm.
    "qvm": 162,
    # test_current_quantized_qvm_split_k.py: affine_qvm_split_k.
    "qvmSplitK": 108,
    # test_current_quantized_qmm_t.py: affine_qmm_t.
    "qmmT": 216,
    # test_current_quantized_qmm_n.py: affine_qmm_n.
    "qmmN": 108,
    # test_current_quantized_qmm_t_splitk.py: affine_qmm_t_splitk.
    "qmmTSplitK": 108,
    # test_current_quantized_gather_qmm.py: affine_gather_qmm_t and
    # affine_gather_qmm_n.
    "gatherQmm": 162,
    # test_current_quantized_qmv_quad.py: affine_qmv_quad.
    "qmvQuad": 216,
    # test_current_quantized_gather_qmm_rhs.py: affine_gather_qmm_rhs_nt and
    # affine_gather_qmm_rhs_nn.
    "gatherQmmRhs": 108,
}

# Kernels of quantized.metal no gate covers yet: every kernel is gated.
EXCLUDED_KERNELS = ()
EXCLUDED_REASON = "every quantized.metal kernel is gated"


def shared_scope(gate):
    """The scope entries every gate's contract shares, as ``gate`` sees them."""
    assert gate in GATE_ENTRY_COUNTS, gate
    others = {
        f"{name}GateEntryCount": count
        for name, count in GATE_ENTRY_COUNTS.items()
        if name != gate
    }
    return {
        **others,
        "excludedEntries": {
            "kernels": list(EXCLUDED_KERNELS),
            "count": UNIT_ENTRY_COUNT - sum(GATE_ENTRY_COUNTS.values()),
            "reason": EXCLUDED_REASON,
        },
    }
