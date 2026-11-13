"""How the fp_quantized.metal gates divide its 518 entries.

Every gate's contract records its own entries, each other gate's entry
count and the kernels no gate covers yet. A new gate adds itself here and
regenerates the other gates' contract scopes.
"""

UNIT_ENTRY_COUNT = 518

# Entries per gate, keyed by the name its siblings' scopes give it.
GATE_ENTRY_COUNTS = {
    # test_current_fp_quantized.py: fp_quantize, fp_dequantize and
    # fp_quantize_dequantize.
    "quantize": 36,
    # test_current_fp_quantized_qmv.py: fp_qmv and fp_gather_qmv.
    "qmv": 54,
    # test_current_fp_quantized_qmv_fast.py: fp_qmv_fast and
    # fp_gather_qmv_fast.
    "qmvFast": 66,
    # test_current_fp_quantized_qvm.py: fp_qvm and fp_gather_qvm.
    "qvm": 54,
    # test_current_fp_quantized_qmv_quad.py: fp_qmv_quad.
    "qmvQuad": 36,
    # test_current_fp_quantized_qmv_wide.py: fp_qmv_wide.
    "qmvWide": 72,
}

# Kernels of fp_quantized.metal no gate covers yet.
EXCLUDED_KERNELS = (
    "fp_qmm_n",
    "fp_gather_qmm_n",
    "fp_qvm_split_k",
    "fp_qmm_t_splitk",
    "fp_gather_qmm_t",
    "fp_qmm_t",
    "fp_gather_qmm_rhs",
)
EXCLUDED_REASON = (
    "the split-K vector-matrix and the matrix-matrix kernels are not gated yet"
)


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
