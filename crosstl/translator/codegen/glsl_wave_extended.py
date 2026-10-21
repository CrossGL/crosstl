"""GLSL spellings of the extended CrossGL wave operations (``crosstl.translator.wave_operations``).

The scan operations already live in the GLSL generator's tables; these are the cluster and ballot operations.
"""

from __future__ import annotations

from ..wave_operations import WAVE_BALLOT_BIT_OPERATIONS, WAVE_CLUSTERED_OPERATIONS

_CLUSTERED = "#extension GL_KHR_shader_subgroup_clustered : require"
_BALLOT = "#extension GL_KHR_shader_subgroup_ballot : require"

GLSL_CLUSTERED_FUNCTIONS = {
    "WaveClusteredSum": "subgroupClusteredAdd",
    "WaveClusteredProduct": "subgroupClusteredMul",
    "WaveClusteredMin": "subgroupClusteredMin",
    "WaveClusteredMax": "subgroupClusteredMax",
    "WaveClusteredBitAnd": "subgroupClusteredAnd",
    "WaveClusteredBitOr": "subgroupClusteredOr",
    "WaveClusteredBitXor": "subgroupClusteredXor",
}

GLSL_BALLOT_BIT_FUNCTIONS = {
    "WaveBallotBitCount": "subgroupBallotBitCount",
    "WaveBallotInclusiveBitCount": "subgroupBallotInclusiveBitCount",
    "WaveBallotExclusiveBitCount": "subgroupBallotExclusiveBitCount",
    "WaveBallotFindLSB": "subgroupBallotFindLSB",
    "WaveBallotFindMSB": "subgroupBallotFindMSB",
    "WaveBallotBitExtract": "subgroupBallotBitExtract",
    "WaveInverseBallot": "subgroupInverseBallot",
}

assert set(GLSL_CLUSTERED_FUNCTIONS) == set(WAVE_CLUSTERED_OPERATIONS)
assert set(GLSL_BALLOT_BIT_FUNCTIONS) == set(WAVE_BALLOT_BIT_OPERATIONS)

GLSL_EXTENDED_WAVE_ARITIES = {
    **{name: 2 for name in GLSL_CLUSTERED_FUNCTIONS},
    **WAVE_BALLOT_BIT_OPERATIONS,
}

GLSL_EXTENDED_WAVE_DIRECT_MAPPINGS = {
    **GLSL_CLUSTERED_FUNCTIONS,
    **GLSL_BALLOT_BIT_FUNCTIONS,
}

GLSL_EXTENDED_WAVE_EXTENSIONS = {
    **{name: _CLUSTERED for name in GLSL_CLUSTERED_FUNCTIONS},
    **{name: _BALLOT for name in GLSL_BALLOT_BIT_FUNCTIONS},
}

GLSL_CLUSTERED_NUMERIC_OPERATIONS = frozenset(
    name
    for name, operator in WAVE_CLUSTERED_OPERATIONS.items()
    if operator in {"sum", "product", "min", "max"}
)
GLSL_CLUSTERED_BITWISE_OPERATIONS = frozenset(
    set(WAVE_CLUSTERED_OPERATIONS) - GLSL_CLUSTERED_NUMERIC_OPERATIONS
)
