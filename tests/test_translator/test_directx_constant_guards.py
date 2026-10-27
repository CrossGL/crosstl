"""Constant boolean guards select the branch software collectives validate."""

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import _codegen


def _source(guard_value):
    return f"""shader ConstantGuards {{
        StructuredBuffer<uint> inputWords @register(t0);
        RWStructuredBuffer<uint> outputWords @register(u1);
        compute {{
            @numthreads(32, 1, 1)
            void main(uvec3 global @gl_GlobalInvocationID) {{
                uint index = global.x;
                const bool has_mask = !{guard_value};
                uint value = inputWords[index];
                if (has_mask) {{
                    if (value == 0u) {{
                        outputWords[index] = 0u;
                        return;
                    }}
                }} else {{
                    value = value + 1u;
                }}
                outputWords[index] = WaveActiveSum(value);
            }}
        }}
    }}"""


def test_refuted_constant_guard_hides_divergent_early_return(tmp_path):
    generated = _codegen().generate_stage(parse(_source("true")), "compute")

    assert "WaveActiveSum" not in generated
    assert "GroupMemoryBarrierWithGroupSync();" in generated
    _compile(generated, tmp_path)


def test_taken_constant_guard_still_rejects_divergent_early_return():
    with pytest.raises(DirectXSoftwareSubgroupError) as failure:
        _codegen().generate_stage(parse(_source("false")), "compute")

    assert failure.value.reason == "early-return-unproven"


def test_constant_guard_written_later_is_not_folded():
    source = _source("true").replace(
        "const bool has_mask = !true;", "bool has_mask = !true; has_mask = true;"
    )

    with pytest.raises(DirectXSoftwareSubgroupError) as failure:
        _codegen().generate_stage(parse(source), "compute")

    assert failure.value.reason == "early-return-unproven"
