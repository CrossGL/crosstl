"""Module- and stage-scope workgroup variables become SPIR-V Workgroup storage, one copy per workgroup.

HLSL ``groupshared`` and CrossGL stage-scope ``shared`` arrays used to be emitted as Private variables, so every
invocation had its own copy and a tiled kernel read values no other invocation wrote. Execution evidence is in the
``h_groupshared`` and ``g_shared_exchange`` conformance cases.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import UnsupportedSPIRVFeatureError

HLSL = """
RWStructuredBuffer<int> o : register(u1);
groupshared int tile[64];
[numthreads(64, 1, 1)]
void main(uint l : SV_GroupIndex) {
    tile[l] = int(l);
    GroupMemoryBarrierWithGroupSync();
    o[l] = tile[63u - l];
}
"""

CROSSGL = """
shader SharedProbe {
    compute probe {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        shared int tile[64];
        buffer int* data;
        void main() {
            uint l = gl_LocalInvocationIndex;
            tile[l] = data[l];
            barrier();
            data[l] = tile[63u - l];
        }
    }
}
"""


def storage_class_of(spirv, name):
    names = {
        line.split()[1]: line.split('"')[1]
        for line in spirv.splitlines()
        if line.startswith("OpName ")
    }
    identifier = next(i for i, value in names.items() if value == name)
    variable = next(
        line
        for line in spirv.splitlines()
        if line.startswith(f"{identifier} = OpVariable")
    )
    return variable.split()[-1]


@pytest.mark.parametrize("suffix,source", [(".hlsl", HLSL), (".cgl", CROSSGL)])
def test_workgroup_array_is_workgroup_storage(tmp_path, suffix, source):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source)
    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert storage_class_of(spirv, "tile") == "Workgroup"


def test_workgroup_variable_initializer_is_refused(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(CROSSGL.replace("shared int tile[64];", "shared int tile = 3;"))
    with pytest.raises(
        UnsupportedSPIRVFeatureError, match="cannot take an initializer"
    ):
        crosstl.translate(str(path), backend="vulkan", format_output=False)
