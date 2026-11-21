"""A translation whose host chooses its workgroup size is validated the way its host compiles it."""

import pytest

import crosstl
from crosstl.translator.codegen.host_workgroup_size_ids import (
    HOST_WORKGROUP_SIZE_MACROS,
)
from tools.coverage import validators

# A Metal kernel that reads its threadgroup size but fixes none, so the host chooses it at dispatch time.
KERNEL = """#include <metal_stdlib>
using namespace metal;
kernel void rotate(device float* data [[buffer(0)]],
                   uint gid [[thread_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint count [[threads_per_threadgroup]]) {
    threadgroup float tile[256];
    tile[lid] = data[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data[gid] = tile[(lid + 1) % count];
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "kernel.metal"
    path.write_text(KERNEL)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_only_a_host_sized_translation_gets_a_size_of_one_in_each_dimension():
    guarded = (
        "#elif defined(CROSSGL_WORKGROUP_SIZE_X) && defined(CROSSGL_WORKGROUP_SIZE_Y)\n"
    )
    assert validators.host_workgroup_defines(guarded) == [
        argument
        for macro in HOST_WORKGROUP_SIZE_MACROS
        for argument in ("-D", f"{macro}=1")
    ]
    assert validators.host_workgroup_defines("layout(local_size_x = 64) in;\n") == []


def test_a_host_sized_glsl_translation_defines_the_size_after_its_version():
    source = "#version 450 core\n#if defined(CROSSGL_WORKGROUP_SIZE_X)\n#endif\nvoid main() {}\n"
    defines = "".join(f"#define {macro} 1\n" for macro in HOST_WORKGROUP_SIZE_MACROS)
    assert validators.host_sized_glsl(source) == (
        f"#version 450 core\n{defines}#line 2\n" + source.split("\n", 1)[1]
    )
    fixed = "#version 450 core\nlayout(local_size_x = 64) in;\n"
    assert validators.host_sized_glsl(fixed) == fixed


def test_dxc_is_given_the_size_the_host_defines(monkeypatch, tmp_path):
    source = _translate(tmp_path, "directx")
    assert "defined(CROSSGL_WORKGROUP_SIZE_X)" in source
    commands = []

    def run(arguments, scratch, *args, **kwargs):
        commands.append(list(arguments))
        return validators.Verdict(True, "")

    monkeypatch.setattr(validators, "run", run)
    validators.validate("hlsl", source)
    [arguments] = commands
    for macro in HOST_WORKGROUP_SIZE_MACROS:
        at = arguments.index(f"{macro}=1")
        assert arguments[at - 1] == "-D"


@pytest.mark.skipif(
    not validators.available("glsl"), reason="glslangValidator is not installed"
)
def test_glslang_accepts_a_host_sized_glsl_translation(monkeypatch, tmp_path):
    source = _translate(tmp_path, "opengl")
    assert validators.validate("glsl", source).accepted
    # Without the host's size the source stops at its #error.
    monkeypatch.setattr(validators, "host_sized_glsl", lambda source: source)
    rejected = validators.validate("glsl", source)
    assert not rejected.accepted
    assert "the host chooses this workgroup size" in rejected.log
