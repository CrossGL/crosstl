"""Mojo output: names a function's own variables hide, and workgroup arrays declared in a kernel body.

Mojo has no global variables, so a helper that reads a module-scope value takes it as an argument
(crosstl/translator/codegen/mojo_kernel_arguments.py). A parameter or local of the same name hides that value only
where it is in scope, a local named like a function is no call of it, and a call made where the caller's own
variable hides the value its callee reads is refused rather than passed the wrong variable. A ``var<workgroup>``
array in a compute entry is the block's ``stack_allocation``; one in a helper is refused, as is a member access on a
value of unknown type whose member is both a swizzle and a structure field.
"""

from __future__ import annotations

import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.refusals import MojoRefusal
from tests.test_translator.test_codegen.test_mojo_codegen import find_mojo_compiler

_ENTRY = """@compute
@host_workgroup_size
fn probe(@group(0) @binding(0) var<storage, read_write> fin: array<float>, @group(0) @binding(1) var<storage, read_write> outp: array<float>) {{
    int i = gl_GlobalInvocationID.x;
{body}
}}
"""


def _mojo(source):
    return MojoCodeGen().generate(parse(source))


def _flat(code):
    return " ".join(code.split())


SHADOWING = """float scale = 2.0;

float twice(float x) {
    return (x * scale);
}

float apply(float scale) {
    var twice = (scale + 1.0);
    return (twice * 4.0);
}

""" + _ENTRY.format(
    body="""    var r = (twice(fin[i]) + apply(fin[i]));
    if ((fin[i] >= 0.0)) {
        float scale = 8.0;
        r += (scale * 64.0);
    }
    outp[i] = (r + (scale * 4096.0));"""
)


def test_a_parameter_and_a_local_named_like_a_function_take_no_module_value():
    mojo = _flat(_mojo(SHADOWING))
    assert "def twice(x: Float32, mut scale: Float32) -> Float32:" in mojo
    assert "def apply(scale: Float32) -> Float32:" in mojo
    # apply's local 'twice' is no call of twice(), so apply takes no 'scale' argument.
    assert "apply(fin[Int(i)])" in mojo
    assert "twice(fin[Int(i)], scale)" in mojo


def test_a_call_where_the_callers_variable_hides_the_module_value_is_refused():
    source = """float scale = 2.0;

float twice(float x) {
    return (x * scale);
}

float apply(float scale) {
    return twice(scale);
}

""" + _ENTRY.format(body="    outp[i] = apply(fin[i]);")
    with pytest.raises(
        MojoRefusal, match="'apply' calls 'twice' where its own 'scale'"
    ):
        _mojo(source)


def test_a_module_value_read_outside_a_nested_local_reaches_the_helper():
    source = """float scale = 2.0;

float pick(float x) {
    if (x > 0.0) {
        float scale = 3.0;
        x = x * scale;
    }
    return x + scale;
}

""" + _ENTRY.format(body="    outp[i] = pick(fin[i]);")
    mojo = _flat(_mojo(source))
    assert "def pick(var x: Float32, mut scale: Float32) -> Float32:" in mojo
    assert "pick(fin[Int(i)], scale)" in mojo


WORKGROUP_LOCAL = _ENTRY.format(body="""    var<workgroup> tile: array<float, 2>;
    tile[0] = fin[i];
    workgroupBarrier();
    outp[i] = tile[0];""")


def test_a_workgroup_array_in_the_kernel_body_is_a_shared_stack_allocation():
    mojo = _mojo(WORKGROUP_LOCAL)
    assert (
        "var tile = stack_allocation[2, Float32, address_space=AddressSpace.SHARED]()"
        in mojo
    )
    assert "from std.memory import stack_allocation" in mojo
    assert "from max.gpu.memory import AddressSpace" in mojo
    assert "StaticTuple" not in mojo


def test_a_workgroup_array_in_a_helper_is_refused():
    source = """float first(float v) {
    var<workgroup> tile: array<float, 2>;
    tile[0] = v;
    return tile[0];
}

""" + _ENTRY.format(body="    outp[i] = first(fin[i]);")
    with pytest.raises(
        MojoRefusal, match="workgroup-shared array 'tile' is declared in a helper"
    ):
        _mojo(source)


def test_an_unknown_value_whose_member_is_a_field_and_a_swizzle_is_refused():
    source = """struct Pair {
    float a;
    float b;
}

""" + _ENTRY.format(
        body="""    var p = Pair(fin[i], 1.0);
    var q = p.copy();
    outp[i] = q.a;"""
    )
    with pytest.raises(
        MojoRefusal, match="'a' could be a structure field or a vector swizzle"
    ):
        _mojo(source)


@pytest.mark.parametrize("source", [SHADOWING, WORKGROUP_LOCAL])
def test_the_mojo_compiler_accepts_the_kernels(tmp_path, source):
    mojo = find_mojo_compiler()
    path = tmp_path / "probe.mojo"
    path.write_text(_mojo(source))
    command = [mojo, "build", "--emit", "llvm", "--disable-warnings", str(path)]
    result = subprocess.run(
        [*command, "-o", str(tmp_path / "probe.ll")],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
