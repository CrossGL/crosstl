"""Module-scope ``var`` and ``var<private>`` variables are per-thread kernel locals on CUDA and HIP.

A module-scope CrossGL ``var`` or ``var<private>`` has one copy per invocation, like a ``static`` one. CUDA and HIP
output declared it as a namespace-scope C++ variable, ``int k3 = 3;``, which is a host variable: nvcc reports
``identifier "k3" is undefined in device code`` and hipcc ``reference to __host__ variable`` for a kernel that reads
or writes it (issue #4825).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import DevicePrivateGlobalUnsupportedError
from crosstl.translator.semantics.device_private_globals import (
    lower_device_private_globals,
)

BACKENDS = ["cuda", "hip"]

KERNEL = """
@compute
@workgroup_size(4, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read_write> dst: array<f32>
) {
    var i: u32 = gl_GlobalInvocationID.x;
    dst[i] = (lut[(i & 3)] + f32(k3));
}
"""

READ_ONLY = {
    "var": "var k3: i32 = 3;\nvar lut: array<f32, 4> = {1.0f, 2.0f, 3.0f, 4.0f};\n",
    "var<private>": (
        "var<private> k3: i32 = 3;\n"
        "var<private> lut: array<f32, 4> = {1.0f, 2.0f, 3.0f, 4.0f};\n"
    ),
}

WRITTEN = """
var counter: f32 = 1.0f;

fn bump() {
    counter = counter * 2.0f + 1.0f;
}

@compute
@workgroup_size(4, 1, 1)
fn k(
    @group(0) @binding(0) var<storage, read> fin: array<f32>,
    @group(0) @binding(1) var<storage, read_write> outp: array<f32>
) {
    var i: u32 = gl_GlobalInvocationID.x;
    var before: f32 = counter;
    counter = fin[i];
    bump();
    outp[i] = counter * 10.0f + before;
}
"""

SIZED_SCALARS = """
var<private> wide: u64;
var<private> small: f16;
var<private> lanes: array<u32, 3>;
var plain: vec3;

@compute
@workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> dst: array<u32>) {
    wide = wide + 1;
    lanes[1] = 2u;
    plain.x = 1.0;
    dst[0] = u32(wide) + lanes[1] + u32(plain.x) + u32(f32(small));
}
"""

POINTER = """
var<private> cursor: ptr<function, f32>;

@compute
@workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> dst: array<f32>) {
    dst[0] = *cursor;
}
"""

UNQUALIFIED_TEXTURE = """
sampler2D tex;

@compute
@workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> dst: array<f32>) {
    dst[0] = 1.0;
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "module_variables.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _kernel(code):
    """The text of the kernel ``k``."""
    start = re.search(r"__global__ void (?:__launch_bounds__\(\d+\) )?k\(", code).start()
    depth = 0
    for index in range(code.index("{", start), len(code)):
        depth += {"{": 1, "}": -1}.get(code[index], 0)
        if depth == 0:
            return code[start : index + 1]
    raise AssertionError("unterminated kernel")


def _compile(tmp_path, code, backend):
    """Compile ``code`` with the target's own compiler when it is installed."""
    if backend == "cuda" and shutil.which("nvcc"):
        (tmp_path / "out.cu").write_text(code)
        command = [shutil.which("nvcc"), "-std=c++17", "--ptx", "out.cu"]
    elif backend == "hip" and shutil.which("hipcc"):
        (tmp_path / "out.hip").write_text(code)
        command = [
            shutil.which("hipcc"),
            "-std=c++17",
            "--offload-device-only",
            "-c",
            "out.hip",
        ]
    else:
        return
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _namespace_scope(code, name):
    """Unindented declarations that name ``name``: the generators indent everything inside a function."""
    return [
        line
        for line in code.splitlines()
        if line[:1] not in ("", " ", "\t", "#", "/", "{", "}")
        and re.search(rf"\b{name}\b", line)
        and line.rstrip().endswith(";")
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("spelling", sorted(READ_ONLY))
def test_read_module_variables_are_kernel_locals(tmp_path, backend, spelling):
    code = _translate(tmp_path, READ_ONLY[spelling] + KERNEL, backend)
    assert _namespace_scope(code, "k3") == []
    assert _namespace_scope(code, "lut") == []
    kernel = _kernel(code)
    assert "int k3 = 3;" in kernel
    assert "float lut[4] = {1.0f, 2.0f, 3.0f, 4.0f};" in kernel
    assert kernel.index("int k3 = 3;") < kernel.index("dst[i] =")
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_written_module_variable_is_passed_to_helpers_by_reference(
    tmp_path, backend
):
    code = _translate(tmp_path, WRITTEN, backend)
    assert _namespace_scope(code, "counter") == []
    assert re.search(r"__device__ void bump\(float& counter\)", code)
    kernel = _kernel(code)
    assert kernel.index("float counter = 1.0f;") < kernel.index("float before")
    assert "bump(counter);" in kernel
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalars_in_every_crossgl_spelling_are_values(tmp_path, backend):
    program = _parse(SIZED_SCALARS)
    assert lower_device_private_globals(program, backend) == 4
    assert program.global_variables == []
    code = _translate(tmp_path, SIZED_SCALARS, backend)
    for name in ("wide", "small", "lanes", "plain"):
        assert _namespace_scope(code, name) == []
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_private_module_variable_that_is_not_a_value_is_refused(backend):
    with pytest.raises(DevicePrivateGlobalUnsupportedError, match="'cursor'"):
        lower_device_private_globals(_parse(POINTER), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_unqualified_resource_is_left_to_the_generator(backend):
    program = _parse(UNQUALIFIED_TEXTURE)
    assert lower_device_private_globals(program, backend) == 0
    assert [v.name for v in program.global_variables] == ["tex"]


@pytest.mark.parametrize("target", ["opengl", "directx", "metal", "vulkan", "slang"])
def test_other_targets_keep_module_variables(target):
    program = _parse(READ_ONLY["var<private>"] + KERNEL)
    assert lower_device_private_globals(program, target) == 0
    assert [v.name for v in program.global_variables] == ["k3", "lut"]
