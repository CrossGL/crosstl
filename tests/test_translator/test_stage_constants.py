"""Constants declared at stage scope, and MSL-style ``constant`` values at program scope, stay constants.

A stage-scope ``const``, ``static const`` or ``constant`` declaration was dropped by the GLSL and HLSL generators
(``use of undeclared identifier``) and declared as a uniform by the Slang one. A program-scope ``constant`` value was
a uniform in Slang, a host variable that device code cannot read in CUDA and HIP, and a WGSL ``var<private>``, whose
initializer may not read another module-scope variable.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.stage_private_variables import (
    hoist_stage_private_variables,
)

STAGE_CONSTANTS = """
shader main {
    constant int base = 3;
    compute {
        const int E = 5;
        constant int F[2] = {6, 7};
        static const float K = 0.5;
        const int H = E * base + 1;
        RWStructuredBuffer<int> out_ @buffer(0);
        int scale(int v) { return v * E + F[1] + int(K) + H; }
        @numthreads(1, 1, 1)
        void main(uint3 gid @gl_GlobalInvocationID) {
            out_[gid.x] = scale(int(gid.x)) + F[0] + E;
        }
    }
}
"""

ENTRY_CONSTANTS = """
shader main {
    compute {
        const int E = 5;
        constant int F = 6;
        static const int G = 7;
        RWStructuredBuffer<int> out_ @buffer(0);
        @numthreads(1, 1, 1)
        void main(uint3 gid @gl_GlobalInvocationID) {
            out_[gid.x] = E + F + G;
        }
    }
}
"""

RUNTIME_CONSTANT = """
shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uniform float u;
        const float twiceU = u * 2.0;
        float helper() { return twiceU; }
        void main() {
            outp[0] = helper() + twiceU;
        }
    }
}
"""

PROGRAM_CONSTANTS = """
shader main {
    constant int base = 3;
    constant float SCALE = 2.5;
    constant int ARR[3] = {1, 2, 3};
    constant int NEXT = base + 1;
    compute {
        RWStructuredBuffer<int> out_ @buffer(0);
        @numthreads(1, 1, 1)
        void main(uint3 gid @gl_GlobalInvocationID) {
            out_[gid.x] = base + int(SCALE) + ARR[gid.x % 3u] + NEXT;
        }
    }
}
"""

STATIC_LOCAL_FROM_CONSTANT = """
shader P {
    constant int base = 3;
    int counter() {
        static int c = base;
        c = c + 1;
        return c;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* outp;
        void main() {
            outp[0] = counter();
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, code, backend):
    """Compile ``code`` with the target's own compiler when it is installed."""
    if backend == "opengl" and shutil.which("glslangValidator"):
        (tmp_path / "out.comp").write_text(code)
        _compile([shutil.which("glslangValidator"), "-S", "comp", "out.comp"], tmp_path)
    elif backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(code)
        _compile(
            [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"],
            tmp_path,
        )
    elif backend == "metal" and shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(code)
        _compile(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif backend == "wgsl" and shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(code)
        _compile([shutil.which("naga"), "out.wgsl"], tmp_path)
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(code)
        _compile(
            [
                shutil.which("spirv-as"),
                "--target-env",
                "vulkan1.1",
                "out.spvasm",
                "-o",
                "out.spv",
            ],
            tmp_path,
        )
        _compile(
            [shutil.which("spirv-val"), "--target-env", "vulkan1.1", "out.spv"],
            tmp_path,
        )


@pytest.mark.parametrize(
    "backend, declarations",
    [
        (
            "opengl",
            (
                "const int E = 5;",
                "const int F[2] = int[2](6, 7);",
                "const float K = 0.5;",
                "const int H = ((E * base) + 1);",
            ),
        ),
        (
            "directx",
            (
                "static const int E = 5;",
                "static const int F[2] = {6, 7};",
                "static const float K = 0.5;",
                "static const int H = ((E * base) + 1);",
            ),
        ),
        (
            "slang",
            (
                "static const int E = 5;",
                "static const int F[2] = {6, 7};",
                "static const float K = 0.5;",
                "static const int H = E * base + 1;",
            ),
        ),
        (
            "metal",
            (
                "constant int E = 5;",
                "constant int F[2] = {6, 7};",
                "constant float K = 0.5;",
                "constant int H = E * base + 1;",
            ),
        ),
    ],
)
def test_stage_constants_become_module_constants(tmp_path, backend, declarations):
    code = _translate(tmp_path, STAGE_CONSTANTS, backend)
    for declaration in declarations:
        assert declaration in code
    # Each is declared ahead of the helper that reads it.
    assert code.index(declarations[-1]) < code.index("int scale(int v)")
    _validate(tmp_path, code, backend)


@pytest.mark.parametrize("backend", ["vulkan", "wgsl"])
def test_stage_constants_compile_for_targets_with_private_storage(tmp_path, backend):
    code = _translate(tmp_path, STAGE_CONSTANTS, backend)
    if backend == "wgsl":
        assert "const H: i32 = ((E * base) + 1);" in code
    _validate(tmp_path, code, backend)


def test_hoisted_stage_constants_are_module_constants():
    program = _parse(STAGE_CONSTANTS)
    stage = next(iter(program.stages.values()))
    assert hoist_stage_private_variables(program, "directx") == 4
    assert [v.name for v in stage.local_variables] == ["out_"]
    moved = {v.name: v for v in program.global_variables}
    for name in ("E", "F", "K", "H"):
        assert moved[name].qualifiers == ["constant"]
        assert moved[name].initial_value is not None


@pytest.mark.parametrize("backend", ["opengl", "directx", "slang", "metal"])
def test_entry_only_stage_constants_are_declared(tmp_path, backend):
    code = _translate(tmp_path, ENTRY_CONSTANTS, backend)
    for name, value in (("E", 5), ("F", 6), ("G", 7)):
        assert re.search(rf"\bint {name} = {value};", code), name
    if backend == "slang":
        # A Slang global without ``static`` would be a uniform parameter.
        assert not re.search(r"(?m)^int [EFG] = ", code)
    _validate(tmp_path, code, backend)


@pytest.mark.parametrize("backend", ["opengl", "directx", "slang"])
def test_runtime_stage_constant_is_assigned_by_the_entry(tmp_path, backend):
    code = _translate(tmp_path, RUNTIME_CONSTANT, backend)
    flat = " ".join(code.split())
    assert re.search(r"(static )?float twiceU( = 0\.0)?;", code)
    assignment = re.search(r"twiceU = \(?u \* 2\.0\)?;", flat)
    entry = re.search(r"void \w*main\(", flat, re.IGNORECASE)
    assert assignment and entry and assignment.start() > entry.start()
    _validate(tmp_path, code, backend)


def test_program_constants_are_slang_static_constants(tmp_path):
    code = _translate(tmp_path, PROGRAM_CONSTANTS, "slang")
    assert "static const int base = 3;" in code
    assert "static const float SCALE = 2.5;" in code
    assert "static const int ARR[3] = {1, 2, 3};" in code
    assert "static const int NEXT = base + 1;" in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_program_constants_are_device_readable(tmp_path, backend):
    code = _translate(tmp_path, PROGRAM_CONSTANTS, backend)
    # A const scalar with a constant initializer is readable from device code; an array needs device memory.
    assert re.search(r"(?m)^const int base = 3;", code)
    assert re.search(r"(?m)^const float SCALE = 2\.5f;", code)
    assert re.search(r"(?m)^__device__ const int ARR\[3\] = \{1, 2, 3\};", code)
    assert not re.search(r"(?m)^int (base|ARR|NEXT)\b", code)


def test_program_constant_reading_a_constant_is_a_wgsl_const(tmp_path):
    code = _translate(tmp_path, PROGRAM_CONSTANTS, "wgsl")
    assert "const base: i32 = 3;" in code
    assert "const NEXT: i32 = (base + 1);" in code
    assert "var<private> base" not in code
    _validate(tmp_path, code, "wgsl")


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_static_local_may_start_from_a_program_constant(tmp_path, backend):
    code = _translate(tmp_path, STATIC_LOCAL_FROM_CONSTANT, backend)
    assert re.search(r"int counter_c = base;", code)
    _validate(tmp_path, code, backend)
