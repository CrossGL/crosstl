"""Metal output renames function-local declarations named like program-scope storage.

Metal receives buffers, threadgroup memory and uniform values as entry parameters and appends each one a helper
reaches to the helper's parameters under the variable's own name, so a helper parameter or local with that name
collides with it or hides it. The generator gives such a declaration a fresh name within its scope.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from tests.runtime_helpers import compile_metal

ROOT = Path(__file__).resolve().parents[2]
METAL_REQUIRE_ENV = "CROSTL_REQUIRE_METAL_SHADOWED_STORAGE"

# Reviewer G4: a CUDA helper parameter named like the kernel's buffer.
CUDA_KERNEL = """__device__ int twice(int o) { return o * 2; }
__device__ int bump(int in) { int o = in + 1; return o; }
__global__ void k(const int* in, int* o) {
    int t = threadIdx.x;
    o[t] = twice(in[t]) + bump(t);
}
"""

HLSL_KERNEL = """RWStructuredBuffer<int> o : register(u0);
int f(int v) { int r = o[0]; { int o = v; r += o; } return r; }
int g(int o) { return o * 2; }
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { o[id.x] = f(int(id.x)) + g(int(id.x)); }
"""

HLSL_CBUFFER = """cbuffer Params : register(b0) { int a; };
RWStructuredBuffer<int> outp : register(u0);
int f(int v) { int r = a; { int a = v; r += a; } return r; }
int g(int a) { return a + 1; }
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) { outp[id.x] = f(int(id.x)) + g(int(id.x)); }
"""

PROGRAM_SCOPE = """@group(0) @binding(0) var<storage, read_write> o: array<i32>;
i32 twice(i32 o) { return o * 2; }
i32 first(i32 v) { var r: i32 = o[0]; { var o: i32 = v; r = r + o; } return r; }
@compute @workgroup_size(4, 1, 1)
fn k() { var t: i32 = gl_LocalInvocationID.x; o[t] = twice(t) + first(t); }
"""

# Every form at once: a parameter named like a buffer, a nested local hiding a
# buffer the helper also reads, a parameter named like threadgroup memory in a
# helper whose callee reads that memory, and an entry block-local named like
# the output buffer the entry writes after the block.
STAGE_BLOCK = """shader S {
    compute {
        layout(local_size_x = 4) in;
        buffer int* a;
        buffer int* o;
        shared int t[4];
        int twice(int o) { return o * 2; }
        int nested(int v) { int r = a[v]; { int a = v * 10; r += a; } return r; }
        int peek(int v) { return t[v]; }
        int through_callee(int t) { return t + peek(t); }
        void main() {
            uint i = gl_LocalInvocationID.x;
            t[i] = int(i) + 100;
            barrier();
            int acc = 0;
            {
                int o = twice(int(i)) + nested(int(i)) + through_callee(int(i));
                acc = o;
            }
            o[i] = acc;
        }
    }
}
"""
STAGE_BLOCK_INPUT = [5, 6, 7, 8]
# twice(i) + a[i] + 10i + (i + t[i]) with t[i] = i + 100.
STAGE_BLOCK_RESULTS = [14 * i + STAGE_BLOCK_INPUT[i] + 100 for i in range(4)]

UNIFORM = """shader S {
    compute {
        layout(local_size_x = 4) in;
        buffer int* outp;
        uniform int a;
        int f(int v) { int r = a; { int a = v; r += a; } return r; }
        int g(int a) { return a + 1; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = f(int(i)) + g(int(i));
        }
    }
}
"""

ENTRY_INITIALIZER = """shader S {
    compute {
        layout(local_size_x = 4) in;
        buffer int* outp;
        buffer int* o;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int o = o[i] * 3;
            outp[i] = o;
        }
    }
}
"""


def translate(tmp_path, source, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source, encoding="utf-8")
    # Unformatted, so the expected spellings do not depend on whether clang-format is installed.
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _signature(code, name):
    match = re.search(rf"\b{name}\((.*?)\)\s*{{", code, re.S)
    assert match, code
    return " ".join(match.group(1).split())


def test_cuda_helper_parameter_named_like_the_kernel_buffer(tmp_path):
    code = translate(tmp_path, CUDA_KERNEL, ".cu")
    assert _signature(code, "twice") == "int o_"
    assert "return o_ * 2;" in code
    # ``in`` is a CrossGL keyword: the buffer is ``in_``, and the parameter that shadows it is renamed too.
    assert re.search(r"int o_ = in__ \+ 1;\s*return o_;", code), code
    assert "device int* o [[buffer(1)]]" in code


def test_nested_local_ends_with_its_block(tmp_path):
    code = translate(tmp_path, HLSL_KERNEL, ".hlsl")
    assert _signature(code, "f") == "int v, device int* o"
    assert "int r = o[0];" in code
    assert "int o_ = v;" in code
    assert _signature(code, "g") == "int o_"


def test_program_scope_resource_names(tmp_path):
    code = translate(tmp_path, PROGRAM_SCOPE, ".cgl")
    assert _signature(code, "twice") == "int o_"
    assert _signature(code, "first") == "int v, device int* o"
    assert "int o_ = v;" in code


def test_threadgroup_parameter_in_a_helper_whose_callee_reads_it(tmp_path):
    code = translate(tmp_path, STAGE_BLOCK, ".cgl")
    assert _signature(code, "through_callee") == ("int t_, threadgroup int (&t)[4]")
    assert "return t_ + peek(t_, t);" in code
    assert re.search(r"int o_ =[^;]*;\s*acc = o_;\s*}\s*o\[i\] = acc;", code), code


def test_uniform_and_constant_buffer_members(tmp_path):
    code = translate(tmp_path, UNIFORM, ".cgl")
    assert _signature(code, "f") == "int v, constant int& a"
    assert "int a_ = v;" in code
    # A helper that never reads the uniform keeps its parameter name.
    assert _signature(code, "g") == "int a"

    code = translate(tmp_path, HLSL_CBUFFER, ".hlsl")
    assert _signature(code, "f") == "int v, constant Params& params"
    assert "int r = params.a;" in code
    assert "int a_ = v;" in code
    assert _signature(code, "g") == "int a"


def test_initializer_reads_the_outer_name(tmp_path):
    code = translate(tmp_path, ENTRY_INITIALIZER, ".cgl")
    # The entry's own block-local renaming runs first here.
    match = re.search(r"int (o_\w*) = o\[i\] \* 3;", code)
    assert match, code
    assert f"outp[i] = {match.group(1)};" in code


def test_the_parsed_program_is_not_changed():
    program = parse(STAGE_BLOCK)
    first = MetalCodeGen().generate(program)
    assert "int o_" in first
    assert MetalCodeGen().generate(program) == first


def _metal_library(tmp_path, source, suffix):
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler requires macOS with xcrun")
    directory = tmp_path / suffix.strip(".")
    directory.mkdir()
    artifact = directory / "kernel.metal"
    artifact.write_text(translate(directory, source, suffix), encoding="utf-8")
    return compile_metal(artifact, ROOT, directory / "metal")


@pytest.mark.parametrize(
    "source, suffix",
    [
        (CUDA_KERNEL, ".cu"),
        (HLSL_KERNEL, ".hlsl"),
        (HLSL_CBUFFER, ".hlsl"),
        (PROGRAM_SCOPE, ".cgl"),
        (STAGE_BLOCK, ".cgl"),
        (UNIFORM, ".cgl"),
        (ENTRY_INITIALIZER, ".cgl"),
    ],
)
def test_renamed_programs_compile_for_metal(tmp_path, source, suffix):
    assert _metal_library(tmp_path, source, suffix).is_file()


def _run(tmp_path, library, entry, inputs, output_count):
    if os.environ.get(METAL_REQUIRE_ENV) != "1":
        pytest.skip(f"set {METAL_REQUIRE_ENV}=1 to require native Metal execution")
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    subprocess.run(["swiftc", str(fixture), "-o", str(runner)], check=True, timeout=300)
    request = tmp_path / "request.json"
    # The runner binds the inputs in order and the output buffer last.
    request.write_text(
        json.dumps(
            {
                "inputs": inputs,
                "outputCount": output_count,
                "threadCount": 1,
                "threadsPerThreadgroup": output_count,
            }
        )
    )
    readback = subprocess.run(
        [str(runner), str(library), entry, str(request)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return json.loads(readback)["values"]


def test_cuda_kernel_computes_on_metal(tmp_path):
    library = _metal_library(tmp_path, CUDA_KERNEL, ".cu")
    inputs = [3, 4, 5, 6]
    values = _run(tmp_path, library, "kernel_k", [inputs], 4)
    assert values == [2 * value + t + 1 for t, value in enumerate(inputs)]


def test_stage_block_program_computes_on_metal(tmp_path):
    library = _metal_library(tmp_path, STAGE_BLOCK, ".cgl")
    values = _run(tmp_path, library, "kernel_main", [STAGE_BLOCK_INPUT], 4)
    assert values == STAGE_BLOCK_RESULTS
