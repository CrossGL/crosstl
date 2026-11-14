"""CrossGL generic pointers ``ptr<T>`` take the address space they point into (generic_pointer_spaces.py)."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.generic_pointer_spaces import (
    assign_generic_pointer_spaces,
    written_generic_pointer_spellings,
)
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.lowering_refusals import LoweringRefused
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = {
    "metal": MetalCodeGen,
    "directx": HLSLCodeGen,
    "opengl": GLSLCodeGen,
}


def parse(code):
    return Parser(Lexer(code).get_tokens()).parse()


def specialize(code):
    reasons = []
    program = assign_generic_pointer_spaces(parse(code), reasons.append)
    return program, reasons


def functions_by_name(program):
    return {function.name: function for function in program.functions}


def entry(body, parameters="", locals_=""):
    return f"""
    @compute
    @workgroup_size(64, 1, 1)
    fn k(
        @group(0) @binding(0) var<storage, read_write> out: array<f32>,
        @group(0) @binding(1) var<storage, read> in_: array<f32>{parameters}
    ) {{
        var<workgroup> tile: array<f32, 64>;
        var local: array<f32, 4>;
        {locals_}
        {body}
    }}
    """


LOAD = "f32 load(ptr<f32> p, i32 i) { return p[i]; }\n"


def test_a_function_called_with_one_space_keeps_its_name():
    program, reasons = specialize(LOAD + entry("out[0] = load(out, 0);"))

    assert reasons == []
    parameter = functions_by_name(program)["load"].parameters[0]
    assert isinstance(parameter.param_type, A.PointerType)
    assert parameter.param_type.address_space == "device"
    assert parameter.qualifiers == ["device"]


def test_each_space_combination_gets_its_own_copy_and_calls_follow_it():
    program, reasons = specialize(
        LOAD
        + entry(
            "out[0] = load(out, 0) + load(tile, 1) + load(local, 2) + load(in_, 3);"
        )
    )

    assert reasons == []
    functions = functions_by_name(program)
    assert "load" not in functions
    spaces = {
        name: (
            functions[name].parameters[0].param_type.address_space,
            functions[name].parameters[0].param_type.access_mode,
            functions[name].parameters[0].qualifiers,
        )
        for name in (
            "load_device",
            "load_threadgroup",
            "load_thread",
            "load_const_device",
        )
    }
    assert spaces == {
        "load_device": ("device", None, ["device"]),
        "load_threadgroup": ("threadgroup", None, ["threadgroup"]),
        "load_thread": ("thread", None, ["thread"]),
        "load_const_device": ("device", "read", ["device", "const"]),
    }
    called = [
        node.function.name
        for node in functions["k"].body.walk()
        if isinstance(node, A.FunctionCallNode)
    ]
    assert sorted(called) == sorted(spaces)


def test_the_original_program_is_not_changed():
    original = parse(LOAD + entry("out[0] = load(out, 0);"))

    specialized = assign_generic_pointer_spaces(original, pytest.fail)

    assert specialized is not original
    assert isinstance(original.functions[0].parameters[0].param_type, A.NamedType)


def test_a_program_without_generic_pointers_is_returned_unchanged():
    original = parse(entry("out[0] = in_[0];"))

    assert assign_generic_pointer_spaces(original, pytest.fail) is original


def test_address_of_arithmetic_and_locals_keep_the_space():
    program, reasons = specialize(
        "void store(ptr<f32> p, f32 v) { (*p) = v; }\n"
        + entry(
            "var t: ptr<f32> = (tile + 1); store(t, 1.0f); store((&local[1]), 2.0f);"
            " store((&out[2]), 3.0f);"
        )
    )

    assert reasons == []
    functions = functions_by_name(program)
    assert {"store_threadgroup", "store_thread", "store_device"} <= set(functions)
    local = next(
        node
        for node in functions["k"].body.walk()
        if isinstance(node, A.VariableNode) and node.name == "t"
    )
    assert local.var_type.address_space == "threadgroup"


def test_a_reassigned_local_and_a_returned_pointer_take_their_values_space():
    program, reasons = specialize(
        "ptr<f32> row(ptr<f32> m, i32 r) { return (m + (r * 4)); }\n"
        "f32 walk(ptr<f32> p, i32 n) {\n"
        "    var q: ptr<f32> = p;\n"
        "    f32 s = 0.0;\n"
        "    for (i32 i = 0; i < n; i++) { s += (*q); q = (q + 2); }\n"
        "    return s;\n"
        "}\n" + entry("out[0] = walk(row(out, 1), 2);")
    )

    assert reasons == []
    functions = functions_by_name(program)
    assert functions["row"].return_type.address_space == "device"
    local = next(
        node
        for node in functions["walk"].body.walk()
        if isinstance(node, A.VariableNode) and node.name == "q"
    )
    assert local.var_type.address_space == "device"


def test_a_local_assigned_two_spaces_is_refused():
    _program, reasons = specialize(
        entry("var p: ptr<f32> = out; if (out[0] > 0.0) { p = tile; } p[0] = 1.0;")
    )

    assert reasons == [
        "generic pointer: 'k.p' points into device and threadgroup memory"
    ]


def test_a_conditional_pointer_into_two_spaces_is_refused():
    _program, reasons = specialize(
        LOAD + entry("out[0] = load(((out[0] > 0.0) ? out : tile), 0);")
    )

    assert reasons[0] == (
        "generic pointer: a conditional expression in 'k' points into device or "
        "threadgroup memory"
    )


def test_a_pointer_read_from_a_structure_member_is_refused():
    _program, reasons = specialize(
        "struct Holder { ptr<f32> data; };\n"
        + LOAD
        + entry("Holder h; out[0] = load(h.data, 0);")
    )

    assert reasons == [
        "generic pointer: the address space that argument 1 of the call to 'load' "
        "in 'k' points into is unknown"
    ]


def test_writing_through_a_pointer_into_read_only_memory_is_refused():
    _program, reasons = specialize(
        "void store(ptr<f32> p) { p[0] = 1.0; }\n" + entry("store(in_);")
    )

    assert reasons == [
        "generic pointer: 'store' writes through 'p', which points into read-only "
        "device memory"
    ]


def test_recursion_through_a_generic_pointer_is_refused():
    _program, reasons = specialize(
        "f32 sum(ptr<f32> p, i32 n) { if (n == 0) { return 0.0; } return (p[0] + sum((p + 1), (n - 1))); }\n"
        + entry("out[0] = sum(out, 4);")
    )

    assert reasons == [
        "generic pointer: 'sum' calls itself with a generic pointer, which a target "
        "without generic pointers cannot specialize"
    ]


def test_unreached_users_of_generic_pointers_are_dropped():
    program, reasons = specialize(
        "void launch(ptr<f32> host, i32 n) { var d: ptr<f32>; }\n"
        "void host_main() { launch(0, 1); }\n"
        "f32 unrelated(f32 x) { return x; }\n" + entry("out[0] = 1.0;")
    )

    assert reasons == []
    assert sorted(functions_by_name(program)) == ["k", "unrelated"]


def test_an_unused_uninitialized_pointer_local_is_removed():
    program, reasons = specialize(entry("out[0] = 1.0;", locals_="var d: ptr<f32>;"))

    assert reasons == []
    assert not any(
        isinstance(node, A.VariableNode) and node.name == "d"
        for node in functions_by_name(program)["k"].body.walk()
    )


def test_a_program_without_an_entry_point_is_refused():
    _program, reasons = specialize(LOAD)

    assert reasons == [
        "generic pointer: the program has no entry point, so the address spaces the "
        "generic pointers of 'load' point into are unknown"
    ]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("void f(ptr<float> p);", ["ptr<float>"]),
        ("ptr < vec4<float> > q;", ["ptr<vec4<float>>"]),
        ('// ptr<float> in a comment\nx = "ptr<int>";', []),
        ("bool b = s.ptr < n && m > 0;", []),
        ("my_ptr<float> v; a->ptr<int> w; ns::ptr<int> u;", []),
    ],
)
def test_written_generic_pointer_spellings(code, expected):
    assert written_generic_pointer_spellings(code) == expected


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_a_generic_pointer_without_a_space_is_refused_rather_than_written(backend):
    program = parse(
        "struct Holder { ptr<f32> data; };\n"
        + LOAD
        + entry("Holder h; out[0] = load(h.data, 0);")
    )

    # HLSL refuses the structure member itself, with its own diagnostic code (a ValueError, as every refusal is).
    with pytest.raises(ValueError, match="generic pointer"):
        GENERATORS[backend]().generate(program)


def test_slang_refuses_pointer_parameters_it_cannot_lower():
    program = parse(LOAD + entry("out[0] = load(out, 0);"))

    with pytest.raises(
        LoweringRefused,
        match="pointer declaration: 'load.p' points into device memory",
    ):
        SlangCodeGen().generate(program)


CUDA_KERNEL = """
__device__ float load2(const float* p, int i) { return p[i] + p[i + 1]; }
__device__ void store(float* p, float v) { *p = v; }
__device__ float walk(const float* p, int n) {
    float s = 0.0f;
    const float* q = p;
    for (int i = 0; i < n; ++i) { s += *q; q = q + 2; }
    return s;
}
__global__ void k(float* out, const float* in) {
    __shared__ float tile[64];
    float local[4];
    local[0] = 0.0f; local[1] = 1.0f; local[2] = 2.0f; local[3] = 3.0f;
    tile[threadIdx.x] = in[threadIdx.x];
    __syncthreads();
    float* t = tile + 1;
    out[threadIdx.x] = load2(tile, 0) + load2(in, 1) + load2(local, 0) + t[0] + walk(in, 4);
    store(&out[2], 1.0f);
    store(&local[1], 2.0f);
    store(t, 4.0f);
}
void launch(float* host, int n) {
    float* d;
    cudaMalloc((void**)&d, n * 4);
    k<<<1, 64>>>(d, d);
    cudaFree(d);
}
"""

SIZE_DEFINES = [
    "-DCROSSGL_WORKGROUP_SIZE_X=64",
    "-DCROSSGL_WORKGROUP_SIZE_Y=1",
    "-DCROSSGL_WORKGROUP_SIZE_Z=1",
]


def compile_natively(backend, code, tmp_path):
    """Compile ``code`` with the target's native checker when it is installed."""
    if backend == "metal":
        tool = shutil.which("xcrun")
        source = tmp_path / "kernel.metal"
        command = [tool, "-sdk", "macosx", "metal", "-c", str(source), "-o"]
        command.append(str(tmp_path / "kernel.air"))
    elif backend == "directx":
        tool = shutil.which("dxc")
        source = tmp_path / "kernel.hlsl"
        command = [tool, "-T", "cs_6_2", "-E", "CSMain", *SIZE_DEFINES, str(source)]
    else:
        # OpenGL SPIR-V defines GL_SPIRV, so the output takes its workgroup size from specialization constants.
        tool = shutil.which("glslangValidator")
        source = tmp_path / "kernel.comp"
        command = [tool, "--target-env", "opengl", "-S", "comp", str(source), "-o"]
        command.append(str(tmp_path / "kernel.spv"))
    if tool is None:
        return
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_cuda_helpers_take_the_spaces_of_their_arguments(backend, tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text(CUDA_KERNEL, encoding="utf-8")

    # Unformatted, so the spellings do not depend on whether clang-format is installed.
    code = crosstl.translate(str(source), backend=backend, format_output=False)

    assert written_generic_pointer_spellings(code) == []
    assert "launch" not in code
    if backend == "metal":
        for signature in (
            "float load2_threadgroup(threadgroup float* p, int i)",
            "float load2_thread(thread float* p, int i)",
            "void store_device(device float* p, float v)",
            "void store_threadgroup(threadgroup float* p, float v)",
            "threadgroup float* t = tile + 1;",
        ):
            assert signature in code
    compile_natively(backend, code, tmp_path)


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_hip_read_only_buffers_reach_const_helpers(backend, tmp_path):
    source = tmp_path / "kernel.hip"
    source.write_text(
        "#include <hip/hip_runtime.h>\n"
        "__device__ float sum2(const float* p, int i) {"
        " const float* q = p + 1; return p[i] + q[i]; }\n"
        "__global__ void k(float* out, const float* in) {"
        " out[0] = sum2(in, 0) + sum2(out, 1); }\n",
        encoding="utf-8",
    )

    code = crosstl.translate(str(source), backend=backend, format_output=False)

    assert written_generic_pointer_spellings(code) == []
    if backend == "metal":
        assert "float sum2_const_device(const device float* p, int i)" in code
        assert "const device float* q = p + 1;" in code
        assert "float sum2_device(device float* p, int i)" in code
    compile_natively(backend, code, tmp_path)


def test_hlsl_refuses_a_pointer_return_type(tmp_path):
    program = parse(
        "ptr<f32> row(ptr<f32> m, i32 r) { return (m + (r * 4)); }\n"
        + entry("out[0] = row(out, 1)[0];")
    )

    with pytest.raises(
        LoweringRefused, match="pointer return: 'row' returns a pointer"
    ):
        HLSLCodeGen().generate(program)
