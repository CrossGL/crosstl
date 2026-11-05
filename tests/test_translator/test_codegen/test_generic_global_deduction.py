"""Generic calls deduce type parameters from global arguments and from ``T*`` parameters given a storage buffer.

``generic<T> T twice(T x)`` called as ``twice(scale)`` on a uniform binds ``T`` from the uniform's type. Generators
collect specializations before they register global types, so OpenGL and SPIR-V refused the call, and Metal emitted
a call to ``twice_float`` without defining it.

``generic<T> T bump(device T* data, int offset)`` called with ``RWStructuredBuffer<int>`` binds ``T`` to the buffer's
element type. Deduction only matched a pointer against a pointer, and a specialization spelled ``device T*`` as
``int*``, which lost the pointer's address space.

A call whose specialization is only resolved after its generic was emitted is refused with
:class:`GenericSpecializationOrderError` instead of naming an undefined function.
"""

import re
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

import crosstl
from crosstl.translator.ast import PointerType, PrimitiveType
from crosstl.translator.codegen.generic_emission_order import (
    GenericSpecializationOrderError,
    mark_generic_function_emitted,
    reset_generic_function_emission,
)
from crosstl.translator.codegen.generic_function_utils import (
    _ensure_generic_function_specialization,
)
from crosstl.translator.codegen.generic_pointer_deduction import (
    concrete_generic_pointer_type,
    structured_buffer_element_type,
)

GLOBAL_ARGUMENTS = textwrap.dedent("""
    shader main {
        generic<T> T twice(T x) { return x + x; }
        uniform float scale;
        compute {
            RWStructuredBuffer<int> ints @buffer(0);
            uniform float bias;
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                ints[gid.x] = int(twice(scale) + twice(bias));
            }
        }
    }
    """)

SHADOWED_GLOBAL = textwrap.dedent("""
    shader main {
        generic<T> T twice(T x) { return x + x; }
        uniform float scale;
        int scaled(int scale) { return twice(scale); }
        compute {
            RWStructuredBuffer<int> ints @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                ints[gid.x] = scaled(int(gid.x));
            }
        }
    }
    """)

POINTER_HELPERS = textwrap.dedent("""
    shader main {
        generic<T> void fill(device T* data, int offset, T value) {
            device T* ptr = data + offset;
            ptr[0] = value;
        }
        generic<T> T bump(device T* data, int offset) {
            device T* ptr = data + offset;
            T old = atomicLoad(ptr[0]);
            atomicStore(ptr[0], old + T(1));
            return old;
        }
        compute {
            RWStructuredBuffer<int> ints @buffer(0);
            RWStructuredBuffer<uint> uints @buffer(1);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                fill(ints, int(gid.x), 7);
                int a = bump(ints, int(gid.x));
                uint b = bump(uints, int(gid.x));
                fill(uints, 1, b + uint(a));
            }
        }
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _definition_pattern(name):
    return re.compile(rf"\b{re.escape(name)}\s*\([^;{{]*\)\s*\{{")


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _compile(tmp_path, backend, code):
    """Compile ``code`` with the target's local toolchain, when it is installed."""
    if backend == "opengl" and shutil.which("glslangValidator"):
        source = tmp_path / "out.comp"
        source.write_text(code, encoding="utf-8")
        result = _run(["glslangValidator", str(source)])
        assert result.returncode == 0, result.stdout + result.stderr
    elif backend == "metal" and shutil.which("xcrun"):
        source = tmp_path / "out.metal"
        source.write_text(code, encoding="utf-8")
        result = _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
            + ["-o", str(tmp_path / "out.air")]
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "out.spvasm"
        assembly.write_text(code, encoding="utf-8")
        binary = tmp_path / "out.spv"
        result = _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)]
        )
        assert result.returncode == 0, result.stderr
        result = _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_global_arguments_bind_type_parameters(tmp_path, backend):
    code = _translate(tmp_path, GLOBAL_ARGUMENTS, backend)

    assert _definition_pattern("twice_float").search(code), code
    assert "twice_float(scale)" in code
    assert "twice_float(bias)" in code
    _compile(tmp_path, backend, code)


def test_spirv_global_arguments_bind_type_parameters(tmp_path):
    code = _translate(tmp_path, GLOBAL_ARGUMENTS, "vulkan")

    assert code.count("OpFunctionCall") == 2
    assert "OpFAdd" in code
    _compile(tmp_path, "vulkan", code)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal", "vulkan"])
def test_parameter_shadowing_a_global_keeps_the_parameter_type(tmp_path, backend):
    code = _translate(tmp_path, SHADOWED_GLOBAL, backend)

    if backend == "vulkan":
        assert "OpIAdd" in code
        assert "OpFAdd" not in code
    else:
        assert _definition_pattern("twice_int").search(code), code
        assert re.search(r"return \(?twice_int\(scale\)\)?;", code), code
    _compile(tmp_path, backend, code)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_pointer_parameter_binds_the_storage_buffer_element_type(tmp_path, backend):
    code = _translate(tmp_path, POINTER_HELPERS, backend)

    assert not re.search(r"\bT\b", code), code
    for name in ("bump_int", "bump_uint", "fill_int", "fill_uint"):
        called = re.findall(rf"\b({re.escape(name)}\w*)\s*\(", code)
        assert called, (name, code)
        for callee in set(called):
            assert _definition_pattern(callee).search(code), (callee, code)
    _compile(tmp_path, backend, code)


def test_hlsl_specialization_keeps_the_pointer_a_buffer_view(tmp_path):
    code = _translate(tmp_path, POINTER_HELPERS, "directx")

    assert re.search(
        r"int bump_int\(RWStructuredBuffer<int> data, \w+ data_offset, int offset\)",
        code,
    ), code
    assert "InterlockedOr(data[uint(ptr_offset)], 0, old);" in code


def test_metal_specialization_keeps_the_pointer_address_space(tmp_path):
    code = _translate(tmp_path, POINTER_HELPERS, "metal")

    assert "uint bump_uint(device uint* data, int offset) {" in code
    assert "device uint* ptr = data + offset;" in code
    assert "atomic_load_explicit(reinterpret_cast<device atomic_uint*>(&ptr[0])" in code


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_pointer_parameter_given_a_value_stays_unresolved(tmp_path, backend):
    source = POINTER_HELPERS.replace(
        "int a = bump(ints, int(gid.x));", "int a = bump(3, int(gid.x));"
    )
    with pytest.raises(ValueError, match="bump"):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_vector_atomic_target_bound_through_a_pointer_is_refused(tmp_path, backend):
    source = POINTER_HELPERS.replace(
        "RWStructuredBuffer<uint> uints", "RWStructuredBuffer<vec2> uints"
    ).replace("uint b = bump(uints, int(gid.x));", "vec2 b = bump(uints, 0);")
    source = source.replace("fill(uints, 1, b + uint(a));", "")
    with pytest.raises(ValueError, match="atomicLoad"):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("RWStructuredBuffer<int>", "int"),
        ("StructuredBuffer<vec4>", "vec4"),
        ("RasterizerOrderedStructuredBuffer<Item>", "Item"),
        ("AppendStructuredBuffer<int>", None),
        ("Buffer<float>", None),
        ("int*", None),
        (None, None),
    ],
)
def test_structured_buffer_element_type(spelling, expected):
    assert structured_buffer_element_type(spelling) == expected


def test_concrete_generic_pointer_type_replaces_only_the_pointee():
    pointer = PointerType(PrimitiveType("T"), address_space="device")

    concrete = concrete_generic_pointer_type(
        pointer, lambda pointee: "int" if pointee.name == "T" else None
    )

    assert isinstance(concrete, PointerType)
    assert concrete.address_space == "device"
    assert concrete.pointee_type.name == "int"
    assert pointer.pointee_type.name == "T"
    assert concrete_generic_pointer_type(pointer, lambda pointee: None) is None
    assert concrete_generic_pointer_type(PrimitiveType("T"), lambda p: "int") is None


def test_specialization_created_after_its_generic_was_emitted_is_refused():
    func = SimpleNamespace(name="twice", params=[], body=None, return_type="T")
    generator = SimpleNamespace(
        generic_function_specializations={},
        generic_function_specialized_names={},
        type_name_string=str,
    )
    reset_generic_function_emission(generator)
    mark_generic_function_emitted(generator, func)

    with pytest.raises(GenericSpecializationOrderError) as caught:
        _ensure_generic_function_specialization(
            generator, func, ("twice", ("float",)), {"T": "float"}
        )

    assert caught.value.function_name == "twice"
    assert caught.value.type_arguments == ("float",)
    assert generator.generic_function_specializations == {}
