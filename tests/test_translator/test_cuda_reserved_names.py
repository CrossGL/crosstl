"""CUDA and HIP output rename program-scope values whose names C++ reserves in the global namespace.

SPIR-V disassembly names the instance of an unnamed block ``_``, ``__0``, ``__1``, and the SPIR-V importer kept those
names, so CUDA output declared ``__device__ int* _;``. NVCC writes that name into PTX unchanged, and PTX cannot spell
a lone underscore: ``ptxas`` failed with ``Parsing error near '_'`` on every such module. C++ reserves every name that
begins with ``_`` in the global namespace, and every name that contains ``__``, so CUDA and HIP output rename them.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_reserved_names import is_reserved_global_name

BACKENDS = ["cuda", "hip"]

# The CrossGL the SPIR-V importer writes for a module whose three storage blocks are unnamed.
SPIRV_BLOCKS = """
shader main {
    struct B1 {
        int iin[];
    };

    RWStructuredBuffer<B1> _ @set(0) @binding(1);

    struct B0 {
        float fin[];
    };

    RWStructuredBuffer<B0> __0 @set(0) @binding(0);

    struct B2 {
        float outp[];
    };

    RWStructuredBuffer<B2> __1 @set(0) @binding(2);

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
      void main() {
        uint i;
        float acc;
        int k;
        i = gl_GlobalInvocationID[0];
        acc = 0.0;
        k = 0;
        while ((k < _[0].iin[i])) {
            acc = (acc + __0[0].fin[i]);
            k = (k + 1);
        }
        __1[0].outp[i] = acc;
        return;
    }
    }
}
"""

# Reserved constants, a constant-buffer member and a struct member, a helper whose parameter shares a renamed name,
# and a program that already spells the first name a rename would pick.
MIXED = """
shader S {
    struct Rec {
        float _x;
        float scaled(float k) { return _x * k; }
    };

    const float _scale = 2.0;
    const float global_scale = 3.0;
    const float a__b = 5.0;

    cbuffer Params {
        float _bias;
    };

    buffer float* data;

    float helper(float _scale) { return _scale + 1.0; }

    compute {
        layout(local_size_x = 64) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            Rec r;
            r._x = data[i];
            data[i] = r.scaled(_scale) + helper(global_scale) + a__b + _bias;
        }
    }
}
"""


def _generate(tmp_path, backend, source):
    path = tmp_path / "reserved.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_reserved_global_names():
    for name in ("_", "__0", "_data", "_Float16", "a__b", "x__"):
        assert is_reserved_global_name(name), name
    for name in ("data", "a_b", "x_", "global_0", "gl_FragCoord"):
        assert not is_reserved_global_name(name), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_unnamed_spirv_blocks_get_unreserved_names(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, SPIRV_BLOCKS))
    for old, new in (("_", "global"), ("__0", "global_0"), ("__1", "global_1")):
        assert f"name={new} kind=buffer" in code, (old, code)
        assert f"name={old} kind=" not in code
    assert "(k < global[i])" in code
    assert "(acc + global_0[i])" in code
    assert "global_1[i] = acc;" in code
    for old in ("_", "__0", "__1"):
        assert f"* {old};" not in code
        assert f"{old}[i]" not in code.replace(f"global{old}", "")


@pytest.mark.parametrize("backend", BACKENDS)
def test_constants_members_and_parameters(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, MIXED))
    # The reserved constants and the constant-buffer member are renamed at their declarations and uses; the
    # existing global_scale keeps its name, so _scale takes the next free one.
    assert "float global_scale_1 = 2.0f;" in code
    assert "float global_scale = 3.0f;" in code
    assert "float global_a_b = 5.0f;" in code
    assert "__constant__ float global_bias;" in code
    assert "Rec_scaled(r, global_scale_1)" in code
    assert "helper(global_scale)" in code
    assert "+ global_a_b) + global_bias)" in code
    assert not re.search(r"\b(_scale|_bias)\b", code)
    assert "a__b" not in code
    # A helper parameter of the renamed name is renamed with it, so its body still reads the parameter.
    assert "global_scale_1 + 1.0" in code
    # A struct member keeps its name, in the struct, in a method and at a member access.
    assert "float _x;" in code
    assert "r._x = data[i];" in code
    assert "receiver._x * k" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_programs_without_reserved_names_are_unchanged(tmp_path, backend):
    source = """
    shader S {
        const float scale = 2.0;
        buffer float* data;
        compute {
            layout(local_size_x = 64) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                float _local = data[i];
                data[i] = _local * scale;
            }
        }
    }
    """
    code = _flat(_generate(tmp_path, backend, source))
    assert not re.search(r"\bglobal_?\w*", code.replace("__global__", ""))
    # A local is not in the global namespace, so it keeps its name.
    assert "_local * scale" in code


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"reserved{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "reserved.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", [SPIRV_BLOCKS, MIXED], ids=["spirv", "mixed"])
def test_nvcc_compiles(tmp_path, source):
    _compile(
        tmp_path,
        "nvcc",
        ["-std=c++17", "-c"],
        _generate(tmp_path, "cuda", source),
        ".cu",
    )


@pytest.mark.parametrize("source", [SPIRV_BLOCKS, MIXED], ids=["spirv", "mixed"])
def test_hipcc_compiles(tmp_path, source):
    _compile(
        tmp_path,
        "hipcc",
        ["-c", "--offload-arch=gfx90a", "-x", "hip"],
        _generate(tmp_path, "hip", source),
        ".hip",
    )
