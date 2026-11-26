"""CUDA and HIP output assign a brace list to a declared array without assigning the array.

The SPIR-V importer writes ``OpStore %local %constant_array`` as ``values = {1.0, 2.0, 4.0, 8.0};`` on an array
declared earlier, and C++ built-in arrays cannot be assigned: NVCC rejected it with ``expression must be a modifiable
lvalue`` and hipcc with ``array type 'float[4]' is not assignable``. A list of constants that names every element is
stored element by element. Any other list initializes a temporary array that is then copied, so the list's elements
are evaluated once, an element that reads the target reads its old value, and the elements a short list leaves out are
value-initialized, as the declaration ``float values[4] = {...};`` would do.
"""

import shutil
import subprocess

import pytest

import crosstl

BACKENDS = ["cuda", "hip"]

CONSTANTS = """
shader main {
    struct B0 {
        float outp[];
    };

    RWStructuredBuffer<B0> results @set(0) @binding(0);

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
      void main() {
        uint i;
        float values[4];
        vec2 pairs[2];
        float grid[2][3];
        i = gl_GlobalInvocationID[0];
        values = {1.0, 2.0, 4.0, -8.0};
        pairs = {vec2(0.5, 1.0), vec2(-1.0, 2.0)};
        grid = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        results[0].outp[i] = values[i & 3u] + pairs[i & 1u].y + grid[1][2];
        return;
    }
    }
}
"""

# Elements that read storage, including the array being assigned, and a list shorter than the array.
READS = """
shader main {
    struct B0 {
        float outp[];
    };

    RWStructuredBuffer<B0> results @set(0) @binding(0);

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;
      void main() {
        uint i;
        float values[3];
        float rest[4];
        i = gl_GlobalInvocationID[0];
        values = {1.0, 2.0, results[0].outp[i]};
        values = {values[2], values[0], values[1]};
        rest = {values[0]};
        results[0].outp[i] = values[0] + rest[3];
        return;
    }
    }
}
"""


def _generate(tmp_path, backend, source):
    path = tmp_path / "literals.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


@pytest.mark.parametrize("backend", BACKENDS)
def test_constant_lists_store_each_element(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, CONSTANTS))
    assert "values = {" not in code
    assert "pairs = {" not in code
    assert "grid = {" not in code
    assert (
        "values[0] = 1.0f; values[1] = 2.0f; values[2] = 4.0f; values[3] = -8.0f;"
        in code
    )
    assert (
        "pairs[0] = make_float2(0.5f, 1.0f); pairs[1] = make_float2(-1.0f, 2.0f);"
        in code
    )
    assert "grid[0][0] = 1.0f; grid[0][1] = 2.0f; grid[0][2] = 3.0f;" in code
    assert "grid[1][0] = 4.0f; grid[1][1] = 5.0f; grid[1][2] = 6.0f;" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_lists_that_read_storage_go_through_a_temporary(tmp_path, backend):
    code = _flat(_generate(tmp_path, backend, READS))
    assert "values = {" not in code
    assert "rest = {" not in code
    # The rotation reads the old values: the list initializes a temporary, which is then copied.
    assert (
        "float __crossgl_array_value_1[3] = {values[2], values[0], values[1]};" in code
    )
    assert (
        "values[__crossgl_array_index_1] = __crossgl_array_value_1[__crossgl_array_index_1];"
        in code
    )
    # A short list value-initializes the remaining elements, as a declaration would.
    assert "float __crossgl_array_value_2[4] = {values[0]};" in code
    assert (
        "rest[__crossgl_array_index_2] = __crossgl_array_value_2[__crossgl_array_index_2];"
        in code
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_impure_target_is_refused(tmp_path, backend):
    source = CONSTANTS.replace(
        "float grid[2][3];", "float grid[2][3];\n        int n;"
    ).replace(
        "grid = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};",
        "n = 0;\n        grid[n++] = {1.0, 2.0, 3.0};",
    )
    with pytest.raises(Exception, match="array assignment"):
        _generate(tmp_path, backend, source)


def _compile(tmp_path, tool, args, code, suffix):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed")
    source = tmp_path / f"literals{suffix}"
    source.write_text(code)
    result = subprocess.run(
        [tool, *args, str(source), "-o", str(tmp_path / "literals.o")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source", [CONSTANTS, READS], ids=["constants", "reads"])
def test_nvcc_compiles(tmp_path, source):
    _compile(
        tmp_path,
        "nvcc",
        ["-std=c++17", "-c"],
        _generate(tmp_path, "cuda", source),
        ".cu",
    )


@pytest.mark.parametrize("source", [CONSTANTS, READS], ids=["constants", "reads"])
def test_hipcc_compiles(tmp_path, source):
    _compile(
        tmp_path,
        "hipcc",
        ["-std=c++17", "-c", "--offload-arch=gfx90a"],
        _generate(tmp_path, "hip", source),
        ".hip",
    )
