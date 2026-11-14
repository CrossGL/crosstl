"""Two structures of one name, and geometry stages that state no primitives, are refused on every target.

The SPIR-V, Slang, WGSL, CUDA and HIP generators wrote both declarations of a structure declared twice with
different members. The SPIR-V generator gave a geometry stage that stated no primitives ``InputPoints``,
``OutputPoints`` and ``OutputVertices 1``.
"""

import re
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused
from crosstl.translator.semantics import (
    ConflictingConstantError,
    ConflictingStructError,
)

BACKENDS = [
    "opengl",
    "directx",
    "metal",
    "vulkan",
    "slang",
    "wgsl",
    "cuda",
    "hip",
    "rust",
    "mojo",
]


def write(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(textwrap.dedent(source))
    return str(path)


CONFLICTING = """
shader S {
    struct A { float x; }
    struct A { int y; }
    compute {
        layout(local_size_x = 1) in;
        void main() { A a; a.x = 1.0; }
    }
}
"""

STAGE_CONFLICTING = """
shader S {
    struct A { float x; }
    compute {
        struct A { int y; }
        layout(local_size_x = 1) in;
        void main() { A a; a.y = 1; }
    }
}
"""

IDENTICAL = """
shader S {
    struct A { float x; }
    struct A { float x; }
    compute {
        layout(local_size_x = 1) in;
        void main() { A a; a.x = 1.0; }
    }
}
"""


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_structures_of_one_name_are_refused(tmp_path, backend):
    with pytest.raises(ConflictingStructError) as refusal:
        crosstl.translate(write(tmp_path, CONFLICTING), backend=backend)
    assert (
        "the structure 'A' is declared twice with different members in the program"
        in str(refusal.value)
    )
    assert refusal.value.code.endswith("-conflicting-struct")


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_stage_structure_conflicting_with_a_program_structure_is_refused(
    tmp_path, backend
):
    with pytest.raises(ConflictingStructError, match="in the compute stage"):
        crosstl.translate(write(tmp_path, STAGE_CONFLICTING), backend=backend)


@pytest.mark.parametrize(
    "backend", ["vulkan", "slang", "wgsl", "cuda", "hip", "metal", "opengl", "directx"]
)
def test_a_repeated_identical_structure_is_written_once(tmp_path, backend):
    output = crosstl.translate(write(tmp_path, IDENTICAL), backend=backend)
    pattern = r"OpTypeStruct" if backend == "vulkan" else r"\bstruct A\b"
    assert len(re.findall(pattern, output)) == 1


GEOMETRY = """
shader S {
    geometry {
        LAYOUTS
        void main() {
            gl_Position = vec4(0.0);
            EmitVertex();
            EndPrimitive();
        }
    }
}
"""

COMPLETE = "layout(triangles) in; layout(triangle_strip, max_vertices = 3) out;"


def test_a_geometry_stage_with_its_primitives_keeps_them(tmp_path):
    output = crosstl.translate(
        write(tmp_path, GEOMETRY.replace("LAYOUTS", COMPLETE)), backend="vulkan"
    )
    # Result ids depend on whether SPIR-V Tools reformatted the module.
    for mode in ("Triangles", "OutputTriangleStrip", "OutputVertices 3"):
        assert re.search(rf"OpExecutionMode %\w+ {mode}\n", output), mode


@pytest.mark.parametrize(
    "layouts,missing",
    [
        ("layout(triangle_strip, max_vertices = 3) out;", "states no input primitive"),
        (
            "layout(triangles) in; layout(max_vertices = 3) out;",
            "states no output primitive",
        ),
        ("layout(triangles) in; layout(triangle_strip) out;", "states no max_vertices"),
    ],
)
def test_spirv_refuses_a_geometry_stage_that_leaves_a_mode_unstated(
    tmp_path, layouts, missing
):
    with pytest.raises(SpirvLoweringRefused) as refusal:
        crosstl.translate(
            write(tmp_path, GEOMETRY.replace("LAYOUTS", layouts)), backend="vulkan"
        )
    message = str(refusal.value)
    assert missing in message
    assert "SPIR-V cannot lower 1 construct" in message


def test_spirv_refuses_an_invalid_max_vertices_once(tmp_path):
    layouts = "layout(triangles) in; layout(triangle_strip, max_vertices = 0) out;"
    with pytest.raises(SpirvLoweringRefused) as refusal:
        crosstl.translate(
            write(tmp_path, GEOMETRY.replace("LAYOUTS", layouts)), backend="vulkan"
        )
    message = str(refusal.value)
    assert "must be a positive integer literal" in message
    assert "states no max_vertices" not in message


CONTROL = """
shader S {
    tessellation_control {
        LAYOUTS
        void main() {
            gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
        }
    }
}
"""


def test_a_tessellation_control_stage_keeps_its_patch_size(tmp_path):
    output = crosstl.translate(
        write(tmp_path, CONTROL.replace("LAYOUTS", "layout(vertices = 4) out;")),
        backend="vulkan",
    )
    assert re.search(r"OpExecutionMode %\d+ OutputVertices 4\b", output)


def test_spirv_refuses_a_tessellation_control_stage_without_a_patch_size(tmp_path):
    # OutputVertices 1 used to be written for a stage that stated no output patch size.
    with pytest.raises(SpirvLoweringRefused) as refusal:
        crosstl.translate(
            write(tmp_path, CONTROL.replace("LAYOUTS", "")), backend="vulkan"
        )
    assert "states no output patch size" in str(refusal.value)


CONSTANTS = """
shader S {
    FIRST
    SECOND
    compute {
        layout(local_size_x = 1) in;
        void main() { float b = float(A); }
    }
}
"""


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "second", ["const int A = 2;", "const float A = 2.0;"], ids=["type", "value"]
)
def test_two_constants_of_one_name_are_refused(tmp_path, backend, second):
    source = CONSTANTS.replace("FIRST", "const float A = 1.0;").replace(
        "SECOND", second
    )
    with pytest.raises(
        ConflictingConstantError, match="the constant 'A' is declared twice"
    ):
        crosstl.translate(write(tmp_path, source), backend=backend)


@pytest.mark.parametrize(
    "backend", ["vulkan", "metal", "opengl", "directx", "slang", "wgsl"]
)
def test_a_repeated_identical_constant_is_written_once(tmp_path, backend):
    source = CONSTANTS.replace("FIRST", "const float A = 1.0;").replace(
        "SECOND", "const float A = 1.0;"
    )
    output = crosstl.translate(write(tmp_path, source), backend=backend)
    if backend == "vulkan":
        assert output.count('OpName %A "A"') + output.count('"A"') >= 1
        assert "%A_0" not in output
    else:
        assert len(re.findall(r"\bA\s*(?::\s*\w+\s*)?=\s*1\.0", output)) == 1
