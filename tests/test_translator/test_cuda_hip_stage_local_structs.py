"""CUDA and HIP output declares the structs a stage declares, before the declarations that name them.

A stage may declare its own structs, among them the block type of a GLSL interface block written inside the stage.
The CUDA and HIP emitters wrote only the shader's structs, so ``buffer Data { ... } data;`` inside ``compute`` came
out as ``Data data;`` with no ``struct Data``, which nvcc and hipcc reject as an undeclared type.
"""

import re

import pytest

import crosstl

BLOCK_KERNEL = """shader StageBlock {
    compute {
        layout(std430, binding = 0) buffer Data { float scale; float values[]; } data;
        layout(local_size_x = 64) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            data.values[i] = data.values[i] * data.scale;
        }
    }
}
"""

LOCAL_STRUCT_KERNEL = """shader StageStruct {
    compute {
        struct Pair { float first; float second; }
        layout(local_size_x = 64) in;
        shared float values[64];
        void main() {
            uint i = gl_LocalInvocationID.x;
            Pair pair;
            pair.first = values[i];
            pair.second = 2.0;
            values[i] = pair.first * pair.second;
        }
    }
}
"""


def translate(tmp_path, backend, source):
    path = tmp_path / "kernel.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def struct_declarations(code, name):
    return [
        match.start() for match in re.finditer(rf"^struct {name}\s*\{{", code, re.M)
    ]


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_interface_block_type_is_declared_before_the_block(tmp_path, backend):
    code = translate(tmp_path, backend, BLOCK_KERNEL)

    assert len(struct_declarations(code, "Data")) == 1
    struct_at = struct_declarations(code, "Data")[0]
    assert code.index("float scale;", struct_at) < code.index("};", struct_at)
    assert struct_at < code.index("Data data;")


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_stage_local_struct_is_declared_before_the_kernel(tmp_path, backend):
    code = translate(tmp_path, backend, LOCAL_STRUCT_KERNEL)

    assert len(struct_declarations(code, "Pair")) == 1
    assert struct_declarations(code, "Pair")[0] < code.index("void compute_main")
    assert "Pair pair;" in code


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_struct_declared_by_shader_and_stage_alike_is_declared_once(tmp_path, backend):
    source = LOCAL_STRUCT_KERNEL.replace(
        "shader StageStruct {",
        "shader StageStruct {\n    struct Pair { float first; float second; }",
    )

    code = translate(tmp_path, backend, source)

    assert len(struct_declarations(code, "Pair")) == 1


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_conflicting_struct_of_one_name_is_refused(tmp_path, backend):
    source = LOCAL_STRUCT_KERNEL.replace(
        "shader StageStruct {",
        "shader StageStruct {\n    struct Pair { int first; int second; int third; }",
    )

    with pytest.raises(ValueError, match="stage-local struct 'Pair'"):
        translate(tmp_path, backend, source)
