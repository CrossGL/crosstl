"""Compute workgroup sizes that the pipeline sets through specialization constants (``local_size_x_id``)."""

import os
import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.specialized_workgroup_size import (
    SpecializedWorkgroupSizeError,
    workgroup_size_specialization_ids,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SPECIALIZED = """shader Probe {
    compute {
        layout(LAYOUT) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = BODY;
        }
    }
}
"""

HELPER_READ = """shader Probe {
    uint groupWidth() {
        return gl_WorkGroupSize.x;
    }
    compute {
        layout(LAYOUT) in;
        buffer uint* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = groupWidth();
        }
    }
}
"""

X_AND_Z = "local_size_x = 64, local_size_x_id = 4, local_size_z_id = 7"
READ_SIZE = "float(gl_WorkGroupSize.x + gl_WorkGroupSize.z)"


def _source(layout, body=READ_SIZE, template=SPECIALIZED):
    return template.replace("LAYOUT", layout).replace("BODY", body)


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def _assemble(tmp_path, code):
    """Validate SPIR-V assembly when SPIRV-Tools is installed; return the binary path or None."""
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return None
    asm, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    asm.write_text(code)
    _run(["spirv-as", "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
    return binary


def _spec_constant(code, spec_id):
    """The default of the ``uint`` specialization constant decorated ``SpecId spec_id``."""
    match = re.search(rf"OpDecorate %(\w+) SpecId {spec_id}\b", code)
    assert match, f"no SpecId {spec_id}"
    name = match.group(1)
    default = re.search(rf"%{name} = OpSpecConstant %\w+ (\d+)", code)
    assert default, f"%{name} is not an OpSpecConstant"
    return name, int(default.group(1))


def _workgroup_size_composite(code):
    match = re.search(r"OpDecorate %(\w+) BuiltIn WorkgroupSize\b", code)
    assert match, "no BuiltIn WorkgroupSize"
    composite = re.search(
        rf"%{match.group(1)} = (OpSpecConstantComposite|OpConstantComposite) %\w+ "
        r"%(\w+) %(\w+) %(\w+)",
        code,
    )
    assert composite, "WorkgroupSize is not a constant composite"
    return composite.group(1), composite.groups()[1:]


def test_parser_keeps_the_specialization_ids_in_the_execution_metadata():
    ast = Parser(Lexer(_source(X_AND_Z)).get_tokens()).parse()
    (stage,) = ast.stages.values()
    config = stage.execution_config
    assert config["local_size_x"] == "64"
    assert config["local_size_x_id"] == "4"
    assert config["local_size_z_id"] == "7"
    assert workgroup_size_specialization_ids(config) == (4, None, 7)


def test_spirv_builds_workgroup_size_from_specialization_constants(tmp_path):
    code = _translate(tmp_path, _source(X_AND_Z), "vulkan")
    x_name, x_default = _spec_constant(code, 4)
    z_name, z_default = _spec_constant(code, 7)
    assert (x_default, z_default) == (64, 1)
    opcode, components = _workgroup_size_composite(code)
    assert opcode == "OpSpecConstantComposite"
    assert components[0] == x_name and components[2] == z_name
    assert re.search(r"OpExecutionMode %\w+ LocalSize 64 1 1\b", code)
    _assemble(tmp_path, code)


def test_spirv_declares_the_specialized_size_when_the_shader_never_reads_it(tmp_path):
    # Without the WorkgroupSize composite the specialization constant would not reach the dispatch.
    code = _translate(tmp_path, _source("local_size_y_id = 2", body="1.0"), "vulkan")
    y_name, y_default = _spec_constant(code, 2)
    assert y_default == 1
    opcode, components = _workgroup_size_composite(code)
    assert opcode == "OpSpecConstantComposite" and components[1] == y_name
    _assemble(tmp_path, code)


@pytest.mark.parametrize(
    "layout, specialized",
    [("local_size_x = 64", False), ("local_size_x = 64, local_size_x_id = 4", True)],
)
def test_spirv_helper_outside_the_stage_reads_the_stage_size(
    tmp_path, layout, specialized
):
    # WorkgroupSize overrides LocalSize, so a composite of ones would run one-invocation workgroups.
    code = _translate(tmp_path, _source(layout, template=HELPER_READ), "vulkan")
    opcode, components = _workgroup_size_composite(code)
    if specialized:
        x_name, x_default = _spec_constant(code, 4)
        assert opcode == "OpSpecConstantComposite"
        assert components[0] == x_name and x_default == 64
    else:
        assert opcode == "OpConstantComposite"
        assert re.search(rf"%{components[0]} = OpConstant %\w+ 64\b", code)
    _assemble(tmp_path, code)


def test_glsl_writes_the_ids_behind_the_spirv_guard(tmp_path):
    code = _translate(tmp_path, _source(X_AND_Z), "opengl")
    fixed = "local_size_x = 64, local_size_y = 1, local_size_z = 1"
    assert (
        "#if defined(GL_SPIRV) || defined(VULKAN)\n"
        f"layout({fixed}, local_size_x_id = 4, local_size_z_id = 7) in;\n"
        f"#else\nlayout({fixed}) in;\n#endif\n"
    ) in code
    if shutil.which("glslangValidator") and shutil.which("spirv-dis"):
        shader, binary = tmp_path / "out.comp", tmp_path / "out.spv"
        shader.write_text(code)
        _run(["glslangValidator", "-V", str(shader), "-o", str(binary)])
        disassembly = _run(["spirv-dis", str(binary)])
        assert _spec_constant(disassembly, 4)[1] == 64
        assert _spec_constant(disassembly, 7)[1] == 1
        # A driver compiling the source for OpenGL keeps the defaults.
        _run(["glslangValidator", str(shader)])


def test_glsl_import_keeps_the_specialized_size_to_spirv(tmp_path):
    glsl = (
        "#version 460\n"
        f"layout({X_AND_Z}) in;\n"
        "layout(std430, binding = 0) buffer B0 { float outp[]; };\n"
        "void main() {\n    uint i = gl_GlobalInvocationID.x;\n"
        "    outp[i] = float(gl_WorkGroupSize.x);\n}\n"
    )
    crossgl = _translate(tmp_path, glsl, "cgl", suffix=".comp")
    assert "local_size_x_id = 4" in crossgl and "local_size_z_id = 7" in crossgl
    code = _translate(tmp_path, crossgl, "vulkan")
    assert _spec_constant(code, 4)[1] == 64
    assert _workgroup_size_composite(code)[0] == "OpSpecConstantComposite"


def test_wgsl_sizes_the_entry_by_overrides(tmp_path):
    code = _translate(tmp_path, _source(X_AND_Z), "wgsl")
    assert "@id(4)\noverride crossgl_workgroup_size_x: u32 = 64u;" in code
    assert "@id(7)\noverride crossgl_workgroup_size_z: u32 = 1u;" in code
    assert (
        "@workgroup_size(crossgl_workgroup_size_x, 1, crossgl_workgroup_size_z)" in code
    )
    assert "vec3<u32>(crossgl_workgroup_size_x, 1u, crossgl_workgroup_size_z)" in code
    if shutil.which("naga"):
        shader = tmp_path / "out.wgsl"
        shader.write_text(code)
        _run(["naga", str(shader)])


def test_wgsl_refuses_an_id_beyond_its_pipeline_constant_range(tmp_path):
    with pytest.raises(SpecializedWorkgroupSizeError) as error:
        _translate(tmp_path, _source("local_size_x_id = 65536"), "wgsl")
    assert error.value.reason == "specialization-id-out-of-range"


def _slangc():
    configured = os.environ.get("SLANGC")
    return (
        configured
        if configured and os.path.exists(configured)
        else shutil.which("slangc")
    )


def test_slang_sizes_numthreads_by_constant_ids(tmp_path):
    code = _translate(tmp_path, _source(X_AND_Z), "slang")
    assert "[vk::constant_id(4)] const int crossgl_workgroup_size_x = 64;" in code
    assert "[vk::constant_id(7)] const int crossgl_workgroup_size_z = 1;" in code
    assert "[numthreads(crossgl_workgroup_size_x, 1, crossgl_workgroup_size_z)]" in code
    assert "uint3(WorkgroupSize())" in code and "gl_WorkGroupSize" not in code
    slangc = _slangc()
    if slangc and shutil.which("spirv-dis"):
        shader, binary = tmp_path / "out.slang", tmp_path / "out.spv"
        shader.write_text(code)
        _run(
            [
                slangc,
                str(shader),
                "-target",
                "spirv",
                "-entry",
                "main",
                "-o",
                str(binary),
            ]
        )
        disassembly = _run(["spirv-dis", str(binary)])
        assert re.search(r"OpDecorate %\w+ SpecId 4\b", disassembly)
        assert "LocalSizeId" in disassembly


def test_slang_reads_a_fixed_workgroup_size_through_its_intrinsic(tmp_path):
    code = _translate(tmp_path, _source("local_size_x = 64"), "slang")
    assert "[numthreads(64, 1, 1)]" in code
    assert "uint3(WorkgroupSize())" in code and "gl_WorkGroupSize" not in code


def test_slang_refuses_the_workgroup_size_in_a_helper(tmp_path):
    with pytest.raises(Exception, match="gl_WorkGroupSize: lowered to WorkgroupSize"):
        _translate(
            tmp_path, _source("local_size_x = 64", template=HELPER_READ), "slang"
        )


def test_hlsl_refuses_a_specialized_size(tmp_path):
    with pytest.raises(SpecializedWorkgroupSizeError) as error:
        _translate(tmp_path, _source(X_AND_Z), "directx")
    assert error.value.target == "HLSL"
    assert error.value.specialization_ids == (4, 7)
    assert error.value.reason == "target-has-no-specialized-workgroup-size"


@pytest.mark.parametrize(
    "layout, reason",
    [
        ("local_size_x_id = 1, local_size_y_id = 1", "duplicate-specialization-id"),
        ("local_size_x_id = -1", "non-literal-specialization-id"),
        ("local_size_x_id = 2147483648", "specialization-id-out-of-range"),
    ],
)
def test_misstated_ids_are_refused(tmp_path, layout, reason):
    for backend in ("vulkan", "opengl", "wgsl", "slang"):
        with pytest.raises(SpecializedWorkgroupSizeError) as error:
            _translate(tmp_path, _source(layout), backend)
        assert error.value.reason == reason
