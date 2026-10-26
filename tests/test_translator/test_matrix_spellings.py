"""Non-square ``floatNxM`` matrices have no single shape in CrossGL and are refused (issue #3209).

``float2x3`` is 2 columns of 3 rows in MSL and 2 rows of 3 columns in HLSL and Slang. The Metal generator kept the
MSL reading, the GLSL generator took the HLSL one and the HLSL and SPIR-V generators mixed the two, so one CrossGL
program meant different matrices on each target. CrossGL spells matrices ``matCxR`` (ADR 0004), the SPIR-V importer
now writes that spelling, and every target refuses the ambiguous ones with a typed diagnostic.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import AmbiguousMatrixSpellingError
from crosstl.translator.semantics.matrix_spellings import ambiguous_matrix_spelling

TARGETS = (
    "metal",
    "opengl",
    "directx",
    "vulkan",
    "slang",
    "cuda",
    "hip",
    "rust",
    "mojo",
)

ISSUE_REPRODUCER = """
shader Nsq {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            float2x3 m = float2x3(float3(data[i], 1.0, 2.0), float3(3.0, 4.0, 5.0));
            float3 v = m * float2(1.0, 2.0);
            data[i] = v.z + m[1][2];
        }
    }
}
"""

CONSTRUCTOR_ONLY = """
shader Nsq {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;

        void main() {
            data[0] = double4x2(1.0)[0][0];
        }
    }
}
"""

SQUARE = """
shader Sq {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* data;

        void main() {
            float3x3 m = float3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
            data[0] = (m * float3(1.0, 0.0, 0.0)).y + m[2][0];
        }
    }
}
"""

SQUARE_DIAGONAL = SQUARE.replace(
    "float3x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)", "float3x3(data[1])"
)

# OpTypeMatrix %v3float 2: two columns of three rows, CrossGL mat2x3; z = 3 * 1 + 1 * 2.
SPIRV_NON_SQUARE = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %Data Block
OpMemberDecorate %Data 0 Offset 0
OpDecorate %data DescriptorSet 0
OpDecorate %data Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%v2float = OpTypeVector %float 2
%v3float = OpTypeVector %float 3
%mat2v3 = OpTypeMatrix %v3float 2
%Data = OpTypeStruct %float
%ptr_data = OpTypePointer StorageBuffer %Data
%ptr_float = OpTypePointer StorageBuffer %float
%ptr_mat = OpTypePointer Function %mat2v3
%data = OpVariable %ptr_data StorageBuffer
%int_0 = OpConstant %int 0
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%f3 = OpConstant %float 3
%col0 = OpConstantComposite %v3float %f1 %f2 %f3
%col1 = OpConstantComposite %v3float %f3 %f2 %f1
%mat = OpConstantComposite %mat2v3 %col0 %col1
%vec = OpConstantComposite %v2float %f1 %f2
%main = OpFunction %void None %fn
%entry = OpLabel
%m = OpVariable %ptr_mat Function
OpStore %m %mat
%loaded = OpLoad %mat2v3 %m
%product = OpMatrixTimesVector %v3float %loaded %vec
%z = OpCompositeExtract %float %product 2
%slot = OpAccessChain %ptr_float %data %int_0
OpStore %slot %z
OpReturn
OpFunctionEnd
"""


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", TARGETS)
def test_issue_reproducer_is_refused_on_every_target(tmp_path, backend):
    with pytest.raises(AmbiguousMatrixSpellingError) as caught:
        _translate(tmp_path, ISSUE_REPRODUCER, backend)
    error = caught.value
    assert error.code == "project.translate.ambiguous-matrix-spelling"
    assert error.type_name == "float2x3"
    assert "mat2x3" in str(error) and "mat3x2" in str(error)


@pytest.mark.parametrize("backend", ("metal", "opengl", "directx"))
def test_a_constructor_without_a_declaration_is_refused(tmp_path, backend):
    with pytest.raises(AmbiguousMatrixSpellingError, match="double4x2"):
        _translate(tmp_path, CONSTRUCTOR_ONLY, backend)


@pytest.mark.parametrize(
    ("name", "readings"),
    (
        ("float2x3", ("mat2x3", "mat3x2")),
        ("float4x2", ("mat4x2", "mat2x4")),
        ("double3x4", ("dmat3x4", "dmat4x3")),
        ("simd_float2x4", ("mat2x4", "mat4x2")),
        ("min16float3x2", ("mat3x2", "mat2x3")),
        ("float3x3", None),
        ("double4x4", None),
        ("int2x3", None),
        ("bool3x2", None),
        ("mat2x3", None),
    ),
)
def test_only_non_square_floating_point_spellings_are_ambiguous(name, readings):
    assert ambiguous_matrix_spelling(name) == readings


@pytest.mark.parametrize(
    "source", (SQUARE, SQUARE_DIAGONAL), ids=("columns", "diagonal")
)
@pytest.mark.parametrize("backend", ("metal", "opengl", "directx", "vulkan"))
def test_square_spellings_stay_aliases_of_matn(tmp_path, backend, source):
    """``float3x3(...)`` fills columns and ``float3x3(s)`` is diagonal, as ``mat3``; HLSL output filled rows."""
    spelled = _translate(tmp_path, source, backend)
    canonical = _translate(
        tmp_path,
        source.replace("float3x3", "mat3").replace("float3(", "vec3("),
        backend,
    )
    assert spelled == canonical


def test_spirv_import_spells_non_square_matrices_mat_c_x_r(tmp_path):
    code = _translate(tmp_path, SPIRV_NON_SQUARE, "cgl", suffix="spvasm")
    assert "mat2x3 m;" in code
    assert "mat2x3(float3(1.0, 2.0, 3.0), float3(3.0, 2.0, 1.0))" in code
    assert "float2x3" not in code


@pytest.mark.parametrize("backend", ("metal", "opengl", "directx", "vulkan"))
def test_spirv_import_with_a_non_square_matrix_translates(tmp_path, backend):
    code = _translate(tmp_path, SPIRV_NON_SQUARE, backend, suffix="spvasm")
    assert code.strip()


def test_spirv_round_trip_with_a_non_square_matrix_validates(tmp_path):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        pytest.skip("spirv-as and spirv-val are not installed")
    code = _translate(tmp_path, SPIRV_NON_SQUARE, "vulkan", suffix="spvasm")
    (tmp_path / "out.spvasm").write_text(code)
    for command in (
        [assembler, "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
        [validator, "--target-env", "vulkan1.1", "out.spv"],
    ):
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
