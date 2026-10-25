"""C-style integer aliases keep their 64-bit width in GLSL and SPIR-V output.

CrossGL's ``long``, ``ulong``, ``size_t`` and ``ptrdiff_t`` are 64-bit integers (the
semantic hub types them ``i64``/``u64``, and the HLSL, Metal and SPIR-V generators emit
them as 64-bit). GLSL used to narrow them to ``int``/``uint``, which silently changed
arithmetic results and the host layout of buffers and uniform blocks. SPIR-V widened a
32-bit unsigned value into a signed 64-bit type with ``OpSConvert``, which sign-extends
values with the high bit set.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.spirv_friendly_names import friendly_spirv

WIDE_ALIAS_COMPUTE = """
shader WideAliases {
    struct Record {
        size_t offset;
        ptrdiff_t delta;
    };

    cbuffer Limits @register(b0) {
        size_t limit;
        long bias;
    };

    ptrdiff_t span(size_t first, size_t last) {
        return ptrdiff_t(last) - ptrdiff_t(first);
    }

    compute {
        buffer ulong totals[];
        layout(local_size_x = 64) in;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            long scaled = long(index) * 3000000000;
            size_t start = size_t(index);
            ptrdiff_t width = span(start, limit);
            Record record;
            record.offset = start;
            record.delta = width;
            totals[index] = ulong(scaled + record.delta + bias) + record.offset;
        }
    }
}
"""


def _glsl(tmp_path, code):
    path = tmp_path / "shader.cgl"
    path.write_text(code)
    return crosstl.translate(str(path), backend="opengl")


def _validate_glsl(tmp_path, code, stage):
    glslang = shutil.which("glslangValidator")
    if glslang is None:
        return
    path = tmp_path / f"shader.{stage}"
    path.write_text(code)
    result = subprocess.run(
        [glslang, "-S", stage, str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _spirv(tmp_path, code):
    path = tmp_path / "shader.cgl"
    path.write_text(code)
    return friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )


def _validate_spirv(tmp_path, assembly):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    source = tmp_path / "shader.spvasm"
    source.write_text(assembly)
    binary = tmp_path / "shader.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("long", "int64_t"),
        ("signed long", "int64_t"),
        ("ptrdiff_t", "int64_t"),
        ("ulong", "uint64_t"),
        ("unsigned long", "uint64_t"),
        ("size_t", "uint64_t"),
    ],
)
def test_glsl_maps_c_integer_aliases_to_64_bit_types(alias, expected):
    assert GLSLCodeGen().map_type(alias) == expected


def test_glsl_keeps_c_integer_aliases_64_bit_in_code_and_storage(tmp_path):
    generated = _glsl(tmp_path, WIDE_ALIAS_COMPUTE)

    assert generated.count("#extension GL_ARB_gpu_shader_int64 : require") == 1
    assert "uint64_t limit;" in generated
    assert "int64_t bias;" in generated
    assert re.search(r"\buint64_t offset;", generated)
    assert re.search(r"\bint64_t delta;", generated)
    assert "uint64_t totals[];" in generated
    assert "int64_t span(uint64_t first, uint64_t last)" in generated
    assert re.search(r"\bint64_t scaled = ", generated)
    assert re.search(r"\buint64_t start = ", generated)
    assert re.search(r"\bint64_t width = ", generated)
    for narrowed in (
        r"\bint scaled\b",
        r"\buint start\b",
        r"\bint width\b",
        r"\buint limit;",
        r"\buint totals\[\]",
    ):
        assert re.search(narrowed, generated) is None, narrowed
    for alias in (r"\blong\b", r"\bulong\b", r"\bsize_t\b", r"\bptrdiff_t\b"):
        assert re.search(alias, generated) is None, alias
    _validate_glsl(tmp_path, generated, "comp")


def test_glsl_es_refuses_c_integer_aliases_instead_of_narrowing(tmp_path):
    shader = """
    shader WideFragment {
        fragment {
            uniform int count;
            vec4 main() @ gl_FragColor {
                size_t wide = size_t(count);
                return vec4(float(wide));
            }
        }
    }
    """
    path = tmp_path / "shader.cgl"
    path.write_text(shader)

    with pytest.raises(Exception) as exc_info:
        crosstl.translate(str(path), backend="webgl")

    message = str(exc_info.value)
    assert "uint64_t" in message or "int64" in message
    assert "integer width" in message or "GL_ARB_gpu_shader_int64" in message


def test_spirv_lowers_c_integer_alias_constructors_to_64_bit_types(tmp_path):
    assembly = _spirv(tmp_path, WIDE_ALIAS_COMPUTE)

    assert "OpCapability Int64" in assembly
    assert re.search(r"%long = OpTypeInt 64 1", assembly)
    assert re.search(r"%ulong = OpTypeInt 64 0", assembly)
    assert "size_t" not in assembly and "ptrdiff_t" not in assembly
    _validate_spirv(tmp_path, assembly)


def test_spirv_widens_unsigned_values_into_signed_64_bit_by_zero_extension(tmp_path):
    shader = """
    shader ZeroExtend {
        compute {
            buffer long results[];
            layout(local_size_x = 64) in;
            void main() {
                uint index = gl_GlobalInvocationID.x;
                uint word = index | 2147483648u;
                results[index] = long(word);
            }
        }
    }
    """
    assembly = _spirv(tmp_path, shader)

    unsigned_ids = set(re.findall(r"(%\w+) = \S+ %uint\b", assembly))
    unsigned_ids.update(re.findall(r"(%\w+) = OpBitwiseOr %uint\b", assembly))
    for source in re.findall(r"= OpSConvert %long (%\w+)", assembly):
        assert source not in unsigned_ids, f"{source} is sign-extended"
    widened = re.search(r"(%\w+) = OpUConvert %ulong (%\w+)", assembly)
    assert widened is not None
    assert re.search(rf"= OpBitcast %long {widened.group(1)}\b", assembly)
    _validate_spirv(tmp_path, assembly)
