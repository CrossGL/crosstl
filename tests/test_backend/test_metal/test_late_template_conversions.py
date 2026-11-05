"""Struct conversions and defaulted template calls reached only through struct template members.

A function template that converts an aggregate with ``T(value)`` is first instantiated after member lowering when
its only caller is a member of a struct template. The aggregate's conversion operator exists by then only as its
free ``Struct__operator_<type>`` helper, and the conversion still has to call it ([dcl.init.general]/16.7,
[over.match.conv]). The same helper instantiated from a kernel and from such a member must be defined once.

A call without template arguments to an all-defaulted constexpr function template names the default
specialization ([temp.arg.explicit]/4). In a static data member initializer or a constructor's member initializer
of a concrete struct, the call must survive the removal of the template's declaration.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalSourceOverloadResolutionError,
)
from crosstl.project import ProjectConfig, translate_project

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

BYTE_STRUCT = """
struct Byte {
  uint8_t bits;
  operator float() thread { return float(bits) * 0.5f; }
};
"""

LATE_CONVERSION = PRELUDE + BYTE_STRUCT + """
template <typename T>
static inline T decode(uint8_t s) {
  return CONVERSION;
}

template <typename T>
struct Reader {
  uint8_t byte;
  T read() thread { return decode<T>(byte); }
};

kernel void probe(device float* out [[buffer(0)]],
                  const device uint8_t* in [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
  Reader<float> reader{in[tid]};
  out[tid] = reader.read() + EXTRA;
}
"""

SPELLINGS = {
    "functional": "T(*(thread Byte*)(&s))",
    "braced": "T{*(thread Byte*)(&s)}",
    "static_cast": "static_cast<T>(*(thread Byte*)(&s))",
    "c_style": "(T)(*(thread Byte*)(&s))",
}

TWO_CONVERSIONS = PRELUDE + """
struct Scale {
  uint8_t bits;
  operator half() const thread { return half(bits); }
  operator float() const thread { return float(bits) * 0.5f; }
};

template <typename T>
static inline T decode(uint8_t s) {
  return T(*(thread Scale*)(&s));
}

template <typename T>
struct Reader {
  uint8_t byte;
  T read() thread { return decode<T>(byte); }
};

kernel void probe(device float* out [[buffer(0)]],
                  device NARROW* narrow_out [[buffer(1)]],
                  const device uint8_t* in [[buffer(2)]],
                  uint tid [[thread_position_in_grid]]) {
  Reader<float> wide{in[tid]};
  Reader<NARROW> narrow{in[tid]};
  out[tid] = wide.read();
  narrow_out[tid] = narrow.read();
}
"""

DEFAULTED_CONSTEXPR = PRELUDE + """
template <int width = 8>
inline constexpr short bytes_per_word() {
  return width / 8;
}

template <typename T, short bits>
struct Loader {
  static constant constexpr const short stride = bytes_per_word();
  const int ld;
  const device uint8_t* src;

  Loader(const device uint8_t* src_, const int ld_, uint tid)
      : ld(ld_), src(src_ + tid * stride) {}

  T load(int i) const thread { return T(src[i * stride]) + T(bits); }
};

template <typename T, short bits>
[[kernel]] void probe(device T* out [[buffer(0)]],
                      const device uint8_t* in [[buffer(1)]],
                      uint tid [[thread_position_in_grid]]) {
  Loader<T, bits> loader(in, 4, tid);
  out[tid] = loader.load(0) + T(bytes_per_word<32>());
}

template [[host_name("probe_float_4")]] [[kernel]] decltype(probe<float, 4>)
    probe<float, 4>;
"""

DXC_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)\s*\(")


def _translate(tmp_path, source, backend, name="probe.metal"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _late_conversion(spelling, extra="0.0f"):
    return LATE_CONVERSION.replace("CONVERSION", SPELLINGS[spelling]).replace(
        "EXTRA", extra
    )


def _helper_body(code, name):
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{(.*?)\}}", code, re.S)
    assert match, code
    return match.group(1)


def _compile(tmp_path, backend, code):
    """Compile ``code`` with the target's compiler when it is installed."""
    if backend == "metal":
        if shutil.which("xcrun") is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "generated.metal"
        command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1", "-c"]
        command += [str(path), "-o", str(tmp_path / "generated.air")]
    elif backend == "opengl":
        compiler = shutil.which("glslangValidator")
        if compiler is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "generated.comp"
        command = [compiler, "--target-env", "opengl", "-S", "comp", str(path)]
        command += ["-o", str(tmp_path / "generated.spv")]
    else:
        compiler = shutil.which("dxc")
        if compiler is None:
            pytest.skip("dxc is not installed")
        entry = DXC_ENTRY.search(code)
        assert entry, code
        path = tmp_path / "generated.hlsl"
        command = [compiler, "-T", "cs_6_2", "-E", entry.group(1)]
        command += ["-enable-16bit-types", str(path)]
        command += ["-Fo", str(tmp_path / "generated.dxil")]
    path.write_text(code)
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_late_template_conversion_calls_the_lowered_helper(tmp_path, spelling):
    code = _translate(tmp_path, _late_conversion(spelling), "metal")
    body = _helper_body(code, "decode_float")
    assert "Byte__operator_float(" in body, body
    assert not re.search(r"\bfloat\s*[({]\s*Byte\s*\{", body), body


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_late_template_conversion_compiles(tmp_path, backend):
    code = _translate(tmp_path, _late_conversion("functional"), backend)
    body = _helper_body(code, "decode_float")
    assert re.search(r"\bByte_{1,2}operator_float\s*\(", body), body
    _compile(tmp_path, backend, code)


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl"])
def test_helper_reached_from_kernel_and_member_is_defined_once(tmp_path, backend):
    source = _late_conversion("functional", extra="decode<float>(in[0])")
    code = _translate(tmp_path, source, backend)
    definitions = re.findall(r"\bdecode_float\s*\([^)]*\)\s*\{", code)
    assert len(definitions) == 1, code
    assert re.search(
        r"\bByte_{1,2}operator_float\s*\(", _helper_body(code, "decode_float")
    )
    _compile(tmp_path, backend, code)


def test_exact_result_type_selects_among_conversions(tmp_path):
    code = _translate(tmp_path, TWO_CONVERSIONS.replace("NARROW", "half"), "metal")
    assert "Scale__operator_float(" in _helper_body(code, "decode_float")
    assert "Scale__operator_half(" in _helper_body(code, "decode_half")
    _compile(tmp_path, "metal", code)


def test_single_inexact_conversion_converts_its_result(tmp_path):
    source = TWO_CONVERSIONS.replace("NARROW", "int").replace(
        "  operator half() const thread { return half(bits); }\n", ""
    )
    code = _translate(tmp_path, source, "metal")
    body = _helper_body(code, "decode_int")
    assert re.search(r"\bint\(Scale__operator_float\(", body), body
    _compile(tmp_path, "metal", code)


def test_ambiguous_inexact_conversion_is_refused(tmp_path):
    # Neither half nor float is int, and both conversions rank alike, so the
    # cast is ambiguous; the Metal compiler rejects the source as well.
    source = TWO_CONVERSIONS.replace("NARROW", "int")
    with pytest.raises(MetalSourceOverloadResolutionError, match="int\\(Scale\\)"):
        _translate(tmp_path, source, "directx")


def test_conversion_of_aggregate_without_conversion_operator_is_refused(tmp_path):
    # A scalar cast of a class value with no conversion function is ill-formed;
    # the Metal compiler rejects it, so translation must not emit it either.
    source = _late_conversion("functional").replace(
        "  operator float() thread { return float(bits) * 0.5f; }\n", ""
    )
    with pytest.raises(MetalSourceOverloadResolutionError, match="no readonly"):
        _translate(tmp_path, source, "metal")


def _translate_project(tmp_path, backend):
    (tmp_path / "loader.metal").write_text(DEFAULTED_CONSTEXPR)
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("loader.metal",),
        targets=(backend,),
        output_dir="out",
        entry_points={"loader.metal": ("probe_float_4",)},
        workgroup_size_rules={"loader.metal": (1, 1, 1)},
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    return (tmp_path / artifact["path"]).read_text()


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_defaulted_constexpr_call_in_struct_members_survives(tmp_path, backend):
    code = _translate_project(tmp_path, backend)
    # The template declaration is gone, so no call may still name it bare.
    assert not re.search(r"\bbytes_per_word\s*\(", code), code
    _compile(tmp_path, backend, code)
