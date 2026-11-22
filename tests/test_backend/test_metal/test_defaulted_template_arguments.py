"""Class-template specializations spelled with and without their defaulted arguments.

``Cell<float>`` and ``Cell<float, void>`` name one specialization when the second parameter defaults to ``void``
([temp.arg.general]/9, [temp.type]/1). The importer must materialize one record for it, so pointers that the source
treats as one type still convert after the round trip.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"

GENERIC_HELPER = """
template <typename T, typename = void>
struct Cell {
  T val;
};

template <typename T>
void put(device Cell<T>* object, T value, uint offset) {
  object[offset].val = value;
}

kernel void store(device Cell<float>* out [[buffer(0)]],
                  uint i [[thread_position_in_grid]]) {
  device Cell<float, void>* cells = out;
  put(cells, float(i), i);
}
"""

DEPENDENT_DEFAULTS = """
template <typename T, typename U = T>
struct Pair { T a; U b; };
template <typename T, int N = 2>
struct Arr { T v[N]; };
kernel void store(device Pair<float>* out [[buffer(0)]],
                  device Arr<float>* arr [[buffer(1)]],
                  device Pair<float, int>* mixed [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
  device Pair<float, float>* p = out;
  p[i].a = float(i);
  p[i].b = float(i) + 1.0f;
  device Arr<float, 1 + 1>* a = arr;
  a[i].v[1] = float(i);
  mixed[i].b = int(i);
}
"""

PARTIAL_SPECIALIZATION = """
template <typename T, typename = void>
struct Cell {
  T val;
  T pad;
};
template <typename T>
struct Cell<T, enable_if_t<is_floating_point_v<T>>> {
  T val;
};
kernel void store(device Cell<float>* out [[buffer(0)]],
                  device Cell<int>* ints [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
  device Cell<float, void>* cells = out;
  cells[i].val = float(i);
  device Cell<int, void>* icells = ints;
  icells[i].pad = int(i);
}
"""


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _records(crossgl):
    return re.findall(r"^\s*struct (\w+) \{", crossgl, re.MULTILINE)


def _run(tool, *args):
    executable = shutil.which(tool)
    if executable is None:
        pytest.skip(f"{tool} is not installed")
    result = subprocess.run([executable, *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _compile_metal(tmp_path, source):
    artifact = tmp_path / "roundtrip.metal"
    artifact.write_text(_translate(tmp_path, source, "metal"), encoding="utf-8")
    _run(
        "xcrun",
        "--sdk",
        "macosx",
        "metal",
        "-Werror",
        "-fno-fast-math",
        "-c",
        str(artifact),
        "-o",
        str(tmp_path / "roundtrip.air"),
    )
    return artifact.read_text(encoding="utf-8")


def test_spellings_with_and_without_defaults_share_one_record(tmp_path):
    crossgl = _translate(tmp_path, GENERIC_HELPER)
    assert _records(crossgl) == ["Cell_float"]
    assert "device Cell_float* cells = out_;" in crossgl


def test_generic_helper_round_trip_compiles_on_metal(tmp_path):
    metal = _compile_metal(tmp_path, GENERIC_HELPER)
    # The helper's specialization of the generic struct and the importer's
    # record are the same type and are defined once.
    assert metal.count("struct Cell_float {") == 1


def test_generic_helper_compiles_for_directx(tmp_path):
    hlsl = tmp_path / "kernel.hlsl"
    hlsl.write_text(_translate(tmp_path, GENERIC_HELPER, "directx"), encoding="utf-8")
    _run("dxc", "-T", "cs_6_0", "-E", "CSMain", str(hlsl), "-Fo", str(tmp_path / "k"))


def test_generic_helper_compiles_for_opengl(tmp_path):
    glsl = tmp_path / "kernel.comp"
    glsl.write_text(_translate(tmp_path, GENERIC_HELPER, "opengl"), encoding="utf-8")
    _run("glslangValidator", "-S", "comp", str(glsl))


def test_dependent_and_value_defaults_canonicalize(tmp_path):
    crossgl = _translate(tmp_path, DEPENDENT_DEFAULTS)
    # Pair<float, float> is Pair<float>; Arr<float, 1 + 1> is Arr<float>.
    # Pair<float, int> is a different specialization.
    assert sorted(_records(crossgl)) == ["Arr_float", "Pair_float", "Pair_float_int"]
    _compile_metal(tmp_path, DEPENDENT_DEFAULTS)


def test_defaults_that_select_a_partial_specialization(tmp_path):
    crossgl = _translate(tmp_path, PARTIAL_SPECIALIZATION)
    records = _records(crossgl)
    # float selects the partial specialization through its defaulted argument;
    # int keeps the primary template. Each spelling pair yields one record.
    assert sorted(records) == ["Cell_float_void", "Cell_int"]
    assert re.search(r"struct Cell_float_void \{\s*float val;\s*\}", crossgl)
    assert re.search(r"struct Cell_int \{\s*int val;\s*int pad;\s*\}", crossgl)
    _compile_metal(tmp_path, PARTIAL_SPECIALIZATION)


# A function template deduced from a local whose type spells the defaulted ``Frag_t`` (or omits it while the
# parameter spells it) receives the one ``Tile_float_1_1`` record ([temp.deduct.type]/9, [temp.arg.general]/9), so
# the member call ``D.frag_at`` lowers and the static call through the nested alias reaches ``Frag<float, 8, 8>``.
TILE_TEMPLATES = """
template <typename T, int R, int C>
struct Frag {
  static METAL_FUNC void add(thread T& d, thread T& a) { d += a; }
};

template <typename T, int M, int N, class Frag_t = Frag<T, 8, 8>>
struct Tile {
  using Frag_type = Frag_t;
  T v[M * N];
  METAL_FUNC thread T& frag_at(const short i, const short j) { return v[i * N + j]; }
};
"""

TILE_ADD = {
    "omitted-parameter": (
        """
template <typename T, int M, int N>
METAL_FUNC void tile_add(thread Tile<T, M, N>& D, thread Tile<T, M, N>& A) {
  for (short m = 0; m < M; ++m)
    for (short n = 0; n < N; ++n)
      Tile<T, M, N>::Frag_type::add(D.frag_at(m, n), A.frag_at(m, n));
}
"""
    ),
    "spelled-parameter": (
        """
template <typename T, int M, int N, class F>
METAL_FUNC void tile_add(thread Tile<T, M, N, F>& D, thread Tile<T, M, N, F>& A) {
  for (short m = 0; m < M; ++m)
    for (short n = 0; n < N; ++n)
      F::add(D.frag_at(m, n), A.frag_at(m, n));
}
"""
    ),
}

TILE_KERNEL = """
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  LOCAL_TYPE P, Q;
  P.v[0] = float(tid);
  Q.v[0] = 2.0f;
  tile_add(P, Q);
  out[tid] = P.v[0];
}
"""

TILE_LOCAL_TYPES = {
    "spelled-local": "Tile<float, 1, 1, Frag<float, 8, 8>>",
    "omitted-local": "Tile<float, 1, 1>",
}


def _tile_program(parameter, local):
    return (
        TILE_TEMPLATES
        + TILE_ADD[parameter]
        + TILE_KERNEL.replace("LOCAL_TYPE", TILE_LOCAL_TYPES[local])
    )


@pytest.mark.parametrize("local", sorted(TILE_LOCAL_TYPES))
@pytest.mark.parametrize("parameter", sorted(TILE_ADD))
@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_deduction_through_a_defaulted_argument(tmp_path, backend, parameter, local):
    code = _translate(tmp_path, _tile_program(parameter, local), backend)
    # One record and one specialization; the member call lowered to the element it returns.
    assert len(re.findall(r"\bstruct Tile_float_1_1\b", code)) == 1, code
    assert not re.search(r"\bfrag_at\b", code), code
    assert re.search(r"Frag_float_8_8_{1,2}add\(D\.v\[", code), code
    assert re.search(r"\btile_add_float_1_1\w*\(P, Q\)", code), code
    suffix = {"opengl": "comp", "directx": "hlsl"}[backend]
    artifact = tmp_path / f"kernel.{suffix}"
    artifact.write_text(code, encoding="utf-8")
    if backend == "opengl":
        _run("glslangValidator", "-S", "comp", str(artifact))
    else:
        _run(
            "dxc", "-T", "cs_6_0", "-E", "k", str(artifact), "-Fo", str(tmp_path / "k")
        )


@pytest.mark.parametrize("local", sorted(TILE_LOCAL_TYPES))
@pytest.mark.parametrize("parameter", sorted(TILE_ADD))
def test_deduction_through_a_defaulted_argument_in_project_metal(
    tmp_path, parameter, local
):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "tile.metal").write_text(PREFIX + _tile_program(parameter, local))
    config = ProjectConfig(
        root=tmp_path,
        source_roots=(".",),
        include_patterns=("tile.metal",),
        targets=("metal",),
        output_dir="out",
    )
    payload = translate_project(config, format_output=False).to_json()
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    (artifact,) = payload["artifacts"]
    metal = (tmp_path / artifact["path"]).read_text(encoding="utf-8")
    assert re.search(r"Frag_float_8_8__add\(D\.v\[", metal), metal
    _run(
        "xcrun",
        "--sdk",
        "macosx",
        "metal",
        "-Werror",
        "-fno-fast-math",
        "-c",
        str(tmp_path / artifact["path"]),
        "-o",
        str(tmp_path / "tile.air"),
    )
