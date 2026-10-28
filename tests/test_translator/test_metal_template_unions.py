"""Template unions and reserved-spelling template arguments materialize for template-hostile targets.

Two Metal template features failed before translation reached the union lowering:

* A struct template instantiated with a functor whose name starts with an underscore
  (``helper<uint, __Add<uint>>``) was refused as ``template-materialization-unsupported``: the
  placeholder scan split the mangled specialization name on underscores and read the bare ``Add``
  fragment as an unresolved template parameter. The specialization registry records the real
  arguments, so the scan now reads those.
* A ``template <typename T> union`` was not collected as a template aggregate, so ``Packed<uchar>``
  never became a concrete union and its members had no inferable type. Its array extent may also be a
  ``sizeof`` expression that folds only after rendering.
"""

import shutil
import subprocess

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.project import ProjectConfig, translate_project

FUNCTOR_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
struct __Add {
  static bool condition(T a, T b) { return true; }
  T operator()(T a, T b) thread { return a + b; }
};

template <typename T, typename Op>
struct update_helper {
  T operator()(T init, T update) thread {
    Op op;
    return op(update, init);
  }
};

template <typename T, typename Op>
METAL_FUNC void update_and_store(device T* object, T update, uint offset) {
  update_helper<T, Op> helper;
  if (Op::condition(update, object[offset])) {
    object[offset] = helper(object[offset], update);
  }
}

kernel void apply_add(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  update_and_store<uint, __Add<uint>>(out, 3u, gid);
}
"""

UNION_SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
union packed_word {
  T val[sizeof(uint) / sizeof(T)];
  uint bits;
};

template <typename T>
uint replace_lane(packed_word<T> init, T update, uint lane) {
  init.val[lane] = update;
  return init.bits;
}

kernel void pack_lanes(
    device uint* out [[buffer(0)]],
    device const ushort* upd [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  packed_word<ushort> expected;
  expected.bits = out[gid / 2u];
  out[gid / 2u] = replace_lane<ushort>(expected, upd[gid], gid % 2u);
}
"""


def _translate(tmp_path, source_text, entry, target):
    root = tmp_path / "project"
    kernels = root / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "kernel.metal").write_text(source_text, encoding="utf-8")
    source = "kernels/kernel.metal"
    config = ProjectConfig(
        root=root,
        source_roots=("kernels",),
        include_patterns=(source,),
        targets=(target,),
        output_dir="out",
        entry_points={source: (entry,)},
        workgroup_size=(32, 1, 1),
    )
    payload = translate_project(config, format_output=False).to_json()
    errors = [d for d in payload["diagnostics"] if d["severity"] == "error"]
    assert payload["summary"]["failedCount"] == 0, errors
    outputs = sorted((root / "out" / target).rglob(f"{entry}.*"))
    assert outputs, list((root / "out").rglob("*"))
    return outputs[0]


def _validate(artifact, target):
    if target == "directx" and shutil.which("dxc"):
        completed = subprocess.run(
            [
                "dxc",
                "-T",
                "cs_6_2",
                "-E",
                "CSMain",
                "-enable-16bit-types",
                str(artifact),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
    if target == "opengl" and shutil.which("glslangValidator"):
        completed = subprocess.run(
            ["glslangValidator", "-S", "comp", str(artifact)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout[-2000:]


def test_placeholder_scan_reads_recorded_specialization_arguments():
    concrete = "update_helper_uint___Add_uint"
    scan = project_pipeline._unresolved_metal_template_placeholders_in_type
    # Without the registry the mangled name splits into a bare ``Add`` fragment.
    assert scan(concrete, {"T", "Op"}, {concrete}, {"__Add", "update_helper"}) == [
        "Add"
    ]
    registry = {concrete: ("update_helper", ("uint", "__Add<uint>"))}
    assert (
        scan(concrete, {"T", "Op"}, {concrete}, {"__Add", "update_helper"}, registry)
        == []
    )
    # A recorded argument that is still a template parameter stays unresolved.
    pending = {concrete: ("update_helper", ("T", "__Add<T>"))}
    assert scan(
        concrete, {"T", "Op"}, {concrete}, {"__Add", "update_helper"}, pending
    ) == ["T"]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_reserved_spelling_functor_argument_materializes(tmp_path, target):
    artifact = _translate(tmp_path, FUNCTOR_SOURCE, "apply_add", target)
    code = artifact.read_text(encoding="utf-8")
    assert "update_helper<" not in code
    _validate(artifact, target)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_template_union_materializes_as_packed_lanes(tmp_path, target):
    artifact = _translate(tmp_path, UNION_SOURCE, "pack_lanes", target)
    code = artifact.read_text(encoding="utf-8")
    assert "CrossGLUnionStorage" in code
    assert "uint16_lane_w1" in code
    assert "packed_word<" not in code
    _validate(artifact, target)
