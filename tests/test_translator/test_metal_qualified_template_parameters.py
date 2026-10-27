"""Late helper deduction treats ``metal::vec<T, N>`` and ``vec<T, N>`` as one type.

A template helper may spell a parameter with the Metal standard-library namespace (``metal::vec<T, 2> b``) while its
caller passes an unqualified ``vec<float, 2>``. When that call sits in a member function of a template struct, project
translation lowers the member to a free function first and then deduces the helper's arguments from the call. Deduction
compared the generic base names textually, so ``metal::vec`` did not match ``vec`` and the explicit ``helper<float>``
call stayed unmaterialized, failing with ``project.translate.template-materialization-unsupported`` on template-hostile
targets.
"""

import shutil
import subprocess

import pytest

import crosstl.project.pipeline as project_pipeline
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import ProjectConfig, translate_project

SOURCE = """
#include <metal_stdlib>
using namespace metal;

template <typename T>
vec<T, 2> cmul(vec<T, 2> a, metal::vec<T, 2> b) {
  return vec<T, 2>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

template <typename T>
struct Loader {
  device vec<T, 2>* out;
  Loader(device vec<T, 2>* out_) thread : out(out_) {}
  void load(uint i, const device vec<T, 2>* w) const thread {
    out[i] = cmul<T>(out[i], w[i]);
  }
};

template <typename T>
[[kernel]] void scale(
    device vec<T, 2>* out [[buffer(0)]],
    const device vec<T, 2>* w [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  Loader<T> loader = Loader<T>(out);
  loader.load(i, w);
}

template [[host_name("scale_float")]] [[kernel]] decltype(scale<float>) scale<float>;
"""

ENTRY = "scale_float"


@pytest.mark.parametrize(
    "expected_type",
    ["metal::vec<T, 2>", "::metal::vec<T, 2>", "const thread metal::vec<T, 2>&"],
)
def test_metal_qualified_vector_parameter_binds_unqualified_argument(expected_type):
    bindings = {}
    assert project_pipeline._collect_metal_template_type_bindings(
        MetalPreprocessor(),
        expected_type=expected_type,
        actual_type="vec<float,2>",
        bindings=bindings,
        template_parameters={"T"},
    )
    assert bindings == {"T": "float"}

    reverse_bindings = {}
    assert project_pipeline._collect_metal_template_type_bindings(
        MetalPreprocessor(),
        expected_type="vec<T, 2>",
        actual_type="metal::vec<half,2>",
        bindings=reverse_bindings,
        template_parameters={"T"},
    )
    assert reverse_bindings == {"T": "half"}


def test_other_namespaces_still_distinguish_generic_types():
    bindings = {}
    assert not project_pipeline._collect_metal_template_type_bindings(
        MetalPreprocessor(),
        expected_type="other::vec<T, 2>",
        actual_type="vec<float,2>",
        bindings=bindings,
        template_parameters={"T"},
    )
    assert bindings == {}
    assert project_pipeline._metal_canonical_generic_base("metal::vec") == "vec"
    assert project_pipeline._metal_canonical_generic_base("::metal::vec") == "vec"
    assert project_pipeline._metal_canonical_generic_base("other::vec") == "other::vec"
    assert (
        project_pipeline._metal_canonical_generic_base("metalx::vec") == "metalx::vec"
    )


def _translate(tmp_path, target):
    root = tmp_path / "project"
    kernels = root / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "scale.metal").write_text(SOURCE, encoding="utf-8")
    source = "kernels/scale.metal"
    config = ProjectConfig(
        root=root,
        source_roots=("kernels",),
        include_patterns=(source,),
        targets=(target,),
        output_dir="out",
        entry_points={source: (ENTRY,)},
        entry_workgroup_size_rules={source: {ENTRY: [32, 1, 1]}},
    )
    payload = translate_project(config, format_output=False).to_json()
    errors = [d for d in payload["diagnostics"] if d["severity"] == "error"]
    assert payload["summary"]["failedCount"] == 0, errors
    assert not errors, errors
    outputs = sorted((root / "out" / target).rglob(f"{ENTRY}.*"))
    assert outputs, list((root / "out").rglob("*"))
    return outputs[0]


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_member_call_to_qualified_vector_helper_materializes(tmp_path, target):
    artifact = _translate(tmp_path, target)
    code = artifact.read_text(encoding="utf-8")
    assert "cmul<" not in code
    assert "cmul" in code

    if target == "directx" and shutil.which("dxc"):
        completed = subprocess.run(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", str(artifact)],
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
