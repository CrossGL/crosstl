"""The rust-gpu probes for Rust features, and the rust-gpu toolchain that compiles them."""

import pytest

from tools.coverage import lowering, probes, rustgpu, semantic, validators
from tools.coverage.features import rust
from tools.coverage.model import VOID, Feature, Param, Signature, Type

from .rust_sources import ARCH, FLOAT, VEC3, VEC3_TYPE


def feature(name, result, *params, spelling=""):
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature("rust", "function", name, signature, spelling=spelling)


SAMPLES = [
    feature("Float::sin", Type("float"), Type("float"), spelling=".sin"),
    feature("Vec3::dot", Type("float"), VEC3_TYPE, VEC3_TYPE, spelling=".dot"),
    feature("Vec3::cross", VEC3_TYPE, VEC3_TYPE, VEC3_TYPE, spelling=".cross"),
    feature("Vec3::cmpeq", Type("bool", (3,)), VEC3_TYPE, VEC3_TYPE, spelling=".cmpeq"),
    feature(
        "atomic_i_add",
        Type("uint"),
        Param(Type("uint"), "inout"),
        Type("uint"),
        spelling="unsafe atomic_i_add::<_, { Scope::Subgroup as u32 }, { Semantics::NONE.bits() }>",
    ),
    feature("workgroup_memory_barrier_with_group_sync", VOID),
    feature("Float::nan", Type("float"), spelling="f32::nan"),
    feature(
        "read_clock_khr",
        Type("uint64"),
        spelling="unsafe read_clock_khr::<{ Scope::Subgroup as u32 }>",
    ),
]


def test_rust_probe_is_a_rust_gpu_compute_entry_point():
    source = probes.build(SAMPLES[1], "rust").source
    assert source.startswith("#![no_std]\nuse spirv_std::arch::*;\n")
    assert "#[spirv(compute(threads(1)))]\npub fn probe(\n" in source
    assert (
        "    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] fin: &[f32],\n"
        in source
    )
    assert "outp: &mut [f32],\n" in source
    assert "let a1: Vec3 = Vec3::new(fin[i + 3], fin[i + 4], fin[i + 5]);" in source
    assert "let r: f32 = a0.dot(a1);" in source
    assert probes.build(SAMPLES[0], "rust").suffix == ".rs"


def test_rust_probe_widens_3_vectors_for_the_output_buffer():
    source = probes.build(SAMPLES[2], "rust").source
    assert "outp: &mut [Vec4]," in source
    assert "outp[i] = r.extend(0.0);" in source


def test_rust_probe_wraps_unsafe_calls_and_passes_inout_by_reference():
    source = probes.build(SAMPLES[4], "rust").source
    assert "unsafe { atomic_i_add::<_, " in source
    assert ">(&mut outp[i], a1) };" in source
    barrier = probes.build(SAMPLES[5], "rust").source
    assert "workgroup_memory_barrier_with_group_sync();\n    outp[i] = 0;" in barrier
    baseline = probes.build(SAMPLES[1], "rust", baseline=True).source
    assert "dot" not in baseline
    assert "outp[i] = outp[i];" in baseline


RAY_TRACING = """
pub unsafe fn report_intersection(hit: f32, hit_kind: u32) -> bool {
    false
}

pub unsafe fn terminate_ray() -> ! {
    loop {}
}

pub unsafe fn execute_callable<T, const ID: usize>(data: &T) {}
"""
PRIMITIVE = "pub unsafe fn emit_vertex() {}\n"


def test_rust_ray_tracing_functions_are_probed_in_a_stage_that_has_them():
    found = rust.parse(
        {"ray_tracing": RAY_TRACING, "primitive": PRIMITIVE, "arch": ARCH}, FLOAT, VEC3
    )
    features = {f.name: f for f in found}
    assert features["report_intersection"].stage == "intersection"
    assert features["terminate_ray"].stage == "any-hit"
    assert features["execute_callable"].stage == "callable"
    assert features["emit_vertex"].stage == "geometry"
    # A function that never returns is called as a statement.
    assert features["terminate_ray"].signature == Signature(VOID, ())
    assert features["kill"].signature == Signature(VOID, ())


STAGE_SAMPLES = [
    Feature(
        "rust",
        "function",
        "report_intersection",
        Signature(Type("bool"), (Param(Type("float")), Param(Type("uint")))),
        spelling="unsafe report_intersection",
        stage="intersection",
    ),
    Feature(
        "rust",
        "function",
        "terminate_ray",
        Signature(VOID, ()),
        spelling="unsafe terminate_ray",
        stage="any-hit",
    ),
    Feature(
        "rust",
        "function",
        "emit_vertex",
        Signature(VOID, ()),
        spelling="unsafe emit_vertex",
        stage="geometry",
    ),
    Feature("rust", "function", "kill", Signature(VOID, ()), stage="fragment"),
    Feature(
        "rust",
        "function",
        "set_mesh_outputs_ext",
        Signature(VOID, (Param(Type("uint")), Param(Type("uint")))),
        spelling="unsafe set_mesh_outputs_ext",
        stage="mesh",
    ),
    Feature(
        "rust",
        "function",
        "emit_mesh_tasks_ext_payload",
        Signature(
            VOID, (*[Param(Type("uint"))] * 3, Param(Type.other("udt"), "inout"))
        ),
        spelling="unsafe emit_mesh_tasks_ext_payload",
        stage="task",
    ),
    Feature(
        "rust",
        "function",
        "execute_callable",
        Signature(VOID, (Param(Type.other("udt"), "inout"),)),
        spelling="unsafe execute_callable::<_, 0>",
        stage="callable",
    ),
]


def test_rust_stage_probes_name_their_execution_model():
    intersection, hit, geometry = (
        probes.build(f, "rust").source for f in STAGE_SAMPLES[:3]
    )
    assert (
        "#[spirv(intersection)]\npub fn probe(\n    #[spirv(storage_buffer"
        in intersection
    )
    assert "Payload" not in intersection
    assert "let r: bool = unsafe { report_intersection(a0, a1) };" in intersection
    assert "pub struct Payload {\n    pub color: Vec4,\n}\n\n#[spirv(any_hit)]" in hit
    assert "    #[spirv(incoming_ray_payload)] payload: &mut Payload,\n" in hit
    assert "    #[spirv(hit_attribute)] attributes: &mut Vec2,\n" in hit
    assert "unsafe { terminate_ray() };" in hit
    assert (
        "#[spirv(geometry(input_points = 1, output_points = 1, output_vertices = 1))]"
        in geometry
    )


MESH_SHADING = """
pub unsafe fn set_mesh_outputs_ext(vertex_count: u32, primitive_count: u32) {}

pub unsafe fn emit_mesh_tasks_ext_payload<T>(
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
    payload: &mut T,
) -> ! {
    loop {}
}
"""


WHERE_VEC3 = """
impl Vec3 {
    pub fn map<F>(self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        self
    }

    pub fn scaled<T>(self, s: T) -> Self
    where
        T: Into<f32>,
    {
        self
    }

    pub fn all_by<F>(self, f: F) -> bool
    where
        F: Fn(f32) -> bool,
    {
        true
    }
}
"""


def test_rust_where_clauses_bound_type_parameters():
    found = rust.parse({}, FLOAT, WHERE_VEC3, frozenset())
    features = {f.name: f for f in found}
    # A closure parameter keeps its bound, and the probe passes a closure that doubles its argument.
    mapped = features["Vec3::map"]
    assert mapped.signature.params[1] == Param(Type.other("Fn(f32) -> f32"))
    assert "a0.map(|x: f32| x + x)" in probes.build(mapped, "rust").source
    with pytest.raises(probes.NoSpelling, match="no closures"):
        probes.build(mapped, "crossgl")
    # A closure whose result differs from its argument has no probe.
    with pytest.raises(
        probes.ProbeError, match=r"parameters of type Fn\(f32\) -> bool"
    ):
        probes.build(features["Vec3::all_by"], "rust")
    # A bound in the where clause types its parameter as a bound in the generics would.
    assert features["Vec3::scaled"].signature.params[1] == Param(Type("float"))


def test_rust_payload_functions_take_the_probe_payload():
    found = rust.parse(
        {"mesh_shading": MESH_SHADING, "ray_tracing": RAY_TRACING}, FLOAT, VEC3
    )
    features = {f.name: f for f in found}
    task = features["emit_mesh_tasks_ext_payload"]
    assert task.stage == "task"
    assert task.signature.params[3] == Param(Type.other("udt"), "inout")
    source = probes.build(task, "rust").source
    assert "#[spirv(task_ext(threads(1)))]" in source
    assert "    #[spirv(task_payload_workgroup_ext)] payload: &mut Payload,\n" in source
    assert "emit_mesh_tasks_ext_payload(a0, a1, a2, payload)" in source
    callable_ = probes.build(features["execute_callable"], "rust").source
    assert "#[spirv(callable)]" in callable_
    assert "    #[spirv(incoming_callable_data)] payload: &mut Payload,\n" in callable_
    assert "execute_callable::<_, 0>(payload)" in callable_
    crossgl = probes.build(features["execute_callable"], "crossgl").source
    assert "struct Payload {" in crossgl
    assert "void main(Payload payload @ callable_data) {" in crossgl


def test_rust_mesh_probe_writes_a_triangle_after_the_call():
    found = rust.parse({"mesh_shading": MESH_SHADING}, FLOAT, VEC3)
    mesh = next(f for f in found if f.name == "set_mesh_outputs_ext")
    assert mesh.stage == "mesh"
    source = probes.build(mesh, "rust").source
    assert (
        "#[spirv(mesh_ext(threads(1), output_vertices = 3, output_primitives_ext = 1, output_triangles_ext))]"
        in source
    )
    assert (
        "    #[spirv(primitive_triangle_indices_ext)] indices: &mut [UVec3; 1],\n"
        in source
    )
    call = source.index("set_mesh_outputs_ext(a0, a1)")
    assert call < source.index("indices[0] = UVec3::new(0, 1, 2);")


def test_rust_mesh_and_task_probes_compile_for_vulkan_1_2(monkeypatch):
    commands = []

    def run(arguments, cwd):
        commands.append(arguments)
        return validators.Verdict(True)

    monkeypatch.setattr(rustgpu, "command", lambda target=rustgpu.TARGET: (target,))
    monkeypatch.setattr(validators, "run", run)
    for stage in ("compute", "closest-hit", "mesh", "task"):
        validators.validate("rust", "", stage)
    assert [arguments[0] for arguments in commands] == [
        "spirv-unknown-vulkan1.1",
        "spirv-unknown-vulkan1.1",
        "spirv-unknown-vulkan1.2",
        "spirv-unknown-vulkan1.2",
    ]


def test_rust_gpu_is_unavailable_without_a_build(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSTL_COVERAGE_RUST_GPU", str(tmp_path))
    rustgpu.command.cache_clear()
    rustgpu.version.cache_clear()
    try:
        assert rustgpu.command() == ()
        assert rustgpu.version() == ""
    finally:
        rustgpu.command.cache_clear()
        rustgpu.version.cache_clear()


def test_rust_gpu_needs_libraries_for_every_target(monkeypatch, tmp_path):
    built = tmp_path / "target"
    (built / "release").mkdir(parents=True)
    (built / "release" / "librustc_codegen_spirv.so").touch()
    (built / "compiletest-target-spec").mkdir()
    deps = built / "compiletest-deps"
    macros = deps / "debug" / "build" / "spirv-std-macros" / "h" / "out"
    macros.mkdir(parents=True)
    (macros / "libspirv_std_macros-h.so").touch()

    def build(target):
        (built / "compiletest-target-spec" / f"{target}.json").touch()
        for crate in ("compiler_builtins", "core", "spirv-std"):
            out = deps / target / "debug" / "build" / crate / "h" / "out"
            out.mkdir(parents=True)
            (out / f"lib{crate.replace('-', '_')}-h.rlib").touch()

    monkeypatch.setenv("CROSSTL_COVERAGE_RUST_GPU", str(tmp_path))
    monkeypatch.setenv("CROSSTL_COVERAGE_RUSTC", "rustc")
    rustgpu.command.cache_clear()
    try:
        build(rustgpu.TARGET)
        assert rustgpu.command() == ()
        rustgpu.command.cache_clear()
        build(rustgpu.MESH_TARGET)
        compute, mesh = rustgpu.command(), rustgpu.command(rustgpu.MESH_TARGET)
        assert compute[-1].endswith("spirv-unknown-vulkan1.1.json")
        assert mesh[-1].endswith("spirv-unknown-vulkan1.2.json")
        features = [
            next(a for a in flags if a.startswith("-Ctarget-feature="))
            for flags in (compute, mesh)
        ]
        assert "+MeshShadingEXT" not in features[0]
        assert "+MeshShadingEXT," in features[1]
        assert features[1].endswith(",+ext:SPV_EXT_mesh_shader")
    finally:
        rustgpu.command.cache_clear()


needs_rust_gpu = pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)

# Fine and coarse derivatives and helper invocations need capabilities beyond Vulkan 1.1's defaults.
FRAGMENT_SAMPLES = [
    Feature(
        "rust",
        "function",
        "Derivative::dfdx_fine",
        Signature(Type("float"), (Param(Type("float")),)),
        spelling=".dfdx_fine",
        stage="fragment",
    ),
    Feature(
        "rust",
        "function",
        "demote_to_helper_invocation",
        Signature(VOID, ()),
        stage="fragment",
    ),
    Feature(
        "rust",
        "function",
        "is_helper_invocation",
        Signature(Type("bool"), ()),
        stage="fragment",
    ),
]


@needs_rust_gpu
@pytest.mark.parametrize("probe_feature", SAMPLES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_rust_probes_compile_with_rust_gpu(probe_feature, baseline):
    probe = probes.build(probe_feature, "rust", baseline=baseline)
    verdict = validators.validate("rust", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


@needs_rust_gpu
@pytest.mark.parametrize("probe_feature", FRAGMENT_SAMPLES, ids=lambda f: f.name)
def test_rust_fragment_probes_compile_with_rust_gpu(probe_feature):
    probe = probes.build(probe_feature, "rust")
    verdict = validators.validate("rust", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


@needs_rust_gpu
@pytest.mark.parametrize("probe_feature", STAGE_SAMPLES, ids=lambda f: f.name)
def test_rust_stage_probes_compile_with_rust_gpu(probe_feature):
    probe = probes.build(probe_feature, "rust")
    verdict = validators.validate(
        "rust", probe.source, probe_feature.stage, entry=probe.entry
    )
    assert verdict.accepted, verdict.log


@needs_rust_gpu
def test_rust_gpu_rejects_code_that_is_not_a_shader_crate():
    verdict = validators.validate("rust", "pub fn f() -> String { String::new() }\n")
    assert not verdict.accepted
    assert "can't find crate for `std`" in verdict.log


@needs_rust_gpu
def test_rust_features_lower_to_spir_v_through_rust_gpu():
    assert lowering.can_lower("rust")
    assert semantic.lower(SAMPLES[0]) == ("GLSL.std.450:Sin", "")
    # glam's methods are Rust arithmetic, so they borrow GLSL's ids through their CrossGL spelling instead.
    dot = Feature("rust", "function", "Vec3::dot", SAMPLES[1].signature, group="vec3")
    assert semantic.lower(dot) == ("", "glam computes it with Rust arithmetic")
