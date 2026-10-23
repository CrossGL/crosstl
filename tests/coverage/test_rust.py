import pytest

from tools.coverage import probes, rustgpu, semantic, validators
from tools.coverage.features import rust
from tools.coverage.model import VOID, Feature, Param, Signature, Type

ARCH = r"""
pub type SubgroupMask = crate::glam::UVec4;

/// Returns the larger of `a` and `b` (see [`signed_max`).
#[inline]
pub fn unsigned_max<T: UnsignedInteger>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

pub unsafe fn atomic_i_add<I: Integer, const SCOPE: u32, const SEMANTICS: u32>(
    ptr: &mut I,
    value: I,
) -> I {
    let mut old = I::default();
    unsafe { asm!("%result = OpAtomicIAdd _ {ptr}", ptr = in(reg) ptr) };
    old
}

pub fn memory_barrier<const MEMORY: u32, const SEMANTICS: u32>() {}

pub fn subgroup_ballot(predicate: bool) -> SubgroupMask {
    SubgroupMask::default()
}

pub fn any<V: Vector<bool, N>, const N: usize>(vector: V) -> bool {
    false
}

pub unsafe fn emit_stream_vertex<const STREAM: i64>() {}

pub unsafe fn vector_extract_dynamic<T: Scalar, const N: usize>(
    vector: impl Vector<T, N>,
    index: usize,
) -> T {
    T::default()
}

pub unsafe fn read_clock_khr<const SCOPE: u32>() -> u64 {
    0
}

pub fn kill() -> ! {
    loop {}
}

macro_rules! macro_subgroup_op {
    ($scalar:ty, $asm_op:literal, $($name:ident, $group_op:expr),+; $docs:literal) => { $(
        #[doc = $docs]
        pub fn $name<I: ScalarOrVector<Scalar = $scalar>>(
            value: I,
        ) -> I {
            value
        }
    )+ };
}

macro_subgroup_op!(impl Float, "OpGroupNonUniformFAdd", subgroup_f_add, GroupOperation::Reduce, subgroup_inclusive_f_add, GroupOperation::InclusiveScan; r"
A floating point add group operation (of all `value` operands, contributed by active invocations.
");
macro_subgroup_op!(bool, "OpGroupNonUniformLogicalAnd", subgroup_logical_and, GroupOperation::Reduce; r"Logical and.");
"""

DERIVATIVE = """
pub unsafe trait Derivative: Sealed + Default {
    fn dfdx(self) -> Self {
        deriv_fn!(OpDPdx, self)
    }
}
"""

FLOAT = """
pub trait FloatCore: Num {
    fn hidden(self) -> Self;
}

pub trait Float: Num + Copy + NumCast + PartialOrd + Neg<Output = Self> {
    /// Returns the `NaN` value.
    fn nan() -> Self;
    fn sin(self) -> Self;
    fn ln(self) -> Self;
    fn fract(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn sin_cos(self) -> (Self, Self);
    fn integer_decode(self) -> (u64, i16, i8);
    fn to_degrees(self) -> Self {
        unimplemented!()
    }
}
"""

VEC3 = """
impl Vec3 {
    #[inline(always)]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn select(mask: BVec3, if_true: Self, if_false: Self) -> Self {
        if_true
    }

    pub const fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn dot(self, rhs: Self) -> f32 {
        0.0
    }

    pub fn lerp(self, rhs: Self, s: f32) -> Self {
        self
    }

    pub fn sin_cos(self) -> (Self, Self) {
        (self, self)
    }

    pub fn max_position(self) -> usize {
        0
    }

    pub fn normalize_and_length(self) -> (Self, f32) {
        (self, 1.0)
    }

    pub fn try_normalize(self) -> Option<Self> {
        None
    }

    fn private_helper(self) -> Self {
        self
    }

    #[cfg(feature = "i32")]
    #[inline]
    #[must_use]
    pub fn as_ivec3(self) -> crate::IVec3 {
        crate::IVec3::new(self.x as i32, self.y as i32, self.z as i32)
    }

    #[cfg(feature = "i16")]
    #[inline]
    #[must_use]
    pub fn as_i16vec3(self) -> crate::I16Vec3 {
        crate::I16Vec3::new(self.x as i16, self.y as i16, self.z as i16)
    }
}
"""


def parse(glsl_names=frozenset({"dFdx", "log", "fract", "mix", "dot"})):
    found = rust.parse(
        {"arch": ARCH, "derivative": DERIVATIVE}, FLOAT, VEC3, glsl_names
    )
    return {f.name: f for f in found}


def test_rust_reads_spirv_std_arch_and_expands_its_macros():
    features = parse()
    arch = sorted(n for n, f in features.items() if f.group == "arch")
    # The macro's scalar argument (bool) and private items are not functions.
    assert arch == [
        "Derivative::dfdx",
        "any",
        "atomic_i_add",
        "emit_stream_vertex",
        "kill",
        "memory_barrier",
        "read_clock_khr",
        "subgroup_ballot",
        "subgroup_f_add",
        "subgroup_inclusive_f_add",
        "subgroup_logical_and",
        "unsigned_max",
        "vector_extract_dynamic",
    ]
    assert features["unsigned_max"].signature == Signature(
        Type("uint"), (Param(Type("uint")), Param(Type("uint")))
    )
    assert features["subgroup_f_add"].signature.result == Type("float")
    assert features["subgroup_logical_and"].signature.result == Type("bool")
    assert features["subgroup_ballot"].signature.result == Type("uint", (4,))
    assert features["any"].signature.params == (Param(Type("bool", (3,))),)
    assert features["kill"].stage == "fragment"
    assert features["Derivative::dfdx"].stage == "fragment"
    assert features["Derivative::dfdx"].spelling == ".dfdx"
    assert features["Derivative::dfdx"].crossgl == "dFdx"


def test_rust_fills_in_constant_generic_parameters():
    features = parse()
    atomic = features["atomic_i_add"]
    assert atomic.signature.params == (
        Param(Type("uint"), "inout"),
        Param(Type("uint")),
    )
    assert atomic.spelling == (
        "unsafe atomic_i_add::<_, { Scope::Subgroup as u32 }, { Semantics::NONE.bits() }>"
    )
    assert "ACQUIRE_RELEASE" in features["memory_barrier"].spelling
    assert features["emit_stream_vertex"].spelling == "unsafe emit_stream_vertex::<0>"
    _, _, note = rust.signature("f", ("const WIDTH: u32", "", ""))
    assert note == "no probe value for the constant parameter WIDTH"


def test_rust_geometry_stream_is_a_crossgl_argument():
    stream = parse()["emit_stream_vertex"]
    assert stream.crossgl == "EmitStreamVertex"
    assert (
        "unsafe { emit_stream_vertex::<0>() };" in probes.build(stream, "rust").source
    )
    assert "EmitStreamVertex(0);" in probes.build(stream, "crossgl").source


def test_rust_reads_float_and_vec3_methods():
    features = parse()
    assert "Float::hidden" not in features
    assert "Vec3::private_helper" not in features
    assert features["Float::sin"].spelling == ".sin"
    assert features["Float::sin"].signature == Signature(
        Type("float"), (Param(Type("float")),)
    )
    assert features["Float::nan"].spelling == "f32::nan"
    assert features["Float::nan"].signature == Signature(Type("float"), ())
    assert features["Float::ln"].crossgl == "log"
    assert features["Float::powi"].signature.params[1] == Param(Type("int"))
    assert features["Float::sin_cos"].signature.unsupported == "(f32, f32)"
    assert features["Float::to_degrees"].group == "float"
    vec3 = Type("float", (3,))
    assert features["Vec3::dot"].signature == Signature(
        Type("float"), (Param(vec3), Param(vec3))
    )
    assert features["Vec3::select"].spelling == "Vec3::select"
    assert features["Vec3::select"].signature.params[0] == Param(Type("bool", (3,)))
    assert features["Vec3::to_array"].signature.unsupported == "[f32; 3]"
    assert features["Vec3::lerp"].crossgl == "mix"
    assert features["Vec3::new"].crossgl == "vec3"


def test_rust_leaves_out_glam_methods_spirv_std_does_not_enable():
    features = parse()
    # spirv-std builds glam with i32, u32 and f64 vectors only.
    assert features["Vec3::as_ivec3"].signature == Signature(
        Type("int", (3,)), (Param(VEC3_TYPE),)
    )
    assert "Vec3::as_i16vec3" not in features


def test_rust_probes_64_bit_integer_results():
    clock = parse()["read_clock_khr"]
    assert clock.signature == Signature(Type("uint64"), ())
    source = probes.build(clock, "rust").source
    assert (
        "let r: u64 = unsafe { read_clock_khr::<{ Scope::Subgroup as u32 }>() };"
        in source
    )
    assert "outp[i] = (r) as u32;" in source
    assert "uint64_t r = read_clock_khr();" in probes.build(clock, "crossgl").source


def test_rust_probes_pointer_sized_integers():
    features = parse()
    position = features["Vec3::max_position"]
    assert position.signature == Signature(Type("usize"), (Param(VEC3_TYPE),))
    source = probes.build(position, "rust").source
    assert "let r: usize = a0.max_position();" in source
    assert "outp[i] = (r) as u32;" in source
    assert position.crossgl == ""
    extract = features["vector_extract_dynamic"]
    source = probes.build(extract, "rust").source
    assert "let a1: usize = (uin[i + 0]) as usize;" in source
    assert "let r: f32 = unsafe { vector_extract_dynamic(a0, a1) };" in source
    # rust-gpu's usize is 32 bits wide, and the Rust importer writes it as CrossGL's uint.
    crossgl = probes.build(extract, "crossgl").source
    assert "uint a1 = uint(uin[i + 0u]);" in crossgl
    assert "float r = a0[a1];" in crossgl
    with pytest.raises(probes.ProbeError, match="no pointer-sized integer type"):
        probes.spell(Type("usize"), "glsl")


def test_rust_has_no_probe_for_vectors_spirv_std_lacks():
    with pytest.raises(probes.ProbeError, match="no int16 vectors"):
        probes.spell(Type("int16", (3,)), "rust")
    assert probes.spell(Type("int16"), "rust") == "i16"


def test_rust_probes_read_the_first_value_of_a_tuple_or_an_option():
    features = parse()
    vec3 = probes.build(features["Vec3::sin_cos"], "rust").source
    assert "let r: Vec3 = a0.sin_cos().0;" in vec3
    assert "outp[i] = r.extend(0.0);" in vec3
    decoded = probes.build(features["Float::integer_decode"], "rust").source
    assert "let r: u64 = a0.integer_decode().0;" in decoded
    assert "outp[i] = (r) as u32;" in decoded
    optional = probes.build(features["Vec3::try_normalize"], "rust").source
    assert "let r: Vec3 = a0.try_normalize().unwrap_or_default();" in optional
    assert features["Vec3::normalize_and_length"].signature.unsupported == (
        "(Vec3, f32)"
    )


def test_rust_sin_cos_is_crossgl_sincos_with_output_parameters():
    features = parse()
    assert features["Float::sin_cos"].crossgl == "sincos"
    source = probes.build(features["Float::sin_cos"], "crossgl").source
    assert "sincos(a0, a1, a2);" in source
    vec3 = probes.build(features["Vec3::sin_cos"], "crossgl").source
    assert "vec3 a1;\n" in vec3
    assert "vec3 a2;\n" in vec3


def test_rust_functions_returning_several_values_have_no_crossgl_spelling():
    features = parse()
    for name in (
        "Float::integer_decode",
        "Vec3::max_position",
        "Vec3::normalize_and_length",
        "Vec3::try_normalize",
    ):
        assert features[name].crossgl == ""
        assert features[name].note.startswith("no CrossGL function returns")
        with pytest.raises(probes.NoSpelling):
            probes.build(features[name], "crossgl")


def test_rust_functions_that_compute_something_else_have_no_crossgl_spelling():
    features = parse()
    # Rust's fract truncates; CrossGL's floors.
    assert features["Float::fract"].crossgl == ""
    assert "trunc" in features["Float::fract"].note
    # A function without a GLSL counterpart is probed under its own name.
    assert features["Float::sin"].crossgl == "sin"
    assert features["unsigned_max"].crossgl == "unsigned_max"


BROADCAST = """
pub unsafe fn subgroup_broadcast<I: ScalarOrVector>(value: I, id: u32) -> I {
    value
}

pub fn subgroup_shuffle<I: ScalarOrVector>(value: I, id: u32) -> I {
    value
}
"""


def test_rust_passes_a_broadcast_lane_id_as_a_literal():
    found = rust.parse({"arch": BROADCAST}, FLOAT, VEC3)
    features = {f.name: f for f in found}
    # Before SPIR-V 1.5 a broadcast's lane id must be a constant; a shuffle's need not be.
    assert features["subgroup_broadcast"].signature.params[1] == Param(
        Type("uint"), "const"
    )
    assert features["subgroup_shuffle"].signature.params[1] == Param(Type("uint"))
    source = probes.build(features["subgroup_broadcast"], "rust").source
    assert "let r: f32 = unsafe { subgroup_broadcast(a0, 1) };" in source


def feature(name, result, *params, spelling=""):
    signature = Signature(
        result, tuple(p if isinstance(p, Param) else Param(p) for p in params)
    )
    return Feature("rust", "function", name, signature, spelling=spelling)


VEC3_TYPE = Type("float", (3,))
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
    assert features["execute_callable"].stage == "closest-hit"
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
]


def test_rust_stage_probes_name_their_execution_model():
    intersection, hit, geometry, _ = (
        probes.build(f, "rust").source for f in STAGE_SAMPLES
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


def test_rust_has_no_mesh_or_task_probe_for_vulkan_1_1():
    for stage in ("mesh", "task"):
        probe_feature = Feature(
            "rust", "function", "f", Signature(VOID, ()), stage=stage
        )
        with pytest.raises(probes.ProbeError, match="need SPIR-V 1.4"):
            probes.build(probe_feature, "rust")


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
    verdict = validators.validate("rust", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


@needs_rust_gpu
def test_rust_gpu_rejects_code_that_is_not_a_shader_crate():
    verdict = validators.validate("rust", "pub fn f() -> String { String::new() }\n")
    assert not verdict.accepted
    assert "can't find crate for `std`" in verdict.log


@needs_rust_gpu
def test_rust_features_lower_to_spir_v_through_rust_gpu():
    assert validators.can_lower("rust")
    assert semantic.lower(SAMPLES[0]) == ("GLSL.std.450:Sin", "")
    # glam's methods are Rust arithmetic, so they borrow GLSL's ids through their CrossGL spelling instead.
    dot = Feature("rust", "function", "Vec3::dot", SAMPLES[1].signature, group="vec3")
    assert semantic.lower(dot) == ("", "glam computes it with Rust arithmetic")
