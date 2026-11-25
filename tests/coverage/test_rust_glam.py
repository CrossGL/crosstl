"""glam's types besides Vec3 and spirv_std::float's packing functions in the Rust feature list."""

import pytest

from tools.coverage import probes, validators
from tools.coverage.features import glam, rust
from tools.coverage.model import Param, Signature, Type

VEC3 = """
impl Vec3 {
    pub fn dot(self, rhs: Self) -> f32 {
        0.0
    }
}
"""
VEC2 = """
impl Vec2 {
    pub fn dot(self, rhs: Self) -> f32 {
        0.0
    }

    pub fn perp(self) -> Self {
        self
    }

    pub fn perp_dot(self, rhs: Self) -> f32 {
        0.0
    }
}
"""
IVEC3 = """
impl IVec3 {
    pub fn dot(self, rhs: Self) -> i32 {
        0
    }

    pub fn perp(self) -> Self {
        self
    }

    pub const fn wrapping_add(self, rhs: Self) -> Self {
        self
    }

    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        None
    }
}
"""
MAT4 = """
impl Mat4 {
    pub const fn from_cols_array(m: &[f32; 16]) -> Self {
        Self::ZERO
    }

    pub const fn from_cols_array_2d(m: &[[f32; 4]; 4]) -> Self {
        Self::ZERO
    }

    pub const fn to_cols_array_2d(&self) -> [[f32; 4]; 4] {
        [[0.0; 4]; 4]
    }

    pub fn col_mut(&mut self, index: usize) -> &mut Vec4 {
        &mut self.x_axis
    }

    pub fn write_cols_to_slice(&self, slice: &mut [f32]) {
        slice[0] = self.x_axis.x;
    }

    pub const fn from_cols_slice(slice: &[f32]) -> Self {
        Self::ZERO
    }

    pub fn from_quat(rotation: Quat) -> Self {
        Self::ZERO
    }

    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        *self
    }

    pub fn try_inverse(&self) -> Option<Self> {
        None
    }

    pub fn to_euler(&self, order: EulerRot) -> (f32, f32, f32) {
        (0.0, 0.0, 0.0)
    }
}
"""
MAT3 = """
impl Mat3 {
    pub fn from_mat4(m: Mat4) -> Self {
        Self::ZERO
    }

    pub fn mul_mat3(&self, rhs: &Self) -> Self {
        *self
    }

    pub fn mul_vec3a(&self, rhs: Vec3A) -> Vec3A {
        rhs
    }
}
"""
QUAT = """
impl Quat {
    pub fn mul_vec3(self, rhs: Vec3) -> Vec3 {
        rhs
    }

    pub fn slerp(self, end: Self, s: f32) -> Self {
        self
    }

    pub fn as_dquat(self) -> DQuat {
        DQuat::IDENTITY
    }

    pub fn from_affine3(a: &crate::Affine3) -> Self {
        Self::IDENTITY
    }

    pub fn from_affine3a(a: &crate::Affine3A) -> Self {
        Self::IDENTITY
    }

    pub fn write_to_slice(self, slice: &mut [f32]) {
        slice[0] = self.x;
    }
}
"""
PACKING = """
#[cfg(target_arch = "spirv")]
impl f32 {
    pub fn powi(self, n: i32) -> f32 {
        intrinsics::powif32(self, n)
    }
}

#[spirv_std_macros::gpu_only]
pub fn vec2_to_f16x2(vec: Vec2) -> u32 {
    0
}

#[spirv_std_macros::gpu_only]
pub fn f32_to_f16(float: f32) -> u32 {
    vec2_to_f16x2(crate::glam::Vec2::new(float, 0.))
}
"""
FLOAT4 = Type("float", (4,))
MAT4_TYPE = Type("float", (4, 4))


def parse(**texts):
    sources = {name: f"impl {name} {{\n}}\n" for name in glam.SOURCES}
    sources.update(Vec3=VEC3, Vec2=VEC2, IVec3=IVEC3, Mat4=MAT4, Mat3=MAT3, Quat=QUAT)
    sources.update(texts)
    return {f.name: f for f in glam.parse(sources, frozenset({"dot"}))}


def test_glam_counts_a_method_once_in_each_family():
    features = parse()
    # Vec3's methods are rust.parse's, and later vectors take a method only when no earlier one has it.
    assert "Vec3::dot" not in features
    assert "Vec2::dot" not in features and "IVec3::dot" not in features
    assert "Vec2::perp" in features and "IVec3::perp" not in features
    assert features["IVec3::wrapping_add"].group == "ivec3"
    # Families are separate: Mat3::mul_mat3 is not Mat4::mul_mat4.
    assert {"Mat4::mul_mat4", "Mat3::mul_mat3", "Mat3::from_mat4"} <= set(features)


def test_glam_sources_are_pinned_for_every_type():
    from tools.coverage import upstream

    for key in glam.SOURCES.values():
        source = upstream.SOURCES[key]
        assert source.repository == "bitshifter/glam-rs"
        assert source.commit == upstream.SOURCES["rust-vec3"].commit
        assert len(source.sha256) == 64


def test_glam_methods_take_crossgl_spellings_or_say_why_not():
    features = parse()
    assert features["Vec2::perp"].crossgl == "vec2(-{0}.y, {0}.x)"
    assert features["IVec3::wrapping_add"].crossgl == "({0} + {1})"
    checked = features["IVec3::checked_add"]
    assert checked.crossgl == "" and "optional" in checked.note
    slerp = features["Quat::slerp"]
    assert slerp.crossgl == "" and slerp.note
    with pytest.raises(probes.NoSpelling):
        probes.build(slerp, "crossgl")


def test_glam_matrices_are_crossgl_matrices():
    features = parse()
    mul = features["Mat4::mul_mat4"]
    # A shared reference is a value the method reads; the probe passes it by reference.
    assert mul.signature == Signature(
        MAT4_TYPE, (Param(MAT4_TYPE), Param(MAT4_TYPE, "address"))
    )
    source = probes.build(mul, "rust").source
    assert "let r: Mat4 = a0.mul_mat4(&a1);" in source
    assert "outp[i] = r.x_axis;" in source
    crossgl = probes.build(mul, "crossgl").source
    assert "mat4 r = (a0 * a1);" in crossgl
    assert (
        "mat3 r = mat3(a0);" in probes.build(
            features["Mat3::from_mat4"], "crossgl"
        ).source
    )


def test_glam_arrays_and_slices_are_read_as_glam_reads_them():
    features = parse()
    array = features["Mat4::from_cols_array"]
    source = probes.build(array, "rust").source
    assert "let a0: [f32; 16] = [fin[i + 0]," in source
    assert "Mat4::from_cols_array(&a0)" in source
    crossgl = probes.build(array, "crossgl").source
    assert "mat4 r = mat4(a0, a1, a2, a3, a4, a5, a6, a7, a8" in crossgl
    # Every invocation reads a slice's first elements, so CrossGL reads them from the buffer.
    crossgl = probes.build(features["Mat4::from_cols_slice"], "crossgl").source
    assert "mat4 r = mat4(fin[0], fin[1]," in crossgl
    # Column arrays are made from the flat array's elements, which CrossGL reads as from_cols_array's.
    columns = features["Mat4::from_cols_array_2d"]
    source = probes.build(columns, "rust").source
    assert "let a0: [f32; 16] = [fin[i + 0]," in source
    assert (
        "Mat4::from_cols_array_2d(&[[a0[0], a0[1], a0[2], a0[3]], [a0[4], a0[5], a0[6], a0[7]], "
        "[a0[8], a0[9], a0[10], a0[11]], [a0[12], a0[13], a0[14], a0[15]]])"
    ) in source
    assert (
        probes.build(columns, "crossgl").source == probes.build(array, "crossgl").source
    )
    # A result's column arrays are stored as its first column, as a matrix is.
    source = probes.build(features["Mat4::to_cols_array_2d"], "rust").source
    assert "let r: Vec4 = Vec4::from((a0.to_cols_array_2d())[0]);" in source
    # A slice the function writes is a local array of one value per component: a matrix's sixteen and a
    # quaternion's four.
    source = probes.build(features["Mat4::write_cols_to_slice"], "rust").source
    assert "let mut a1: [f32; 16] = Default::default();" in source
    assert "a0.write_cols_to_slice(&mut a1);" in source
    source = probes.build(features["Quat::write_to_slice"], "rust").source
    assert "let mut a1: [f32; 4] = Default::default();" in source
    assert "Quat::from_vec4(a0).write_to_slice(&mut a1);" in source


def test_glam_column_references_are_read_through():
    source = probes.build(parse()["Mat4::col_mut"], "rust").source
    assert "let mut a0: Mat4 = Mat4::from_cols(" in source
    assert "let r: Vec4 = *(a0.col_mut(a1));" in source


def test_glam_affine_transforms_are_made_from_their_matrix():
    features = parse()
    source = probes.build(features["Quat::from_affine3"], "rust").source
    assert "let a0: Mat3 = Mat3::from_cols(" in source
    assert (
        "Vec4::from(Quat::from_affine3(&Affine3 { matrix3: a0, translation: Vec3::ZERO }))"
        in source
    )
    source = probes.build(features["Quat::from_affine3a"], "rust").source
    assert (
        "Quat::from_affine3a(&Affine3A { matrix3: Mat3A::from(a0), translation: Vec3A::ZERO })"
        in source
    )


def test_glam_quaternions_are_crossgl_vec4s():
    features = parse()
    rotate = features["Quat::mul_vec3"]
    assert rotate.counterpart == Signature(
        Type("float", (3,)), (Param(FLOAT4), Param(Type("float", (3,))))
    )
    source = probes.build(rotate, "rust").source
    # A quaternion argument is made from the vector of its components.
    assert "let a0: Vec4 = Vec4::new(" in source
    assert "Quat::from_vec4(a0).mul_vec3(a1)" in source
    crossgl = probes.build(rotate, "crossgl").source
    assert "vec4 a0 = vec4(" in crossgl
    assert "cross(a0.xyz, a1) * (a0.w * 2.0)" in crossgl
    # A quaternion result is stored as the vector of its components.
    source = probes.build(features["Mat4::from_quat"], "rust").source
    assert "Mat4::from_quat(Quat::from_vec4(a0))" in source
    source = probes.build(features["Quat::as_dquat"], "rust").source
    assert "DVec4::from(Quat::from_vec4(a0).as_dquat())" in source
    assert features["Quat::as_dquat"].counterpart.result == Type("double", (4,))


def test_glam_aligned_types_are_made_from_and_read_as_plain_ones():
    vec3a = parse()["Mat3::mul_vec3a"]
    source = probes.build(vec3a, "rust").source
    assert "Vec3::from(a0.mul_vec3a(Vec3A::from(a1)))" in source
    assert vec3a.counterpart.params[1] == Param(Type("float", (3,)))


@pytest.mark.skipif(
    not validators.available("rust"), reason="rust-gpu is not installed"
)
@pytest.mark.parametrize(
    "name",
    [
        "Mat4::from_cols_array_2d",
        "Mat4::to_cols_array_2d",
        "Mat4::col_mut",
        "Quat::from_affine3",
        "Quat::from_affine3a",
    ],
)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_glam_array_reference_and_affine_probes_compile_with_rust_gpu(name, baseline):
    probe = probes.build(parse()[name], "rust", baseline=baseline)
    verdict = validators.validate("rust", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_glam_options_and_tuples_of_matrices_are_read():
    features = parse()
    source = probes.build(features["Mat4::try_inverse"], "rust").source
    assert "let r: Mat4 = a0.try_inverse().unwrap_or_default();" in source
    source = probes.build(features["Mat4::to_euler"], "rust").source
    assert "a0.to_euler(EulerRot::XYZ).0" in source


def test_rust_packing_functions_are_glsl_packing_functions():
    found = {f.name: f for f in rust.packing(PACKING, frozenset({"packHalf2x16"}))}
    # f32's powi is the method a Float::powi probe already calls.
    assert set(found) == {"vec2_to_f16x2", "f32_to_f16"}
    packed = found["vec2_to_f16x2"]
    assert packed.group == "packing" and packed.crossgl == "packHalf2x16"
    source = probes.build(packed, "rust").source
    assert "let r: u32 = spirv_std::float::vec2_to_f16x2(a0);" in source
    half = found["f32_to_f16"]
    assert half.crossgl == "packHalf2x16(vec2({0}, 0.0))"
    assert (
        "uint r = packHalf2x16(vec2(a0, 0.0));" in probes.build(half, "crossgl").source
    )
