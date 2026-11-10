"""Excerpts of spirv-std that the Rust feature and probe tests read, and the ``Vec3`` type they declare."""

from tools.coverage.model import Type

VEC3_TYPE = Type("float", (3,))

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

pub unsafe fn vector_insert_dynamic<T: Scalar, V: Vector<T, N>, const N: usize>(
    vector: V,
    index: usize,
    element: T,
) -> V {
    vector
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
