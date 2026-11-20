// An excerpt of crates/spirv-std/src/image.rs from Rust-GPU/rust-gpu 62401280 (spirv-std 0.11.0),
// read by tests/coverage/test_rust_spirv_std.py.
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        DIM,
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        { Sampled::Yes as u32 },
        FORMAT,
        COMPONENTS,
    >
{
    /// Fetch a single texel with a sampler set at compile time
    #[doc(alias = "OpImageFetch")]
    pub fn fetch<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        self.fetch_with_lod(coordinate, 0)
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        DIM,
        DEPTH,
        ARRAYED,
        { Multisampled::False as u32 },
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
    pub fn gather<F>(
        &self,
        sampler: Sampler,
        coordinate: impl ImageCoordinate<F, DIM, ARRAYED>,
        component: u32,
    ) -> SampledType::Vec4
    where
        Self: HasGather,
        F: Float,
    {
        let mut result = SampledType::Vec4::default();
        unsafe {
            asm! {
                "%result = OpImageGather typeof*{result} %sampledImage %coordinate {component}",
                "OpStore {result} %result",
            }
        }
        result
    }

    /// Sample texels at `coord` from the image using `sampler`.
    pub fn sample<F>(
        &self,
        sampler: Sampler,
        coord: impl ImageCoordinate<F, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        F: Float,
    {
        unsafe {
            asm!(
                "%result = OpImageSampleImplicitLod typeof*{0} %sampledImage %coord",
                "OpStore {0} %result",
            );
        }
    }

    #[doc(alias = "OpImageSampleDrefExplicitLod")]
    pub fn sample_depth_reference_by_lod<F>(
        &self,
        sampler: Sampler,
        coordinate: impl ImageCoordinate<F, DIM, ARRAYED>,
        depth_reference: f32,
        lod: f32,
    ) -> SampledType
    where
        F: Float,
    {
        unsafe {
            asm!("%result = OpImageSampleDrefExplicitLod typeof*{1} %sampledImage %coordinate %dref Lod %lod");
        }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        DIM,
        DEPTH,
        { Arrayed::False as u32 },
        { Multisampled::False as u32 },
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
    pub fn sample_with_project_coordinate<F>(
        &self,
        sampler: Sampler,
        project_coordinate: impl ImageCoordinate<F, DIM, { Arrayed::True as u32 }>,
    ) -> SampledType::SampleResult
    where
        F: Float,
    {
        unsafe {
            asm!("%result = OpImageSampleProjImplicitLod typeof*{1} %sampledImage %project_coordinate");
        }
    }

    pub fn sample_depth_reference_with_project_coordinate_by_lod<F>(
        &self,
        sampler: Sampler,
        coordinate: impl ImageCoordinate<F, DIM, { Arrayed::True as u32 }>,
        depth_reference: f32,
        lod: f32,
    ) -> SampledType
    where
        F: Float,
    {
        unsafe {
            asm!("%result = OpImageSampleProjDrefExplicitLod typeof*{1} %sampledImage %coordinate %dref Lod %lod");
        }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        DIM,
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        { Sampled::No as u32 },
        FORMAT,
        COMPONENTS,
    >
{
    pub fn read<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        unsafe { asm!("%result = OpImageRead typeof*{0} %image %coordinate") }
    }

    pub unsafe fn write<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
        texels: SampledType::SampleResult,
    ) where
        I: Integer,
    {
        unsafe { asm!("OpImageWrite %image %coordinate %texels") }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        DIM,
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        { Sampled::Unknown as u32 },
        FORMAT,
        COMPONENTS,
    >
{
    pub fn read<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        unsafe { asm!("%result = OpImageRead typeof*{0} %image %coordinate") }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    Image<
        SampledType,
        { Dimensionality::SubpassData as u32 },
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        { Sampled::No as u32 },
        FORMAT,
        COMPONENTS,
    >
{
    pub fn read_subpass<I>(
        &self,
        coordinate: impl ImageCoordinateSubpassData<I, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        unsafe { asm!("%result = OpImageRead typeof*{0} %image %coordinate") }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
> Image<SampledType, DIM, DEPTH, ARRAYED, MULTISAMPLED, SAMPLED, FORMAT, COMPONENTS>
{
    pub fn query_lod(
        &self,
        sampler: Sampler,
        coord: impl ImageCoordinate<f32, DIM, { Arrayed::False as u32 }>,
    ) -> SampledType::Vec2
    where
        Self: HasQueryLevels,
    {
        unsafe { asm!("%result = OpImageQueryLod typeof*{0} %sampledImage %coord") }
    }

    pub fn query_size<Size: ImageSizeQuery<u32, DIM, ARRAYED> + Default>(&self) -> Size
    where
        Self: HasQuerySize,
    {
        unsafe { asm!("%result = OpImageQuerySize typeof*{0} %image") }
    }
}

pub struct SampledImage<I> {
    _image: I,
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const SAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    SampledImage<
        Image<
            SampledType,
            DIM,
            DEPTH,
            ARRAYED,
            { Multisampled::False as u32 },
            SAMPLED,
            FORMAT,
            COMPONENTS,
        >,
    >
{
    pub fn sample_by_lod<F>(
        &self,
        coord: impl ImageCoordinate<F, DIM, ARRAYED>,
        lod: f32,
    ) -> SampledType::SampleResult
    where
        F: Float,
    {
        unsafe { asm!("%result = OpImageSampleExplicitLod typeof*{0} %sampledImage %coord Lod %lod") }
    }
}

/// This is a marker trait to represent the constraints on `OpImageGather` too complex to be
/// represented by const generics. Specifically:
///
/// "Its `OpTypeImage` must have a Dim of 2D, Cube, or Rect. The MS operand of the underlying
/// `OpTypeImage` must be 0."
pub trait HasGather {}
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DEPTH: u32,
    const FORMAT: u32,
    const ARRAYED: u32,
    const SAMPLED: u32,
    const COMPONENTS: u32,
> HasGather
    for Image<
        SampledType,
        { Dimensionality::TwoD as u32 },
        DEPTH,
        ARRAYED,
        { Multisampled::False as u32 },
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
}

pub trait HasQueryLevels {}
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DEPTH: u32,
    const FORMAT: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const SAMPLED: u32,
    const COMPONENTS: u32,
> HasQueryLevels
    for Image<
        SampledType,
        { Dimensionality::TwoD as u32 },
        DEPTH,
        ARRAYED,
        MULTISAMPLED,
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
}

/// "Its Dim operand must be 1D, 2D, 3D, Cube, Rect, or Buffer. Additionally, if its Dim is 1D, 2D,
/// 3D, or Cube, it must also have either an MS of 1 or a Sampled of 0 or 2."
pub trait HasQuerySize {}
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DEPTH: u32,
    const FORMAT: u32,
    const ARRAYED: u32,
    const SAMPLED: u32,
    const COMPONENTS: u32,
> HasQuerySize
    for Image<
        SampledType,
        { Dimensionality::OneD as u32 },
        DEPTH,
        ARRAYED,
        { Multisampled::True as u32 },
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
}
impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DEPTH: u32,
    const FORMAT: u32,
    const ARRAYED: u32,
    const SAMPLED: u32,
    const COMPONENTS: u32,
> HasQuerySize
    for Image<
        SampledType,
        { Dimensionality::TwoD as u32 },
        DEPTH,
        ARRAYED,
        { Multisampled::True as u32 },
        SAMPLED,
        FORMAT,
        COMPONENTS,
    >
{
}
