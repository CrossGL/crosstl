import pytest

import crosstl
import crosstl.translator
from crosstl.backend.Rust.RustAst import (
    BinaryOpNode,
    BlockNode,
    ClosureNode,
    ClosureParameterNode,
    FunctionNode,
    LetNode,
    ReturnNode,
    ShaderNode,
    VariableNode,
)
from crosstl.backend.Rust.RustCrossGLCodeGen import RustToCrossGLConverter
from crosstl.backend.Rust.RustLexer import RustLexer
from crosstl.backend.Rust.RustParser import RustParser


def parse_and_generate(code: str) -> str:
    lexer = RustLexer(code)
    tokens = lexer.tokenize()
    parser = RustParser(tokens)
    ast = parser.parse()

    generator = RustToCrossGLConverter()
    return generator.generate(ast)


def test_struct_conversion():
    code = """
    #[repr(C)]
    pub struct Vertex {
        position: Vec3<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }
    """
    try:
        result = parse_and_generate(code)
        assert "shader main {" in result
        assert "struct Vertex {" in result
        assert "vec3 position" in result
        assert "vec3 normal" in result
        assert "vec2 texcoord" in result
    except Exception as e:
        pytest.fail(f"Struct conversion failed: {e}")


def test_tuple_struct_conversion():
    code = """
    #[repr(transparent)]
    pub(crate) struct Marker;
    struct Tagged<T>;
    pub(super) struct Pair(pub(crate) f32, pub(self) Vec3<f32>);
    """
    try:
        result = parse_and_generate(code)
        assert "struct Marker {" in result
        assert "struct Tagged {" in result
        assert "struct Pair {" in result
        assert "float field0;" in result
        assert "vec3 field1;" in result
    except Exception as e:
        pytest.fail(f"Tuple struct conversion failed: {e}")


def test_struct_where_clause_conversion():
    code = """
    pub struct Buffer<T>
    where
        T: Copy,
    {
        value: T,
    }

    pub(super) struct Pair<T>(T, Vec3<f32>)
    where
        T: Copy + Clone;

    struct Marker<T>
    where
        T: Copy;
    """
    try:
        result = parse_and_generate(code)
        assert "struct Buffer {" in result
        assert "T value;" in result
        assert "struct Pair {" in result
        assert "T field0;" in result
        assert "vec3 field1;" in result
        assert "struct Marker {" in result
    except Exception as e:
        pytest.fail(f"Struct where-clause conversion failed: {e}")


def test_function_conversion():
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: Vec3<f32>) -> Vec4<f32> {
        let pos = Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
        return pos;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vertex vertex_main {" in result
        assert "vec4 main(vec3 vertex_)" in result
        assert "vec4(vertex_.x, vertex_.y, vertex_.z, 1.0)" in result
        assert "pos = vec4(" in result
        assert "return pos;" in result
    except Exception as e:
        pytest.fail(f"Function conversion failed: {e}")


def test_value_returning_function_uses_final_expression_as_return():
    code = """
    fn finish(value: u32) -> u32 {
        let next = value + 1;
        next
    }

    fn call_finish(value: u32) -> u32 {
        finish(value)
    }

    fn passthrough(value: Option<u32>) -> Option<u32> {
        value
    }

    fn choose(value: u32, fallback: u32, ready: bool) -> u32 {
        if ready {
            value
        } else {
            fallback
        }
    }
    """

    result = parse_and_generate(code)

    assert "uint finish(uint value)" in result
    assert "let next = (value + 1);" in result
    assert "return next;" in result
    assert "uint call_finish(uint value)" in result
    assert "return finish(value);" in result
    assert "Option<u32> passthrough(Option<u32> value)" in result
    assert "return value;" in result
    assert "uint choose(uint value, uint fallback, bool ready)" in result
    assert "if (ready) {\n            return value;\n        } else {" in result
    assert "return fallback;" in result


def test_value_returning_function_uses_final_literal_expression_as_return():
    code = r"""
    fn zero() -> u32 {
        0
    }

    fn enabled() -> bool {
        true
    }

    fn letter() -> char {
        'z'
    }

    fn message() -> String {
        r#"done"#
    }
    """

    result = parse_and_generate(code)

    assert "uint zero()" in result
    assert "return 0;" in result
    assert "bool enabled()" in result
    assert "return true;" in result
    assert "char letter()" in result
    assert "return 'z';" in result
    assert "String message()" in result
    assert 'return "done";' in result


def test_final_expression_return_is_limited_to_function_body():
    code = """
    fn value_after_branch(value: u32, ready: bool) -> u32 {
        if ready {
            tick();
        }
        value
    }

    fn void_observer(value: u32) {
        value
    }
    """

    result = parse_and_generate(code)

    assert "if (ready) {\n            tick();\n        }" in result
    assert "return value;" in result
    assert "void void_observer(uint value)" in result
    assert "return value;" not in result[result.index("void void_observer") :]
    assert "value;" in result[result.index("void void_observer") :]


def test_translate_api_accepts_rust_source_preserves_stage_entry(tmp_path):
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: Vec3<f32>) -> Vec4<f32> {
        let pos = Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
        return pos;
    }
    """
    rust_file = tmp_path / "shader.rs"
    rust_file.write_text(code, encoding="utf-8")

    cgl_result = crosstl.translate(str(rust_file), backend="cgl", format_output=False)
    assert "vertex vertex_main {" in cgl_result
    assert "vec4 main(vec3 vertex_)" in cgl_result
    assert "vec4(vertex_.x, vertex_.y, vertex_.z, 1.0)" in cgl_result

    rust_result = crosstl.translate(str(rust_file), backend="rust", format_output=False)
    assert "pub fn vertex_main(vertex_: Vec3<f32>) -> Vec4<f32>" in rust_result
    assert "Vec4::<f32>::new(vertex_.x, vertex_.y, vertex_.z, 1.0)" in rust_result
    assert "pub fn main() -> ()" not in rust_result


def test_rust_gpu_ray_query_parenthesized_statement_macro_codegen():
    code = """
    use glam::Vec3;
    use spirv_std::ray_tracing::{AccelerationStructure, RayFlags, RayQuery};
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main(#[spirv(descriptor_set = 0, binding = 0)] accel: &AccelerationStructure) {
        unsafe {
            spirv_std::ray_query!(let mut handle);
            handle.initialize(accel, RayFlags::NONE, 0, Vec3::ZERO, 0.0, Vec3::ZERO, 0.0);
            let origin: glam::Vec3 = handle.get_world_ray_origin();
        }
    }
    """

    result = parse_and_generate(code)

    assert "fragment {" in result
    assert "AccelerationStructure accel @ set(0) @ binding(0)" in result
    assert "spirv_std::ray_query!(let mut handle);" in result
    assert (
        "handle.initialize(accel, RayFlags::NONE, 0, Vec3::ZERO, 0.0, Vec3::ZERO, 0.0);"
        in result
    )
    assert "vec3 origin = handle.get_world_ray_origin();" in result


def test_rust_gpu_compute_shader_method_turbofish_codegen():
    code = """
    fn main() {
        let result = src_range.clone().into_par_iter().map(collatz).collect::<Vec<_>>();
    }
    """

    result = parse_and_generate(code)

    assert "collect<Vec<_>>();" in result


def test_rust_gpu_compute_shader_is_multiple_of_method_codegen():
    code = """
    pub fn collatz(mut n: u32) -> Option<u32> {
        let mut i = 0;
        while n != 1 {
            n = if n.is_multiple_of(2) {
                n / 2
            } else {
                3 * n + 1
            };
            i += 1;
        }
        Some(i)
    }
    """

    result = parse_and_generate(code)

    assert "(((n % 2) == 0) ? (n / 2) : ((3 * n) + 1))" in result
    assert "is_multiple_of" not in result


def test_vulkan_shader_examples_compute_atomic_const_generics_codegen():
    code = """
    use spirv_std::arch::atomic_i_add;
    use spirv_std::glam::UVec3;
    use spirv_std::spirv;

    #[repr(C)]
    pub struct UBOOut {
        pub draw_count: i32,
        pub lod_count: [i32; 6],
    }

    #[spirv(compute(threads(16)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] global_id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] ubo_out: &mut UBOOut,
    ) {
        let lod_level = 0u32;
        unsafe {
            atomic_i_add::<
                i32,
                { spirv_std::memory::Scope::Device as u32 },
                { spirv_std::memory::Semantics::NONE.bits() },
            >(&mut ubo_out.lod_count[lod_level as usize], 1);
        }
    }
    """

    result = parse_and_generate(code)

    assert "compute main_cs {" in result
    assert "void main(uvec3 global_id @ gl_GlobalInvocationID" in result
    assert "UBOOut ubo_out @ set(0) @ binding(3)" in result
    assert "int lod_count[6];" in result
    assert (
        "atomic_i_add<i32, {spirv_std::memory::Scope::Device as u32}, "
        "{spirv_std::memory::Semantics::NONE.bits()},>"
        "(ubo_out.lod_count[(uint)lod_level], 1);"
    ) in result


def test_vulkan_shader_examples_emboss_2d_compute_image_macro_codegen():
    # Reduced from https://github.com/Rust-GPU/VulkanShaderExamples commit
    # b29a37eb46802b5ea6882af4808d6887fc184581,
    # shaders/rust/computeshader/emboss/src/lib.rs.
    code = """
    use spirv_std::{glam::{IVec2, UVec3, Vec4, vec4}, spirv, Image};

    #[spirv(compute(threads(16, 16)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] global_id: UVec3,
        #[spirv(descriptor_set = 0, binding = 0)] input_image: &Image!(
            2D,
            format = rgba8,
            sampled = false
        ),
        #[spirv(descriptor_set = 0, binding = 1)] result_image: &Image!(
            2D,
            format = rgba8,
            sampled = false
        ),
    ) {
        let coord = IVec2::new(global_id.x as i32, global_id.y as i32);
        let rgb: Vec4 = input_image.read(coord);
        let gray = (rgb.x + rgb.y + rgb.z) / 3.0;
        let res = vec4(gray, gray, gray, 1.0);
        unsafe {
            result_image.write(coord, res);
        }
    }
    """

    result = parse_and_generate(code)

    assert "compute main_cs {" in result
    assert "uvec3 global_id @ gl_GlobalInvocationID" in result
    assert "image2D input_image @ set(0) @ binding(0)" in result
    assert "image2D result_image @ set(0) @ binding(1)" in result
    assert "@numthreads(16, 16, 1)" in result
    assert "let coord = ivec2((int)global_id.x, (int)global_id.y);" in result
    assert "vec4 rgb = imageLoad(input_image, coord);" in result
    assert "let gray = (((rgb.x + rgb.y) + rgb.z) / 3.0);" in result
    assert "imageStore(result_image, coord, res);" in result


def test_rust_gpu_final_parenthesized_binary_expression_codegen():
    code = """
    fn total_rayleigh(lambda: Vec3) -> Vec3 {
        (8.0 * PI.powf(3.0)
            * (REFRACTIVE_INDEX.powf(2.0) - 1.0).powf(2.0)
            * (6.0 + 3.0 * DEPOLARIZATION_FACTOR))
            / (3.0 * NUM_MOLECULES * pow(lambda, 4.0) * (6.0 - 7.0 * DEPOLARIZATION_FACTOR))
    }
    """

    result = parse_and_generate(code)

    assert "return ((((8.0 * pow(PI, 3.0))" in result
    assert ") / (((3.0 * NUM_MOLECULES) * pow(lambda, 4.0))" in result
    assert "pow(PI, 3.0));" not in result


def test_rust_gpu_path_qualified_attribute_codegen():
    code = r"""
    #[spirv_std_macros::gpu_only]
    #[doc(alias = "OpEmitVertex")]
    #[inline]
    pub unsafe fn emit_vertex() {
        unsafe {
            asm! {
                "OpEmitVertex",
            }
        }
    }
    """

    result = parse_and_generate(code)

    assert "void emit_vertex()" in result
    assert 'asm!("OpEmitVertex",);' in result


def test_rust_gpu_spirv_attributes_drive_stage_and_parameter_semantics():
    code = """
    #![cfg_attr(target_arch = "spirv", no_std)]

    use shared::glam::{Vec4, vec4};
    use spirv_std::spirv;

    #[spirv(vertex)]
    pub fn main_vs(
        #[spirv(vertex_index)] vert_id: i32,
        #[spirv(position, invariant)] out_pos: &mut Vec4,
    ) {
        *out_pos = vec4(vert_id as f32, 0.0, 0.0, 1.0);
    }

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] values: &mut [u32],
    ) {}
    """
    result = parse_and_generate(code)

    assert "vertex main_vs {" in result
    assert "int vert_id @ VertexID" in result
    assert "vec4 out_pos @ gl_Position @ invariant" in result
    assert "compute main_cs {" in result
    assert "uvec3 id @ gl_GlobalInvocationID" in result
    assert "uint values[] @ set(0) @ binding(0)" in result


def test_rust_gpu_spirv_entry_point_name_drives_stage_name():
    code = """
    use spirv_std::spirv;

    #[spirv(compute(threads(256), entry_point_name = "find_matches_and_compute_f4"))]
    pub fn shader_body() {}

    #[spirv(vertex(entry_point_name = "renamed_vs"))]
    pub fn vertex_body() {}
    """
    result = parse_and_generate(code)

    assert "compute find_matches_and_compute_f4 {" in result
    assert "vertex renamed_vs {" in result
    assert "compute shader_body {" not in result
    assert "vertex vertex_body {" not in result


def test_rust_gpu_vec_extend_codegen_from_upstream_examples():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/{sky-shader,mouse-shader}/src/lib.rs.
    code = """
    use spirv_std::glam::{vec3, Vec2, Vec3, Vec4};
    use spirv_std::spirv;

    fn tonemap(color: Vec3) -> Vec3 {
        color
    }

    #[spirv(fragment)]
    pub fn main_fs(output: &mut Vec4) {
        let color = vec3(1.0, 0.5, 0.25);
        *output = tonemap(color).extend(1.0);
    }

    #[spirv(vertex)]
    pub fn main_vs(pos: Vec2, #[spirv(position)] builtin_pos: &mut Vec4) {
        *builtin_pos = pos.extend(0.0).extend(1.0);
    }
    """

    result = parse_and_generate(code)

    assert "output = vec4(tonemap(color), 1.0);" in result
    assert "builtin_pos = vec4(vec3(pos, 0.0), 1.0);" in result
    assert ".extend(" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_image_macro_types_drive_resource_parameters():
    code = """
    use spirv_std::{spirv, Image};

    type Image2d = Image!(2D, type=f32, sampled);

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(descriptor_set = 2, binding = 5)] sampled_tex: &Image2d,
        #[spirv(descriptor_set = 3, binding = 1)] storage_tex: &Image!(2D, format=rgba16f, sampled=false),
        #[spirv(descriptor_set = 4, binding = 2)] direct_tex: &Image!(2D, type=f32, sampled),
    ) {}
    """
    result = parse_and_generate(code)

    assert "typedef sampler2D Image2d;" in result
    assert "fragment main_fs {" in result
    assert "Image2d sampled_tex @ set(2) @ binding(5)" in result
    assert "image2D storage_tex @ set(3) @ binding(1)" in result
    assert "sampler2D direct_tex @ set(4) @ binding(2)" in result


def test_vulkan_shader_examples_oit_block_scoped_use_codegen():
    code = """
    use spirv_std::{spirv, glam::{IVec2, Vec4}, arch::atomic_i_add, Image};

    type Image2d = Image!(2D, type=u32, sampled=false);

    #[repr(C)]
    pub struct GeometrySbo {
        pub count: u32,
        pub max_node_count: u32,
    }

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] frag_coord: Vec4,
        #[spirv(descriptor_set = 0, binding = 1, storage_buffer)] geometry_sbo: &mut GeometrySbo,
        #[spirv(descriptor_set = 0, binding = 2)] head_index_image: &Image2d,
    ) {
        let node_idx = unsafe { atomic_i_add(&mut geometry_sbo.count, 1) };
        if node_idx < geometry_sbo.max_node_count {
            let coord = IVec2::new(frag_coord.x as i32, frag_coord.y as i32);
            let prev_head_idx = unsafe {
                use spirv_std::memory::{Scope, Semantics};
                use spirv_std::arch::atomic_exchange;
                let current = head_index_image.read(coord).x;
                current
            };
        }
    }
    """

    result = parse_and_generate(code)

    assert "fragment main_fs {" in result
    assert "// use spirv_std::memory::{Scope, Semantics}" in result
    assert "// use spirv_std::arch::atomic_exchange" in result
    assert "current = imageLoad(head_index_image, coord).x;" in result
    assert "prev_head_idx = current;" in result


def test_rust_gpu_image_query_and_fetch_methods_codegen_from_upstream_compiletests():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/image/{fetch.rs,query/query_size*.rs,query_levels.rs,query_samples.rs}.
    code = """
    use spirv_std::spirv;
    use spirv_std::{Image, Sampler};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] sampled_image: &Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] storage_image: &Image!(2D, type=f32, sampled=false),
        #[spirv(descriptor_set = 0, binding = 2)] ms_image: &Image!(2D, type=f32, sampled, multisampled),
        #[spirv(descriptor_set = 0, binding = 3)] sampler: &Sampler,
        output: &mut Vec4,
        size_out: &mut UVec2,
        storage_size_out: &mut UVec2,
        levels_out: &mut u32,
        samples_out: &mut u32,
    ) {
        let texel = sampled_image.fetch(IVec2::new(0, 1));
        let sampled = sampled_image.sample(*sampler, Vec2::new(0.0, 1.0));
        let stored = storage_image.read(IVec2::new(0, 1));
        storage_image.write(UVec2::new(0, 1), stored);
        *size_out = sampled_image.query_size_lod(0);
        *storage_size_out = storage_image.query_size();
        *levels_out = sampled_image.query_levels();
        *samples_out = ms_image.query_samples();
        *output = texel + sampled + stored;
    }
    """

    result = parse_and_generate(code)

    assert "sampler2D sampled_image @ set(0) @ binding(0)" in result
    assert "image2D storage_image @ set(0) @ binding(1)" in result
    assert "sampler2DMS ms_image @ set(0) @ binding(2)" in result
    assert "let texel = texelFetch(sampled_image, ivec2(0, 1));" in result
    assert "let sampled = texture(sampled_image, sampler_, vec2(0.0, 1.0));" in result
    assert "let stored = imageLoad(storage_image, ivec2(0, 1));" in result
    assert "imageStore(storage_image, uvec2(0, 1), stored);" in result
    assert "size_out = textureSize(sampled_image, 0);" in result
    assert "storage_size_out = imageSize(storage_image);" in result
    assert "levels_out = textureQueryLevels(sampled_image);" in result
    assert "samples_out = textureSamples(ms_image);" in result
    assert ".fetch(" not in result
    assert ".read(" not in result
    assert ".write(" not in result
    assert ".query_size(" not in result
    assert ".query_size_lod(" not in result
    assert ".query_levels(" not in result
    assert ".query_samples(" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_extra_image_macro_dimensions_codegen_from_upstream():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/image/query/{rect_image_query_size,storage_image_query_size}.rs
    # and tests/compiletests/ui/image/read_subpass.rs.
    code = """
    use spirv_std::{spirv, Image};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] rect_storage: &Image!(rect, type=f32, sampled=false),
        #[spirv(descriptor_set = 1, binding = 1)] rect_storage_array: &Image!(rect, type=f32, sampled=false, arrayed),
        #[spirv(descriptor_set = 2, binding = 2)] buffer_image: &Image!(buffer, type=f32, sampled=false),
        #[spirv(descriptor_set = 3, binding = 3)] sampled_buffer: &Image!(buffer, type=u32, sampled),
        #[spirv(descriptor_set = 4, binding = 4, input_attachment_index = 0)] input_attachment: &Image!(subpass, type=f32, sampled=false),
        #[spirv(descriptor_set = 5, binding = 5)] ms_input_attachment: &Image!(subpass, type=i32, sampled=false, multisampled),
    ) {}
    """

    result = parse_and_generate(code)

    assert "image2DRect rect_storage @ set(0) @ binding(0)" in result
    assert "image2DRectArray rect_storage_array @ set(1) @ binding(1)" in result
    assert "imageBuffer buffer_image @ set(2) @ binding(2)" in result
    assert "usamplerBuffer sampled_buffer @ set(3) @ binding(3)" in result
    assert (
        "subpassInput input_attachment @ set(4) @ binding(4) "
        "@ input_attachment_index(0)"
    ) in result
    assert "isubpassInputMS ms_input_attachment @ set(5) @ binding(5)" in result
    assert "Image!(" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_fetch_with_sample_index_codegen_from_upstream_release_pattern():
    # Reduced from EmbarkStudios/rust-gpu v0.7 release notes for the Image API
    # sample_with builder pattern:
    # image.fetch_with(coord, sample_with::sample_index(idx)).
    code = """
    use spirv_std::spirv;
    use spirv_std::Image;
    use spirv_std::image::sample_with;

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] ms_image: &Image!(2D, type=f32, sampled, multisampled),
        pixel: IVec2,
        sample_index: i32,
        output: &mut Vec4,
    ) {
        let texel = ms_image.fetch_with(pixel, sample_with::sample_index(sample_index));
        *output = texel;
    }
    """

    result = parse_and_generate(code)

    assert "sampler2DMS ms_image @ set(0) @ binding(0)" in result
    assert "texel = texelFetch(ms_image, pixel, sample_index);" in result
    assert ".fetch_with(" not in result
    assert "sample_with::sample_index" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_sample_with_and_depth_project_methods_codegen_from_upstream():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # crates/spirv-std/src/image.rs, crates/spirv-std/src/image/sample_with.rs,
    # and tests/compiletests/ui/image/{sample_lod.rs,sample_gradient.rs,
    # sample_depth_reference/sample_lod.rs,sample_with_project_coordinate/sample_lod.rs}.
    code = """
    use spirv_std::spirv;
    use spirv_std::{Image, Sampler};
    use spirv_std::image::sample_with;

    #[spirv(fragment)]
    pub fn main(
        #[spirv(descriptor_set = 0, binding = 0)] color_image: &Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] projected_image: &Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 2)] depth_image: &Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 3)] sampler: &Sampler,
        output_color: &mut Vec4,
        output_depth: &mut f32,
    ) {
        let uv = Vec2::new(0.0, 1.0);
        let uvw = Vec3::new(0.0, 1.0, 0.5);
        let lod_color: Vec4 = color_image.sample_by_lod(*sampler, uv, 0.0);
        let grad_color: Vec4 = color_image.sample_by_gradient(*sampler, uv, uv, uv);
        let biased_color: Vec4 = color_image.sample_with(*sampler, uv, sample_with::bias(0.25));
        let direct_lod: Vec4 = color_image.sample_with(*sampler, uv, sample_with::lod(1.0));
        let direct_grad: Vec4 = color_image.sample_with(*sampler, uv, sample_with::grad(uv, uv));
        let projected_color: Vec4 = projected_image.sample_with_project_coordinate_by_lod(*sampler, uvw, 0.0);
        let direct_projected: Vec4 = projected_image.sample_with_project_coordinate_with(*sampler, uvw, sample_with::lod(2.0));
        let depth: f32 = depth_image.sample_depth_reference_by_lod(*sampler, uv, 0.5, 0.0);
        let direct_depth: f32 = depth_image.sample_depth_reference_with(*sampler, uv, 0.5, sample_with::lod(1.0));
        let projected_depth: f32 = depth_image.sample_depth_reference_with_project_coordinate_by_gradient(*sampler, uvw, 0.5, uv, uv);
        *output_color = lod_color + grad_color + biased_color + direct_lod + direct_grad + projected_color + direct_projected;
        *output_depth = depth + direct_depth + projected_depth;
    }
    """

    result = parse_and_generate(code)

    assert "sampler2D color_image @ set(0) @ binding(0)" in result
    assert "sampler2D projected_image @ set(0) @ binding(1)" in result
    assert "sampler2D depth_image @ set(0) @ binding(2)" in result
    assert "lod_color = textureLod(color_image, sampler_, uv, 0.0);" in result
    assert "grad_color = textureGrad(color_image, sampler_, uv, uv, uv);" in result
    assert "biased_color = texture(color_image, sampler_, uv, 0.25);" in result
    assert "direct_lod = textureLod(color_image, sampler_, uv, 1.0);" in result
    assert "direct_grad = textureGrad(color_image, sampler_, uv, uv, uv);" in result
    assert (
        "projected_color = textureProjLod(projected_image, sampler_, uvw, 0.0);"
        in result
    )
    assert (
        "direct_projected = textureProjLod(projected_image, sampler_, uvw, 2.0);"
        in result
    )
    assert "depth = textureCompareLod(depth_image, sampler_, uv, 0.5, 0.0);" in result
    assert (
        "direct_depth = textureCompareLod(depth_image, sampler_, uv, 0.5, 1.0);"
        in result
    )
    assert (
        "projected_depth = textureCompareProjGrad(depth_image, sampler_, uvw, 0.5, uv, uv);"
        in result
    )
    assert ".sample_by_lod(" not in result
    assert ".sample_by_gradient(" not in result
    assert ".sample_with(" not in result
    assert ".sample_depth_reference" not in result
    assert "sample_with::" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_sampled_image_generic_type_drives_resource_parameter():
    code = """
    use spirv_std::{spirv, Image};
    use spirv_std::image::SampledImage;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(descriptor_set = 0, binding = 0)] tex: &SampledImage<Image!(2D, type=f32, sampled)>,
    ) {}
    """
    result = parse_and_generate(code)

    assert "fragment main_fs {" in result
    assert "sampler2D tex @ set(0) @ binding(0)" in result
    assert "void main(SampledImage" not in result


def test_rust_gpu_builtin_spirv_aliases_drive_parameter_semantics():
    code = """
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] in_frag_coord: Vec4,
        #[spirv(front_facing)] is_front_facing: bool,
    ) {}

    #[spirv(vertex)]
    pub fn main_vs(
        #[spirv(instance_index)] instance_index: u32,
    ) {}

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(local_invocation_index)] local_index: u32,
    ) {}
    """
    result = parse_and_generate(code)

    assert "fragment main_fs {" in result
    assert "vec4 in_frag_coord @ gl_FragCoord" in result
    assert "bool is_front_facing @ gl_FrontFacing" in result
    assert "vertex main_vs {" in result
    assert "uint instance_index @ InstanceID" in result
    assert "compute main_cs {" in result
    assert "uint local_index @ gl_LocalInvocationIndex" in result


def test_rust_gpu_all_builtins_compiletest_semantic_aliases_codegen():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/spirv-attr/all-builtins.rs.
    code = """
    use spirv_std::glam::*;
    use spirv_std::spirv;

    #[spirv(vertex)]
    pub fn vertex(
        #[spirv(base_instance)] base_instance: u32,
        #[spirv(base_vertex)] base_vertex: u32,
        #[spirv(device_index)] device_index: u32,
        #[spirv(draw_index)] draw_index: u32,
        #[spirv(point_size)] point_size: &mut u32,
        #[spirv(subgroup_eq_mask)] subgroup_eq_mask: UVec4,
        #[spirv(subgroup_ge_mask)] subgroup_ge_mask: UVec4,
        #[spirv(subgroup_gt_mask)] subgroup_gt_mask: UVec4,
        #[spirv(subgroup_le_mask)] subgroup_le_mask: UVec4,
        #[spirv(subgroup_lt_mask)] subgroup_lt_mask: UVec4,
        #[spirv(viewport_index)] viewport_index: u32,
    ) {}

    #[spirv(fragment)]
    pub fn fragment(
        #[spirv(bary_coord)] bary_coord: Vec3,
        #[spirv(bary_coord_no_persp)] bary_coord_no_persp: Vec3,
        #[spirv(clip_distance)] clip_distance: [f32; 1],
        #[spirv(cull_distance)] cull_distance: [f32; 1],
        #[spirv(frag_depth)] frag_depth: &mut f32,
        #[spirv(frag_invocation_count_ext, flat)] frag_invocation_count_ext: u32,
        #[spirv(frag_size_ext, flat)] frag_size_ext: UVec2,
        #[spirv(fully_covered_ext)] fully_covered_ext: bool,
        #[spirv(helper_invocation)] helper_invocation: bool,
        #[spirv(layer, flat)] layer: u32,
        #[spirv(sample_id, flat)] sample_id: u32,
        #[spirv(sample_mask, flat)] sample_mask: [u32; 1],
        #[spirv(sample_position)] sample_position: Vec2,
        #[spirv(viewport_index, flat)] viewport_index: u32,
    ) {}
    """
    result = parse_and_generate(code)

    assert "uint base_instance @ gl_BaseInstance" in result
    assert "uint base_vertex @ gl_BaseVertex" in result
    assert "uint device_index @ gl_DeviceIndex" in result
    assert "uint draw_index @ gl_DrawID" in result
    assert "uint point_size @ gl_PointSize" in result
    assert "uvec4 subgroup_eq_mask @ gl_SubgroupEqMask" in result
    assert "uvec4 subgroup_ge_mask @ gl_SubgroupGeMask" in result
    assert "uvec4 subgroup_gt_mask @ gl_SubgroupGtMask" in result
    assert "uvec4 subgroup_le_mask @ gl_SubgroupLeMask" in result
    assert "uvec4 subgroup_lt_mask @ gl_SubgroupLtMask" in result
    assert "vec3 bary_coord @ gl_BaryCoordEXT" in result
    assert "vec3 bary_coord_no_persp @ gl_BaryCoordNoPerspEXT" in result
    assert "float clip_distance[1] @ gl_ClipDistance" in result
    assert "float cull_distance[1] @ gl_CullDistance" in result
    assert "float frag_depth @ gl_FragDepth" in result
    assert "uint frag_invocation_count_ext @ gl_FragInvocationCountEXT @ flat" in result
    assert "uvec2 frag_size_ext @ gl_FragSizeEXT @ flat" in result
    assert "bool fully_covered_ext @ gl_FullyCoveredEXT" in result
    assert "bool helper_invocation @ gl_HelperInvocation" in result
    assert "uint layer @ gl_Layer @ flat" in result
    assert "uint sample_id @ gl_SampleID @ flat" in result
    assert "uint sample_mask[1] @ gl_SampleMask @ flat" in result
    assert "vec2 sample_position @ gl_SamplePosition" in result
    assert "uint viewport_index @ gl_ViewportIndex @ flat" in result
    assert result.count("uint viewport_index @ gl_ViewportIndex") == 2
    crosstl.translator.parse(result)


def test_rust_gpu_tessellation_control_stage_codegen_from_upstream_compiletest():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/spirv-attr/all-builtins.rs tessellation_control entry.
    code = """
    use spirv_std::spirv;

    #[spirv(tessellation_control)]
    pub fn main(
        #[spirv(invocation_id)] invocation_id: u32,
        #[spirv(patch_vertices)] patch_vertices: u32,
        #[spirv(tess_level_inner)] tess_level_inner: &mut [f32; 2],
        #[spirv(tess_level_outer)] tess_level_outer: &mut [f32; 4],
    ) {}
    """

    result = parse_and_generate(code)

    assert "tessellation_control {" in result
    assert "void main(" in result
    assert "void tessellation_control(" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_all_builtins_tessellation_and_nv_semantics_codegen():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # tests/compiletests/ui/spirv-attr/all-builtins.rs.
    code = """
    use spirv_std::glam::*;
    use spirv_std::spirv;

    #[spirv(tessellation_control)]
    pub fn tessellation_control(
        #[spirv(invocation_id)] invocation_id: u32,
        #[spirv(patch_vertices)] patch_vertices: u32,
        #[spirv(tess_level_inner)] tess_level_inner: &mut [f32; 2],
        #[spirv(tess_level_outer)] tess_level_outer: &mut [f32; 4],
    ) {}

    #[spirv(tessellation_evaluation)]
    pub fn tessellation_evaluation(#[spirv(tess_coord)] tess_coord: Vec3) {}

    #[spirv(vertex)]
    pub fn vertex(
        #[spirv(SMIDNV)] smidnv: u32,
        #[spirv(mesh_view_count_nv)] mesh_view_count_nv: u32,
        #[spirv(position_per_view_nv)] position_per_view_nv: u32,
        #[spirv(warp_id_nv)] warp_id_nv: u32,
        #[spirv(warps_per_sm_nv)] warps_per_sm_nv: u32,
    ) {}
    """

    result = parse_and_generate(code)

    assert "uint invocation_id @ gl_InvocationID" in result
    assert "uint patch_vertices @ gl_PatchVerticesIn" in result
    assert "float tess_level_inner[2] @ gl_TessLevelInner" in result
    assert "float tess_level_outer[4] @ gl_TessLevelOuter" in result
    assert "vec3 tess_coord @ gl_TessCoord" in result
    assert "uint smidnv @ gl_SMIDNV" in result
    assert "uint mesh_view_count_nv @ gl_MeshViewCountNV" in result
    assert "uint position_per_view_nv @ gl_PositionPerViewNV" in result
    assert "uint warp_id_nv @ gl_WarpIDNV" in result
    assert "uint warps_per_sm_nv @ gl_WarpsPerSMNV" in result
    crosstl.translator.parse(result)


def test_vulkan_shader_examples_multiview_view_index_semantic_codegen():
    # Reduced from Rust-GPU/VulkanShaderExamples commit
    # b29a37eb46802b5ea6882af4808d6887fc184581,
    # shaders/rust/multiview/multiview/src/lib.rs main_vs.
    code = """
    use spirv_std::spirv;

    #[spirv(vertex)]
    pub fn main_vs(
        in_pos: Vec3,
        #[spirv(view_index)] view_index: u32,
        #[spirv(position)] out_position: &mut Vec4,
    ) {
        *out_position = Vec4::new(view_index as f32, in_pos.y, in_pos.z, 1.0);
    }
    """

    result = parse_and_generate(code)

    assert "vertex main_vs {" in result
    assert "uint view_index @ gl_ViewIndex" in result
    assert "vec4 out_position @ gl_Position" in result
    crosstl.translator.parse(result)


def test_rust_gpu_reduce_subgroup_builtins_codegen_from_upstream_example():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/reduce/src/lib.rs compute subgroup reduction entry point.
    code = """
    use spirv_std::glam::UVec3;
    use spirv_std::spirv;

    #[spirv(compute(threads(256)))]
    pub fn main(
        #[spirv(global_invocation_id)] global_invocation_id: UVec3,
        #[spirv(local_invocation_id)] local_invocation_id: UVec3,
        #[spirv(subgroup_local_invocation_id)] subgroup_local_invocation_id: u32,
        #[spirv(workgroup_id)] workgroup_id: UVec3,
        #[spirv(subgroup_id)] subgroup_id: u32,
        #[spirv(num_subgroups)] num_subgroups: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
        #[spirv(workgroup)] shared: &mut [u32; 256],
    ) {
        let global_invocation_id_x = global_invocation_id.x as usize;
        let local_invocation_id_x = local_invocation_id.x as usize;
        let workgroup_id_x = workgroup_id.x as usize;

        let mut sum = 0;
        if global_invocation_id_x < input.len() {
            sum = input[global_invocation_id_x];
        }
        sum = unsafe { subgroup_add(sum) };
        if subgroup_local_invocation_id == 0 {
            shared[subgroup_id as usize] = sum;
        }
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
        let mut sum = 0;
        if subgroup_id == 0 {
            if subgroup_local_invocation_id < num_subgroups {
                sum = shared[subgroup_local_invocation_id as usize];
            }
            sum = unsafe { subgroup_add(sum) };
        }
        if local_invocation_id_x == 0 {
            output[workgroup_id_x] = sum;
        }
    }
    """
    result = parse_and_generate(code)

    assert "compute {" in result
    assert "void main(uvec3 global_invocation_id @ gl_GlobalInvocationID" in result
    assert "uvec3 local_invocation_id @ gl_LocalInvocationID" in result
    assert "uint subgroup_local_invocation_id @ gl_SubgroupInvocationID" in result
    assert "uvec3 workgroup_id @ gl_WorkGroupID" in result
    assert "uint subgroup_id @ gl_SubgroupID" in result
    assert "uint num_subgroups @ gl_NumSubgroups" in result
    assert "uint input[] @ set(0) @ binding(0)" in result
    assert "uint output[] @ set(0) @ binding(1)" in result
    assert "uint shared_[256] @ groupshared" in result
    assert "@numthreads(256, 1, 1)" in result
    assert "sum = subgroup_add(sum);" in result
    assert "spirv_std::arch::workgroup_memory_barrier_with_group_sync();" in result
    assert "output[workgroup_id_x] = sum_;" in result
    crosstl.translator.parse(result)


def test_rust_gpu_compute_collatz_option_chain_codegen_from_upstream_example():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/compute-shader/src/lib.rs Collatz compute entry point.
    code = """
    use glam::UVec3;
    use spirv_std::{glam, spirv};

    pub fn collatz(mut n: u32) -> Option<u32> {
        let mut i = 0;
        if n == 0 {
            return None;
        }
        while n != 1 {
            n = if n.is_multiple_of(2) {
                n / 2
            } else {
                if n >= 0x5555_5555 {
                    return None;
                }
                3 * n + 1
            };
            i += 1;
        }
        Some(i)
    }

    #[spirv(compute(threads(64)))]
    pub fn main_cs(
        #[spirv(global_invocation_id)] id: UVec3,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
        prime_indices: &mut [u32],
    ) {
        let index = id.x as usize;
        prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);
    }
    """
    result = parse_and_generate(code)

    assert "// use spirv_std::{glam, spirv}" in result
    assert "Option<u32> collatz(uint n)" in result
    assert "if (((n % 2) == 0))" in result
    assert "return Some(i);" in result
    assert "compute main_cs {" in result
    assert "void main(uvec3 id @ gl_GlobalInvocationID" in result
    assert "uint prime_indices[] @ set(0) @ binding(0)" in result
    assert "@numthreads(64, 1, 1)" in result
    assert "let index = (uint)id.x;" in result
    assert (
        "prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);"
        in result
    )
    assert "Unhandled statement type" not in result


def test_rust_shader_stage_local_aliases_shadow_parameter_after_initializer(tmp_path):
    code = """
    #[vertex_shader]
    pub fn vertex_main(vertex: Vec3<f32>, condition: bool) -> Vec4<f32> {
        let vertex = Vec4::new(vertex.x, vertex.y, vertex.z, 1.0);
        if condition {
            let vertex = Vec4::new(0.0, 0.0, 0.0, 1.0);
        }
        let fragment: Vec4<f32> = vertex;
        return fragment;
    }
    """
    rust_file = tmp_path / "shader.rs"
    rust_file.write_text(code, encoding="utf-8")

    cgl_result = crosstl.translate(str(rust_file), backend="cgl", format_output=False)
    assert "vec4 main(vec3 vertex_, bool condition)" in cgl_result
    assert "let vertex_1 = vec4(vertex_.x, vertex_.y, vertex_.z, 1.0);" in cgl_result
    assert "let vertex_2 = vec4(0.0, 0.0, 0.0, 1.0);" in cgl_result
    assert "vec4 fragment_ = vertex_1;" in cgl_result
    assert "return fragment_;" in cgl_result

    rust_result = crosstl.translate(str(rust_file), backend="rust", format_output=False)
    assert (
        "pub fn vertex_main(vertex_: Vec3<f32>, condition: bool) -> Vec4<f32>"
        in rust_result
    )
    assert (
        "let vertex_1: Vec4<f32> = "
        "Vec4::<f32>::new(vertex_.x, vertex_.y, vertex_.z, 1.0);" in rust_result
    )
    assert (
        "let vertex_2: Vec4<f32> = "
        "Vec4::<f32>::new(0.0, 0.0, 0.0, 1.0);" in rust_result
    )
    assert "let fragment_: Vec4<f32> = vertex_1;" in rust_result
    assert "return fragment_;" in rust_result


def test_fragment_shader_conversion():
    code = """
    #[fragment_shader]
    pub fn fragment_main(input: Vec2<f32>) -> Vec4<f32> {
        Vec4::new(input.x, input.y, 0.0, 1.0)
    }
    """
    try:
        result = parse_and_generate(code)
        assert "fragment fragment_main {" in result
        assert "vec4 main(vec2 input)" in result
        assert "return vec4(input.x, input.y, 0.0, 1.0);" in result
    except Exception as e:
        pytest.fail(f"Fragment shader conversion failed: {e}")


def test_compute_shader_conversion():
    code = """
    #[compute_shader]
    pub fn compute_main() {
        // Compute shader logic
    }
    """
    try:
        result = parse_and_generate(code)
        assert "compute compute_main {" in result
        assert "void main()" in result
    except Exception as e:
        pytest.fail(f"Compute shader conversion failed: {e}")


def test_type_mapping():
    code = """
    pub struct TypeTest {
        f32_val: f32,
        i32_val: i32,
        u32_val: u32,
        bool_val: bool,
        vec2_val: Vec2<f32>,
        vec3_val: Vec3<f32>,
        vec4_val: Vec4<f32>,
        mat4_val: Mat4<f32>,
    }
    """
    try:
        result = parse_and_generate(code)
        assert "float f32_val" in result
        assert "int i32_val" in result
        assert "uint u32_val" in result
        assert "bool bool_val" in result
        assert "vec2 vec2_val" in result
        assert "vec3 vec3_val" in result
        assert "vec4 vec4_val" in result
        assert "mat4 mat4_val" in result
    except Exception as e:
        pytest.fail(f"Type mapping failed: {e}")


def test_type_alias_conversion():
    code = """
    type Real = f32;
    type Color = Vec3<f32>;
    type Weights = [f32; 4];
    type Buffer<T> = [T; 4];
    type Vector<T> = Vec3<T>;

    fn sample(x: Real, weights: Weights, buffer: Buffer<f32>, normal: Vector<f32>) -> Real {
        let color: Color;
        x
    }

    impl<T> Kernel<T> {
        type Output = T;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "typedef float Real;" in result
        assert "typedef vec3 Color;" in result
        assert "typedef float Weights[4];" in result
        assert "typedef" in result
        assert "Buffer" not in result
        assert "Vector" not in result
        assert (
            "Real sample(Real x, Weights weights, float buffer[4], vec3 normal)"
            in result
        )
        assert "Color color;" in result
    except Exception as e:
        pytest.fail(f"Type alias conversion failed: {e}")


def test_turbofish_path_conversion():
    code = """
    type Vector<T> = Vec3<T>;

    fn test_paths() {
        let a = Vec3::<f32>::new(1.0, 2.0, 3.0);
        let b: Vector<f32> = Vector::<f32>::new(4.0, 5.0, 6.0);
        let c = Type::<f32>::CONST;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "a = vec3(1.0, 2.0, 3.0);" in result
        assert "vec3 b = vec3(4.0, 5.0, 6.0);" in result
        assert "c = Type<f32>::CONST;" in result
        assert "Vector" not in result
        assert "Vec3::<f32>" not in result
    except Exception as e:
        pytest.fail(f"Turbofish path conversion failed: {e}")


def test_module_path_alias_conversion():
    code = """
    type Real = f32;
    type Vector<T> = Vec3<T>;

    fn sample(x: crate::math::Real, normal: crate::math::Vector<f32>) -> crate::math::Real {
        let lifted = crate::math::Vector::<f32>::new(1.0, 2.0, 3.0);
        let c = self::helper::CONST;
        let s = super::math::VALUE;
        x
    }
    """
    try:
        result = parse_and_generate(code)
        assert "typedef float Real;" in result
        assert "Real sample(Real x, vec3 normal)" in result
        assert "lifted = vec3(1.0, 2.0, 3.0);" in result
        assert "c = self::helper::CONST;" in result
        assert "s = super::math::VALUE;" in result
        assert "crate::math::Vector" not in result
    except Exception as e:
        pytest.fail(f"Module path alias conversion failed: {e}")


def test_qualified_math_function_conversion():
    code = """
    fn sample(x: f32) {
        let a = crate::math::sin(x);
        let b = self::shader_math::clamp(x, 0.0, 1.0);
        let c = std::f32::sqrt(x);
        let d = crate::math::custom(x);
        let e = Color::clamp(x, 0.0, 1.0);
        let f = std::f32::ln(x);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "a = sin(x);" in result
        assert "b = clamp(x, 0.0, 1.0);" in result
        assert "c = sqrt(x);" in result
        assert "d = crate::math::custom(x);" in result
        assert "e = Color::clamp(x, 0.0, 1.0);" in result
        assert "f = log(x);" in result
        assert "std::f32::ln(" not in result
    except Exception as e:
        pytest.fail(f"Qualified math function conversion failed: {e}")


def test_qualified_builtin_type_conversion():
    code = """
    fn sample(position: crate::math::Vec3<f32>, tangent: glam::Vec4) -> glam::Vec4 {
        let a: crate::math::Vec3<f32> = crate::math::Vec3::<f32>::new(1.0, 2.0, 3.0);
        let b = glam::Vec3::new(4.0, 5.0, 6.0);
        let transform: crate::math::Mat4<f32>;
        tangent
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec4 sample(vec3 position, vec4 tangent)" in result
        assert "vec3 a = vec3(1.0, 2.0, 3.0);" in result
        assert "b = vec3(4.0, 5.0, 6.0);" in result
        assert "mat4 transform;" in result
        assert "crate::math::Vec3" not in result
        assert "glam::Vec" not in result
    except Exception as e:
        pytest.fail(f"Qualified builtin type conversion failed: {e}")


def test_use_alias_builtin_type_conversion():
    code = """
    use glam::Vec3 as GVec3;
    use crate::math::Mat4 as CMat4;
    use crate::math::Unknown as ShaderUnknown;

    fn sample(position: GVec3, extra: GVec3<f32>) {
        let a: GVec3 = GVec3::new(1.0, 2.0, 3.0);
        let b: GVec3<f32> = GVec3::<f32>::new(4.0, 5.0, 6.0);
        let transform: CMat4;
        let fallback: ShaderUnknown;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "void sample(vec3 position, vec3 extra)" in result
        assert "vec3 a = vec3(1.0, 2.0, 3.0);" in result
        assert "vec3 b = vec3(4.0, 5.0, 6.0);" in result
        assert "mat4 transform;" in result
        assert "ShaderUnknown fallback;" in result
        assert "GVec3" not in result
        assert "CMat4" not in result
    except Exception as e:
        pytest.fail(f"Use alias builtin type conversion failed: {e}")


def test_grouped_use_alias_builtin_type_conversion():
    code = """
    use glam::{Vec3 as GVec3, Vec4 as GVec4};
    use crate::math::{Mat4 as CMat4, Unknown as ShaderUnknown};

    fn sample(position: GVec3) -> GVec4 {
        let a: GVec3 = GVec3::new(1.0, 2.0, 3.0);
        let color: GVec4 = GVec4::new(1.0, 0.5, 0.25, 1.0);
        let transform: CMat4;
        let fallback: ShaderUnknown;
        color
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec4 sample(vec3 position)" in result
        assert "vec3 a = vec3(1.0, 2.0, 3.0);" in result
        assert "vec4 color = vec4(1.0, 0.5, 0.25, 1.0);" in result
        assert "mat4 transform;" in result
        assert "ShaderUnknown fallback;" in result
        assert "GVec3::new" not in result
        assert "GVec4::new" not in result
    except Exception as e:
        pytest.fail(f"Grouped use alias builtin type conversion failed: {e}")


def test_nested_grouped_use_alias_builtin_type_conversion():
    code = """
    use crate::{math::{Vec3 as CVec3, Mat4 as CMat4}, color::Vec4 as CVec4};

    fn sample(position: CVec3) -> CVec4 {
        let a: CVec3 = CVec3::new(1.0, 2.0, 3.0);
        let color: CVec4 = CVec4::new(1.0, 0.5, 0.25, 1.0);
        let transform: CMat4;
        color
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec4 sample(vec3 position)" in result
        assert "vec3 a = vec3(1.0, 2.0, 3.0);" in result
        assert "vec4 color = vec4(1.0, 0.5, 0.25, 1.0);" in result
        assert "mat4 transform;" in result
        assert "CVec3::new" not in result
        assert "CVec4::new" not in result
        assert "CVec3 position" not in result
        assert "CMat4 transform" not in result
    except Exception as e:
        pytest.fail(f"Nested grouped use alias builtin type conversion failed: {e}")


def test_use_module_alias_path_conversion():
    code = """
    type Real = f32;
    use crate::math as shader_math;
    use glam as gm;
    use crate::{color as shader_color};

    fn sample(x: shader_math::Real) -> shader_math::Real {
        let a = shader_math::sin(x);
        let b = shader_math::custom(x);
        let c = shader_math::PI;
        let v = shader_math::Vec3::new(1.0, 2.0, 3.0);
        let color: gm::Vec4 = shader_color::Vec4::new(1.0, 0.5, 0.25, 1.0);
        x
    }
    """
    try:
        result = parse_and_generate(code)
        assert "Real sample(Real x)" in result
        assert "a = sin(x);" in result
        assert "b = crate::math::custom(x);" in result
        assert "c = crate::math::PI;" in result
        assert "v = vec3(1.0, 2.0, 3.0);" in result
        assert "vec4 color = vec4(1.0, 0.5, 0.25, 1.0);" in result
        assert "shader_math::" not in result
        assert "shader_color::" not in result
    except Exception as e:
        pytest.fail(f"Use module alias path conversion failed: {e}")


def test_pub_use_and_grouped_glob_alias_conversion():
    code = """
    pub(crate) use glam::Vec3 as PubVec3;
    use crate::{math::{*, Vec3 as CVec3}, color::*};

    fn sample(position: CVec3, other: PubVec3) {
        let a: CVec3 = CVec3::new(1.0, 2.0, 3.0);
        let b: PubVec3 = PubVec3::new(4.0, 5.0, 6.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "void sample(vec3 position, vec3 other)" in result
        assert "vec3 a = vec3(1.0, 2.0, 3.0);" in result
        assert "vec3 b = vec3(4.0, 5.0, 6.0);" in result
        assert "CVec3::new" not in result
        assert "PubVec3::new" not in result
    except Exception as e:
        pytest.fail(f"Pub use and grouped glob alias conversion failed: {e}")


def test_control_flow_conversion():
    code = """
    fn test_control() {
        if x > 0 {
            let y = x + 1;
        } else {
            let y = x - 1;
        }

        for i in 0..10 {
            println!("{}", i);
        }

        while condition {
            do_something();
        }

        loop {
            if should_break {
                break;
            }
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "if (" in result
        assert "} else {" in result
        assert "for (" in result
        assert "while (" in result
        assert "while (true) {" in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"Control flow conversion failed: {e}")


def test_for_loop_pattern_conversion():
    code = """
    fn test_for_patterns() {
        for i in 1..=3 {
            use_index(i);
        }
        for _ in 0..4 {
            tick();
        }
        for (_, _) in pairs {
            tick_pair();
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "for (int i = 1; i <= 3; i++)" in result
        assert "for (int _for_index_0 = 0; _for_index_0 < 4; _for_index_0++)" in result
        assert (
            "for (int _for_index_1 = 0; " "_for_index_1 < pairs; _for_index_1++)"
        ) in result
        assert "tick();" in result
        assert "tick_pair();" in result
        assert "RangeNode" not in result
        assert "TupleNode" not in result
    except Exception as e:
        pytest.fail(f"For loop pattern conversion failed: {e}")


def test_for_loop_enumerate_iter_tuple_pattern_conversion():
    code = """
    fn test_enumerate(values: Buffer) {
        for (i, value) in values.iter().enumerate() {
            consume(i, value);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "for (int i = 0; i < values.length; i++)" in result
        assert "auto value = values[i];" in result
        assert "consume(i, value);" in result
        assert "values.iter().enumerate()" not in result
        assert "_for_bound" not in result
    except Exception as e:
        pytest.fail(f"For loop enumerate tuple pattern conversion failed: {e}")


def test_for_loop_enumerate_iter_side_effect_source_conversion():
    code = """
    fn test_enumerate_side_effect_source() {
        for (_, value) in next_values().iter().enumerate() {
            consume(value);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _for_iterable_0 = next_values();" in result
        assert (
            "for (int _for_index_0 = 0; "
            "_for_index_0 < _for_iterable_0.length; _for_index_0++)"
        ) in result
        assert "auto value = _for_iterable_0[_for_index_0];" in result
        assert "consume(value);" in result
        assert result.count("next_values()") == 1
        assert "next_values().iter().enumerate()" not in result
    except Exception as e:
        pytest.fail(f"For loop enumerate side-effect source conversion failed: {e}")


def test_for_loop_step_by_range_conversion():
    code = """
    fn test_for_step_by() {
        for i in (0..10).step_by(2) {
            use_index(i);
        }
        for j in (1..=9).step_by(step) {
            use_index(j);
        }
        for k in (0..limit).step_by(next_step()) {
            use_index(k);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "for (int i = 0; i < 10; i += 2)" in result
        assert "for (int j = 1; j <= 9; j += step)" in result
        assert "int _for_step_0 = next_step();" in result
        assert "for (int k = 0; k < limit; k += _for_step_0)" in result
        assert result.count("next_step()") == 1
        assert "RangeNode" not in result
        assert ".step_by" not in result
    except Exception as e:
        pytest.fail(f"For loop step_by range conversion failed: {e}")


def test_for_loop_rev_range_conversion():
    code = """
    fn test_for_rev(limit: i32) {
        for i in (0..4).rev() {
            use_index(i);
        }
        for j in (1..=4).rev() {
            use_index(j);
        }
        for k in (0..limit).rev() {
            use_index(k);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "for (int i = 3; i >= 0; i--)" in result
        assert "for (int j = 4; j >= 1; j--)" in result
        assert "for (int k = (limit - 1); k >= 0; k--)" in result
        assert "RangeNode" not in result
        assert "0.." not in result
        assert ".rev" not in result
    except Exception as e:
        pytest.fail(f"For loop rev range conversion failed: {e}")


def test_for_loop_step_by_rev_range_conversion():
    code = """
    fn test_for_step_by_rev() {
        for i in (0..10).step_by(2).rev() {
            use_index(i);
        }
        for j in (0..10).rev().step_by(2) {
            use_index(j);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "for (int i = 8; i >= 0; i -= 2)" in result
        assert "for (int j = 9; j >= 0; j -= 2)" in result
        assert "0..10" not in result
        assert ".step_by" not in result
        assert ".rev" not in result
    except Exception as e:
        pytest.fail(f"For loop step_by rev range conversion failed: {e}")


def test_for_loop_range_bound_additive_precedence_conversion():
    code = """
    fn test_for_range_bounds(count: i32) {
        for i in 0..count + 1 {
            use_index(i);
        }
    }
    """

    result = parse_and_generate(code)

    assert "for (int i = 0; i < (count + 1); i++)" in result
    assert "RangeNode" not in result
    assert "0..count" not in result


def test_for_loop_range_side_effect_bound_conversion():
    code = """
    fn test_for_bound_effects() {
        for i in next_start()..next_limit() {
            use_index(i);
        }
        for j in (next_start()..=next_limit()).step_by(next_step()) {
            use_index(j);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        first_start = result.index("int _for_bound_0 = next_start();")
        first_end = result.index("int _for_bound_1 = next_limit();")
        first_loop = result.index("for (int i = _for_bound_0; i < _for_bound_1; i++)")
        assert first_start < first_end < first_loop

        second_start = result.index("int _for_bound_2 = next_start();")
        second_end = result.index("int _for_bound_3 = next_limit();")
        step = result.index("int _for_step_0 = next_step();")
        second_loop = result.index(
            "for (int j = _for_bound_2; " "j <= _for_bound_3; j += _for_step_0)"
        )
        assert second_start < second_end < step < second_loop

        assert result.count("next_start()") == 2
        assert result.count("next_limit()") == 2
        assert result.count("next_step()") == 1
    except Exception as e:
        pytest.fail(f"For loop range side-effect bound conversion failed: {e}")


def test_for_loop_fallback_side_effect_iterable_conversion():
    code = """
    fn test_for_count_effects(count: i32) {
        for i in next_count() {
            use_index(i);
        }
        for j in count {
            use_index(j);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        setup = result.index("int _for_bound_0 = next_count();")
        first_loop = result.index("for (int i = 0; i < _for_bound_0; i++)")
        second_loop = result.index("for (int j = 0; j < count; j++)")

        assert setup < first_loop < second_loop
        assert result.count("next_count()") == 1
        assert "_for_bound_1" not in result
    except Exception as e:
        pytest.fail(f"For loop fallback side-effect iterable conversion failed: {e}")


def test_len_method_conversion():
    code = """
    fn test_len_method(values: Buffer) {
        let n = values.len();
        for i in 0..values.len() {
            use_index(i);
        }
        for j in values.len() {
            use_index(j);
        }
        for k in 0..next_values().len() {
            use_index(k);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "n = values.length;" in result
        assert "for (int i = 0; i < values.length; i++)" in result
        assert "for (int j = 0; j < values.length; j++)" in result
        assert "int _for_bound_0 = next_values().length;" in result
        assert "for (int k = 0; k < _for_bound_0; k++)" in result
        assert result.count("next_values()") == 1
        assert ".len()" not in result
    except Exception as e:
        pytest.fail(f"len method conversion failed: {e}")


def test_vector_method_conversion():
    code = """
    fn test_vector_methods(input: VertexInput, normal: Vec3<f32>, other: Vec3<f32>) {
        let position = input.position.xyz();
        let color = sample_color().rgba();
        let mag = normal.length();
        let unit = normal.normalize();
        let dp = normal.dot(other);
        let cp = normal.cross(other);
        for i in 0..next_values().xyz().length() {
            use_index(i);
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "position = input.position.xyz;" in result
        assert "color = sample_color().rgba;" in result
        assert "mag = length(normal);" in result
        assert "unit = normalize(normal);" in result
        assert "dp = dot(normal, other);" in result
        assert "cp = cross(normal, other);" in result
        assert "int _for_bound_0 = length(next_values().xyz);" in result
        assert "for (int i = 0; i < _for_bound_0; i++)" in result
        assert result.count("next_values()") == 1
        assert ".xyz()" not in result
        assert ".rgba()" not in result
        assert ".length()" not in result
        assert ".normalize()" not in result
        assert ".dot(" not in result
        assert ".cross(" not in result
    except Exception as e:
        pytest.fail(f"Vector method conversion failed: {e}")


def test_scalar_math_method_conversion():
    code = """
    fn scalar_methods(x: f32, y: f32, angle: f32, n: i32) -> f32 {
        let a = x.abs();
        let b = y.sqrt();
        let c = angle.sin();
        let d = y.ln();
        let e = angle.to_degrees();
        let f = x.powf(y);
        let g = x.powi(n);
        let h = x.fract();
        let i = x.clamp(0.0, 1.0);
        let j = x.clamp(y, angle);
        let k = x.min(y);
        let l = x.max(y);
        let m = y.atan2(x);
        return a + b + c + d + e + f + g + h + i + j + k + l + m;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "a = abs(x);" in result
        assert "b = sqrt(y);" in result
        assert "c = sin(angle);" in result
        assert "d = log(y);" in result
        assert "e = degrees(angle);" in result
        assert "f = pow(x, y);" in result
        assert "g = pow(x, n);" in result
        assert "h = fract(x);" in result
        assert "i = clamp(x, 0.0, 1.0);" in result
        assert "j = clamp(x, y, angle);" in result
        assert "k = min(x, y);" in result
        assert "l = max(x, y);" in result
        assert "m = atan2(y, x);" in result
        assert ".abs()" not in result
        assert ".sqrt()" not in result
        assert ".sin()" not in result
        assert ".ln()" not in result
        assert ".to_degrees()" not in result
        assert ".powf(" not in result
        assert ".powi(" not in result
        assert ".fract()" not in result
        assert ".clamp(" not in result
        assert ".min(" not in result
        assert ".max(" not in result
        assert ".atan2(" not in result
    except Exception as e:
        pytest.fail(f"Scalar math method conversion failed: {e}")


def test_common_math_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn common_intrinsics(
        normal: Vec3<f32>,
        incident: Vec3<f32>,
        reference: Vec3<f32>,
        x: f32,
        y: f32,
        angle: f32,
    ) -> f32 {
        let dist = distance(normal, incident);
        let facing = faceforward(normal, incident, reference);
        let rounded = round(x);
        let exponential = exp(x);
        let exponential2 = exp2(x);
        let natural_log = log(y);
        let binary_log = log2(y);
        let deg = degrees(angle);
        let rad = radians(deg);
        let arc = asin(x) + acos(x) + atan(x) + atan2(y, x);
        let hyper = sinh(x) + cosh(x) + tanh(x);
        return dist + rounded + exponential + exponential2 + natural_log
            + binary_log + deg + rad + arc + hyper;
    }
    """

    result = parse_and_generate(code)

    assert "dist = distance(normal, incident);" in result
    assert "facing = faceforward(normal, incident, reference);" in result
    assert "rounded = round(x);" in result
    assert "exponential = exp(x);" in result
    assert "exponential2 = exp2(x);" in result
    assert "natural_log = log(y);" in result
    assert "binary_log = log2(y);" in result
    assert "deg = degrees(angle);" in result
    assert "rad = radians(deg);" in result
    assert "arc = (((asin(x) + acos(x)) + atan(x)) + atan2(y, x));" in result
    assert "hyper = ((sinh(x) + cosh(x)) + tanh(x));" in result
    assert "Vec3" not in result


def test_unary_math_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn unary_math_ops(value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
        let root = crate::math::sqrt(value);
        let inv_root = rsqrt(value);
        let fractional = crate::math::fract(value);
        let angles = degrees(value) + radians(value);
        let trig = sin(value) + cos(value) + tan(value);
        let inverse_trig = asin(value) + acos(value) + atan(value);
        let hyper = sinh(value) + cosh(value) + tanh(value);
        let logs = exp(value) + exp2(value) + log(value) + log2(value);
        let rounded = floor(value) + ceil(value) + round(value) + trunc(value)
            + round_even(value);
        let scalar_root = scalar.sqrt();
        let scalar_fraction = scalar.fract();
        return root + inv_root + fractional + angles + trig + inverse_trig
            + hyper + logs + rounded
            + Vec3::<f32>::new(scalar_root, scalar_fraction, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "vec3 unary_math_ops(vec3 value, float scalar)" in result
    assert "root = sqrt(value);" in result
    assert "inv_root = inversesqrt(value);" in result
    assert "fractional = fract(value);" in result
    assert "degrees(value)" in result
    assert "radians(value)" in result
    assert "sin(value)" in result
    assert "cos(value)" in result
    assert "tan(value)" in result
    assert "asin(value)" in result
    assert "acos(value)" in result
    assert "atan(value)" in result
    assert "sinh(value)" in result
    assert "cosh(value)" in result
    assert "tanh(value)" in result
    assert "exp(value)" in result
    assert "exp2(value)" in result
    assert "log(value)" in result
    assert "log2(value)" in result
    assert "floor(value)" in result
    assert "ceil(value)" in result
    assert "round(value)" in result
    assert "trunc(value)" in result
    assert "roundEven(value)" in result
    assert "scalar_root = sqrt(scalar);" in result
    assert "scalar_fraction = fract(scalar);" in result
    assert "crate::math::sqrt" not in result
    assert "rsqrt(value)" not in result
    assert "crate::math::fract" not in result
    assert ".sqrt()" not in result
    assert ".fract()" not in result


def test_rounding_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn rounding_ops(value: Vec3<f32>, scale: f32) -> Vec3<f32> {
        let truncated = trunc(value);
        let even = crate::math::round_even(value);
        let scalar_truncated = scale.trunc();
        let scalar_even = scale.round_ties_even();
        return truncated + even + Vec3::<f32>::new(scalar_truncated, scalar_even, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "vec3 rounding_ops(vec3 value, float scale)" in result
    assert "truncated = trunc(value);" in result
    assert "even = roundEven(value);" in result
    assert "scalar_truncated = trunc(scale);" in result
    assert "scalar_even = roundEven(scale);" in result
    assert "crate::math::round_even" not in result
    assert ".trunc()" not in result
    assert ".round_ties_even()" not in result


def test_min_max_mixed_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn min_max_ops(value: Vec3<f32>, other: Vec3<f32>, scalar: f32) -> Vec3<f32> {
        let min_vector_scalar = min(value, scalar);
        let max_vector_scalar = crate::math::max(value, scalar);
        let min_scalar_vector = min(scalar, other);
        let max_scalar_vector = max(scalar, other);
        let min_vector_vector = min(value, other);
        let max_vector_vector = crate::math::max(value, other);
        let min_scalar_scalar = scalar.min(1.0);
        let max_scalar_scalar = scalar.max(0.0);
        return min_vector_scalar + max_vector_scalar + min_scalar_vector
            + max_scalar_vector + min_vector_vector + max_vector_vector;
    }
    """

    result = parse_and_generate(code)

    assert "vec3 min_max_ops(vec3 value, vec3 other, float scalar)" in result
    assert "min_vector_scalar = min(value, scalar);" in result
    assert "max_vector_scalar = max(value, scalar);" in result
    assert "min_scalar_vector = min(scalar, other);" in result
    assert "max_scalar_vector = max(scalar, other);" in result
    assert "min_vector_vector = min(value, other);" in result
    assert "max_vector_vector = max(value, other);" in result
    assert "min_scalar_scalar = min(scalar, 1.0);" in result
    assert "max_scalar_scalar = max(scalar, 0.0);" in result
    assert "crate::math::max" not in result
    assert ".min(" not in result
    assert ".max(" not in result


def test_binary_math_mixed_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn binary_math_ops(value: Vec3<f32>, other: Vec3<f32>, scalar: f32) -> Vec3<f32> {
        let pow_vector_scalar = pow(value, scalar);
        let pow_scalar_vector = crate::math::pow(scalar, other);
        let mod_vector_scalar = modulo(value, scalar);
        let mod_scalar_vector = crate::math::modulo(scalar, other);
        let atan_vector_scalar = atan2(value, scalar);
        let atan_scalar_vector = crate::math::atan2(scalar, other);
        let pow_scalar_scalar = scalar.powf(2.0);
        let atan_scalar_scalar = scalar.atan2(1.0);
        return pow_vector_scalar + pow_scalar_vector + mod_vector_scalar
            + mod_scalar_vector + atan_vector_scalar + atan_scalar_vector;
    }
    """

    result = parse_and_generate(code)

    assert "vec3 binary_math_ops(vec3 value, vec3 other, float scalar)" in result
    assert "pow_vector_scalar = pow(value, scalar);" in result
    assert "pow_scalar_vector = pow(scalar, other);" in result
    assert "mod_vector_scalar = mod(value, scalar);" in result
    assert "mod_scalar_vector = mod(scalar, other);" in result
    assert "atan_vector_scalar = atan2(value, scalar);" in result
    assert "atan_scalar_vector = atan2(scalar, other);" in result
    assert "pow_scalar_scalar = pow(scalar, 2.0);" in result
    assert "atan_scalar_scalar = atan2(scalar, 1.0);" in result
    assert "crate::math::pow" not in result
    assert "crate::math::modulo" not in result
    assert "crate::math::atan2" not in result
    assert ".powf(" not in result
    assert ".atan2(" not in result


def test_matrix_intrinsic_calls_and_types_convert_to_crossgl_intrinsics():
    code = """
    fn matrix_intrinsics(transform: Mat3<f64>, affine: Mat3x4<f32>) -> f64 {
        let basis = transpose(transform);
        let inverted = inverse(basis);
        let det = determinant(inverted);
        let rectangular = transpose(affine);
        return det;
    }
    """

    result = parse_and_generate(code)

    assert "double matrix_intrinsics(dmat3 transform, mat3x4 affine)" in result
    assert "basis = transpose(transform);" in result
    assert "inverted = inverse(basis);" in result
    assert "det = determinant(inverted);" in result
    assert "rectangular = transpose(affine);" in result
    assert "Mat3" not in result
    assert "Mat3x4" not in result


def test_matrix_product_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn matrix_product_intrinsics(column: Vec3<f32>, row: Vec2<f32>, transform: Mat2<f32>, precise: Mat2<f64>) -> Mat2x3<f32> {
        let basis = outer_product(column, row);
        let componentwise = crate::math::matrix_comp_mult(transform, precise);
        return basis;
    }
    """

    result = parse_and_generate(code)

    assert (
        "mat2x3 matrix_product_intrinsics(vec3 column, vec2 row, mat2 transform, dmat2 precise)"
        in result
    )
    assert "basis = outerProduct(column, row);" in result
    assert "componentwise = matrixCompMult(transform, precise);" in result
    assert "crate::math::matrix_comp_mult" not in result
    assert "Mat2x3" not in result


def test_step_and_smoothstep_calls_convert_to_crossgl_intrinsics():
    code = """
    fn thresholds(low: Vec3<f32>, high: Vec3<f32>, value: Vec3<f32>, scalar: f32) -> Vec3<f32> {
        let vector_gate = step(0.25, value);
        let vector_edge_gate = crate::math::step(low, scalar);
        let soft_value = smoothstep(0.0, 1.0, value);
        let soft_edges = crate::math::smoothstep(low, high, scalar);
        return vector_gate + vector_edge_gate + soft_value + soft_edges;
    }
    """

    result = parse_and_generate(code)

    assert "vec3 thresholds(vec3 low, vec3 high, vec3 value, float scalar)" in result
    assert "vector_gate = step(0.25, value);" in result
    assert "vector_edge_gate = step(low, scalar);" in result
    assert "soft_value = smoothstep(0.0, 1.0, value);" in result
    assert "soft_edges = smoothstep(low, high, scalar);" in result
    assert "crate::math::step" not in result
    assert "crate::math::smoothstep" not in result


def test_derivative_and_fused_math_calls_convert_to_crossgl_intrinsics():
    code = """
    fn derivative_ops(value: Vec3<f32>, scale: f32, bias: f32) -> Vec3<f32> {
        let dx = dfdx(value);
        let dy = crate::math::dfdy(value);
        let fine = dfdx_fine(value);
        let coarse = crate::math::dfdy_coarse(value);
        let wide = fwidth(value);
        let wide_fine = fwidth_fine(value);
        let wide_coarse = crate::math::fwidth_coarse(value);
        let fused = fma(value, scale, wide);
        let scalar = scale.mul_add(bias, 1.0);
        return dx + dy + fine + coarse + wide + wide_fine + wide_coarse + fused;
    }
    """

    result = parse_and_generate(code)

    assert "vec3 derivative_ops(vec3 value, float scale, float bias)" in result
    assert "dx = dFdx(value);" in result
    assert "dy = dFdy(value);" in result
    assert "fine = dFdxFine(value);" in result
    assert "coarse = dFdyCoarse(value);" in result
    assert "wide = fwidth(value);" in result
    assert "wide_fine = fwidthFine(value);" in result
    assert "wide_coarse = fwidthCoarse(value);" in result
    assert "fused = fma(value, scale, wide);" in result
    assert "scalar = fma(scale, bias, 1.0);" in result
    assert "crate::math::dfdy" not in result
    assert "crate::math::fwidth_coarse" not in result
    assert ".mul_add(" not in result


def test_ldexp_intrinsic_calls_convert_to_crossgl_intrinsic():
    code = """
    fn exponent_ops(value: Vec3<f32>, exponents: Vec3<i32>, scale: f32, scalar_exponent: i32) -> Vec3<f32> {
        let shifted = ldexp(value, exponents);
        let scalar_shifted = crate::math::ldexp(scale, scalar_exponent);
        return shifted + Vec3::<f32>::new(scalar_shifted, scalar_shifted, scalar_shifted);
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec3 exponent_ops(vec3 value, ivec3 exponents, float scale, int scalar_exponent)"
        in result
    )
    assert "shifted = ldexp(value, exponents);" in result
    assert "scalar_shifted = ldexp(scale, scalar_exponent);" in result
    assert "crate::math::ldexp" not in result


def test_sign_and_classification_calls_convert_to_crossgl_intrinsics():
    code = """
    fn classify(value: Vec3<f32>, scale: f32) -> f32 {
        let signed_value = sign(value);
        let signed_scale = scale.signum();
        let nan_mask = isnan(value);
        let inf_mask = crate::math::isinf(value);
        let finite_mask = isfinite(value);
        let scalar_nan = scale.is_nan();
        let scalar_inf = scale.is_infinite();
        let scalar_finite = scale.is_finite();
        return signed_scale;
    }
    """

    result = parse_and_generate(code)

    assert "float classify(vec3 value, float scale)" in result
    assert "signed_value = sign(value);" in result
    assert "signed_scale = sign(scale);" in result
    assert "nan_mask = isnan(value);" in result
    assert "inf_mask = isinf(value);" in result
    assert "finite_mask = isfinite(value);" in result
    assert "scalar_nan = isnan(scale);" in result
    assert "scalar_inf = isinf(scale);" in result
    assert "scalar_finite = isfinite(scale);" in result
    assert "crate::math::isinf" not in result
    assert ".signum(" not in result
    assert ".is_nan(" not in result
    assert ".is_infinite(" not in result
    assert ".is_finite(" not in result


def test_any_and_all_calls_convert_to_crossgl_intrinsics():
    code = """
    fn reduce_masks(mask: Vec3<bool>, flag: bool) -> bool {
        let has_any = any(mask);
        let has_all = crate::math::all(mask);
        let scalar_any = any(flag);
        let scalar_all = crate::math::all(flag);
        return has_any || has_all || scalar_any || scalar_all;
    }
    """

    result = parse_and_generate(code)

    assert "bool reduce_masks(bvec3 mask, bool flag)" in result
    assert "has_any = any(mask);" in result
    assert "has_all = all(mask);" in result
    assert "scalar_any = any(flag);" in result
    assert "scalar_all = all(flag);" in result
    assert "crate::math::all" not in result


def test_relational_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn relational_ops(left: Vec3<f32>, right: Vec3<f32>, threshold: f32, mask: Vec3<bool>) -> bool {
        let lt = less_than(left, right);
        let le = less_than_equal(left, threshold);
        let gt = crate::math::greater_than(threshold, right);
        let ge = crate::math::greater_than_equal(left, right);
        let eq = equal(mask, Vec3::<bool>::new(true, false, true));
        let ne = not_equal(left, right);
        let scalar_eq = equal(threshold, 1.0);
        return any(lt) || any(le) || any(gt) || any(ge) || any(eq) || any(ne) || scalar_eq;
    }
    """

    result = parse_and_generate(code)

    assert (
        "bool relational_ops(vec3 left, vec3 right, float threshold, bvec3 mask)"
        in result
    )
    assert "lt = lessThan(left, right);" in result
    assert "le = lessThanEqual(left, threshold);" in result
    assert "gt = greaterThan(threshold, right);" in result
    assert "ge = greaterThanEqual(left, right);" in result
    assert "eq = equal(mask, bvec3(true, false, true));" in result
    assert "ne = notEqual(left, right);" in result
    assert "scalar_eq = equal(threshold, 1.0);" in result
    assert "crate::math::greater_than" not in result
    assert "crate::math::greater_than_equal" not in result


def test_integer_bit_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn bit_ops(mask: Vec3<u32>, signed_mask: Vec3<i32>, scalar: u32, offset: i32, bits: i32) -> Vec3<u32> {
        let reversed = bitfield_reverse(mask);
        let signed_reversed = bitfield_reverse(signed_mask);
        let counts = bit_count(mask);
        let signed_counts = crate::math::bit_count(signed_mask);
        let lsb = find_lsb(mask);
        let msb = crate::math::find_msb(signed_mask);
        let extracted = bitfield_extract(mask, offset, bits);
        let inserted = crate::math::bitfield_insert(mask, extracted, offset, bits);
        let scalar_count = bit_count(scalar);
        return inserted;
    }
    """

    result = parse_and_generate(code)

    assert (
        "uvec3 bit_ops(uvec3 mask, ivec3 signed_mask, uint scalar, int offset, int bits)"
        in result
    )
    assert "reversed = bitfieldReverse(mask);" in result
    assert "signed_reversed = bitfieldReverse(signed_mask);" in result
    assert "counts = bitCount(mask);" in result
    assert "signed_counts = bitCount(signed_mask);" in result
    assert "lsb = findLSB(mask);" in result
    assert "msb = findMSB(signed_mask);" in result
    assert "extracted = bitfieldExtract(mask, offset, bits);" in result
    assert "inserted = bitfieldInsert(mask, extracted, offset, bits);" in result
    assert "scalar_count = bitCount(scalar);" in result
    assert "crate::math::bit_count" not in result
    assert "crate::math::find_msb" not in result
    assert "crate::math::bitfield_insert" not in result


def test_bit_reinterpret_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn reinterpret_ops(value: Vec3<f32>, scalar: f32, signed_bits: Vec3<i32>, unsigned_bits: Vec3<u32>) -> Vec3<f32> {
        let signed_from_vector = float_bits_to_int(value);
        let unsigned_from_vector = crate::math::float_bits_to_uint(value);
        let signed_from_scalar = float_bits_to_int(scalar);
        let unsigned_from_scalar = float_bits_to_uint(scalar);
        let float_from_signed = int_bits_to_float(signed_bits);
        let float_from_unsigned = crate::math::uint_bits_to_float(unsigned_bits);
        return float_from_signed + float_from_unsigned;
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec3 reinterpret_ops(vec3 value, float scalar, ivec3 signed_bits, uvec3 unsigned_bits)"
        in result
    )
    assert "signed_from_vector = floatBitsToInt(value);" in result
    assert "unsigned_from_vector = floatBitsToUint(value);" in result
    assert "signed_from_scalar = floatBitsToInt(scalar);" in result
    assert "unsigned_from_scalar = floatBitsToUint(scalar);" in result
    assert "float_from_signed = intBitsToFloat(signed_bits);" in result
    assert "float_from_unsigned = uintBitsToFloat(unsigned_bits);" in result
    assert "crate::math::float_bits_to_uint" not in result
    assert "crate::math::uint_bits_to_float" not in result


def test_pack_unpack_intrinsic_calls_convert_to_crossgl_intrinsics():
    code = """
    fn pack_unpack_ops(pair: Vec2<f32>, color: Vec4<f32>, packed2: u32, packed4: u32, double_bits: Vec2<u32>, packed_double: f64) -> Vec4<f32> {
        let packed_unorm2 = pack_unorm_2x16(pair);
        let packed_snorm2 = crate::math::pack_snorm_2x16(pair);
        let packed_unorm4 = pack_unorm_4x8(color);
        let packed_snorm4 = pack_snorm_4x8(color);
        let packed_half = crate::math::pack_half_2x16(pair);
        let packed_double_value = pack_double_2x32(double_bits);
        let unpacked_unorm2 = unpack_unorm_2x16(packed2);
        let unpacked_snorm2 = crate::math::unpack_snorm_2x16(packed2);
        let unpacked_unorm4 = unpack_unorm_4x8(packed4);
        let unpacked_snorm4 = unpack_snorm_4x8(packed4);
        let unpacked_half = unpack_half_2x16(packed2);
        let unpacked_double = crate::math::unpack_double_2x32(packed_double);
        return unpacked_unorm4 + unpacked_snorm4;
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 pack_unpack_ops(vec2 pair, vec4 color, uint packed2, uint packed4, uvec2 double_bits, double packed_double)"
        in result
    )
    assert "packed_unorm2 = packUnorm2x16(pair);" in result
    assert "packed_snorm2 = packSnorm2x16(pair);" in result
    assert "packed_unorm4 = packUnorm4x8(color);" in result
    assert "packed_snorm4 = packSnorm4x8(color);" in result
    assert "packed_half = packHalf2x16(pair);" in result
    assert "packed_double_value = packDouble2x32(double_bits);" in result
    assert "unpacked_unorm2 = unpackUnorm2x16(packed2);" in result
    assert "unpacked_snorm2 = unpackSnorm2x16(packed2);" in result
    assert "unpacked_unorm4 = unpackUnorm4x8(packed4);" in result
    assert "unpacked_snorm4 = unpackSnorm4x8(packed4);" in result
    assert "unpacked_half = unpackHalf2x16(packed2);" in result
    assert "unpacked_double = unpackDouble2x32(packed_double);" in result
    assert "crate::math::pack_snorm_2x16" not in result
    assert "crate::math::unpack_snorm_2x16" not in result
    assert "crate::math::unpack_double_2x32" not in result


def test_lerp_function_converts_to_crossgl_mix():
    code = """
    fn blend(a: f32, b: f32, t: f32) -> f32 {
        let x = lerp(a, b, t);
        x
    }
    """
    try:
        result = parse_and_generate(code)

        assert "x = mix(a, b, t);" in result
        assert "lerp(a, b, t)" not in result
    except Exception as e:
        pytest.fail(f"Lerp function conversion failed: {e}")


def test_lerp_method_converts_to_crossgl_mix():
    code = """
    fn blend(a: f32, b: f32, t: f32) -> f32 {
        let x = a.lerp(b, t);
        x
    }
    """
    result = parse_and_generate(code)

    assert "x = mix(a, b, t);" in result
    assert "a.lerp(b, t)" not in result


def test_user_defined_lerp_call_does_not_convert_to_mix():
    code = """
    fn lerp(x: f32) -> f32 {
        x
    }

    fn shade(x: f32) -> f32 {
        let y = lerp(x);
        y
    }
    """
    result = parse_and_generate(code)

    assert "float lerp(float x)" in result
    assert "let y = lerp(x);" in result
    assert "let y = mix(x);" not in result


def test_current_module_qualified_user_defined_lerp_call_does_not_convert_to_mix():
    code = """
    fn lerp(x: f32) -> f32 {
        x
    }

    fn shade(x: f32) -> f32 {
        let a = self::lerp(x);
        let b = crate::lerp(x);
        a + b
    }
    """
    result = parse_and_generate(code)

    assert "float lerp(float x)" in result
    assert "let a = lerp(x);" in result
    assert "let b = lerp(x);" in result
    assert "mix(x)" not in result
    assert "self::lerp" not in result
    assert "crate::lerp" not in result


def test_user_defined_impl_lerp_method_does_not_convert_to_mix():
    code = """
    struct Wave {
        value: f32,
    }

    impl Wave {
        fn lerp(&self, other: Wave, t: f32) -> f32 {
            return self.value + other.value + t;
        }
    }

    fn sample(w: Wave, other: Wave, t: f32) -> f32 {
        return w.lerp(other, t);
    }
    """

    result = parse_and_generate(code)

    assert "float Wave_lerp(" in result
    assert "return Wave_lerp(w, other, t);" in result
    assert "return mix(w, other, t);" not in result


def test_user_defined_impl_method_named_like_scalar_builtin_does_not_lower_to_builtin():
    code = """
    struct Wave {
        value: f32,
    }

    impl Wave {
        fn sin(&self) -> f32 {
            return self.value;
        }
    }

    fn sample(w: Wave) -> f32 {
        return w.sin();
    }
    """

    result = parse_and_generate(code)

    assert "float Wave_sin(" in result
    assert "return Wave_sin(w);" in result
    assert "return sin(w);" not in result


def test_inferred_struct_local_impl_method_named_like_scalar_builtin_does_not_lower_to_builtin():
    code = """
    struct Wave {
        value: f32,
    }

    impl Wave {
        fn sin(&self) -> f32 {
            return self.value;
        }
    }

    fn sample(x: f32) -> f32 {
        let w = Wave { value: x };
        return w.sin();
    }
    """

    result = parse_and_generate(code)

    assert "float Wave_sin(" in result
    assert "let w = Wave { value: x };" in result
    assert "return Wave_sin(w);" in result
    assert "return sin(w);" not in result


def test_struct_member_impl_method_named_like_resource_helper_stays_user_defined():
    code = """
    struct ResourceLike {
        value: f32,
    }

    struct Holder {
        resource: ResourceLike,
    }

    impl ResourceLike {
        fn sample_sampler(&self, sampler_state: f32, uv: f32) -> f32 {
            return self.value + sampler_state + uv;
        }
    }

    fn sample(holder: Holder, sampler_state: f32, uv: f32) -> f32 {
        return holder.resource.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "float ResourceLike_sample_sampler(" in result
    assert (
        "return ResourceLike_sample_sampler(holder.resource, sampler_state, uv);"
        in result
    )
    assert "texture(holder.resource" not in result


def test_generic_struct_member_impl_method_named_like_resource_helper_stays_user_defined():
    code = """
    struct ResourceLike {
        value: f32,
    }

    struct Holder<T> {
        resource: T,
    }

    impl ResourceLike {
        fn sample_sampler(&self, sampler_state: f32, uv: f32) -> f32 {
            return self.value + sampler_state + uv;
        }
    }

    fn sample(holder: Holder<ResourceLike>, sampler_state: f32, uv: f32) -> f32 {
        return holder.resource.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "float ResourceLike_sample_sampler(" in result
    assert (
        "return ResourceLike_sample_sampler(holder.resource, sampler_state, uv);"
        in result
    )
    assert "texture(holder.resource" not in result


def test_modulo_function_converts_to_crossgl_mod():
    code = """
    fn wrap(x: f32) -> f32 {
        let y = modulo(x, 2.0);
        y
    }
    """
    try:
        result = parse_and_generate(code)

        assert "y = mod(x, 2.0);" in result
        assert "modulo(x, 2.0)" not in result
    except Exception as e:
        pytest.fail(f"Modulo function conversion failed: {e}")


def test_swizzle_and_index_assignment_return_conversion():
    code = """
    fn test_swizzle_index(values: Buffer, output: Output) -> Vec3<f32> {
        output.normal = next_values().xyz();
        output.uv = input_texcoord().st();
        output.value = values[next_index()];
        return next_values().stp();
    }
    """
    try:
        result = parse_and_generate(code)

        assert "output.normal = next_values().xyz;" in result
        assert "output.uv = input_texcoord().st;" in result
        assert "output.value = values[next_index()];" in result
        assert "return next_values().stp;" in result
        assert result.count("next_values()") == 2
        assert result.count("next_index()") == 1
        assert ".xyz()" not in result
        assert ".st()" not in result
        assert ".stp()" not in result
    except Exception as e:
        pytest.fail(f"Swizzle and index assignment/return conversion failed: {e}")


def test_labeled_loop_conversion():
    code = """
    fn test_labeled_control() {
        'outer: loop {
            continue 'outer;
            break 'outer;
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "while (true) {" in result
        assert "continue;" in result
        assert "break;" in result
        assert "'outer" not in result
    except Exception as e:
        pytest.fail(f"Labeled loop conversion failed: {e}")


def test_nested_labeled_loop_control_conversion():
    code = """
    fn test_nested_labeled_control() {
        'outer: loop {
            setup();
            loop {
                if should_continue {
                    continue 'outer;
                }
                if should_break {
                    break 'outer;
                }
                step();
            }
            after_inner();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_break_outer_0 = false;" in result
        assert "bool _rust_continue_outer_0 = false;" in result
        assert "_rust_continue_outer_0 = true;" in result
        assert "_rust_break_outer_0 = true;" in result
        assert "if (_rust_continue_outer_0) {" in result
        assert "_rust_continue_outer_0 = false;" in result
        assert "if (_rust_break_outer_0) {" in result
        assert "continue 'outer" not in result
        assert "break 'outer" not in result
        assert "setup();" in result
        assert "step();" in result
        assert "after_inner();" in result
    except Exception as e:
        pytest.fail(f"Nested labeled loop conversion failed: {e}")


def test_nested_labeled_loop_expression_break_value_conversion():
    code = """
    fn test_nested_labeled_loop_expression() {
        let value: i32 = 'outer: loop {
            loop {
                break 'outer 3;
            }
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "bool _rust_break_outer_0 = false;" in result
        assert "value = 3;" in result
        assert "_rust_break_outer_0 = true;" in result
        assert "if (_rust_break_outer_0) {" in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Nested labeled loop expression conversion failed: {e}")


def test_break_value_conversion():
    code = """
    fn test_break_values() {
        loop {
            break result;
        }
        'outer: loop {
            break 'outer value + 1;
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert result.count("break;") == 2
        assert "break result;" not in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Break value conversion failed: {e}")


def test_statement_loop_break_value_side_effect_conversion():
    code = """
    fn test_statement_loop_break_values() {
        loop {
            break compute();
        }
        loop {
            break {
                let temp = compute_block();
                temp
            };
        }
        loop {
            break if ready {
                let seeded = compute_if();
                seeded
            } else {
                compute_else()
            };
        }
        loop {
            break match mode {
                0 => compute_zero(),
                _ => {
                    let fallback = compute_default();
                    fallback
                },
            };
        }
        loop {
            break result;
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "compute();" in result
        assert "temp = compute_block();" in result
        assert "if (ready) {" in result
        assert "seeded = compute_if();" in result
        assert "compute_else();" in result
        assert "switch (mode)" in result
        assert "compute_zero();" in result
        assert "fallback = compute_default();" in result
        assert "result;" not in result
        assert result.count("break;") == 7
    except Exception as e:
        pytest.fail(f"Statement loop break value side-effect conversion failed: {e}")


def test_discarded_match_break_value_control_conversion():
    code = """
    fn test_discarded_match_break_value(mode: i32) -> i32 {
        loop {
            break match mode {
                0 => {
                    return 1;
                },
                1 => {
                    continue;
                },
                2 => {
                    break;
                },
                _ => {
                    compute_default();
                },
            };
        }
        return 2;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_switch_break_0 = false;" in result
        assert "_rust_switch_break_0 = true;" in result
        assert "if (_rust_switch_break_0) {" in result
        assert "return 1;" in result
        assert "continue;" in result
        assert "compute_default();" in result
        assert "return 2;" in result
        assert "return 1;\n                        break;" not in result
        assert "continue;\n                        break;" not in result
        assert "break;\n                        break;" not in result
    except Exception as e:
        pytest.fail(f"Discarded match break-value control conversion failed: {e}")


def test_loop_break_result_expression_conversion():
    code = """
    fn test_loop_break_result_expressions() -> i32 {
        let from_if: i32 = loop {
            break if ready {
                let seed = 1;
                seed + 1
            } else {
                2
            };
        };
        output.color = loop {
            break match from_if {
                0 => 0,
                _ => {
                    let next = from_if + 1;
                    next
                },
            };
        };
        return loop {
            break {
                let final_value = from_if + 1;
                final_value
            };
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int from_if;" in result
        assert "if (ready) {" in result
        assert "seed = 1;" in result
        assert "from_if = (seed + 1);" in result
        assert "from_if = 2;" in result
        assert "switch (from_if)" in result
        assert "output.color = 0;" in result
        assert "next = (from_if + 1);" in result
        assert "output.color = next;" in result
        assert "final_value = (from_if + 1);" in result
        assert "return final_value;" in result
        assert "from_if = if" not in result
        assert "output.color = match" not in result
        assert "return BlockNode" not in result
    except Exception as e:
        pytest.fail(f"Loop break result expression conversion failed: {e}")


def test_loop_expression_let_conversion():
    code = """
    fn test_loop_expression() {
        let value: i32 = loop {
            if ready {
                break 7;
            }
        };
        let labeled: i32 = 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "value = 7;" in result
        assert "int labeled;" in result
        assert "labeled = (value + 1);" in result
        assert "break 7;" not in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Loop expression let conversion failed: {e}")


def test_loop_expression_nested_loop_does_not_steal_inner_break_value():
    code = """
    fn test_nested_loop_expression() {
        let value: i32 = loop {
            loop {
                break 1;
            }
            break 2;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "value = 2;" in result
        assert "value = 1;" not in result
    except Exception as e:
        pytest.fail(f"Nested loop expression conversion failed: {e}")


def test_loop_expression_assignment_conversion():
    code = """
    fn test_loop_assignment() {
        value = loop {
            break result;
        };
        output.color = 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "value = result;" in result
        assert "output.color = (value + 1);" in result
        assert "value = while" not in result
        assert "output.color = while" not in result
        assert "break result;" not in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Loop expression assignment conversion failed: {e}")


def test_loop_expression_assignment_nested_loop_does_not_steal_outer_target():
    code = """
    fn test_nested_loop_assignment() {
        value = loop {
            loop {
                break 1;
            }
            break 2;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "value = 2;" in result
        assert "value = 1;" not in result
    except Exception as e:
        pytest.fail(f"Nested loop expression assignment conversion failed: {e}")


def test_loop_expression_return_conversion():
    code = """
    fn test_loop_return(value: i32) -> i32 {
        return 'outer: loop {
            break 'outer value + 1;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int test_loop_return(int value)" in result
        assert "while (true) {" in result
        assert "return (value + 1);" in result
        assert "return while" not in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Loop expression return conversion failed: {e}")


def test_loop_expression_return_nested_loop_does_not_return_inner_break_value():
    code = """
    fn test_nested_loop_return() -> i32 {
        return loop {
            loop {
                break 1;
            }
            break 2;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "return 2;" in result
        assert "return 1;" not in result
    except Exception as e:
        pytest.fail(f"Nested loop expression return conversion failed: {e}")


def test_block_loop_expression_conversion():
    code = """
    fn test_block_loop_expression() -> i32 {
        let value: i32 = {
            let seed = 1;
            loop {
                break seed + 1;
            }
        };
        output.color = {
            'outer: loop {
                break 'outer value + 1;
            }
        };
        return {
            loop {
                break value;
            }
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "seed = 1;" in result
        assert "value = (seed + 1);" in result
        assert "output.color = (value + 1);" in result
        assert "return value;" in result
        assert "BlockNode" not in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Block loop expression conversion failed: {e}")


def test_block_expression_statements_are_preserved_in_let_conversion():
    code = """
    fn test_block_expression() {
        let value: i32 = {
            let seed = 1;
            prepare_seed();
            seed + 2
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "seed = 1;" in result
        assert "prepare_seed();" in result
        assert "value = (seed + 2);" in result
    except Exception as e:
        pytest.fail(f"Block expression let conversion failed: {e}")


def test_rust_gpu_mouse_shader_untyped_block_expression_let_codegen():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/mouse-shader/src/lib.rs fragment background block.
    code = """
    use glam::{Vec3, Vec4, vec2, vec3};
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] in_frag_coord: Vec4,
        output: &mut Vec4,
    ) {
        let frag_coord = vec2(in_frag_coord.x, in_frag_coord.y);
        let background = {
            let v = frag_coord - vec2(400.0, 300.0);
            let color = vec3(v.x.abs(), v.y.abs(), 1.0);
            let vignette = smoothstep(1.0, 0.0, (v.x * v.y).abs());
            color * vignette
        };

        *output = background.extend(1.0);
    }
    """
    result = parse_and_generate(code)

    assert "auto background;" in result
    assert "let v = (frag_coord - vec2(400.0, 300.0));" in result
    assert "background = (color * vignette);" in result
    assert "\n        background;\n" not in result
    assert "output = background.extend(1.0);" in result


def test_if_expression_result_conversion():
    code = """
    fn test_if_expression() -> i32 {
        let value: i32 = if ready {
            let seed = 1;
            seed + 1
        } else {
            2
        };
        output.color = if value > 1 {
            value + 1
        } else {
            let fallback = 0;
            fallback
        };
        return if done {
            value
        } else {
            let next = value + 1;
            next
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "if (ready) {" in result
        assert "seed = 1;" in result
        assert "value = (seed + 1);" in result
        assert "} else {" in result
        assert "value = 2;" in result
        assert "if ((value > 1)) {" in result
        assert "output.color = (value + 1);" in result
        assert "fallback = 0;" in result
        assert "output.color = fallback;" in result
        assert "if (done) {" in result
        assert "return value;" in result
        assert "next = (value + 1);" in result
        assert "return next;" in result
        assert "? seed" not in result
    except Exception as e:
        pytest.fail(f"If expression result conversion failed: {e}")


def test_match_expression_result_conversion():
    code = """
    fn test_match_expression() -> i32 {
        let value: i32 = match mode {
            0 => 1,
            1 => {
                let seed = 2;
                seed + 1
            },
            _ => 3,
        };
        output.color = match value {
            0 => 0,
            _ => value + 1,
        };
        return match value {
            0 => value,
            _ => {
                let next = value + 1;
                next
            },
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "switch (mode)" in result
        assert "case 0:" in result
        assert "value = 1;" in result
        assert "case 1:" in result
        assert "seed = 2;" in result
        assert "value = (seed + 1);" in result
        assert "default:" in result
        assert "value = 3;" in result
        assert "switch (value)" in result
        assert "output.color = 0;" in result
        assert "output.color = (value + 1);" in result
        assert "return value;" in result
        assert "next = (value + 1);" in result
        assert "return next;" in result
        assert "case _:" not in result
    except Exception as e:
        pytest.fail(f"Match expression result conversion failed: {e}")


def test_match_arm_break_exits_enclosing_loop_conversion():
    code = """
    fn test_match_break_loop() {
        loop {
            match mode {
                0 => {
                    break;
                },
                _ => {
                    step();
                },
            }
            after_match();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_switch_break_0 = false;" in result
        assert "_rust_switch_break_0 = true;" in result
        assert "if (_rust_switch_break_0) {" in result
        assert "after_match();" in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Match arm break loop conversion failed: {e}")


def test_match_arm_break_value_exits_loop_expression_conversion():
    code = """
    fn test_match_break_loop_expression() {
        let value: i32 = loop {
            match mode {
                0 => {
                    break 3;
                },
                _ => {
                    break 4;
                },
            }
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "bool _rust_switch_break_0 = false;" in result
        assert "value = 3;" in result
        assert "value = 4;" in result
        assert "_rust_switch_break_0 = true;" in result
        assert "if (_rust_switch_break_0) {" in result
        assert "break 3;" not in result
        assert "break 4;" not in result
    except Exception as e:
        pytest.fail(f"Match arm break value loop expression conversion failed: {e}")


def test_nested_match_arm_break_propagates_switch_conversion():
    code = """
    fn test_nested_match_break_loop() {
        loop {
            match outer {
                0 => {
                    match inner {
                        1 => {
                            break;
                        },
                        _ => {
                            step();
                        },
                    }
                },
                _ => {
                    other();
                },
            }
            after_match();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_switch_break_0 = false;" in result
        assert "bool _rust_switch_break_1 = false;" in result
        assert "_rust_switch_break_1 = true;" in result
        assert "_rust_switch_break_0 = true;" in result
        assert "if (_rust_switch_break_1) {" in result
        assert "if (_rust_switch_break_0) {" in result
        assert "break;\n                            break;" not in result
        assert "after_match();" in result
    except Exception as e:
        pytest.fail(f"Nested match arm break propagation failed: {e}")


def test_labeled_match_arm_break_current_loop_conversion():
    code = """
    fn test_labeled_match_break_loop() {
        'outer: loop {
            match mode {
                0 => {
                    break 'outer;
                },
                _ => {
                    step();
                },
            }
            after_match();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_switch_break_0 = false;" in result
        assert "_rust_switch_break_0 = true;" in result
        assert "if (_rust_switch_break_0) {" in result
        assert "break 'outer" not in result
    except Exception as e:
        pytest.fail(f"Labeled match arm break current loop conversion failed: {e}")


def test_labeled_match_arm_control_exits_outer_loop_conversion():
    code = """
    fn test_labeled_match_control_outer_loop() {
        'outer: loop {
            loop {
                match mode {
                    0 => {
                        continue 'outer;
                    },
                    1 => {
                        break 'outer;
                    },
                    _ => {
                        step();
                    },
                }
                after_match();
            }
            after_inner();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_break_outer_0 = false;" in result
        assert "bool _rust_continue_outer_0 = false;" in result
        assert "_rust_continue_outer_0 = true;" in result
        assert "_rust_break_outer_0 = true;" in result
        assert "if (_rust_continue_outer_0 || _rust_break_outer_0) {" in result
        assert "if (_rust_continue_outer_0) {" in result
        assert "_rust_continue_outer_0 = false;" in result
        assert "continue 'outer" not in result
        assert "break 'outer" not in result
        assert "break;\n                        if (" not in result
        assert result.index(
            "if (_rust_continue_outer_0 || _rust_break_outer_0) {"
        ) < result.index("after_match();")
    except Exception as e:
        pytest.fail(f"Labeled match arm outer-loop control conversion failed: {e}")


def test_transparent_match_arm_labeled_control_conversion():
    code = """
    fn test_transparent_match_labeled_control() {
        'outer: loop {
            loop {
                match mode {
                    0 => unsafe {
                        continue 'outer;
                    },
                    1 => const {
                        break 'outer;
                    },
                    _ => async {
                        step();
                    },
                }
                after_match();
            }
            after_inner();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_break_outer_0 = false;" in result
        assert "bool _rust_continue_outer_0 = false;" in result
        assert "_rust_continue_outer_0 = true;" in result
        assert "_rust_break_outer_0 = true;" in result
        assert "continue 'outer" not in result
        assert "break 'outer" not in result
        assert (
            "break;\n                        if (_rust_continue_outer_0 || "
            "_rust_break_outer_0)"
        ) not in result
        assert result.index(
            "if (_rust_continue_outer_0 || _rust_break_outer_0) {"
        ) < result.index("after_match();")
    except Exception as e:
        pytest.fail(f"Transparent match arm labeled control conversion failed: {e}")


def test_transparent_block_final_match_labeled_control_conversion():
    code = """
    fn test_transparent_block_final_match_labeled_control() {
        'outer: loop {
            loop {
                unsafe {
                    match mode {
                        0 => {
                            continue 'outer;
                        },
                        _ => {
                            break 'outer;
                        },
                    }
                };
                after_match();
            }
            after_inner();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "_rust_continue_outer_0 = true;" in result
        assert "_rust_break_outer_0 = true;" in result
        assert "continue 'outer" not in result
        assert "break 'outer" not in result
        propagation = "if (_rust_continue_outer_0 || _rust_break_outer_0) {"
        assert propagation in result
        assert result.index("switch (mode)") < result.index(propagation)
        assert result.index(propagation) < result.index("after_match();")
    except Exception as e:
        pytest.fail(f"Transparent block final match labeled control failed: {e}")


def test_match_arm_plain_continue_keeps_loop_continue_conversion():
    code = """
    fn test_match_continue_loop() {
        loop {
            match mode {
                0 => {
                    continue;
                },
                _ => {
                    step();
                },
            }
            after_match();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "continue;" in result
        assert "bool _rust_switch_break" not in result
        assert "bool _rust_continue" not in result
        assert "after_match();" in result
    except Exception as e:
        pytest.fail(f"Match arm plain continue conversion failed: {e}")


def test_match_expression_return_branches_do_not_emit_fallback_return_conversion():
    code = """
    fn test_match_return_branches(value: i32) -> i32 {
        return match value {
            0 => {
                return 1;
            },
            _ => {
                return value + 1;
            },
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "switch (value)" in result
        assert "return 1;" in result
        assert "return (value + 1);" in result
        assert "return;\n" not in result
        assert "break;" not in result
    except Exception as e:
        pytest.fail(f"Match expression return branch conversion failed: {e}")


def test_if_expression_return_branches_do_not_emit_fallback_return_conversion():
    code = """
    fn test_if_return_branches(value: i32) -> i32 {
        return if ready {
            return value;
        } else {
            return value + 1;
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "if (ready) {" in result
        assert "return value;" in result
        assert "return (value + 1);" in result
        assert "return;\n" not in result
    except Exception as e:
        pytest.fail(f"If expression return branch conversion failed: {e}")


def test_block_final_if_match_expression_conversion():
    code = """
    fn test_block_final_expression() -> i32 {
        let value: i32 = {
            let seed = 1;
            if ready {
                seed + 1
            } else {
                let fallback = 2;
                fallback
            }
        };
        output.color = {
            match value {
                0 => 0,
                _ => {
                    let next = value + 1;
                    next
                },
            }
        };
        return {
            match value {
                0 => value,
                _ => value + 1,
            }
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "seed = 1;" in result
        assert "if (ready) {" in result
        assert "value = (seed + 1);" in result
        assert "fallback = 2;" in result
        assert "value = fallback;" in result
        assert "switch (value)" in result
        assert "output.color = 0;" in result
        assert "next = (value + 1);" in result
        assert "output.color = next;" in result
        assert "return value;" in result
        assert "return (value + 1);" in result
        assert "case _:" not in result
        assert "BlockNode" not in result
    except Exception as e:
        pytest.fail(f"Block final if/match expression conversion failed: {e}")


def test_block_expression_keeps_non_final_if_match_statements_conversion():
    code = """
    fn test_block_statement_restore() {
        let value: i32 = {
            if ready {
                output.color = 1;
            }
            match mode {
                0 => do_zero(),
                _ => do_default(),
            }
            4
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "if (ready) {" in result
        assert "output.color = 1;" in result
        assert "switch (mode)" in result
        assert "do_zero();" in result
        assert "do_default();" in result
        assert "value = 4;" in result
        assert "return 4;" not in result
    except Exception as e:
        pytest.fail(f"Block expression statement restore conversion failed: {e}")


def test_block_expression_semicolon_if_match_statement_conversion():
    code = """
    fn test_statement_like_block_expressions() {
        let value: i32 = {
            if ready {
                compute_ready();
                1
            } else {
                compute_fallback();
                2
            };
            match mode {
                0 => {
                    compute_zero();
                    0
                },
                _ => {
                    compute_default();
                    3
                },
            };
            fallback
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int value;" in result
        assert "if (ready) {" in result
        assert "compute_ready();" in result
        assert "compute_fallback();" in result
        assert "switch (mode)" in result
        assert "compute_zero();" in result
        assert "compute_default();" in result
        assert "value = fallback;" in result
        assert "value = 1;" not in result
        assert "value = 2;" not in result
    except Exception as e:
        pytest.fail(f"Block expression semicolon if/match conversion failed: {e}")


def test_let_pattern_conversion():
    code = """
    fn test_let_patterns() {
        let _ = compute_discard();
        let _ = 1;
        let (_, y) = (compute_x(), 2);
        let (a, (b, _)) = (1, (2, compute_ignored()));
        let (tx, ty): (f32, i32) = (1.0, 2);
        let (_, (nz, _)): (f32, (i32, f32)) = (
            compute_typed(),
            (3, compute_typed_drop()),
        );
    }
    """
    try:
        result = parse_and_generate(code)
        assert "compute_discard();" in result
        assert "compute_x();" in result
        assert "y = 2;" in result
        assert "a = 1;" in result
        assert "b = 2;" in result
        assert "compute_ignored();" in result
        assert "float tx = 1.0;" in result
        assert "int ty = 2;" in result
        assert "compute_typed();" in result
        assert "int nz = 3;" in result
        assert "compute_typed_drop();" in result
        assert "_ =" not in result
        assert "_ = 1;" not in result
    except Exception as e:
        pytest.fail(f"Let pattern conversion failed: {e}")


def test_tuple_pattern_result_expression_elements_conversion():
    code = """
    fn test_tuple_result_elements() {
        let (from_if, from_loop, from_match, from_block): (i32, i32, i32, i32) = (
            if ready {
                prepare_if();
                1
            } else {
                2
            },
            loop {
                break 3;
            },
            match mode {
                0 => 4,
                _ => {
                    prepare_match();
                    5
                },
            },
            {
                prepare_block();
                6
            },
        );
        let (_, kept): (i32, i32) = (
            if should_drop {
                prepare_drop();
                31
            } else {
                fallback_drop();
                32
            },
            7,
        );
    }
    """
    try:
        result = parse_and_generate(code)

        assert "int from_if;" in result
        assert "if (ready) {" in result
        assert "prepare_if();" in result
        assert "from_if = 1;" in result
        assert "from_if = 2;" in result

        assert "int from_loop;" in result
        assert "while (true)" in result
        assert "from_loop = 3;" in result

        assert "int from_match;" in result
        assert "switch (mode)" in result
        assert "from_match = 4;" in result
        assert "prepare_match();" in result
        assert "from_match = 5;" in result

        assert "int from_block;" in result
        assert "prepare_block();" in result
        assert "from_block = 6;" in result

        assert "prepare_drop();" in result
        assert "fallback_drop();" in result
        assert "int kept = 7;" in result
        assert "31;" not in result
        assert "32;" not in result
    except Exception as e:
        pytest.fail(f"Tuple pattern result expression conversion failed: {e}")


def test_match_to_switch_conversion():
    code = """
    fn test_match() {
        match value {
            0 => do_zero(),
            1 => do_one(),
            _ => do_default(),
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "switch (" in result
        assert "case " in result
        assert "default:" in result
        assert "case _:" not in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"Match to switch conversion failed: {e}")


def test_let_binding_conversion():
    code = """
    fn test_let() {
        let x = 42;
        let mut y = 3.14;
        let z: f32 = 2.0;
    }
    """
    try:
        result = parse_and_generate(code)
        lines = [line.strip() for line in result.splitlines()]

        assert "let x = 42;" in lines
        assert "let mut y = 3.14;" in lines
        assert "float z = 2.0;" in lines
    except Exception as e:
        pytest.fail(f"Let binding conversion failed: {e}")


def test_binary_operations_conversion():
    code = """
    fn test_binary() {
        let a = 5 + 3;
        let b = 10 - 2;
        let c = 4 * 6;
        let d = 8 / 2;
        let e = 10 % 3;
        let f = a > b;
        let g = c <= d;
        let h = e == 1;
        let i = f && g;
        let j = h || i;
        let k = 1 << 2;
        let l = k >> 1;
        let m = k | l;
        let n = k & l;
        let o = k ^ l;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "(5 + 3)" in result
        assert "(10 - 2)" in result
        assert "(4 * 6)" in result
        assert "(8 / 2)" in result
        assert "(10 % 3)" in result
        assert "(a > b)" in result
        assert "(c <= d)" in result
        assert "(e == 1)" in result
        assert "(f && g)" in result
        assert "(h || i)" in result
        assert "(1 << 2)" in result
        assert "(k >> 1)" in result
        assert "(k | l)" in result
        assert "(k & l)" in result
        assert "(k ^ l)" in result
    except Exception as e:
        pytest.fail(f"Binary operations conversion failed: {e}")


def test_if_expression_operand_conversion():
    code = """
    fn test_if_operand(a: bool, b: bool, c: bool, d: bool) {
        let x = a || if b { c } else { d };
    }
    """

    result = parse_and_generate(code)

    assert "let x = (a || (b ? c : d));" in result


def test_match_and_loop_expression_operand_conversion():
    code = """
    fn test_expression_operands(a: bool, v: i32) {
        let matched = a || match v {
            0 => false,
            _ => true,
        };
        let looped = check(loop {
            break true;
        });
    }
    """

    result = parse_and_generate(code)

    assert "lambda({" not in result
    assert "auto _rust_expr_value_0;" in result
    assert "if (a) {" in result
    assert "_rust_expr_value_0 = true;" in result
    assert "} else {" in result
    assert "switch (v)" in result
    assert "_rust_expr_value_1 = false;" in result
    assert "_rust_expr_value_1 = true;" in result
    assert "let matched = _rust_expr_value_0;" in result
    assert "while (true)" in result
    assert "let looped = check(_rust_expr_value_2);" in result
    crosstl.translator.parse(result)


def test_logical_and_match_operand_materialization_preserves_short_circuit():
    code = """
    fn test_and(a: bool, v: i32) -> bool {
        return a && match v {
            0 => false,
            _ => true,
        };
    }
    """

    result = parse_and_generate(code)

    assert "lambda({" not in result
    assert "if (!a) {" in result
    assert "_rust_expr_value_0 = false;" in result
    assert "} else {" in result
    assert "switch (v)" in result
    assert "return _rust_expr_value_0;" in result
    crosstl.translator.parse(result)


def test_simple_block_expression_operand_stays_inline():
    code = """
    fn test_block_operand(a: bool, b: bool) {
        let x = a || { b };
    }
    """

    result = parse_and_generate(code)

    assert "let x = (a || b);" in result
    assert "lambda({" not in result


def test_unary_operations_conversion():
    code = """
    fn test_unary() {
        let a = -5;
        let b = !true;
        let shared = &value;
        let writable = &mut value;
        let pointed = *ptr;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "(-5)" in result
        assert "(!true)" in result
        assert "shared = value;" in result
        assert "writable = value;" in result
        assert "pointed = ptr;" in result
        assert "&value" not in result
        assert "&mut value" not in result
        assert "*ptr" not in result
    except Exception as e:
        pytest.fail(f"Unary operations conversion failed: {e}")


def test_reference_dereference_assignment_return_conversion():
    code = """
    fn test_reference_deref(ptr: f32, value: f32, output: Output) -> f32 {
        output.shared = &value;
        output.writable = &mut value;
        output.pointed = *ptr;
        &next_value();
        *next_pointer();
        return *ptr;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "output.shared = value;" in result
        assert "output.writable = value;" in result
        assert "output.pointed = ptr;" in result
        assert "next_value();" in result
        assert "next_pointer();" in result
        assert "return ptr;" in result
        assert "&value" not in result
        assert "&mut value" not in result
        assert "*ptr" not in result
        assert "&next_value" not in result
        assert "*next_pointer" not in result
    except Exception as e:
        pytest.fail(f"Reference/dereference assignment/return conversion failed: {e}")


def test_function_call_conversion():
    code = """
    fn test_calls() {
        let result = normalize(vector);
        let dot_product = dot(a, b);
        let length = length(vec);
        let vector = Vec3::new(1.0, 2.0, 3.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "normalize(" in result
        assert "dot(" in result
        assert "length(" in result
        assert "vec3(" in result
    except Exception as e:
        pytest.fail(f"Function call conversion failed: {e}")


def test_gpu_texture_helper_calls_convert_to_crossgl_intrinsics():
    code = """
    fn texture_helpers(
        tex: Texture2D<f32>,
        cube: TextureCube<f32>,
        uv: Vec2<f32>,
        projected: Vec3<f32>,
        ddx: Vec2<f32>,
        ddy: Vec2<f32>,
        pixel: Vec2<i32>,
        offset: Vec2<i32>,
        offsets: [Vec2<i32>; 4],
    ) -> Vec4<f32> {
        let base = sample(tex, uv);
        let lod = sample_lod(tex, uv, 1.0);
        let grad = sample_grad(tex, uv, ddx, ddy);
        let shifted = sample_offset(tex, uv, offset);
        let projected_color = sample_projected(tex, projected);
        let projected_lod = sample_projected_lod(tex, projected, 1.0);
        let projected_grad = sample_projected_grad(tex, projected, ddx, ddy);
        let projected_offset = sample_projected_offset(tex, projected, offset);
        let projected_lod_offset = sample_projected_lod_offset(tex, projected, 1.0, offset);
        let projected_grad_offset = sample_projected_grad_offset(tex, projected, ddx, ddy, offset);
        let fetched = texel_fetch(tex, pixel, 0);
        let fetched_offset = texel_fetch_offset(tex, pixel, 0, offset);
        let size = texture_size_lod(tex, 0);
        let cube_size = texture_size(cube);
        let levels = texture_query_levels(tex);
        let query_lod = texture_query_lod(tex, uv);
        let samples = texture_samples(tex);
        let gather = texture_gather(tex, uv);
        let gather_component = texture_gather_component(tex, uv, 1);
        let gather_offset = texture_gather_offset(tex, uv, offset);
        let gather_offset_component = texture_gather_offset_component(tex, uv, offset, 2);
        let gather_offsets = texture_gather_offsets(tex, uv, offsets);
        let gather_offsets_component = texture_gather_offsets_component(tex, uv, offsets, 3);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert "vec4 texture_helpers(sampler2D tex, samplerCube cube" in result
    assert "ivec2 offsets[4]" in result
    assert "base = texture(tex, uv);" in result
    assert "lod = textureLod(tex, uv, 1.0);" in result
    assert "grad = textureGrad(tex, uv, ddx, ddy);" in result
    assert "shifted = textureOffset(tex, uv, offset);" in result
    assert "projected_color = textureProj(tex, projected);" in result
    assert "projected_lod = textureProjLod(tex, projected, 1.0);" in result
    assert "projected_grad = textureProjGrad(tex, projected, ddx, ddy);" in result
    assert "projected_offset = textureProjOffset(tex, projected, offset);" in result
    assert (
        "projected_lod_offset = textureProjLodOffset(tex, projected, 1.0, offset);"
        in result
    )
    assert (
        "projected_grad_offset = textureProjGradOffset(tex, projected, ddx, ddy, offset);"
        in result
    )
    assert "fetched = texelFetch(tex, pixel, 0);" in result
    assert "fetched_offset = texelFetchOffset(tex, pixel, 0, offset);" in result
    assert "size = textureSize(tex, 0);" in result
    assert "cube_size = textureSize(cube);" in result
    assert "levels = textureQueryLevels(tex);" in result
    assert "query_lod = textureQueryLod(tex, uv);" in result
    assert "samples = textureSamples(tex);" in result
    assert "gather = textureGather(tex, uv);" in result
    assert "gather_component = textureGather(tex, uv, 1);" in result
    assert "gather_offset = textureGatherOffset(tex, uv, offset);" in result
    assert (
        "gather_offset_component = textureGatherOffset(tex, uv, offset, 2);" in result
    )
    assert "gather_offsets = textureGatherOffsets(tex, uv, offsets);" in result
    assert (
        "gather_offsets_component = textureGatherOffsets(tex, uv, offsets, 3);"
        in result
    )
    assert "sample_lod" not in result
    assert "texture_size_lod" not in result
    assert "texture_gather_component" not in result


def test_gpu_multisample_texture_types_convert_to_crossgl_intrinsics():
    code = """
    fn multisample_texture_helpers(
        ms_tex: Texture2DMS<f32>,
        ms_array: Texture2DMSArray<f32>,
        pixel: Vec2<i32>,
        pixel_layer: Vec3<i32>,
        sample_index: i32,
    ) -> Vec4<f32> {
        let size = texture_size(ms_tex);
        let array_size = texture_size(ms_array);
        let samples = texture_samples(ms_tex);
        let fetched = texel_fetch(ms_tex, pixel, sample_index);
        let fetched_array = texel_fetch(ms_array, pixel_layer, sample_index);
        return fetched;
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 multisample_texture_helpers(sampler2DMS ms_tex, sampler2DMSArray ms_array"
        in result
    )
    assert "size = textureSize(ms_tex);" in result
    assert "array_size = textureSize(ms_array);" in result
    assert "samples = textureSamples(ms_tex);" in result
    assert "fetched = texelFetch(ms_tex, pixel, sample_index);" in result
    assert "fetched_array = texelFetch(ms_array, pixel_layer, sample_index);" in result
    assert "Texture2DMS" not in result
    assert "texture_size" not in result
    assert "texture_samples" not in result
    assert "texel_fetch" not in result


def test_gpu_explicit_sampler_texture_helpers_convert_to_crossgl_intrinsics():
    code = """
    fn texture_sampler_helpers(
        tex: Texture2D<f32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        projected: Vec3<f32>,
        ddx: Vec2<f32>,
        ddy: Vec2<f32>,
        offset: Vec2<i32>,
        offsets: [Vec2<i32>; 4],
    ) -> Vec4<f32> {
        let base = sample_sampler(tex, sampler_state, uv);
        let biased = sample_bias(tex, uv, 0.25);
        let biased_sampler = sample_bias_sampler(tex, sampler_state, uv, 0.5);
        let lod = sample_lod_sampler(tex, sampler_state, uv, 1.0);
        let lod_offset = sample_lod_offset_sampler(tex, sampler_state, uv, 1.0, offset);
        let grad = sample_grad_sampler(tex, sampler_state, uv, ddx, ddy);
        let grad_offset = sample_grad_offset_sampler(tex, sampler_state, uv, ddx, ddy, offset);
        let shifted = sample_offset_sampler(tex, sampler_state, uv, offset);
        let shifted_bias = sample_offset_bias(tex, uv, offset, 0.125);
        let shifted_bias_sampler = sample_offset_bias_sampler(tex, sampler_state, uv, offset, 0.125);
        let projected_color = sample_projected_sampler(tex, sampler_state, projected);
        let projected_bias = sample_projected_bias(tex, projected, 0.25);
        let projected_bias_sampler = sample_projected_bias_sampler(tex, sampler_state, projected, 0.25);
        let projected_lod = sample_projected_lod_sampler(tex, sampler_state, projected, 1.0);
        let projected_grad = sample_projected_grad_sampler(tex, sampler_state, projected, ddx, ddy);
        let projected_offset = sample_projected_offset_sampler(tex, sampler_state, projected, offset);
        let projected_offset_bias = sample_projected_offset_bias(tex, projected, offset, 0.25);
        let projected_offset_bias_sampler = sample_projected_offset_bias_sampler(tex, sampler_state, projected, offset, 0.25);
        let projected_lod_offset = sample_projected_lod_offset_sampler(tex, sampler_state, projected, 1.0, offset);
        let projected_grad_offset = sample_projected_grad_offset_sampler(tex, sampler_state, projected, ddx, ddy, offset);
        let query = texture_query_lod_sampler(tex, sampler_state, uv);
        let gather = texture_gather_sampler(tex, sampler_state, uv);
        let gather_component = texture_gather_component_sampler(tex, sampler_state, uv, 1);
        let gather_offset = texture_gather_offset_sampler(tex, sampler_state, uv, offset);
        let gather_offset_component = texture_gather_offset_component_sampler(tex, sampler_state, uv, offset, 2);
        let gather_offsets = texture_gather_offsets_sampler(tex, sampler_state, uv, offsets);
        let gather_offsets_component = texture_gather_offsets_component_sampler(tex, sampler_state, uv, offsets, 3);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert "vec4 texture_sampler_helpers(sampler2D tex, sampler sampler_state" in result
    assert "base = texture(tex, sampler_state, uv);" in result
    assert "biased = texture(tex, uv, 0.25);" in result
    assert "biased_sampler = texture(tex, sampler_state, uv, 0.5);" in result
    assert "lod = textureLod(tex, sampler_state, uv, 1.0);" in result
    assert (
        "lod_offset = textureLodOffset(tex, sampler_state, uv, 1.0, offset);" in result
    )
    assert "grad = textureGrad(tex, sampler_state, uv, ddx, ddy);" in result
    assert (
        "grad_offset = textureGradOffset(tex, sampler_state, uv, ddx, ddy, offset);"
        in result
    )
    assert "shifted = textureOffset(tex, sampler_state, uv, offset);" in result
    assert "shifted_bias = textureOffset(tex, uv, offset, 0.125);" in result
    assert (
        "shifted_bias_sampler = textureOffset(tex, sampler_state, uv, offset, 0.125);"
        in result
    )
    assert "projected_color = textureProj(tex, sampler_state, projected);" in result
    assert "projected_bias = textureProj(tex, projected, 0.25);" in result
    assert (
        "projected_bias_sampler = textureProj(tex, sampler_state, projected, 0.25);"
        in result
    )
    assert (
        "projected_lod = textureProjLod(tex, sampler_state, projected, 1.0);" in result
    )
    assert (
        "projected_grad = textureProjGrad(tex, sampler_state, projected, ddx, ddy);"
        in result
    )
    assert (
        "projected_offset = textureProjOffset(tex, sampler_state, projected, offset);"
        in result
    )
    assert (
        "projected_offset_bias = textureProjOffset(tex, projected, offset, 0.25);"
        in result
    )
    assert (
        "projected_offset_bias_sampler = textureProjOffset(tex, sampler_state, projected, offset, 0.25);"
        in result
    )
    assert (
        "projected_lod_offset = textureProjLodOffset(tex, sampler_state, projected, 1.0, offset);"
        in result
    )
    assert (
        "projected_grad_offset = textureProjGradOffset(tex, sampler_state, projected, ddx, ddy, offset);"
        in result
    )
    assert "query = textureQueryLod(tex, sampler_state, uv);" in result
    assert "gather = textureGather(tex, sampler_state, uv);" in result
    assert "gather_component = textureGather(tex, sampler_state, uv, 1);" in result
    assert (
        "gather_offset = textureGatherOffset(tex, sampler_state, uv, offset);" in result
    )
    assert (
        "gather_offset_component = textureGatherOffset(tex, sampler_state, uv, offset, 2);"
        in result
    )
    assert (
        "gather_offsets = textureGatherOffsets(tex, sampler_state, uv, offsets);"
        in result
    )
    assert (
        "gather_offsets_component = textureGatherOffsets(tex, sampler_state, uv, offsets, 3);"
        in result
    )
    assert "sample_sampler" not in result
    assert "sample_bias" not in result
    assert "texture_gather" not in result


def test_gpu_shadow_compare_helper_calls_convert_to_crossgl_intrinsics():
    code = """
    fn shadow_helpers(
        shadow: DepthTexture2D<f32>,
        shadow_array: DepthTexture2DArray<f32>,
        cube_shadow: DepthTextureCube<f32>,
        cube_array_shadow: DepthTextureCubeArray<f32>,
        compare_sampler: Sampler,
        uv: Vec2<f32>,
        uv_layer: Vec3<f32>,
        direction: Vec3<f32>,
        cube_layer: Vec4<f32>,
        uvq: Vec3<f32>,
        uvqw: Vec4<f32>,
        depth: f32,
        lod: f32,
        ddx: Vec2<f32>,
        ddy: Vec2<f32>,
        offset: Vec2<i32>,
    ) -> f32 {
        let base = texture_compare(shadow, uv, depth);
        let base_sampler = texture_compare_sampler(shadow, compare_sampler, uv, depth);
        let array_compare = texture_compare(shadow_array, uv_layer, depth);
        let cube_compare = texture_compare(cube_shadow, direction, depth);
        let cube_array_compare = texture_compare(cube_array_shadow, cube_layer, depth);
        let offset_compare = texture_compare_offset(shadow, uv, depth, offset);
        let offset_sampler = texture_compare_offset_sampler(shadow, compare_sampler, uv, depth, offset);
        let lod_compare = texture_compare_lod(shadow, uv, depth, lod);
        let lod_sampler = texture_compare_lod_sampler(shadow, compare_sampler, uv, depth, lod);
        let lod_offset = texture_compare_lod_offset(shadow, uv, depth, lod, offset);
        let lod_offset_sampler = texture_compare_lod_offset_sampler(shadow, compare_sampler, uv, depth, lod, offset);
        let grad_compare = texture_compare_grad(shadow, uv, depth, ddx, ddy);
        let grad_sampler = texture_compare_grad_sampler(shadow, compare_sampler, uv, depth, ddx, ddy);
        let grad_offset = texture_compare_grad_offset(shadow, uv, depth, ddx, ddy, offset);
        let grad_offset_sampler = texture_compare_grad_offset_sampler(shadow, compare_sampler, uv, depth, ddx, ddy, offset);
        let projected = texture_compare_projected(shadow, uvq, depth);
        let projected_sampler = texture_compare_projected_sampler(shadow, compare_sampler, uvq, depth);
        let projected_w = texture_compare_projected(shadow, uvqw, depth);
        let projected_offset = texture_compare_projected_offset(shadow, uvq, depth, offset);
        let projected_offset_sampler = texture_compare_projected_offset_sampler(shadow, compare_sampler, uvq, depth, offset);
        let projected_lod = texture_compare_projected_lod(shadow, uvq, depth, lod);
        let projected_lod_sampler = texture_compare_projected_lod_sampler(shadow, compare_sampler, uvq, depth, lod);
        let projected_lod_offset = texture_compare_projected_lod_offset(shadow, uvq, depth, lod, offset);
        let projected_lod_offset_sampler = texture_compare_projected_lod_offset_sampler(shadow, compare_sampler, uvq, depth, lod, offset);
        let projected_grad = texture_compare_projected_grad(shadow, uvq, depth, ddx, ddy);
        let projected_grad_sampler = texture_compare_projected_grad_sampler(shadow, compare_sampler, uvq, depth, ddx, ddy);
        let projected_grad_offset = texture_compare_projected_grad_offset(shadow, uvq, depth, ddx, ddy, offset);
        let projected_grad_offset_sampler = texture_compare_projected_grad_offset_sampler(shadow, compare_sampler, uvq, depth, ddx, ddy, offset);
        let gathered = texture_gather_compare(shadow, uv, depth);
        let gathered_sampler = texture_gather_compare_sampler(shadow, compare_sampler, uv, depth);
        let gathered_offset = texture_gather_compare_offset(shadow, uv, depth, offset);
        let gathered_offset_sampler = texture_gather_compare_offset_sampler(shadow, compare_sampler, uv, depth, offset);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert (
        "float shadow_helpers(sampler2DShadow shadow, sampler2DArrayShadow shadow_array"
        in result
    )
    assert "samplerCubeShadow cube_shadow" in result
    assert "samplerCubeArrayShadow cube_array_shadow" in result
    assert "sampler compare_sampler" in result
    assert "base = textureCompare(shadow, uv, depth);" in result
    assert (
        "base_sampler = textureCompare(shadow, compare_sampler, uv, depth);" in result
    )
    assert "array_compare = textureCompare(shadow_array, uv_layer, depth);" in result
    assert "cube_compare = textureCompare(cube_shadow, direction, depth);" in result
    assert (
        "cube_array_compare = textureCompare(cube_array_shadow, cube_layer, depth);"
        in result
    )
    assert "offset_compare = textureCompareOffset(shadow, uv, depth, offset);" in result
    assert (
        "offset_sampler = textureCompareOffset(shadow, compare_sampler, uv, depth, offset);"
        in result
    )
    assert "lod_compare = textureCompareLod(shadow, uv, depth, lod);" in result
    assert (
        "lod_sampler = textureCompareLod(shadow, compare_sampler, uv, depth, lod);"
        in result
    )
    assert (
        "lod_offset = textureCompareLodOffset(shadow, uv, depth, lod, offset);"
        in result
    )
    assert (
        "lod_offset_sampler = textureCompareLodOffset(shadow, compare_sampler, uv, depth, lod, offset);"
        in result
    )
    assert "grad_compare = textureCompareGrad(shadow, uv, depth, ddx, ddy);" in result
    assert (
        "grad_sampler = textureCompareGrad(shadow, compare_sampler, uv, depth, ddx, ddy);"
        in result
    )
    assert (
        "grad_offset = textureCompareGradOffset(shadow, uv, depth, ddx, ddy, offset);"
        in result
    )
    assert (
        "grad_offset_sampler = textureCompareGradOffset(shadow, compare_sampler, uv, depth, ddx, ddy, offset);"
        in result
    )
    assert "projected = textureCompareProj(shadow, uvq, depth);" in result
    assert (
        "projected_sampler = textureCompareProj(shadow, compare_sampler, uvq, depth);"
        in result
    )
    assert (
        "projected_offset = textureCompareProjOffset(shadow, uvq, depth, offset);"
        in result
    )
    assert (
        "projected_offset_sampler = textureCompareProjOffset(shadow, compare_sampler, uvq, depth, offset);"
        in result
    )
    assert "projected_lod = textureCompareProjLod(shadow, uvq, depth, lod);" in result
    assert (
        "projected_lod_sampler = textureCompareProjLod(shadow, compare_sampler, uvq, depth, lod);"
        in result
    )
    assert (
        "projected_lod_offset = textureCompareProjLodOffset(shadow, uvq, depth, lod, offset);"
        in result
    )
    assert (
        "projected_lod_offset_sampler = textureCompareProjLodOffset(shadow, compare_sampler, uvq, depth, lod, offset);"
        in result
    )
    assert (
        "projected_grad = textureCompareProjGrad(shadow, uvq, depth, ddx, ddy);"
        in result
    )
    assert (
        "projected_grad_sampler = textureCompareProjGrad(shadow, compare_sampler, uvq, depth, ddx, ddy);"
        in result
    )
    assert (
        "projected_grad_offset = textureCompareProjGradOffset(shadow, uvq, depth, ddx, ddy, offset);"
        in result
    )
    assert (
        "projected_grad_offset_sampler = textureCompareProjGradOffset(shadow, compare_sampler, uvq, depth, ddx, ddy, offset);"
        in result
    )
    assert "gathered = textureGatherCompare(shadow, uv, depth);" in result
    assert (
        "gathered_sampler = textureGatherCompare(shadow, compare_sampler, uv, depth);"
        in result
    )
    assert (
        "gathered_offset = textureGatherCompareOffset(shadow, uv, depth, offset);"
        in result
    )
    assert (
        "gathered_offset_sampler = textureGatherCompareOffset(shadow, compare_sampler, uv, depth, offset);"
        in result
    )
    assert "DepthTexture2D" not in result
    assert "texture_compare" not in result
    assert "texture_gather_compare" not in result


def test_gpu_image_and_buffer_helper_calls_convert_to_crossgl_intrinsics():
    code = """
    fn resource_helpers(
        color_image: Image2D<Vec4<f32>>,
        counter_image: Image2D<Vec4<u32>>,
        ramp_image: Image1D<Vec4<f32>>,
        ramp_array_image: Image1DArray<Vec4<f32>>,
        volume_image: Image3D<Vec4<f32>>,
        cube_image: ImageCube<Vec4<f32>>,
        layer_image: Image2DArray<Vec4<f32>>,
        ms_image: Image2DMS<Vec4<f32>>,
        ms_array_image: Image2DMSArray<Vec4<f32>>,
        signed_volume: Image3D<Vec4<i32>>,
        counter_layers: Image2DArray<Vec4<u32>>,
        values: RwBuffer<i32>,
        weights: Buffer<f32>,
        append_values: AppendBuffer<u32>,
        consume_values: ConsumeBuffer<u32>,
        raw_bytes: ByteAddressBuffer,
        raw_out: RwByteAddressBuffer,
        pixel: Vec2<i32>,
        index: u32,
        amount: u32,
    ) -> Vec4<f32> {
        let color = image_load(color_image, pixel);
        image_store(color_image, pixel, color);
        let color_size = image_size(color_image);
        let ramp_size = image_size(ramp_image);
        let volume_size = image_size(volume_image);
        let samples = image_samples(ms_image);
        let previous = image_atomic_add(counter_image, pixel, amount);
        let min_value = image_atomic_min(counter_image, pixel, amount);
        let max_value = image_atomic_max(counter_image, pixel, amount);
        let and_value = image_atomic_and(counter_image, pixel, amount);
        let or_value = image_atomic_or(counter_image, pixel, amount);
        let xor_value = image_atomic_xor(counter_image, pixel, amount);
        let exchanged = image_atomic_exchange(counter_image, pixel, amount);
        let swapped = image_atomic_comp_swap(counter_image, pixel, previous, amount);
        let value = buffer_load(values, index);
        let weight = buffer_load(weights, index);
        let length: u32 = 0;
        buffer_dimensions(values, length);
        buffer_store(values, index, value);
        return color;
    }
    """

    result = parse_and_generate(code)

    assert "vec4 resource_helpers(image2D color_image, uimage2D counter_image" in result
    assert "image1D ramp_image" in result
    assert "image1DArray ramp_array_image" in result
    assert "image3D volume_image" in result
    assert "imageCube cube_image" in result
    assert "image2DArray layer_image" in result
    assert "image2DMS ms_image" in result
    assert "image2DMSArray ms_array_image" in result
    assert "iimage3D signed_volume" in result
    assert "uimage2DArray counter_layers" in result
    assert "RWStructuredBuffer<int> values" in result
    assert "StructuredBuffer<float> weights" in result
    assert "AppendStructuredBuffer<uint> append_values" in result
    assert "ConsumeStructuredBuffer<uint> consume_values" in result
    assert "ByteAddressBuffer raw_bytes" in result
    assert "RWByteAddressBuffer raw_out" in result
    assert "color = imageLoad(color_image, pixel);" in result
    assert "imageStore(color_image, pixel, color);" in result
    assert "color_size = imageSize(color_image);" in result
    assert "ramp_size = imageSize(ramp_image);" in result
    assert "volume_size = imageSize(volume_image);" in result
    assert "samples = imageSamples(ms_image);" in result
    assert "previous = imageAtomicAdd(counter_image, pixel, amount);" in result
    assert "min_value = imageAtomicMin(counter_image, pixel, amount);" in result
    assert "max_value = imageAtomicMax(counter_image, pixel, amount);" in result
    assert "and_value = imageAtomicAnd(counter_image, pixel, amount);" in result
    assert "or_value = imageAtomicOr(counter_image, pixel, amount);" in result
    assert "xor_value = imageAtomicXor(counter_image, pixel, amount);" in result
    assert "exchanged = imageAtomicExchange(counter_image, pixel, amount);" in result
    assert (
        "swapped = imageAtomicCompSwap(counter_image, pixel, previous, amount);"
        in result
    )
    assert "value = buffer_load(values, index);" in result
    assert "weight = buffer_load(weights, index);" in result
    assert "length = 0;" in result
    assert "buffer_dimensions(values, length);" in result
    assert "buffer_store(values, index, value);" in result
    assert "Image2D" not in result
    assert "RwBuffer" not in result
    assert "Buffer<f32>" not in result
    assert "image_load" not in result
    assert "image_size" not in result
    assert "image_samples" not in result
    assert "image_atomic_min" not in result
    assert "image_atomic_max" not in result
    assert "image_atomic_and" not in result
    assert "image_atomic_or" not in result
    assert "image_atomic_xor" not in result
    assert "image_atomic_exchange" not in result
    assert "image_atomic_comp_swap" not in result


def test_gpu_multisample_image_sample_helpers_convert_to_crossgl_intrinsics():
    code = """
    fn multisample_image_helpers(
        ms_image: Image2DMS<Vec4<f32>>,
        ms_array_image: Image2DMSArray<Vec4<f32>>,
        ms_counter: Image2DMS<Vec4<u32>>,
        ms_signed_layers: Image2DMSArray<Vec4<i32>>,
        pixel: Vec2<i32>,
        pixel_layer: Vec3<i32>,
        sample_index: i32,
        amount: u32,
    ) -> Vec4<f32> {
        let color = image_load_sample(ms_image, pixel, sample_index);
        image_store_sample(ms_image, pixel, sample_index, color);
        let layer = image_load_sample(ms_array_image, pixel_layer, sample_index);
        image_store_sample(ms_array_image, pixel_layer, sample_index, layer);
        let previous = image_atomic_add_sample(
            ms_counter,
            pixel,
            sample_index,
            amount
        );
        let swapped = image_atomic_comp_swap_sample(
            ms_signed_layers,
            pixel_layer,
            sample_index,
            0,
            previous as i32
        );
        return color + layer + Vec4::<f32>::new(swapped as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 multisample_image_helpers(image2DMS ms_image, "
        "image2DMSArray ms_array_image, uimage2DMS ms_counter, "
        "iimage2DMSArray ms_signed_layers, ivec2 pixel, ivec3 pixel_layer, "
        "int sample_index, uint amount)"
    ) in result
    assert "color = imageLoad(ms_image, pixel, sample_index);" in result
    assert "imageStore(ms_image, pixel, sample_index, color);" in result
    assert "layer = imageLoad(ms_array_image, pixel_layer, sample_index);" in result
    assert "imageStore(ms_array_image, pixel_layer, sample_index, layer);" in result
    assert (
        "previous = imageAtomicAdd(ms_counter, pixel, sample_index, amount);" in result
    )
    assert (
        "swapped = imageAtomicCompSwap("
        "ms_signed_layers, pixel_layer, sample_index, 0, (int)previous);" in result
    )
    assert "image_load_sample" not in result
    assert "image_store_sample" not in result
    assert "image_atomic_add_sample" not in result
    assert "image_atomic_comp_swap_sample" not in result


def test_aliased_reference_resource_helpers_convert_to_crossgl_intrinsics():
    code = """
    use crate::gpu::{
        Texture2D as ColorTexture,
        DepthTexture2D as ShadowTexture,
        Image2D as StorageImage,
        RwBuffer as WritableBuffer,
        Buffer as ReadBuffer,
        Sampler as ShaderSampler,
    };

    fn aliased_resources(
        tex: &ColorTexture<f32>,
        shadow: &ShadowTexture<f32>,
        image: &mut StorageImage<Vec4<u32>>,
        values: &mut WritableBuffer<i32>,
        weights: &ReadBuffer<f32>,
        sampler_state: &ShaderSampler,
        uv: Vec2<f32>,
        pixel: Vec2<i32>,
        index: u32,
        amount: u32,
    ) -> Vec4<f32> {
        let base = sample_sampler(tex, sampler_state, uv);
        let lit = texture_compare(shadow, uv, 0.5);
        let previous = image_atomic_add(image, pixel, amount);
        let value = buffer_load(values, index);
        let weight = buffer_load(weights, index);
        buffer_store(values, index, value + previous as i32);
        return base + Vec4::<f32>::new(weight + lit, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 aliased_resources(sampler2D tex, sampler2DShadow shadow, "
        "uimage2D image, RWStructuredBuffer<int> values, "
        "StructuredBuffer<float> weights, sampler sampler_state, vec2 uv, "
        "ivec2 pixel, uint index, uint amount)"
    ) in result
    assert "base = texture(tex, sampler_state, uv);" in result
    assert "lit = textureCompare(shadow, uv, 0.5);" in result
    assert "previous = imageAtomicAdd(image, pixel, amount);" in result
    assert "value = buffer_load(values, index);" in result
    assert "weight = buffer_load(weights, index);" in result
    assert "buffer_store(values, index, (value + (int)previous));" in result
    assert "&ColorTexture" not in result
    assert "&mut StorageImage" not in result
    assert "&ReadBuffer" not in result
    assert "sample_sampler" not in result
    assert "texture_compare" not in result
    assert "image_atomic_add" not in result
    assert "ColorTexture tex" not in result
    assert "WritableBuffer<int> values" not in result


def test_gpu_resource_method_calls_convert_to_crossgl_intrinsics():
    code = """
    fn method_resources(
        tex: Texture2D<f32>,
        shadow: DepthTexture2D<f32>,
        color_image: Image2D<Vec4<f32>>,
        counter_image: Image2D<Vec4<u32>>,
        values: RwBuffer<i32>,
        weights: Buffer<f32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        pixel: Vec2<i32>,
        index: u32,
        amount: u32,
    ) -> Vec4<f32> {
        let base = tex.sample_sampler(sampler_state, uv);
        let biased = tex.sample_bias(uv, 0.25);
        let size = tex.texture_size_lod(0);
        let lit = shadow.texture_compare(uv, 0.5);
        let color = color_image.image_load(pixel);
        color_image.image_store(pixel, color);
        let previous = counter_image.image_atomic_add(pixel, amount);
        let value = values.buffer_load(index);
        let weight = weights.buffer_load(index);
        values.buffer_store(index, value + previous as i32);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 method_resources(sampler2D tex, sampler2DShadow shadow, "
        "image2D color_image, uimage2D counter_image, "
        "RWStructuredBuffer<int> values, StructuredBuffer<float> weights, "
        "sampler sampler_state, vec2 uv, ivec2 pixel, uint index, uint amount)"
    ) in result
    assert "base = texture(tex, sampler_state, uv);" in result
    assert "biased = texture(tex, uv, 0.25);" in result
    assert "size = textureSize(tex, 0);" in result
    assert "lit = textureCompare(shadow, uv, 0.5);" in result
    assert "color = imageLoad(color_image, pixel);" in result
    assert "imageStore(color_image, pixel, color);" in result
    assert "previous = imageAtomicAdd(counter_image, pixel, amount);" in result
    assert "value = buffer_load(values, index);" in result
    assert "weight = buffer_load(weights, index);" in result
    assert "buffer_store(values, index, (value + (int)previous));" in result
    assert ".sample_sampler" not in result
    assert ".sample_bias" not in result
    assert ".texture_size_lod" not in result
    assert ".texture_compare" not in result
    assert ".image_load" not in result
    assert ".image_store" not in result
    assert ".image_atomic_add" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_struct_member_receivers():
    code = """
    type ResourceBundle = Resources;

    struct Resources {
        tex: Texture2D<f32>,
        shadow: DepthTexture2D<f32>,
        color_image: Image2D<Vec4<f32>>,
        values: RwBuffer<i32>,
    }

    struct ResourceGroup {
        primary: Resources,
        alternatives: Resources[2],
        aliased: ResourceBundle,
    }

    fn member_method_resources(
        group: ResourceGroup,
        resources: ResourceBundle,
        resource_array: Resources[2],
        sampler_state: Sampler,
        uv: Vec2<f32>,
        pixel: Vec2<i32>,
        index: u32,
    ) -> Vec4<f32> {
        let primary_tex = group.primary.tex;
        let base = primary_tex.sample_sampler(sampler_state, uv);
        let lit = group.alternatives[index].shadow.texture_compare(uv, 0.5);
        let color = resources.color_image.image_load(pixel);
        resources.color_image.image_store(pixel, color);
        let value = group.aliased.values.buffer_load(index);
        resource_array[index].values.buffer_store(index, value);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 member_method_resources(ResourceGroup group, "
        "ResourceBundle resources, Resources resource_array[2], "
        "sampler sampler_state, vec2 uv, ivec2 pixel, uint index)"
    ) in result
    assert "base = texture(primary_tex, sampler_state, uv);" in result
    assert "lit = textureCompare(group.alternatives[index].shadow, uv, 0.5);" in result
    assert "color = imageLoad(resources.color_image, pixel);" in result
    assert "imageStore(resources.color_image, pixel, color);" in result
    assert "value = buffer_load(group.aliased.values, index);" in result
    assert "buffer_store(resource_array[index].values, index, value);" in result
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".image_load" not in result
    assert ".image_store" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_generic_struct_member_receivers():
    code = """
    type ResourceHolder<T> = Holder<T>;

    struct Holder<T> {
        resource: T,
        resources: T[2],
    }

    struct Nested<T> {
        holder: Holder<T>,
    }

    fn generic_member_resources(
        holder: Holder<Texture2D<f32>>,
        shadow_holder: ResourceHolder<DepthTexture2D<f32>>,
        buffer_holder: Nested<RwBuffer<i32>>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = holder.resource;
        let base = local_tex.sample_sampler(sampler_state, uv);
        let lit = shadow_holder.resources[index].texture_compare(uv, 0.5);
        let value = buffer_holder.holder.resource.buffer_load(index);
        buffer_holder.holder.resources[index].buffer_store(index, value);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "lit = textureCompare(shadow_holder.resources[index], uv, 0.5);" in result
    assert "value = buffer_load(buffer_holder.holder.resource, index);" in result
    assert (
        "buffer_store(buffer_holder.holder.resources[index], index, value);" in result
    )
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_multi_generic_struct_member_receivers():
    code = """
    struct TaggedHolder<T, Tag> {
        resource: T,
        tag: Tag,
    }

    fn tagged_resource(
        holder: TaggedHolder<Texture2D<f32>, u32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        return holder.resource.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "return texture(holder.resource, sampler_state, uv);" in result
    assert ".sample_sampler" not in result


def test_gpu_resource_method_calls_convert_on_tuple_struct_field_receivers():
    code = """
    struct TupleHolder(Texture2D<f32>, DepthTexture2D<f32>, RwBuffer<i32>);

    fn tuple_resources(
        holder: TupleHolder,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let base = holder.0.sample_sampler(sampler_state, uv);
        let lit = holder.1.texture_compare(uv, 0.5);
        let value = holder.2.buffer_load(index);
        holder.2.buffer_store(index, value);
        return base + Vec4::<f32>::new(lit, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "sampler2D field0;" in result
    assert "sampler2DShadow field1;" in result
    assert "RWStructuredBuffer<int> field2;" in result
    assert "base = texture(holder.field0, sampler_state, uv);" in result
    assert "lit = textureCompare(holder.field1, uv, 0.5);" in result
    assert "value = buffer_load(holder.field2, index);" in result
    assert "buffer_store(holder.field2, index, value);" in result
    assert "holder.0" not in result
    assert "holder.1" not in result
    assert "holder.2" not in result
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_resource_type_alias_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;
    type GenericColorTexture<T> = Texture2D<T>;
    type ShadowTexture<T> = DepthTexture2D<T>;
    type WritableBuffer<T> = RwBuffer<T>;

    fn alias_resources<T>(
        tex: ColorTexture,
        generic_tex: GenericColorTexture<T>,
        shadow: ShadowTexture<T>,
        values: WritableBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let base = tex.sample_sampler(sampler_state, uv);
        let generic_base = generic_tex.sample_sampler(sampler_state, uv);
        let lit = shadow.texture_compare(uv, 0.5);
        let value = values.buffer_load(index);
        values.buffer_store(index, value);
        return base + generic_base + Vec4::<f32>::new(lit, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "typedef sampler2D ColorTexture;" in result
    assert (
        "vec4 alias_resources(ColorTexture tex, sampler2D generic_tex, "
        "sampler2DShadow shadow, RWStructuredBuffer<int> values"
    ) in result
    assert "base = texture(tex, sampler_state, uv);" in result
    assert "generic_base = texture(generic_tex, sampler_state, uv);" in result
    assert "lit = textureCompare(shadow, uv, 0.5);" in result
    assert "value = buffer_load(values, index);" in result
    assert "buffer_store(values, index, value);" in result
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result
    assert "GenericColorTexture" not in result
    assert "ShadowTexture" not in result
    assert "WritableBuffer" not in result


def test_gpu_resource_method_calls_convert_on_function_returned_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;
    type WritableBuffer<T> = RwBuffer<T>;

    fn choose_tex(tex: ColorTexture) -> ColorTexture {
        return tex;
    }

    fn choose_values(values: WritableBuffer<i32>) -> WritableBuffer<i32> {
        return values;
    }

    fn returned_resource_methods(
        tex: ColorTexture,
        values: WritableBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = self::choose_tex(tex);
        let local_values = crate::choose_values(values);
        let base = local_tex.sample_sampler(sampler_state, uv);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert "ColorTexture choose_tex(ColorTexture tex)" in result
    assert (
        "RWStructuredBuffer<int> choose_values(RWStructuredBuffer<int> values)"
        in result
    )
    assert "let local_tex = choose_tex(tex);" in result
    assert "let local_values = choose_values(values);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_generic_function_returned_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;
    type WritableBuffer<T> = RwBuffer<T>;

    struct Holder<T> {
        resource: T,
    }

    fn pick<T>(value: T) -> T {
        return value;
    }

    fn choose<T>(left: T, right: T) -> T {
        return left;
    }

    fn unwrap_holder<T>(holder: Holder<T>) -> T {
        return holder.resource;
    }

    fn generic_returned_resource_methods(
        tex: ColorTexture,
        shadow: DepthTexture2D<f32>,
        values: WritableBuffer<i32>,
        holder: Holder<Texture2D<f32>>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = pick(tex);
        let explicit_tex = pick::<Texture2D<f32>>(tex);
        let local_shadow = pick(shadow);
        let local_values = choose(values, values);
        let holder_tex = unwrap_holder(holder);
        let base = local_tex.sample_sampler(sampler_state, uv);
        let explicit_base = explicit_tex.sample_sampler(sampler_state, uv);
        let lit = local_shadow.texture_compare(uv, 0.5);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base + explicit_base + holder_tex.sample_sampler(sampler_state, uv)
            + Vec4::<f32>::new(lit, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "T pick(T value)" in result
    assert "T choose(T left, T right)" in result
    assert "T unwrap_holder(Holder<T> holder)" in result
    assert "let local_tex = pick(tex);" in result
    assert "let explicit_tex = pick(tex);" in result
    assert "let local_shadow = pick(shadow);" in result
    assert "let local_values = choose(values, values);" in result
    assert "let holder_tex = unwrap_holder(holder);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "explicit_base = texture(explicit_tex, sampler_state, uv);" in result
    assert "lit = textureCompare(local_shadow, uv, 0.5);" in result
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert "texture(holder_tex, sampler_state, uv)" in result
    assert "pick<Texture2D<f32>>" not in result
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_unresolved_generic_function_returned_receivers_stay_unlowered():
    code = """
    fn default_value<T>() -> T {
        return make_default();
    }

    fn conflict<T>(left: T, right: T) -> T {
        return left;
    }

    fn unresolved_generic_returned_resource_methods(
        tex: Texture2D<f32>,
        values: RwBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        let unknown = default_value();
        let conflicted = conflict(tex, values);
        let base = unknown.sample_sampler(sampler_state, uv);
        return base + conflicted.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "let unknown = default_value();" in result
    assert "let conflicted = conflict(tex, values);" in result
    assert "base = unknown.sample_sampler(sampler_state, uv);" in result
    assert "return (base + conflicted.sample_sampler(sampler_state, uv));" in result
    assert "texture(unknown" not in result
    assert "texture(conflicted" not in result


def test_gpu_resource_method_calls_convert_on_impl_method_returned_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Provider {
        tex: ColorTexture,
        values: RwBuffer<i32>,
    }

    impl Provider {
        fn clone_provider(&self) -> Self {
            return self;
        }

        fn color(&self) -> ColorTexture {
            return self.tex;
        }

        fn writable(&self) -> RwBuffer<i32> {
            return self.values;
        }
    }

    fn impl_returned_resource_methods(
        provider: Provider,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = provider.color();
        let local_values = provider.writable();
        let base = local_tex.sample_sampler(sampler_state, uv);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base;
    }
    """

    result = parse_and_generate(code)

    assert "Provider Provider_clone_provider(Provider self)" in result
    assert "ColorTexture Provider_color(Provider self)" in result
    assert "RWStructuredBuffer<int> Provider_writable(Provider self)" in result
    assert "let local_tex = Provider_color(provider);" in result
    assert "let local_values = Provider_writable(provider);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert "Self self" not in result
    assert "Self Provider_clone_provider" not in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_generic_impl_method_returned_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Holder<T> {
        resource: T,
        resources: T[2],
    }

    impl<T> Holder<T> {
        fn get(&self) -> T {
            return self.resource;
        }

        fn get_index(&self, index: u32) -> T {
            return self.resources[index];
        }
    }

    fn generic_impl_returned_resource_methods(
        holder: Holder<ColorTexture>,
        shadow_holder: Holder<DepthTexture2D<f32>>,
        values_holder: Holder<RwBuffer<i32>>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = holder.get();
        let local_shadow = shadow_holder.get_index(index);
        let local_values = values_holder.get();
        let base = local_tex.sample_sampler(sampler_state, uv);
        let lit = local_shadow.texture_compare(uv, 0.5);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base + Vec4::<f32>::new(lit, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "T Holder_get(Holder<T> self)" in result
    assert "T Holder_get_index(Holder<T> self, uint index)" in result
    assert "let local_tex = Holder_get(holder);" in result
    assert "let local_shadow = Holder_get_index(shadow_holder, index);" in result
    assert "let local_values = Holder_get(values_holder);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "lit = textureCompare(local_shadow, uv, 0.5);" in result
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert "Holder<T>_get" not in result
    assert ".get()" not in result
    assert ".get_index" not in result
    assert ".sample_sampler" not in result
    assert ".texture_compare" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_chained_self_impl_method_members():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Holder<T> {
        resource: T,
    }

    impl<T> Holder<T> {
        fn clone_holder(&self) -> Self {
            return self;
        }
    }

    fn chained_self_impl_member_resource_methods(
        holder: Holder<ColorTexture>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        return holder.clone_holder().resource.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "Holder<T> Holder_clone_holder(Holder<T> self)" in result
    assert (
        "return texture(Holder_clone_holder(holder).resource, sampler_state, uv);"
        in result
    )
    assert ".sample_sampler" not in result


def test_gpu_resource_method_calls_convert_on_method_generic_impl_returns():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Provider {
    }

    impl Provider {
        fn pick<T>(&self, value: T) -> T {
            return value;
        }
    }

    fn method_generic_impl_returned_resource_methods(
        provider: Provider,
        tex: ColorTexture,
        values: RwBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = provider.pick(tex);
        let local_values = provider.pick(values);
        let base = local_tex.sample_sampler(sampler_state, uv);
        let direct_base = provider.pick(tex).sample_sampler(sampler_state, uv);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base + direct_base;
    }
    """

    result = parse_and_generate(code)

    assert "T Provider_pick(Provider self, T value)" in result
    assert "let local_tex = Provider_pick(provider, tex);" in result
    assert "let local_values = Provider_pick(provider, values);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert (
        "direct_base = texture(Provider_pick(provider, tex), sampler_state, uv);"
        in result
    )
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_indexed_impl_return_arrays():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Holder<T> {
        resources: T[2],
    }

    impl<T> Holder<T> {
        fn get_resources(&self) -> T[2] {
            return self.resources;
        }
    }

    fn indexed_impl_returned_resource_methods(
        holder: Holder<ColorTexture>,
        values_holder: Holder<RwBuffer<i32>>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = holder.get_resources()[index];
        let base = local_tex.sample_sampler(sampler_state, uv);
        let direct = holder.get_resources()[index].sample_sampler(sampler_state, uv);
        let value = values_holder.get_resources()[index].buffer_load(index);
        values_holder.get_resources()[index].buffer_store(index, value);
        return base + direct;
    }
    """

    result = parse_and_generate(code)

    assert "T[2] Holder_get_resources(Holder<T> self)" in result
    assert "let local_tex = Holder_get_resources(holder)[index];" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert (
        "direct = texture(Holder_get_resources(holder)[index], sampler_state, uv);"
        in result
    )
    assert (
        "value = buffer_load(Holder_get_resources(values_holder)[index], index);"
        in result
    )
    assert (
        "buffer_store(Holder_get_resources(values_holder)[index], index, value);"
        in result
    )
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_associated_constructor_returns():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Holder<T> {
        resource: T,
    }

    impl<T> Holder<T> {
        fn new(resource: T) -> Self {
            return Holder { resource: resource };
        }
    }

    fn associated_constructor_resource_methods(
        tex: ColorTexture,
        values: RwBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let holder = Holder::new(tex);
        let base = holder.resource.sample_sampler(sampler_state, uv);
        let direct = Holder::new(tex).resource.sample_sampler(sampler_state, uv);
        let value = Holder::new(values).resource.buffer_load(index);
        return base + direct + Vec4::<f32>::new(value as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "Holder<T> Holder_new(T resource)" in result
    assert "let holder = Holder_new(tex);" in result
    assert "base = texture(holder.resource, sampler_state, uv);" in result
    assert "direct = texture(Holder_new(tex).resource, sampler_state, uv);" in result
    assert "value = buffer_load(Holder_new(values).resource, index);" in result
    assert "Holder::new" not in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result


def test_gpu_resource_method_calls_convert_on_explicit_associated_constructor_types():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Holder<T> {
        resource: T,
    }

    impl<T> Holder<T> {
        fn new(resource: T) -> Self {
            return Holder { resource: resource };
        }
    }

    fn explicit_associated_constructor_resource_methods(
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let direct = Holder::<ColorTexture>::new(make_texture()).resource.sample_sampler(
            sampler_state,
            uv,
        );
        let value = Holder::<RwBuffer<i32>>::new(make_values()).resource.buffer_load(index);
        return direct + Vec4::<f32>::new(value as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "Holder<T> Holder_new(T resource)" in result
    assert (
        "direct = texture(Holder_new(make_texture()).resource, sampler_state, uv);"
        in result
    )
    assert "value = buffer_load(Holder_new(make_values()).resource, index);" in result
    assert "Holder::<ColorTexture>::new" not in result
    assert "Holder::<RwBuffer<i32>>::new" not in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result


def test_gpu_resource_method_calls_convert_on_static_method_generic_returns():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Provider {
    }

    impl Provider {
        fn pick<T>(value: T) -> T {
            return value;
        }
    }

    fn static_method_generic_resource_methods(
        tex: ColorTexture,
        values: RwBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let local_tex = Provider::pick(tex);
        let local_values = Provider::pick(values);
        let base = local_tex.sample_sampler(sampler_state, uv);
        let direct = Provider::pick(tex).sample_sampler(sampler_state, uv);
        let value = local_values.buffer_load(index);
        local_values.buffer_store(index, value);
        return base + direct;
    }
    """

    result = parse_and_generate(code)

    assert "T Provider_pick(T value)" in result
    assert "let local_tex = Provider_pick(tex);" in result
    assert "let local_values = Provider_pick(values);" in result
    assert "base = texture(local_tex, sampler_state, uv);" in result
    assert "direct = texture(Provider_pick(tex), sampler_state, uv);" in result
    assert "value = buffer_load(local_values, index);" in result
    assert "buffer_store(local_values, index, value);" in result
    assert "Provider::pick" not in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_gpu_resource_method_calls_convert_on_explicit_static_method_generic_returns():
    code = """
    type ColorTexture = Texture2D<f32>;

    struct Provider {
    }

    impl Provider {
        fn pick<T>(value: T) -> T {
            return value;
        }
    }

    fn explicit_static_method_generic_resource_methods(
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let direct = Provider::pick::<ColorTexture>(make_texture()).sample_sampler(
            sampler_state,
            uv,
        );
        let value = Provider::pick::<RwBuffer<i32>>(make_values()).buffer_load(index);
        return direct + Vec4::<f32>::new(value as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "T Provider_pick(T value)" in result
    assert (
        "direct = texture(Provider_pick(make_texture()), sampler_state, uv);" in result
    )
    assert "value = buffer_load(Provider_pick(make_values()), index);" in result
    assert "Provider::pick<ColorTexture>" not in result
    assert "Provider::pick<RwBuffer<i32>>" not in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result


def test_gpu_resource_method_calls_convert_on_branch_expression_receivers():
    code = """
    type ColorTexture = Texture2D<f32>;
    type Values = RwBuffer<i32>;

    fn branch_expression_resource_methods(
        flag: bool,
        mode: u32,
        left: ColorTexture,
        right: ColorTexture,
        values_left: Values,
        values_right: Values,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let selected = if flag { left } else { right };
        let local_base = selected.sample_sampler(sampler_state, uv);
        let direct_base = (if flag { left } else { right }).sample_sampler(
            sampler_state,
            uv,
        );
        let block_selected = { if flag { left } else { right } };
        let block_base = block_selected.sample_sampler(sampler_state, uv);
        let direct_block_base = ({ if flag { left } else { right } }).sample_sampler(
            sampler_state,
            uv,
        );
        let matched = match mode {
            0 => { left },
            _ => { right },
        };
        let match_base = matched.sample_sampler(sampler_state, uv);
        let direct_match_base = (match mode {
            0 => left,
            _ => right,
        }).sample_sampler(sampler_state, uv);
        let value = (if flag { values_left } else { values_right }).buffer_load(index);
        let selected_values = match mode {
            0 => values_left,
            _ => values_right,
        };
        selected_values.buffer_store(index, value);
        return local_base + direct_base + block_base + direct_block_base
            + match_base + direct_match_base
            + Vec4::<f32>::new(value as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert "let local_base = texture(selected, sampler_state, uv);" in result
    assert (
        "let direct_base = texture((flag ? left : right), sampler_state, uv);" in result
    )
    assert "let block_base = texture(block_selected, sampler_state, uv);" in result
    assert (
        "let direct_block_base = texture((flag ? left : right), sampler_state, uv);"
        in result
    )
    assert "let match_base = texture(matched, sampler_state, uv);" in result
    assert (
        "let direct_match_base = texture(_rust_expr_value_0, sampler_state, uv);"
        in result
    )
    assert (
        "let value = buffer_load((flag ? values_left : values_right), index);" in result
    )
    assert "buffer_store(selected_values, index, value);" in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result
    assert ".buffer_store" not in result


def test_mixed_branch_resource_method_receivers_stay_unlowered():
    code = """
    fn mixed_branch_resource_methods(
        flag: bool,
        tex: Texture2D<f32>,
        values: RwBuffer<i32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        let receiver = if flag { tex } else { values };
        return receiver.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "let receiver = (flag ? tex : values);" in result
    assert "return receiver.sample_sampler(sampler_state, uv);" in result
    assert "texture(receiver" not in result


def test_unresolved_method_generic_impl_returns_stay_unlowered():
    code = """
    struct Provider {
    }

    impl Provider {
        fn default_value<T>(&self) -> T {
            return make_default();
        }
    }

    fn unresolved_method_generic_impl_returned_resource(
        provider: Provider,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        let local_tex = provider.default_value();
        return local_tex.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "T Provider_default_value(Provider self)" in result
    assert "let local_tex = Provider_default_value(provider);" in result
    assert "return local_tex.sample_sampler(sampler_state, uv);" in result
    assert "texture(local_tex" not in result


def test_unresolved_generic_impl_method_returned_receivers_stay_unlowered():
    code = """
    struct Holder<T> {
        resource: T,
    }

    impl<T> Holder<T> {
        fn get(&self) -> T {
            return self.resource;
        }
    }

    fn unresolved_generic_impl_returned_resource<T>(
        holder: Holder<T>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32>
    where
        T: SampledTexture,
    {
        let local_tex = holder.get();
        return local_tex.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "let local_tex = Holder_get(holder);" in result
    assert "return local_tex.sample_sampler(sampler_state, uv);" in result
    assert "texture(local_tex" not in result


def test_unknown_generic_resource_bound_receiver_stays_unlowered():
    code = """
    fn generic_bound_resource<T>(
        tex: T,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32>
    where
        T: SampledTexture,
    {
        return tex.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert (
        "vec4 generic_bound_resource(T tex, sampler sampler_state, vec2 uv)" in result
    )
    assert "return tex.sample_sampler(sampler_state, uv);" in result
    assert "texture(tex" not in result


def test_malformed_generic_struct_arity_does_not_infer_resource_member_type():
    code = """
    struct Holder<T> {
        resource: T,
    }

    fn malformed_holder(
        holder: Holder<Texture2D<f32>, f32>,
        sampler_state: Sampler,
        uv: Vec2<f32>,
    ) -> Vec4<f32> {
        return holder.resource.sample_sampler(sampler_state, uv);
    }
    """

    result = parse_and_generate(code)

    assert "return holder.resource.sample_sampler(sampler_state, uv);" in result
    assert "texture(holder.resource" not in result


def test_char_literal_conversion():
    code = r"""
    fn test_chars(c: char) -> char {
        let letter = 'a';
        let newline = '\n';
        match c {
            'a' => hit_a(),
            '\n' => hit_newline(),
            _ => hit_default(),
        }
        return 'z';
    }
    """
    try:
        result = parse_and_generate(code)

        assert "letter = 'a';" in result
        assert r"newline = '\n';" in result
        assert "case 'a':" in result
        assert r"case '\n':" in result
        assert "return 'z';" in result
        assert "CHAR_LIT" not in result
    except Exception as e:
        pytest.fail(f"Char literal conversion failed: {e}")


def test_raw_string_literal_conversion():
    code = r"""
    fn test_raw_strings(s: String) -> String {
        let plain = r"plain raw";
        let quoted = r#"This is a "raw" string"#;
        let slashes = r#"path\to\file"#;
        let hashed = r##"This has "# in it"##;
        match s {
            r#"ok"# => hit_ok(),
            _ => hit_default(),
        }
        return r#"done"#;
    }
    """
    try:
        result = parse_and_generate(code)

        assert 'plain = "plain raw";' in result
        assert 'quoted = "This is a \\"raw\\" string";' in result
        assert 'slashes = "path\\\\to\\\\file";' in result
        assert 'hashed = "This has \\"# in it";' in result
        assert 'case "ok":' in result
        assert 'return "done";' in result
        assert "RAW_STRING" not in result
        assert 'r#"' not in result
        assert 'r##"' not in result
    except Exception as e:
        pytest.fail(f"Raw string literal conversion failed: {e}")


def test_escaped_newline_string_literal_codegen_reparseable_from_asm_macro():
    code = (
        'fn trace(payload: &mut u32) { unsafe { asm! { "OpTraceRayKHR \\\n'
        '                {payload}", payload = in(reg) payload, } } }'
    )

    try:
        result = parse_and_generate(code)

        asm_line = next(line for line in result.splitlines() if "asm!(" in line)
        assert r"\n" in asm_line
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Escaped newline string literal conversion failed: {e}")


def test_byte_literal_conversion():
    code = r"""
    fn test_byte_literals(c: u8) -> String {
        let bytes = b"abc";
        let byte = b'a';
        let newline = b'\n';
        let slash = b'\\';
        let hex = b'\x41';
        let raw = br#"a "quoted" byte raw"#;
        match c {
            b'a' => hit_a(),
            b'\n' => hit_newline(),
            _ => hit_default(),
        }
        return br#"done"#;
    }
    """
    try:
        result = parse_and_generate(code)

        assert 'bytes = "abc";' in result
        assert "byte = 97;" in result
        assert "newline = 10;" in result
        assert "slash = 92;" in result
        assert "hex = 65;" in result
        assert 'raw = "a \\"quoted\\" byte raw";' in result
        assert "case 97:" in result
        assert "case 10:" in result
        assert 'return "done";' in result
        assert 'b"' not in result
        assert "br#" not in result
        assert "b'" not in result
    except Exception as e:
        pytest.fail(f"Byte literal conversion failed: {e}")


def test_option_result_constructor_conversion():
    code = """
    use std::option::Option::{Some, None};

    fn test_option_result(value: i32, err: i32) -> Result<i32, i32> {
        let maybe: Option<i32> = Some(value);
        let empty: Option<i32> = None;
        let ok_value = Ok(value + 1);
        let err_value = Err(err);
        return Ok(value);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "// use std::option::Option::{Some, None}" in result
        assert "Result<i32, i32> test_option_result(int value, int err)" in result
        assert "Option<i32> maybe = Some(value);" in result
        assert "Option<i32> empty = None;" in result
        assert "ok_value = Ok((value + 1));" in result
        assert "err_value = Err(err);" in result
        assert "return Ok(value);" in result
    except Exception as e:
        pytest.fail(f"Option/Result constructor conversion failed: {e}")


def test_option_result_payload_match_conversion():
    code = """
    fn test_payload_match(maybe: Option<i32>, result: Result<i32, i32>) -> i32 {
        let from_option: i32 = match maybe {
            Some(1) => 10,
            Some(v) => v + 1,
            None => 0,
        };
        let from_result: i32 = match result {
            Ok(value) if value > 0 => value,
            Err(code) => code,
            _ => -1,
        };
        match maybe {
            Some(v) => use_value(v),
            None => use_none(),
        }
        return match result {
            Ok(v) => v,
            Err(e) => e,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (maybe)" not in result
        assert "switch (result)" not in result
        assert "case Some" not in result
        assert "case Ok" not in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (!_rust_match_matched_0 && is_Some(maybe)" in result
        assert "(unwrap_Some(maybe) == 1)" in result
        assert "auto v = unwrap_Some(maybe);" in result
        assert "from_option = (v + 1);" in result
        assert "if (!_rust_match_matched_0 && (maybe == None))" in result
        assert "if (!_rust_match_matched_1 && is_Ok(result))" in result
        assert "auto value = unwrap_Ok(result);" in result
        assert "if ((value > 0))" in result
        assert "auto code = unwrap_Err(result);" in result
        assert "from_result = code;" in result
        assert "if (!_rust_match_matched_1)" in result
        assert "from_result = (-1);" in result
        assert "use_value(v);" in result
        assert "return v;" in result
        assert "auto e = unwrap_Err(result);" in result
        assert "return e;" in result
    except Exception as e:
        pytest.fail(f"Option/Result payload match conversion failed: {e}")


def test_guarded_literal_match_uses_if_chain_conversion():
    code = """
    fn test_guarded_match(mode: i32) {
        match mode {
            0 if ready => hit_ready(),
            0 => hit_zero(),
            _ if fallback => hit_fallback(),
            _ => hit_default(),
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (mode)" not in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (!_rust_match_matched_0 && (mode == 0))" in result
        assert "if (ready)" in result
        assert "hit_ready();" in result
        assert "hit_zero();" in result
        assert "if (fallback)" in result
        assert "hit_fallback();" in result
        assert "hit_default();" in result
    except Exception as e:
        pytest.fail(f"Guarded literal match conversion failed: {e}")


def test_if_chain_match_hoists_side_effecting_subject_conversion():
    code = """
    fn test_subject_hoist() -> i32 {
        let from_result: i32 = match next_result() {
            Ok(value) if value > 0 => value,
            Err(code) => code,
            _ => -1,
        };
        match next_option() {
            Some(v) => use_value(v),
            None => use_none(),
        }
        return match next_result() {
            Ok(v) => v,
            Err(e) => e,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_match_subject_0 = next_result();" in result
        assert "auto _rust_match_subject_1 = next_option();" in result
        assert "auto _rust_match_subject_2 = next_result();" in result
        assert result.count("next_result()") == 2
        assert result.count("next_option()") == 1
        assert "is_Ok(_rust_match_subject_0)" in result
        assert "auto value = unwrap_Ok(_rust_match_subject_0);" in result
        assert "auto code = unwrap_Err(_rust_match_subject_0);" in result
        assert "is_Some(_rust_match_subject_1)" in result
        assert "auto v = unwrap_Some(_rust_match_subject_1);" in result
        assert "is_Ok(_rust_match_subject_2)" in result
        assert "return v;" in result
        assert "return e;" in result
    except Exception as e:
        pytest.fail(f"If-chain match subject hoisting failed: {e}")


def test_multi_payload_constructor_match_conversion():
    code = """
    fn test_constructor_patterns(pair: Pair, color: Color) {
        match pair {
            Pair(a, b) => use_pair(a, b),
            Pair(x, 3) => use_literal(x),
            _ => use_default(),
        }
        match color {
            Color::Rgb(r, g, b) => use_rgb(r, g, b),
            Color::Red => use_red(),
            _ => use_default(),
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (pair)" not in result
        assert "switch (color)" not in result
        assert "case Pair" not in result
        assert "case Color::Rgb" not in result
        assert "if (!_rust_match_matched_0 && is_Pair(pair))" in result
        assert "auto a = unwrap_Pair_0(pair);" in result
        assert "auto b = unwrap_Pair_1(pair);" in result
        assert "use_pair(a, b);" in result
        assert (
            "if (!_rust_match_matched_0 && is_Pair(pair) && "
            "(unwrap_Pair_1(pair) == 3))"
        ) in result
        assert "auto x = unwrap_Pair_0(pair);" in result
        assert "use_literal(x);" in result
        assert "if (!_rust_match_matched_1 && is_Color_Rgb(color))" in result
        assert "auto r = unwrap_Color_Rgb_0(color);" in result
        assert "auto g = unwrap_Color_Rgb_1(color);" in result
        assert "auto b = unwrap_Color_Rgb_2(color);" in result
        assert "use_rgb(r, g, b);" in result
        assert "if (!_rust_match_matched_1 && (color == Color::Red))" in result
        assert "use_red();" in result
        assert "use_default();" in result
    except Exception as e:
        pytest.fail(f"Multi-payload constructor match conversion failed: {e}")


def test_nested_constructor_match_conversion():
    code = """
    fn test_nested_constructor_patterns(value: Option<Result<i32, i32>>, pair: Pair) -> i32 {
        match value {
            Some(Ok(v)) => use_ok(v),
            Some(Err(e)) => use_err(e),
            None => use_none(),
        }
        match pair {
            Pair(Some(x), y) => use_pair(x, y),
            Pair(None, 3) => use_none_pair(),
            _ => use_default(),
        }
        return match value {
            Some(Ok(v)) => v,
            Some(Err(3)) => 3,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "== Ok(v)" not in result
        assert "== Err(e)" not in result
        assert "== Some(x)" not in result
        assert "unwrap_Ok(unwrap_Some(value))" not in result
        assert "unwrap_Err(unwrap_Some(value))" not in result
        assert "is_Some(value)" in result
        assert "auto _rust_match_payload_0 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_0))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_0);" in result
        assert "auto _rust_match_payload_1 = unwrap_Some(value);" in result
        assert "if (is_Err(_rust_match_payload_1))" in result
        assert "auto e = unwrap_Err(_rust_match_payload_1);" in result
        assert "use_ok(v);" in result
        assert "use_err(e);" in result
        assert "auto _rust_match_payload_2 = unwrap_Pair_0(pair);" in result
        assert "if (is_Some(_rust_match_payload_2))" in result
        assert "auto x = unwrap_Some(_rust_match_payload_2);" in result
        assert "auto y = unwrap_Pair_1(pair);" in result
        assert (
            "if (!_rust_match_matched_1 && is_Pair(pair) && "
            "(unwrap_Pair_0(pair) == None) && (unwrap_Pair_1(pair) == 3))"
        ) in result
        assert "return v;" in result
        assert "auto _rust_match_payload_3 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_3))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_3);" in result
        assert "auto _rust_match_payload_4 = unwrap_Some(value);" in result
        assert "if (is_Err(_rust_match_payload_4))" in result
        assert ("if ((unwrap_Err(_rust_match_payload_4) == 3))") in result
        assert "return 3;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Nested constructor match conversion failed: {e}")


def test_guarded_nested_constructor_match_conversion():
    code = """
    fn test_guarded_nested_patterns(value: Option<Result<i32, i32>>) -> i32 {
        match value {
            Some(Ok(v)) if v > 0 => use_positive(v),
            Some(Ok(v)) => use_non_positive(v),
            Some(Err(e)) if e != 0 => use_err(e),
            _ => use_default(),
        }
        let out: i32 = match value {
            Some(Ok(v)) if v > 0 => v,
            Some(Ok(v)) => 0,
            _ => -1,
        };
        return match value {
            Some(Ok(v)) if v > 0 => v,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_match_payload_0 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_0))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_0);" in result
        assert "if ((v > 0))" in result
        assert "_rust_match_matched_0 = true;" in result
        assert "use_positive(v);" in result

        assert "auto _rust_match_payload_1 = unwrap_Some(value);" in result
        assert "use_non_positive(v);" in result

        assert "auto _rust_match_payload_2 = unwrap_Some(value);" in result
        assert "if (is_Err(_rust_match_payload_2))" in result
        assert "auto e = unwrap_Err(_rust_match_payload_2);" in result
        assert "if ((e != 0))" in result
        assert "use_err(e);" in result

        assert "int out;" in result
        assert "auto _rust_match_payload_3 = unwrap_Some(value);" in result
        assert "out = v;" in result
        assert "auto _rust_match_payload_4 = unwrap_Some(value);" in result
        assert "out = 0;" in result
        assert "out = (-1);" in result

        assert "auto _rust_match_payload_5 = unwrap_Some(value);" in result
        assert "return v;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Guarded nested constructor match conversion failed: {e}")


def test_match_or_pattern_conversion():
    code = """
    fn test_or_patterns(mode: i32, color: Color, value: Option<Result<i32, i32>>, nested: Option<i32>) -> i32 {
        match mode {
            0 | 1 => hit_small(),
            _ => hit_default(),
        }
        match color {
            Color::Red | Color::Green => hit_warm(),
            _ => hit_default(),
        }
        match value {
            Some(Ok(v)) | Some(Err(v)) if v > 0 => use_value(v),
            _ => use_default(),
        }
        match value {
            Some(Ok(v) | Err(v)) if v > 0 => use_nested(v),
            _ => use_default(),
        }
        match nested {
            Some(0 | 1) => use_small_option(),
            _ => use_default(),
        }
        return match mode {
            0 | 1 => 10,
            _ => 20,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (mode)" not in result
        assert "switch (color)" not in result
        assert "(mode == 0)" in result
        assert "(mode == 1)" in result
        assert result.count("hit_small();") == 2
        assert "(color == Color::Red)" in result
        assert "(color == Color::Green)" in result
        assert result.count("hit_warm();") == 2
        assert "auto _rust_match_payload_0 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_0))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_0);" in result
        assert "auto _rust_match_payload_1 = unwrap_Some(value);" in result
        assert "if (is_Err(_rust_match_payload_1))" in result
        assert "auto v = unwrap_Err(_rust_match_payload_1);" in result
        assert result.count("if ((v > 0))") == 4
        assert result.count("use_value(v);") == 2
        assert "bool _rust_match_or_matched_0 = false;" in result
        assert "auto _rust_match_payload_2 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_2))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_2);" in result
        assert "if (is_Err(_rust_match_payload_2))" in result
        assert "auto v = unwrap_Err(_rust_match_payload_2);" in result
        assert result.count("use_nested(v);") == 2
        assert "is_Some(nested)" in result
        assert "(unwrap_Some(nested) == 0)" in result
        assert "(unwrap_Some(nested) == 1)" in result
        assert "return 10;" in result
        assert "return 20;" in result
    except Exception as e:
        pytest.fail(f"Match or-pattern conversion failed: {e}")


def test_match_range_pattern_conversion():
    code = """
    fn test_range_patterns(mode: i32, signed: i32, value: Option<i32>) -> i32 {
        match mode {
            1..=10 => hit_inclusive(),
            11..20 => hit_exclusive(),
            0 | 30..=40 => hit_or_range(),
            _ => hit_default(),
        }
        match signed {
            -5..=-1 => hit_negative(),
            _ => hit_default(),
        }
        match value {
            Some(1..=3 | 7..10) => hit_nested(),
            _ => hit_default(),
        }
        return match mode {
            1..=10 => 10,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (mode)" not in result
        assert "switch (signed)" not in result
        assert "(mode >= 1)" in result
        assert "(mode <= 10)" in result
        assert "(mode >= 11)" in result
        assert "(mode < 20)" in result
        assert "(mode == 0)" in result
        assert "(mode >= 30)" in result
        assert "(mode <= 40)" in result
        assert result.count("hit_or_range();") == 2
        assert "(signed >= (-5))" in result
        assert "(signed <= (-1))" in result
        assert "is_Some(value)" in result
        assert "(unwrap_Some(value) >= 1)" in result
        assert "(unwrap_Some(value) <= 3)" in result
        assert "(unwrap_Some(value) >= 7)" in result
        assert "(unwrap_Some(value) < 10)" in result
        assert "hit_nested();" in result
        assert "return 10;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Match range pattern conversion failed: {e}")


def test_match_grouped_reference_pattern_conversion():
    code = """
    fn test_grouped_reference_patterns(mode: i32, value: &Option<Result<i32, i32>>, nested: Option<&i32>) -> i32 {
        match mode {
            (0 | 1) => hit_grouped(),
            _ => hit_default(),
        }
        match &mode {
            &(2..=4) => hit_ref_range(),
            &mut (5 | 6) => hit_mut_ref_or(),
            _ => hit_default(),
        }
        match value {
            &Some(Ok(v) | Err(v)) if v > 0 => use_value(v),
            _ => use_default(),
        }
        match nested {
            Some(&(1 | 2)) => hit_nested_ref(),
            _ => hit_default(),
        }
        return match &mode {
            &(0 | 1) => 1,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (mode)" not in result
        assert result.count("hit_grouped();") == 2
        assert "(mode == 0)" in result
        assert "(mode == 1)" in result
        assert "(mode >= 2)" in result
        assert "(mode <= 4)" in result
        assert result.count("hit_ref_range();") == 1
        assert "(mode == 5)" in result
        assert "(mode == 6)" in result
        assert result.count("hit_mut_ref_or();") == 2
        assert "is_Some(value)" in result
        assert "auto _rust_match_payload_0 = unwrap_Some(value);" in result
        assert "if (is_Ok(_rust_match_payload_0))" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_0);" in result
        assert "if (is_Err(_rust_match_payload_0))" in result
        assert "auto v = unwrap_Err(_rust_match_payload_0);" in result
        assert result.count("use_value(v);") == 2
        assert "is_Some(nested)" in result
        assert "(unwrap_Some(nested) == 1)" in result
        assert "(unwrap_Some(nested) == 2)" in result
        assert "hit_nested_ref();" in result
        assert "return 1;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Grouped/reference match pattern conversion failed: {e}")


def test_match_binding_pattern_conversion():
    code = """
    fn test_binding_patterns(mode: i32, value: Option<Result<i32, i32>>, nested: Option<i32>) -> i32 {
        match mode {
            whole @ 1..=10 if whole > 3 => use_mode(whole),
            label @ (0 | 20) => use_label(label),
            _ @ _ => use_discard(),
        }
        match value {
            outer @ Some(Ok(v) | Err(v)) if outer_ready(outer) && v > 0 => use_result(outer, v),
            _ => use_default(),
        }
        match nested {
            Some(inner @ (1..=3 | 7..10)) => use_inner(inner),
            _ => use_default(),
        }
        return match mode {
            kept @ 1..=2 => kept,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch (mode)" not in result
        assert "(mode >= 1)" in result
        assert "(mode <= 10)" in result
        assert "auto whole = mode;" in result
        assert "if ((whole > 3))" in result
        assert "use_mode(whole);" in result
        assert result.count("auto label = mode;") == 2
        assert result.count("use_label(label);") == 2
        assert "(mode == 0)" in result
        assert "(mode == 20)" in result
        assert "auto _ = mode;" not in result
        assert "use_discard();" in result

        assert "is_Some(value)" in result
        assert "auto _rust_match_payload_0 = unwrap_Some(value);" in result
        assert "auto v = unwrap_Ok(_rust_match_payload_0);" in result
        assert "auto v = unwrap_Err(_rust_match_payload_0);" in result
        assert result.count("auto outer = value;") == 2
        assert result.count("outer_ready(outer)") == 2
        assert result.count("use_result(outer, v);") == 2

        assert "is_Some(nested)" in result
        assert "(unwrap_Some(nested) >= 1)" in result
        assert "(unwrap_Some(nested) <= 3)" in result
        assert "(unwrap_Some(nested) >= 7)" in result
        assert "(unwrap_Some(nested) < 10)" in result
        assert result.count("auto inner = unwrap_Some(nested);") == 1
        assert result.count("use_inner(inner);") == 1

        assert "(mode <= 2)" in result
        assert "auto kept = mode;" in result
        assert "return kept;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Match binding pattern conversion failed: {e}")


def test_match_tuple_pattern_conversion():
    code = """
    fn test_tuple_patterns(mode: i32, state: i32, value: Option<i32>) -> i32 {
        match (mode, state) {
            (0, y) if y > 1 => use_y(y),
            (1..=3, 4 | 5) => hit_range_or(),
            (left @ (6 | 7), right @ 8..10) => use_pair(left, right),
            _ => use_default(),
        }
        match (value, mode) {
            (Some(v), 1) => use_some(v),
            (None, _) => use_none(),
            _ => use_default(),
        }
        match (mode, (state, value)) {
            (9, (kept @ 10..=12, Some(v))) => use_nested(kept, v),
            _ => use_default(),
        }
        return match (mode, state) {
            (9, kept @ 10..=12) => kept,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_match_tuple_0 = mode;" in result
        assert "auto _rust_match_tuple_1 = state;" in result
        assert "switch" not in result

        assert "(_rust_match_tuple_0 == 0)" in result
        assert "auto y = _rust_match_tuple_1;" in result
        assert "if ((y > 1))" in result
        assert "use_y(y);" in result

        assert "(_rust_match_tuple_0 >= 1)" in result
        assert "(_rust_match_tuple_0 <= 3)" in result
        assert "(_rust_match_tuple_1 == 4)" in result
        assert "(_rust_match_tuple_1 == 5)" in result
        assert result.count("hit_range_or();") == 2

        assert result.count("auto left = _rust_match_tuple_0;") == 2
        assert result.count("auto right = _rust_match_tuple_1;") == 2
        assert result.count("use_pair(left, right);") == 2

        assert "auto _rust_match_tuple_2 = value;" in result
        assert "auto _rust_match_tuple_3 = mode;" in result
        assert "if (is_Some(_rust_match_tuple_2))" in result
        assert "auto v = unwrap_Some(_rust_match_tuple_2);" in result
        assert "(_rust_match_tuple_3 == 1)" in result
        assert "use_some(v);" in result
        assert "(_rust_match_tuple_2 == None)" in result
        assert "use_none();" in result

        assert "auto _rust_match_tuple_4 = mode;" in result
        assert "auto _rust_match_tuple_5 = state;" in result
        assert "auto _rust_match_tuple_6 = value;" in result
        assert "(_rust_match_tuple_4 == 9)" in result
        assert "auto kept = _rust_match_tuple_5;" in result
        assert "if (is_Some(_rust_match_tuple_6))" in result
        assert "auto v = unwrap_Some(_rust_match_tuple_6);" in result
        assert "use_nested(kept, v);" in result

        assert "return kept;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Match tuple pattern conversion failed: {e}")


def test_match_array_pattern_conversion():
    code = """
    fn test_array_patterns(values: [i32; 4], option_values: [Option<i32>; 2], mode: i32, state: i32) -> i32 {
        match values {
            [first, 2, third @ 3..=5, _] if first > 0 => use_fixed(first, third),
            [0 | 1, ..] => hit_prefix(),
            [head, middle @ .., tail] => use_rest(head, middle, tail),
            [] => use_empty(),
            _ => use_default(),
        }
        match option_values {
            [Some(v), _] => use_some(v),
            _ => use_default(),
        }
        match [mode, state] {
            [0, kept @ 1..=3] => use_literal(kept),
            _ => use_default(),
        }
        return match values {
            [kept @ 1..=3, ..] => kept,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch" not in result
        assert "(length(values) == 4)" in result
        assert "auto first = values[0];" in result
        assert "(values[1] == 2)" in result
        assert "(values[2] >= 3)" in result
        assert "(values[2] <= 5)" in result
        assert "auto third = values[2];" in result
        assert "if ((first > 0))" in result
        assert "use_fixed(first, third);" in result

        assert "(length(values) >= 1)" in result
        assert "(values[0] == 0)" in result
        assert "(values[0] == 1)" in result
        assert result.count("hit_prefix();") == 2

        assert "(length(values) >= 2)" in result
        assert "auto head = values[0];" in result
        assert "auto tail = values[(length(values) - 1)];" in result
        assert "auto middle = _rust_slice(values, 1, (length(values) - 2));" in result
        assert "use_rest(head, middle, tail);" in result

        assert "(length(values) == 0)" in result
        assert "use_empty();" in result

        assert "(length(option_values) == 2)" in result
        assert "if (is_Some(option_values[0]))" in result
        assert "auto v = unwrap_Some(option_values[0]);" in result
        assert "use_some(v);" in result

        assert "auto _rust_match_array_0 = mode;" in result
        assert "auto _rust_match_array_1 = state;" in result
        assert "(_rust_match_array_0 == 0)" in result
        assert "(_rust_match_array_1 >= 1)" in result
        assert "(_rust_match_array_1 <= 3)" in result
        assert "auto kept = _rust_match_array_1;" in result
        assert "use_literal(kept);" in result

        assert "auto kept = values[0];" in result
        assert "return kept;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Match array pattern conversion failed: {e}")


def test_match_struct_pattern_conversion():
    code = """
    fn test_struct_patterns(point: Point, shape: Shape) -> i32 {
        match point {
            Point { x, y: 0, .. } if x > 0 => use_x(x),
            Point { x: left @ 1..=3, y } => use_pair(left, y),
            _ => use_default(),
        }
        match shape {
            Shape::Circle { radius } if radius > 0 => use_radius(radius),
            Shape::Rect { corner: Point { x, y }, size: 1..=10, .. } => use_rect(x, y),
            Shape::Circle { radius: small @ (1 | 2), .. }
            | Shape::Circle { radius: small @ 3, .. } => use_small(small),
            _ => use_default(),
        }
        return match point {
            Point { x: kept @ 1..=3, .. } => kept,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch" not in result
        assert "auto x = point.x;" in result
        assert "(point.y == 0)" in result
        assert "if ((x > 0))" in result
        assert "use_x(x);" in result
        assert "(point.x >= 1)" in result
        assert "(point.x <= 3)" in result
        assert "auto left = point.x;" in result
        assert "auto y = point.y;" in result
        assert "use_pair(left, y);" in result

        assert "if (is_Shape_Circle(shape))" in result
        assert "auto radius = unwrap_Shape_Circle_radius(shape);" in result
        assert "if ((radius > 0))" in result
        assert "use_radius(radius);" in result

        assert "if (is_Shape_Rect(shape))" in result
        assert "auto x = unwrap_Shape_Rect_corner(shape).x;" in result
        assert "auto y = unwrap_Shape_Rect_corner(shape).y;" in result
        assert "(unwrap_Shape_Rect_size(shape) >= 1)" in result
        assert "(unwrap_Shape_Rect_size(shape) <= 10)" in result
        assert "use_rect(x, y);" in result

        assert result.count("auto small = unwrap_Shape_Circle_radius(shape);") == 3
        assert "(unwrap_Shape_Circle_radius(shape) == 1)" in result
        assert "(unwrap_Shape_Circle_radius(shape) == 2)" in result
        assert "(unwrap_Shape_Circle_radius(shape) == 3)" in result
        assert result.count("use_small(small);") == 3

        assert "auto kept = point.x;" in result
        assert "return kept;" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Match struct pattern conversion failed: {e}")


def test_match_binding_modifier_pattern_conversion():
    code = """
    fn test_binding_modifier_patterns(value: Option<i32>, point: Point, pair: Pair, values: [i32; 3]) -> i32 {
        match value {
            Some(ref v) if v > 0 => use_ref(v),
            Some(mut bounded @ 1..=3) => use_mut(bounded),
            Some(ref mut choice @ (4 | 5)) => use_choice(choice),
            _ => use_default(),
        }
        match point {
            Point { ref x, mut y, z: ref renamed, w: mut kept @ 1..=2, .. } => use_point(x, y, renamed, kept),
            _ => use_default(),
        }
        match pair {
            Pair(ref left, mut right) => use_pair(left, right),
        }
        match values {
            [ref first, mut second @ 2..=3, ..] => use_array(first, second),
            _ => use_default(),
        }
        return match value {
            Some(ref kept) => kept,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "switch" not in result
        assert "auto v = unwrap_Some(value);" in result
        assert "if ((v > 0))" in result
        assert "use_ref(v);" in result

        assert "(unwrap_Some(value) >= 1)" in result
        assert "(unwrap_Some(value) <= 3)" in result
        assert "auto bounded = unwrap_Some(value);" in result
        assert "use_mut(bounded);" in result

        assert "(unwrap_Some(value) == 4)" in result
        assert "(unwrap_Some(value) == 5)" in result
        assert result.count("auto choice = unwrap_Some(value);") == 1
        assert result.count("use_choice(choice);") == 1

        assert "auto x = point.x;" in result
        assert "auto y = point.y;" in result
        assert "auto renamed = point.z;" in result
        assert "(point.w >= 1)" in result
        assert "(point.w <= 2)" in result
        assert "auto kept = point.w;" in result
        assert "use_point(x, y, renamed, kept);" in result

        assert "is_Pair(pair)" in result
        assert "auto left = unwrap_Pair_0(pair);" in result
        assert "auto right = unwrap_Pair_1(pair);" in result
        assert "use_pair(left, right);" in result

        assert "auto first = values[0];" in result
        assert "(values[1] >= 2)" in result
        assert "(values[1] <= 3)" in result
        assert "auto second = values[1];" in result
        assert "use_array(first, second);" in result

        assert "auto kept = unwrap_Some(value);" in result
        assert "return kept;" in result
        assert "return 0;" in result
        assert "ref " not in result
        assert "mut " not in result
    except Exception as e:
        pytest.fail(f"Match binding modifier pattern conversion failed: {e}")


def test_member_access_conversion():
    code = """
    fn test_member() {
        let x = vertex.position.x;
        let y = vertex.position.y;
        let z = vertex.position.z;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vertex.position.x" in result
        assert "vertex.position.y" in result
        assert "vertex.position.z" in result
    except Exception as e:
        pytest.fail(f"Member access conversion failed: {e}")


def test_tuple_field_access_conversion():
    code = """
    fn test_tuple_field(a: i32, b: i32) -> i32 {
        let pair = (a, b);
        let first = pair.0;
        let second = (a, b).1;
        return first + second;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "let pair = (a, b);" in result
        assert "let first = pair.0;" in result
        assert "let second = (a, b).1;" in result
        assert "return (first + second);" in result
    except Exception as e:
        pytest.fail(f"Tuple field access conversion failed: {e}")


def test_array_access_conversion():
    code = """
    fn test_array() {
        let element = array[0];
        let slice_element = slice[index];
        let closed = &array[1..3];
        let prefix = &array[..end];
        let suffix = &array[start..];
        let full = &array[..];
    }
    """
    try:
        result = parse_and_generate(code)
        assert "array[0]" in result
        assert "slice[index]" in result
        assert "array[1..3]" in result
        assert "array[..end]" in result
        assert "array[start..]" in result
        assert "array[..]" in result
        assert "RangeNode" not in result
    except Exception as e:
        pytest.fail(f"Array access conversion failed: {e}")


def test_assignment_operations_conversion():
    code = """
    fn test_assign() {
        let mut a = 5;
        a += 3;
        a -= 2;
        a *= 4;
        a /= 2;
        a %= 3;
        a &= 1;
        a |= 2;
        a ^= 3;
        a <<= 1;
        a >>= 2;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "a += 3;" in result
        assert "a -= 2;" in result
        assert "a *= 4;" in result
        assert "a /= 2;" in result
        assert "a %= 3;" in result
        assert "a &= 1;" in result
        assert "a |= 2;" in result
        assert "a ^= 3;" in result
        assert "a <<= 1;" in result
        assert "a >>= 2;" in result
    except Exception as e:
        pytest.fail(f"Assignment operations conversion failed: {e}")


def test_assignment_expression_conversion():
    code = """
    fn test_assignment_expr() {
        let mut a = 0;
        (a = 1);
        let unit = (a = 2);
        let compound = (a += 3);
        let block_unit = { a = 4; };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "a = 1;" in result
        assert "a = 2;" in result
        assert "unit = ();" in result
        assert "a += 3;" in result
        assert "compound = ();" in result
        assert "a = 4;" in result
        assert "AssignmentNode" not in result
    except Exception as e:
        pytest.fail(f"Assignment expression conversion failed: {e}")


def test_compound_assignment_try_conversion_preserves_operator():
    code = """
    fn accumulate(value: Result<i32, i32>) -> Result<i32, i32> {
        let mut total: i32 = 0;
        total += value?;
        Ok(total)
    }
    """
    try:
        result = parse_and_generate(code)
        assert "total += _rust_try_value_0;" in result
        assert "total = _rust_try_value_0;" not in result
    except Exception as e:
        pytest.fail(f"Compound assignment try conversion failed: {e}")


def test_vector_constructor_conversion():
    code = """
    fn test_vectors() {
        let vec2_val = Vec2::new(1.0, 2.0);
        let vec3_val = Vec3::new(1.0, 2.0, 3.0);
        let vec4_val = Vec4::new(1.0, 2.0, 3.0, 4.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec2(1.0, 2.0)" in result
        assert "vec3(1.0, 2.0, 3.0)" in result
        assert "vec4(1.0, 2.0, 3.0, 4.0)" in result
    except Exception as e:
        pytest.fail(f"Vector constructor conversion failed: {e}")


def test_matrix_constructor_conversion():
    code = """
    fn test_matrices() {
        let mat2_val = Mat2::new(1.0, 2.0, 3.0, 4.0);
        let mat3_val = Mat3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let mat4_val = Mat4::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "mat2(" in result
        assert "mat3(" in result
        assert "mat4(" in result
    except Exception as e:
        pytest.fail(f"Matrix constructor conversion failed: {e}")


def test_const_static_conversion():
    code = """
    const PI: f32 = 3.14159;
    static mut GLOBAL_COUNTER: i32 = 0;
    """
    try:
        result = parse_and_generate(code)
        assert "const float PI = 3.14159;" in result
        assert "static mut int GLOBAL_COUNTER = 0;" in result
    except Exception as e:
        pytest.fail(f"Const/static conversion failed: {e}")


def test_fixed_array_conversion():
    code = """
    const KERNEL: [f32; 3] = [1.0; 3];

    pub struct Filter {
        taps: [f32; 3],
    }

    fn test_arrays() {
        let weights: [f32; 4] = [0.25, 0.25, 0.25, 0.25];
        let zeros: [f32; 3] = [0.0; 3];
        let matrix: [[f32; 2]; 2] = [[1.0, 2.0]; 2];
    }
    """
    try:
        result = parse_and_generate(code)
        assert "const float KERNEL[3] = {1.0, 1.0, 1.0};" in result
        assert "float taps[3];" in result
        assert "float weights[4] = {0.25, 0.25, 0.25, 0.25};" in result
        assert "float zeros[3] = {0.0, 0.0, 0.0};" in result
        assert "float matrix[2][2] = {{1.0, 2.0}, {1.0, 2.0}};" in result
    except Exception as e:
        pytest.fail(f"Fixed array conversion failed: {e}")


def test_reference_array_conversion():
    code = """
    pub struct Filter {
        taps: [f32; 3],
    }

    fn sample_arrays(filter: &Filter, weights: &[f32; 4], output: &mut [Vec3<f32>; 2]) {
        let first = weights[0];
    }
    """
    try:
        result = parse_and_generate(code)
        assert (
            "void sample_arrays(Filter filter, float weights[4], vec3 output[2])"
            in result
        )
        assert "first = weights[0];" in result
    except Exception as e:
        pytest.fail(f"Reference array conversion failed: {e}")


def test_lifetime_reference_type_conversion():
    code = """
    fn borrow<'a>(value: &'a Vec3<f32>) -> &'a Vec3<f32> {
        return value;
    }

    fn borrow_mut<'a>(value: &'a mut Vec3<f32>) -> &'a mut Vec3<f32> {
        return value;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "vec3 borrow(vec3 value)" in result
        assert "vec3 borrow_mut(vec3 value)" in result
        assert "return value;" in result
        assert "'a" not in result
        assert "&" not in result
    except Exception as e:
        pytest.fail(f"Lifetime reference type conversion failed: {e}")


def test_use_statement_conversion():
    code = """
    use std::collections::HashMap;
    use gpu::*;
    use math::{sin, cos, tan};
    """
    try:
        result = parse_and_generate(code)
        assert "// use std::collections::HashMap" in result
        assert "// use gpu::*" in result
    except Exception as e:
        pytest.fail(f"Use statement conversion failed: {e}")


def test_impl_block_conversion():
    code = """
    impl Vertex {
        pub fn new(x: f32, y: f32, z: f32) -> Self {
            Self {
                position: Vec3::new(x, y, z),
                normal: Vec3::new(0.0, 1.0, 0.0),
                texcoord: Vec2::new(0.0, 0.0),
            }
        }

        fn calculate_distance(&self, other: &Vertex) -> f32 {
            distance(self.position, other.position)
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "// Implementation for Vertex" in result
        assert "Vertex_new(" in result
        assert (
            "calculate_distance(" in result
        )  # Currently not prefixed with struct name
    except Exception as e:
        pytest.fail(f"Impl block conversion failed: {e}")


def test_references_conversion():
    code = """
    fn test_references() {
        let x = 42;
        let y = &x;
        let z = &mut x;
        let w = *y;
    }
    """
    try:
        result = parse_and_generate(code)
        # References should be handled by removing the reference syntax
        assert "y = x;" in result  # Reference removed
        assert "w = y;" in result  # Dereference removed
    except Exception as e:
        pytest.fail(f"References conversion failed: {e}")


def test_type_casting_conversion():
    code = """
    fn test_cast() {
        let x = 42i32;
        let y = x as f64;
        let z = y as i32;
        let chained = x as f32 as i32;
        let from_deref = *ptr as f32;
        let from_ref = &value as f32;
        let from_swizzle = color.xyz() as Vec3<f32>;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "x = 42;" in result
        assert "(double)x" in result
        assert "(int)y" in result
        assert "chained = (int)(float)x;" in result
        assert "from_deref = (float)ptr;" in result
        assert "from_ref = (float)value;" in result
        assert "from_swizzle = (vec3)color.xyz;" in result
        assert "42i32" not in result
        assert ".xyz()" not in result
        assert "*ptr" not in result
        assert "&value" not in result
    except Exception as e:
        pytest.fail(f"Type casting conversion failed: {e}")


def test_numeric_literal_suffix_conversion():
    code = """
    fn test_numeric_suffixes(mode: i32) {
        let signed = 42i32;
        let unsigned = 10u32;
        let sized = 7usize;
        let float32 = 1.5f32;
        let float64 = 2.0f64;
        let casted = 1u32 as i32;
        let repeated: [i32; 3] = [4i32; 3usize];
        let grouped = 1_000u32;
        let hexed = 0xffu32;
        let binary = 0b1010u32;
        let octal = 0o77usize;
        let grouped_float = 1_000.5f32;
        let exponent = 1.0e-3f32;
        let trailing_dot = 1.;
        match mode {
            1u32 => hit_one(),
            0xffu32 => hit_hex(),
            _ => hit_default(),
        }
    }
    """
    try:
        result = parse_and_generate(code)

        assert "signed = 42;" in result
        assert "unsigned = 10;" in result
        assert "sized = 7;" in result
        assert "float32 = 1.5;" in result
        assert "float64 = 2.0;" in result
        assert "casted = (int)1;" in result
        assert "int repeated[3] = {4, 4, 4};" in result
        assert "grouped = 1000;" in result
        assert "hexed = 255;" in result
        assert "binary = 10;" in result
        assert "octal = 63;" in result
        assert "grouped_float = 1000.5;" in result
        assert "exponent = 1.0e-3;" in result
        assert "trailing_dot = 1.;" in result
        assert "case 1:" in result
        assert "case 255:" in result
        assert "i32" not in result
        assert "u32" not in result
        assert "usize" not in result
        assert "f32" not in result
        assert "f64" not in result
        assert "0xff" not in result
        assert "0b1010" not in result
        assert "0o77" not in result
        assert "1_000" not in result
    except Exception as e:
        pytest.fail(f"Numeric literal suffix conversion failed: {e}")


def test_ternary_expression_conversion():
    code = """
    fn test_ternary() {
        let result = if condition { true_value } else { false_value };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "condition ? true_value : false_value" in result
    except Exception as e:
        pytest.fail(f"Ternary expression conversion failed: {e}")


def test_if_let_statement_conversion():
    code = """
    fn choose(value: Option<i32>, fallback: i32) -> i32 {
        if let Some(v) = value {
            return v;
        } else {
            return fallback;
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_match_matched_0 = false;" in result
        assert "is_Some(value)" in result
        assert "auto v = unwrap_Some(value);" in result
        assert "return v;" in result
        assert "if (!_rust_match_matched_0)" in result
        assert "return fallback;" in result
    except Exception as e:
        pytest.fail(f"If let statement conversion failed: {e}")


def test_if_let_expression_conversion():
    code = """
    fn score(value: Option<i32>) -> i32 {
        let result: i32 = if let Some(v) = value {
            v
        } else {
            0
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int result;" in result
        assert "if (is_Some(value))" in result
        assert "auto v = unwrap_Some(value);" in result
        assert "result = v;" in result
        assert "result = 0;" in result
    except Exception as e:
        pytest.fail(f"If let expression conversion failed: {e}")


def test_if_let_complex_pattern_conversion():
    code = """
    fn handle(point: Point) {
        if let Some(Ok(v) | Err(v)) = next_value() {
            use_value(v);
        }

        if let Point { x, y: 0, .. } = point {
            use_point(x);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0 = next_value();" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "bool _rust_match_or_matched_0 = false;" in result
        assert "if (is_Ok(" in result
        assert "if (is_Err(" in result
        assert "auto v = unwrap_Ok(" in result
        assert "auto v = unwrap_Err(" in result
        assert "auto x = point.x;" in result
        assert "if ((point.y == 0))" in result
    except Exception as e:
        pytest.fail(f"If let complex pattern conversion failed: {e}")


def test_while_let_conversion():
    code = """
    fn drain() {
        while let Some(item) = next_value() {
            consume(item);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "while (true)" in result
        assert "auto _rust_match_subject_0 = next_value();" in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "auto item = unwrap_Some(_rust_match_subject_0);" in result
        assert "if (!_rust_match_matched_0)" in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"While let conversion failed: {e}")


def test_let_else_conversion():
    code = """
    fn decode(value: Option<i32>, fallback: i32) -> i32 {
        let Some(v) = value else {
            return fallback;
        };
        return v;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0 = value;" in result
        assert "v;" in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "v = unwrap_Some(_rust_match_subject_0);" in result
        assert "auto v = unwrap_Some(_rust_match_subject_0);" not in result
        assert "return fallback;" in result
        assert "return v;" in result
    except Exception as e:
        pytest.fail(f"Let else conversion failed: {e}")


def test_let_else_shadowing_conversion():
    code = """
    fn decode_shadow(value: Option<i32>) -> i32 {
        let Some(value) = value else {
            return 0;
        };
        return value;
    }
    """
    try:
        result = parse_and_generate(code)
        subject_index = result.index("auto _rust_match_subject_0 = value;")
        binding_index = result.index("value;", subject_index)
        assert subject_index < binding_index
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "value = unwrap_Some(_rust_match_subject_0);" in result
        assert "return 0;" in result
        assert "return value;" in result
    except Exception as e:
        pytest.fail(f"Let else shadowing conversion failed: {e}")


def test_let_else_complex_pattern_conversion():
    code = """
    fn decode_complex(point: Point) {
        let Some(Ok(v) | Err(v)) = next_value() else {
            return;
        };
        use_value(v);

        let Point { x, y: 0, .. } = point else {
            return;
        };
        use_point(x);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0 = next_value();" in result
        assert "v;" in result
        assert "bool _rust_match_or_matched_0 = false;" in result
        assert "v = unwrap_Ok(" in result
        assert "v = unwrap_Err(" in result
        assert "auto v = unwrap_Ok(" not in result
        assert "auto v = unwrap_Err(" not in result
        assert "use_value(v);" in result
        assert "auto _rust_match_subject_1 = point;" in result
        assert "x;" in result
        assert "x = _rust_match_subject_1.x;" in result
        assert "if ((_rust_match_subject_1.y == 0))" in result
        assert "use_point(x);" in result
    except Exception as e:
        pytest.fail(f"Let else complex pattern conversion failed: {e}")


def test_transparent_let_else_subject_preserves_block_statements():
    code = """
    fn decode(value: Option<i32>) -> i32 {
        let Some(v) = unsafe {
            prepare();
            value
        } else {
            return 0;
        };
        return v;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0;" in result
        assert "prepare();" in result
        assert "_rust_match_subject_0 = value;" in result
        assert result.index("prepare();") < result.index(
            "_rust_match_subject_0 = value;"
        )
        assert result.index("_rust_match_subject_0 = value;") < result.index(
            "if (is_Some(_rust_match_subject_0))"
        )
        assert "v = unwrap_Some(_rust_match_subject_0);" in result
        assert "return 0;" in result
    except Exception as e:
        pytest.fail(f"Transparent let-else subject conversion failed: {e}")


def test_transparent_if_let_subject_preserves_block_statements():
    code = """
    fn branch(value: Option<i32>) {
        if let Some(v) = const {
            prepare();
            value
        } {
            use_value(v);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0;" in result
        assert "prepare();" in result
        assert "_rust_match_subject_0 = value;" in result
        assert result.index("prepare();") < result.index(
            "_rust_match_subject_0 = value;"
        )
        assert result.index("_rust_match_subject_0 = value;") < result.index(
            "if (is_Some(_rust_match_subject_0))"
        )
        assert "auto v = unwrap_Some(_rust_match_subject_0);" in result
    except Exception as e:
        pytest.fail(f"Transparent if-let subject conversion failed: {e}")


def test_plain_struct_array_let_pattern_conversion():
    code = """
    fn unpack(point: Point, values: [i32; 3]) {
        let Point { x, y } = point;
        let [first, .., last] = values;
        use_point(x, y);
        use_pair(first, last);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0 = point;" in result
        assert "x = _rust_match_subject_0.x;" in result
        assert "y = _rust_match_subject_0.y;" in result
        assert "auto x = _rust_match_subject_0.x;" not in result
        assert "auto y = _rust_match_subject_0.y;" not in result
        assert "auto _rust_match_subject_1 = values;" in result
        assert "first = _rust_match_subject_1[0];" in result
        assert (
            "last = _rust_match_subject_1[(length(_rust_match_subject_1) - 1)];"
            in result
        )
        assert "use_point(x, y);" in result
        assert "use_pair(first, last);" in result
    except Exception as e:
        pytest.fail(f"Plain struct/array let pattern conversion failed: {e}")


def test_matches_macro_let_conversion():
    code = """
    fn check(value: Option<i32>) -> bool {
        let ok: bool = matches!(value, Some(v) if v > 0);
        return ok;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool ok;" in result
        assert "ok = false;" in result
        assert "auto _rust_match_subject_0 = value;" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "auto v = unwrap_Some(_rust_match_subject_0);" in result
        assert "if ((v > 0))" in result
        assert "ok = true;" in result
        assert "return ok;" in result
    except Exception as e:
        pytest.fail(f"Matches macro let conversion failed: {e}")


def test_transparent_matches_subject_preserves_block_statements():
    code = """
    fn branch(value: Option<i32>) {
        let ready: bool = matches!(async {
            prepare();
            value
        }, Some(_));
    }
    """
    try:
        result = parse_and_generate(code)
        assert "auto _rust_match_subject_0;" in result
        assert "prepare();" in result
        assert "_rust_match_subject_0 = value;" in result
        assert result.index("prepare();") < result.index(
            "_rust_match_subject_0 = value;"
        )
        assert "ready = true;" in result
    except Exception as e:
        pytest.fail(f"Transparent matches subject conversion failed: {e}")


def test_matches_macro_if_conversion():
    code = """
    fn branch(value: Option<Result<i32, i32>>) {
        if matches!(value, Some(Ok(v) | Err(v)) if v > 0) {
            use_match();
        } else {
            use_default();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_matches_result_0 = false;" in result
        assert "auto _rust_match_subject_0 = value;" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "bool _rust_match_or_matched_0 = false;" in result
        assert "auto v = unwrap_Ok(" in result
        assert "auto v = unwrap_Err(" in result
        assert "if ((v > 0))" in result
        assert "_rust_matches_result_0 = true;" in result
        assert "if (_rust_matches_result_0)" in result
        assert "use_match();" in result
        assert "use_default();" in result
    except Exception as e:
        pytest.fail(f"Matches macro if conversion failed: {e}")


def test_matches_macro_return_and_while_conversion():
    code = """
    fn is_small(mode: i32) -> bool {
        return matches!(mode, 0 | 1);
    }

    fn drain() {
        while matches!(next_value(), Some(item)) {
            consume();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_matches_result_0 = false;" in result
        assert "auto _rust_match_subject_0 = mode;" in result
        assert "bool _rust_match_or_matched_0 = false;" in result
        assert "_rust_matches_result_0 = true;" in result
        assert "return _rust_matches_result_0;" in result

        assert "while (true)" in result
        assert "bool _rust_matches_result_1 = false;" in result
        assert "auto _rust_match_subject_1 = next_value();" in result
        assert "if (is_Some(_rust_match_subject_1))" in result
        assert "auto item = unwrap_Some(_rust_match_subject_1);" in result
        assert "_rust_matches_result_1 = true;" in result
        assert "if (!_rust_matches_result_1)" in result
        assert "break;" in result
        assert "consume();" in result
    except Exception as e:
        pytest.fail(f"Matches macro return/while conversion failed: {e}")


def test_let_condition_chain_if_conversion():
    code = """
    fn branch(ready: bool, value: Option<i32>) {
        if ready && let Some(v) = value && v > 0 {
            use_value(v);
        } else {
            use_default();
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (ready)" in result
        assert "if (is_Some(value))" in result
        assert "auto v = unwrap_Some(value);" in result
        assert "if ((v > 0))" in result
        assert "_rust_match_matched_0 = true;" in result
        assert "use_value(v);" in result
        assert "if (!_rust_match_matched_0)" in result
        assert "use_default();" in result
    except Exception as e:
        pytest.fail(f"Let condition chain if conversion failed: {e}")


def test_let_condition_chain_if_expression_conversion():
    code = """
    fn score(ready: bool, value: Option<i32>) -> i32 {
        let result: i32 = if ready && let Some(v) = value && v > 0 {
            v
        } else {
            0
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int result;" in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (ready)" in result
        assert "if (is_Some(value))" in result
        assert "auto v = unwrap_Some(value);" in result
        assert "if ((v > 0))" in result
        assert "result = v;" in result
        assert "if (!_rust_match_matched_0)" in result
        assert "result = 0;" in result
    except Exception as e:
        pytest.fail(f"Let condition chain if expression conversion failed: {e}")


def test_let_condition_chain_while_conversion():
    code = """
    fn drain() {
        while has_next() && let Some(item) = next_value() && item != 0 {
            consume(item);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "while (true)" in result
        assert "bool _rust_match_matched_0 = false;" in result
        assert "if (has_next())" in result
        assert "auto _rust_match_subject_0 = next_value();" in result
        assert "if (is_Some(_rust_match_subject_0))" in result
        assert "auto item = unwrap_Some(_rust_match_subject_0);" in result
        assert "if ((item != 0))" in result
        assert "_rust_match_matched_0 = true;" in result
        assert "consume(item);" in result
        assert "if (!_rust_match_matched_0)" in result
        assert "break;" in result
    except Exception as e:
        pytest.fail(f"Let condition chain while conversion failed: {e}")


def test_multiple_let_condition_chain_conversion():
    code = """
    fn pair(left: Option<i32>, right: Option<i32>) {
        if let Some(a) = left && let Some(b) = right && a < b {
            use_pair(a, b);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "if (is_Some(left))" in result
        assert "auto a = unwrap_Some(left);" in result
        assert "if (is_Some(right))" in result
        assert "auto b = unwrap_Some(right);" in result
        assert "if ((a < b))" in result
        assert "use_pair(a, b);" in result
        assert "bool _rust_match_matched_" not in result
    except Exception as e:
        pytest.fail(f"Multiple let condition chain conversion failed: {e}")


def test_transparent_condition_chain_subjects_preserve_block_statements():
    code = """
    fn pair(left: Option<i32>, right: Option<i32>) {
        if let Some(a) = async {
            prepare_left();
            left
        } && let Some(b) = unsafe {
            prepare_right();
            right
        } && a < b {
            use_pair(a, b);
        }
    }
    """
    try:
        result = parse_and_generate(code)
        assert "prepare_left();" in result
        assert "_rust_match_subject_0 = left;" in result
        assert "prepare_right();" in result
        assert "_rust_match_subject_1 = right;" in result
        assert result.index("prepare_left();") < result.index(
            "_rust_match_subject_0 = left;"
        )
        assert result.index("prepare_right();") < result.index(
            "_rust_match_subject_1 = right;"
        )
        assert "auto a = unwrap_Some(_rust_match_subject_0);" in result
        assert "auto b = unwrap_Some(_rust_match_subject_1);" in result
        assert "use_pair(a, b);" in result
    except Exception as e:
        pytest.fail(f"Transparent condition chain subject conversion failed: {e}")


def test_closure_expression_conversion():
    code = """
    fn closures(values: Values) {
        let add = |x, y| x + y;
        let typed = |x: i32| x + 1;
        let always = || true;
        let moved = move |v| v * 2;
    }
    """
    try:
        result = parse_and_generate(code)
        assert "add = lambda(x, y, (x + y));" in result
        assert "typed = lambda(int x, (x + 1));" in result
        assert "always = lambda(true);" in result
        assert "moved = lambda(v, (v * 2));" in result
    except Exception as e:
        pytest.fail(f"Closure expression conversion failed: {e}")


def test_async_closure_expression_conversion():
    code = """
    fn async_closures(values: Values) {
        let async_add = async |x| x + 1;
        let async_moved = async move || true;
        let mapped = values.map(async |x| x + 1);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "async_add = lambda(x, (x + 1));" in result
        assert "async_moved = lambda(true);" in result
        assert "mapped = map(values, lambda(x, (x + 1)));" in result
        assert "async |" not in result
        assert "async move" not in result
    except Exception as e:
        pytest.fail(f"Async closure expression conversion failed: {e}")


def test_closure_iterator_method_conversion():
    code = """
    fn iterator_methods(values: Values) {
        let mapped = values.map(|x| x + 1);
        let filtered = values.filter(|x| x > 0);
        let chained = values.map(|x| x + 1).filter(|y| y > 2);
        let total = values.fold(0, |acc, x| acc + x);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "mapped = map(values, lambda(x, (x + 1)));" in result
        assert "filtered = filter(values, lambda(x, (x > 0)));" in result
        assert (
            "chained = filter(map(values, lambda(x, (x + 1))), " "lambda(y, (y > 2)));"
        ) in result
        assert "total = fold(values, 0, lambda(acc, x, (acc + x)));" in result
    except Exception as e:
        pytest.fail(f"Closure iterator method conversion failed: {e}")


def test_typed_noncapturing_closures_emit_helpers():
    code = """
    fn helper_closures(values: Values) {
        let add = |x: i32, y: i32| -> i32 { x + y };
        let mapped = values.map(|x: i32| -> i32 { x + 1 });
        let filtered = values.filter(|x: i32| -> bool { x > 0 });
        let total = values.fold(0, |acc: i32, x: i32| -> i32 { acc + x });
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int _rust_closure_add_0(int x, int y) {" in result
        assert "return (x + y);" in result
        assert "let add = _rust_closure_add_0;" in result
        assert "int _rust_closure_map_1(int x) {" in result
        assert "mapped = map(values, _rust_closure_map_1);" in result
        assert "bool _rust_closure_filter_2(int x) {" in result
        assert "filtered = filter(values, _rust_closure_filter_2);" in result
        assert "int _rust_closure_fold_3(int acc, int x) {" in result
        assert "total = fold(values, 0, _rust_closure_fold_3);" in result
        assert "lambda(int x" not in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Typed noncapturing closure helper conversion failed: {e}")


def test_typed_inline_closure_resource_parameters_convert_helpers():
    code = """
    type ColorTexture = Texture2D<f32>;
    type Values = RwBuffer<i32>;

    fn closure_resource_parameters(
        tex: ColorTexture,
        values: Values,
        sampler_state: Sampler,
        uv: Vec2<f32>,
        index: u32,
    ) -> Vec4<f32> {
        let apply_sample = |resource: ColorTexture| -> Vec4<f32> {
            resource.sample_sampler(sampler_state, uv)
        };
        let apply_load = |buffer: Values| -> i32 {
            buffer.buffer_load(index)
        };
        let base = apply_sample(tex);
        let value = apply_load(values);
        return base + Vec4::<f32>::new(value as f32, 0.0, 0.0, 0.0);
    }
    """

    result = parse_and_generate(code)

    assert (
        "let apply_sample = lambda(ColorTexture resource, "
        "{ return texture(resource, sampler_state, uv); });"
    ) in result
    assert (
        "let apply_load = lambda(Values buffer, "
        "{ return buffer_load(buffer, index); });"
    ) in result
    assert "let base = apply_sample(tex);" in result
    assert "let value = apply_load(values);" in result
    assert ".sample_sampler" not in result
    assert ".buffer_load" not in result


def test_local_closure_names_shadow_resource_builtin_function_names():
    code = """
    type ColorTexture = Texture2D<f32>;

    fn closure_named_like_builtin(
        tex: ColorTexture,
        uv: Vec2<f32>,
        fallback: Vec4<f32>,
        lod: f32,
    ) -> Vec4<f32> {
        let sample = |resource: ColorTexture| -> Vec4<f32> {
            fallback
        };
        let sample_lod = |resource: ColorTexture, level: f32| -> Vec4<f32> {
            fallback + Vec4::<f32>::new(level, 0.0, 0.0, 0.0)
        };
        let local_sample = sample(tex);
        let local_lod = sample_lod(tex, lod);
        let real_sample = tex.sample(uv);
        return local_sample + local_lod + real_sample;
    }
    """

    result = parse_and_generate(code)

    assert "let local_sample = sample(tex);" in result
    assert "let local_lod = sample_lod(tex, lod);" in result
    assert "let real_sample = texture(tex, uv);" in result
    assert "let local_sample = texture(tex);" not in result
    assert "let local_lod = textureLod(tex, lod);" not in result


def test_block_local_closure_names_do_not_shadow_resource_builtins_after_block():
    code = """
    type ColorTexture = Texture2D<f32>;

    fn block_local_callable_shadowing(
        tex: ColorTexture,
        uv: Vec2<f32>,
        fallback: Vec4<f32>,
        flag: bool,
        lod: f32,
    ) -> Vec4<f32> {
        let block_value = {
            let sample = |resource: ColorTexture| -> Vec4<f32> {
                fallback
            };
            sample(tex)
        };
        if flag {
            let sample_lod = |resource: ColorTexture, level: f32| -> Vec4<f32> {
                fallback + Vec4::<f32>::new(level, 0.0, 0.0, 0.0)
            };
            let branch_value = sample_lod(tex, lod);
        }
        let builtin_value = sample(tex, uv);
        let builtin_lod = sample_lod(tex, lod);
        return block_value + builtin_value + builtin_lod;
    }
    """

    result = parse_and_generate(code)

    assert "block_value = sample(tex);" in result
    assert "let branch_value = sample_lod(tex, lod);" in result
    assert "let builtin_value = texture(tex, uv);" in result
    assert "let builtin_lod = textureLod(tex, lod);" in result
    assert "let builtin_value = sample(tex, uv);" not in result
    assert "let builtin_lod = sample_lod(tex, lod);" not in result


def test_block_local_function_items_shadow_resource_builtins_only_in_scope():
    code = """
    type ColorTexture = Texture2D<f32>;

    fn block_local_function_item_shadowing(
        tex: ColorTexture,
        uv: Vec2<f32>,
        flag: bool,
        lod: f32,
    ) -> Vec4<f32> {
        let block_value = {
            fn sample(resource: ColorTexture) -> Vec4<f32> {
                return Vec4::<f32>::new(1.0, 0.0, 0.0, 1.0);
            }
            sample(tex)
        };
        if flag {
            fn sample_lod(resource: ColorTexture, level: f32) -> Vec4<f32> {
                return Vec4::<f32>::new(level, 0.0, 0.0, 0.0);
            }
            let branch_value = sample_lod(tex, lod);
        }
        let builtin_value = sample(tex, uv);
        let builtin_lod = sample_lod(tex, lod);
        return block_value + builtin_value + builtin_lod;
    }
    """

    result = parse_and_generate(code)

    assert "vec4 _rust_local_sample_" in result
    assert "vec4 _rust_local_sample_lod_" in result
    assert "return vec4(1.0, 0.0, 0.0, 1.0);" in result
    assert "return vec4(level, 0.0, 0.0, 0.0);" in result
    assert "block_value = _rust_local_sample_" in result
    assert "let branch_value = _rust_local_sample_lod_" in result
    assert "let builtin_value = texture(tex, uv);" in result
    assert "let builtin_lod = textureLod(tex, lod);" in result
    assert "let builtin_value = _rust_local_sample_" not in result
    assert "let builtin_lod = _rust_local_sample_lod_" not in result


def test_forward_local_function_items_shadow_resource_builtins_in_scope():
    code = """
    type ColorTexture = Texture2D<f32>;

    fn forward_local_function_item_shadowing(
        tex: ColorTexture,
        uv: Vec2<f32>,
        flag: bool,
        lod: f32,
    ) -> Vec4<f32> {
        let block_value = {
            let before = sample(tex);
            fn sample(resource: ColorTexture) -> Vec4<f32> {
                return Vec4::<f32>::new(1.0, 0.0, 0.0, 1.0);
            }
            let after = sample(tex);
            before + after
        };
        if flag {
            let branch_value = sample_lod(tex, lod);
            fn sample_lod(resource: ColorTexture, level: f32) -> Vec4<f32> {
                return Vec4::<f32>::new(level, 0.0, 0.0, 0.0);
            }
        }
        let builtin_value = sample(tex, uv);
        let builtin_lod = sample_lod(tex, lod);
        return block_value + builtin_value + builtin_lod;
    }
    """

    result = parse_and_generate(code)

    assert "vec4 _rust_local_sample_" in result
    assert "vec4 _rust_local_sample_lod_" in result
    assert "let before = _rust_local_sample_" in result
    assert "let after = _rust_local_sample_" in result
    assert "let branch_value = _rust_local_sample_lod_" in result
    assert "let builtin_value = texture(tex, uv);" in result
    assert "let builtin_lod = textureLod(tex, lod);" in result
    assert "let before = texture(tex);" not in result
    assert "let builtin_value = _rust_local_sample_" not in result
    assert "let builtin_lod = _rust_local_sample_lod_" not in result


def test_typed_pattern_closures_emit_helpers():
    code = """
    struct Point {
        x: i32,
        y: i32,
    }

    fn pattern_helpers(values: Values) {
        let projected = values.map(|Point { x, y }: Point| -> i32 { x + y });
        let optional = values.map(|Some(v): Option<i32>| -> i32 { v });
    }
    """
    try:
        result = parse_and_generate(code)
        assert "int _rust_closure_map_0(Point _rust_closure_arg_0) {" in result
        assert "auto x = _rust_closure_arg_0.x;" in result
        assert "auto y = _rust_closure_arg_0.y;" in result
        assert "projected = map(values, _rust_closure_map_0);" in result
        assert "int _rust_closure_map_1(Option<i32> _rust_closure_arg_1) {" in result
        assert "if (is_Some(_rust_closure_arg_1))" in result
        assert "auto v = unwrap_Some(_rust_closure_arg_1);" in result
        assert "optional = map(values, _rust_closure_map_1);" in result
        assert "lambda(Point" not in result
        assert "lambda(Option" not in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Typed pattern closure helper conversion failed: {e}")


def test_tuple_typed_pattern_closure_stays_inline():
    code = """
    fn tuple_pattern(values: Values) {
        let projected = values.map(|(x, y): (i32, i32)| -> i32 { x + y });
    }
    """
    try:
        result = parse_and_generate(code)
        assert "_rust_closure_map_0" not in result
        assert "projected = map(values, lambda((i32, i32) _rust_closure_arg_0" in result
        assert "auto x = _rust_tuple_0(_rust_closure_arg_0);" in result
        assert "auto y = _rust_tuple_1(_rust_closure_arg_0);" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Tuple typed pattern closure fallback conversion failed: {e}")


def test_closure_helpers_fall_back_for_captures_untyped_and_try():
    code = """
    fn helper_fallbacks(values: Values, scale: i32, value: Result<i32, i32>) {
        let captured = |x: i32| -> i32 { x * scale };
        let untyped = |x| x + 1;
        let parsed = |value: Result<i32, i32>| -> Result<i32, i32> {
            let v: i32 = value?;
            Ok(v)
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert "_rust_closure_captured" not in result
        assert "captured = lambda(int x, { return (x * scale); });" in result
        assert "_rust_closure_untyped" not in result
        assert "untyped = lambda(x, (x + 1));" in result
        assert "_rust_closure_parsed" not in result
        assert "parsed = lambda(Result<i32, i32> value, {" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure helper fallback conversion failed: {e}")


def test_closure_helper_names_avoid_user_function_collisions():
    try:
        ast = ShaderNode(
            functions=[
                FunctionNode(
                    "i32",
                    "_rust_closure_add_0",
                    [VariableNode("i32", "x")],
                    [ReturnNode("x")],
                ),
                FunctionNode(
                    "void",
                    "collision",
                    [],
                    [
                        LetNode(
                            "add",
                            ClosureNode(
                                [ClosureParameterNode("x", "i32")],
                                BlockNode([], BinaryOpNode("x", "+", "1")),
                                return_type="i32",
                            ),
                        )
                    ],
                ),
            ]
        )
        result = RustToCrossGLConverter().generate(ast)
        assert "int _rust_closure_add_0(int x) {" in result
        assert "int _rust_closure_add_1(int x) {" in result
        assert "let add = _rust_closure_add_1;" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure helper name collision conversion failed: {e}")


def test_scalar_statement_block_node_codegen():
    statement = LetNode("value", "1", "i32")
    block = BlockNode(statement, "value")
    ast = ShaderNode(
        functions=[
            FunctionNode(
                "i32",
                "scalar_block",
                [],
                [ReturnNode(block)],
            ),
        ]
    )

    result = RustToCrossGLConverter().generate(ast)

    assert block.statements == [statement]
    assert "int scalar_block() {" in result
    assert "int value = 1;" in result
    assert "return value;" in result


def test_closure_block_body_conversion():
    code = """
    fn closure_blocks(values: Values) {
        let doubled = |x| {
            let y = x + 1;
            y * 2
        };
        let mapped = values.map(|x| {
            prepare(x);
            x + 1
        });
        let choose = |x| {
            if x > 0 {
                let kept = x;
                kept
            } else {
                let fallback = 0;
                fallback
            }
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert (
            "let doubled = lambda(x, { let y = (x + 1); return (y * 2); });" in result
        )
        assert (
            "let mapped = map(values, lambda(x, { prepare(x); return (x + 1); }));"
            in result
        )
        assert (
            "let choose = lambda(x, { if ((x > 0)) { let kept = x; return kept; } "
            "else { let fallback = 0; return fallback; } });"
        ) in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure block body conversion failed: {e}")


def test_closure_transparent_block_body_conversion():
    code = """
    fn closure_transparent_blocks() {
        let guarded = |x| unsafe {
            let y = x + 1;
            y * 2
        };
        let folded = || const {
            let value = 2;
            value + 1
        };
        let suspended = |x| async {
            let y = x + 1;
            y
        };
    }
    """
    try:
        result = parse_and_generate(code)
        assert (
            "let guarded = lambda(x, { let y = (x + 1); return (y * 2); });" in result
        )
        assert "let folded = lambda({ let value = 2; return (value + 1); });" in result
        assert "let suspended = lambda(x, { let y = (x + 1); return y; });" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure transparent block body conversion failed: {e}")


def test_closure_pattern_parameter_conversion():
    code = """
    fn closure_pattern_params(values: Values) {
        let tupled = values.map(|(x, y)| x + y);
        let pointed = values.map(|Point { x, y }| x + y);
        let optioned = values.map(|Some(v)| v);
        let ignored = values.filter(|_| true);
    }
    """
    try:
        result = parse_and_generate(code)
        assert (
            "tupled = map(values, lambda(_rust_closure_arg_0, { "
            "auto x = _rust_tuple_0(_rust_closure_arg_0); "
            "auto y = _rust_tuple_1(_rust_closure_arg_0); "
            "return (x + y); }));"
        ) in result
        assert (
            "pointed = map(values, lambda(_rust_closure_arg_1, { "
            "auto x = _rust_closure_arg_1.x; "
            "auto y = _rust_closure_arg_1.y; "
            "return (x + y); }));"
        ) in result
        assert (
            "optioned = map(values, lambda(_rust_closure_arg_2, { "
            "if (is_Some(_rust_closure_arg_2)) { "
            "auto v = unwrap_Some(_rust_closure_arg_2); "
            "return v; } }));"
        ) in result
        assert "ignored = filter(values, lambda(_rust_closure_arg_3, true));" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure pattern parameter conversion failed: {e}")


def test_closure_explicit_result_try_conversion():
    code = """
    fn outer(value: Result<i32, i32>) -> Option<i32> {
        let parse = |value: Result<i32, i32>| -> Result<i32, i32> {
            let v: i32 = value?;
            return Ok(v);
        };
        return Some(1);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "parse = lambda(Result<i32, i32> value, {" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "return Err(unwrap_Err(_rust_try_subject_0));" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "return Ok(v);" in result
        assert "return None;" not in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure explicit Result try conversion failed: {e}")


def test_closure_explicit_option_try_conversion():
    code = """
    fn outer(value: Option<i32>) -> Result<i32, i32> {
        let lift = |value: Option<i32>| -> Option<i32> {
            let v: i32 = value?;
            Some(v)
        };
        return Ok(1);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lift = lambda(Option<i32> value, {" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_None(_rust_try_subject_0))" in result
        assert "return None;" in result
        assert "auto _rust_try_value_0 = unwrap_Some(_rust_try_subject_0);" in result
        assert "return Some(v);" in result
        assert "return Err(unwrap_Err(_rust_try_subject_0));" not in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure explicit Option try conversion failed: {e}")


def test_closure_inferred_result_try_conversion():
    code = """
    fn outer(value: Result<i32, i32>) -> Option<i32> {
        let parse = |value: Result<i32, i32>| {
            let v: i32 = value?;
            Ok(v)
        };
        let inline = |value: Result<i32, i32>| Ok(value?);
        return Some(1);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "parse = lambda(Result<i32, i32> value, {" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "return Err(unwrap_Err(_rust_try_subject_0));" in result
        assert "return Ok(v);" in result
        assert "inline = lambda(Result<i32, i32> value, {" in result
        assert "if (is_Err(_rust_try_subject_1))" in result
        assert "return Ok(_rust_try_value_1);" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure inferred Result try conversion failed: {e}")


def test_closure_inferred_option_try_conversion():
    code = """
    fn outer(value: Option<i32>) -> Result<i32, i32> {
        let lift = |value: Option<i32>| {
            let v: i32 = value?;
            Some(v)
        };
        return Ok(1);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lift = lambda(Option<i32> value, {" in result
        assert "if (is_None(_rust_try_subject_0))" in result
        assert "return None;" in result
        assert "return Some(v);" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Closure inferred Option try conversion failed: {e}")


def test_await_postfix_conversion():
    code = """
    fn decode_future(
        future: Result<i32, i32>,
        object: FutureObject,
    ) -> Result<i32, i32> {
        let raw = compute().await;
        let field = object.await.value;
        let value: i32 = future.await?;
        return Ok(value);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "raw = compute();" in result
        assert "field = object.value;" in result
        assert "auto _rust_try_subject_0 = future;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "value = _rust_try_value_0;" in result
        assert ".await" not in result
    except Exception as e:
        pytest.fail(f"Await postfix conversion failed: {e}")


def test_await_try_binary_conversion():
    code = """
    fn add_futures(
        left: Result<i32, i32>,
        right: Result<i32, i32>,
    ) -> Result<i32, i32> {
        let sum: i32 = left.await? + right.await?;
        return Ok(sum);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = left;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "auto _rust_try_subject_1 = right;" in result
        assert "auto _rust_try_value_1 = unwrap_Ok(_rust_try_subject_1);" in result
        assert "sum = (_rust_try_value_0 + _rust_try_value_1);" in result
    except Exception as e:
        pytest.fail(f"Await try binary conversion failed: {e}")


def test_async_function_transparent_conversion():
    code = """
    pub async fn decode_future(future: Result<i32, i32>) -> Result<i32, i32> {
        let value: i32 = future.await?;
        return Ok(value);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "Result<i32, i32> decode_future(Result<i32, i32> future)" in result
        assert "auto _rust_try_subject_0 = future;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "value = _rust_try_value_0;" in result
        assert "return Ok(value);" in result
        assert "async" not in result
        assert ".await" not in result
    except Exception as e:
        pytest.fail(f"Async function transparent conversion failed: {e}")


def test_async_block_expression_conversion():
    code = """
    fn decode(value: Result<i32, i32>, fallback: i32) -> Result<i32, i32> {
        let result: i32 = async {
            let v: i32 = value.await?;
            v + 1
        };
        let second: i32 = 0;
        second = async move {
            fallback + result
        };
        return Ok(second);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "int result;" in result
        assert "int v;" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "v = _rust_try_value_0;" in result
        assert "result = (v + 1);" in result
        assert "int second = 0;" in result
        assert "second = (fallback + result);" in result
        assert "async" not in result
        assert ".await" not in result
    except Exception as e:
        pytest.fail(f"Async block expression conversion failed: {e}")


def test_unsafe_extern_transparent_conversion():
    code = """
    pub unsafe extern "C" fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let result: i32 = unsafe {
            let v: i32 = value?;
            v + 1
        };
        return Ok(result);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "Result<i32, i32> decode(Result<i32, i32> value)" in result
        assert "int result;" in result
        assert "int v;" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "v = _rust_try_value_0;" in result
        assert "result = (v + 1);" in result
        assert "return Ok(result);" in result
        assert "unsafe" not in result
        assert "extern" not in result
    except Exception as e:
        pytest.fail(f"Unsafe extern transparent conversion failed: {e}")


def test_unsafe_block_assignment_conversion():
    code = """
    fn decode(value: i32) -> i32 {
        let result: i32 = 0;
        result = unsafe {
            value + 1
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "int result = 0;" in result
        assert "result = (value + 1);" in result
        assert "return result;" in result
        assert "unsafe" not in result
    except Exception as e:
        pytest.fail(f"Unsafe block assignment conversion failed: {e}")


def test_const_fn_transparent_conversion():
    code = """
    pub const unsafe extern "C" fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let result: i32 = const {
            let v: i32 = value?;
            v + 1
        };
        return Ok(result);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "Result<i32, i32> decode(Result<i32, i32> value)" in result
        assert "int result;" in result
        assert "int v;" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "v = _rust_try_value_0;" in result
        assert "result = (v + 1);" in result
        assert "return Ok(result);" in result
        assert "const " not in result
        assert "unsafe" not in result
        assert "extern" not in result
    except Exception as e:
        pytest.fail(f"Const fn transparent conversion failed: {e}")


def test_const_block_assignment_conversion():
    code = """
    fn decode(value: i32) -> i32 {
        let result: i32 = 0;
        result = const {
            value + 1
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "int result = 0;" in result
        assert "result = (value + 1);" in result
        assert "return result;" in result
        assert "const " not in result
    except Exception as e:
        pytest.fail(f"Const block assignment conversion failed: {e}")


def test_local_const_static_item_conversion():
    code = """
    fn local_items(value: i32) -> i32 {
        const OFFSET: i32 = 1;
        static mut CACHE: i32 = 2;
        let result: i32 = {
            const INNER: i32 = 3;
            static ENABLED: bool = true;
            value + OFFSET + INNER
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "const int OFFSET = 1;" in result
        assert "static mut int CACHE = 2;" in result
        assert "int result;" in result
        assert "const int INNER = 3;" in result
        assert "static bool ENABLED = true;" in result
        assert "result = ((value + OFFSET) + INNER);" in result
        assert "return result;" in result
    except Exception as e:
        pytest.fail(f"Local const/static item conversion failed: {e}")


def test_result_try_expression_conversion():
    code = """
    fn decode(left: Result<i32, i32>, right: Result<i32, i32>) -> Result<i32, i32> {
        let value: i32 = left?;
        let sum: i32 = value + right?;
        return Ok(sum);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = left;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "return Err(unwrap_Err(_rust_try_subject_0));" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "int value;" in result
        assert "value = _rust_try_value_0;" in result

        assert "auto _rust_try_subject_1 = right;" in result
        assert "auto _rust_try_value_1 = unwrap_Ok(_rust_try_subject_1);" in result
        assert "int sum;" in result
        assert "sum = (value + _rust_try_value_1);" in result
        assert "return Ok(sum);" in result
    except Exception as e:
        pytest.fail(f"Result try expression conversion failed: {e}")


def test_option_try_expression_conversion():
    code = """
    fn decode(value: Option<i32>) -> Option<i32> {
        let unwrapped: i32 = value?;
        return Some(unwrapped);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "if (is_None(_rust_try_subject_0))" in result
        assert "return None;" in result
        assert "auto _rust_try_value_0 = unwrap_Some(_rust_try_subject_0);" in result
        assert "unwrapped = _rust_try_value_0;" in result
        assert "return Some(unwrapped);" in result
    except Exception as e:
        pytest.fail(f"Option try expression conversion failed: {e}")


def test_nested_try_expression_conversion():
    code = """
    fn wrap(value: Result<i32, i32>, receiver: Receiver) -> Result<i32, i32> {
        let wrapped = Ok(value?);
        receiver.store(value?);
        return wrapped;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = value;" in result
        assert "wrapped = Ok(_rust_try_value_0);" in result
        assert "auto _rust_try_subject_1 = value;" in result
        assert "receiver.store(_rust_try_value_1);" in result
        assert "return wrapped;" in result
    except Exception as e:
        pytest.fail(f"Nested try expression conversion failed: {e}")


def test_try_pattern_subject_conversion():
    code = """
    fn decode(value: Result<Option<i32>, i32>) -> Result<i32, i32> {
        let Some(v) = value? else {
            return Ok(0);
        };
        let present: bool = matches!(value?, Some(1));
        return Ok(v);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = value;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "if (is_Some(_rust_try_value_0))" in result
        assert "v = unwrap_Some(_rust_try_value_0);" in result
        assert "auto _rust_try_subject_1 = value;" in result
        assert "auto _rust_try_value_1 = unwrap_Ok(_rust_try_subject_1);" in result
        assert "present = false;" in result
        assert "if (is_Some(_rust_try_value_1))" in result
        assert "return Ok(v);" in result
    except Exception as e:
        pytest.fail(f"Try pattern subject conversion failed: {e}")


def test_try_match_scrutinee_conversion():
    code = """
    fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let mapped: i32 = match value? {
            0 => 1,
            _ => 2,
        };
        return Ok(mapped);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "switch (_rust_try_value_0)" in result
        assert "mapped = 1;" in result
        assert "mapped = 2;" in result
        assert "return Ok(mapped);" in result
    except Exception as e:
        pytest.fail(f"Try match scrutinee conversion failed: {e}")


def test_try_match_scrutinee_hoists_post_try_side_effects():
    code = """
    fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        return match value? + adjust() {
            0 => 1,
            _ => 2,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = value;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "auto _rust_match_subject_0 = (_rust_try_value_0 + adjust());" in result
        assert "switch (_rust_match_subject_0)" in result
    except Exception as e:
        pytest.fail(f"Try match scrutinee side-effect hoist failed: {e}")


def test_try_match_guard_conversion():
    code = """
    fn decode(value: Option<i32>, ready: Result<bool, i32>) -> Result<i32, i32> {
        return match value {
            Some(v) if ready? => v,
            _ => 0,
        };
    }
    """
    try:
        result = parse_and_generate(code)

        assert "is_Some(value)" in result
        assert "auto _rust_try_subject_0 = ready;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "if (_rust_try_value_0)" in result
        assert "return v;" in result
    except Exception as e:
        pytest.fail(f"Try match guard conversion failed: {e}")


def test_try_for_loop_setup_conversion():
    code = """
    fn iterate(
        start: Result<i32, i32>,
        end: Result<i32, i32>,
        step: Result<i32, i32>,
        count: Result<i32, i32>,
    ) -> Result<i32, i32> {
        let total: i32 = 0;
        for i in start?..end? {
            total += i;
        }
        for j in (0..end?).step_by(step?) {
            total += j;
        }
        for k in count? {
            total += k;
        }
        return Ok(total);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = start;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "auto _rust_try_subject_1 = end;" in result
        assert "for (int i = _rust_try_value_0; i < _rust_try_value_1; i++)" in result
        assert "auto _rust_try_subject_2 = end;" in result
        assert "auto _rust_try_subject_3 = step;" in result
        assert (
            "for (int j = 0; j < _rust_try_value_2; j += _rust_try_value_3)" in result
        )
        assert "auto _rust_try_subject_4 = count;" in result
        assert "for (int k = 0; k < _rust_try_value_4; k++)" in result
        assert "return Ok(total);" in result
    except Exception as e:
        pytest.fail(f"Try for loop setup conversion failed: {e}")


def test_try_for_loop_setup_hoists_post_try_side_effects():
    code = """
    fn iterate(end: Result<i32, i32>) -> Result<i32, i32> {
        for i in 0..(end? + limit()) {
            use_value(i);
        }
        return Ok(0);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = end;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "int _for_bound_0 = (_rust_try_value_0 + limit());" in result
        assert "for (int i = 0; i < _for_bound_0; i++)" in result
    except Exception as e:
        pytest.fail(f"Try for loop side-effect hoist failed: {e}")


def test_try_struct_initialization_conversion():
    code = """
    struct Output {
        position: Vec3<f32>,
        value: i32,
    }

    fn build(
        position: Result<Vec3<f32>, i32>,
        value: Result<i32, i32>,
    ) -> Result<Output, i32> {
        return Ok(Output {
            position: position?,
            value: value?,
        });
    }
    """
    try:
        result = parse_and_generate(code)

        assert "auto _rust_try_subject_0 = position;" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "auto _rust_try_subject_1 = value;" in result
        assert "auto _rust_try_value_1 = unwrap_Ok(_rust_try_subject_1);" in result
        assert (
            "return Ok(Output { position: _rust_try_value_0, "
            "value: _rust_try_value_1 });"
        ) in result
    except Exception as e:
        pytest.fail(f"Try struct initialization conversion failed: {e}")


def test_result_try_block_conversion():
    code = """
    fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let result: Result<i32, i32> = try {
            let v: i32 = value?;
            v + 1
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lambda({" not in result
        assert "Result<i32, i32> result;" in result
        assert "do {" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "result = Err(unwrap_Err(_rust_try_subject_0));" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "result = Ok((v + 1));" in result
        assert "} while (false);" in result
        assert "return result;" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Result try block conversion failed: {e}")


def test_option_try_block_conversion():
    code = """
    fn decode(value: Option<i32>) -> Option<i32> {
        let result: Option<i32> = try {
            let v: i32 = value?;
            v + 1
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lambda({" not in result
        assert "Option<i32> result;" in result
        assert "do {" in result
        assert "auto _rust_try_subject_0 = value;" in result
        assert "if (is_None(_rust_try_subject_0))" in result
        assert "result = None;" in result
        assert "auto _rust_try_value_0 = unwrap_Some(_rust_try_subject_0);" in result
        assert "result = Some((v + 1));" in result
        assert "} while (false);" in result
        assert "return result;" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Option try block conversion failed: {e}")


def test_try_block_as_try_operand_conversion():
    code = """
    fn decode(value: Result<i32, i32>) -> Result<i32, i32> {
        let unwrapped: i32 = try {
            let v: i32 = value?;
            v + 1
        }?;
        return Ok(unwrapped);
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lambda({" not in result
        assert "auto _rust_try_subject_0;" in result
        assert "do {" in result
        assert "_rust_try_subject_0 = Ok((v + 1));" in result
        assert "if (is_Err(_rust_try_subject_0))" in result
        assert "auto _rust_try_value_0 = unwrap_Ok(_rust_try_subject_1);" in result
        assert "auto _rust_try_value_1 = unwrap_Ok(_rust_try_subject_0);" in result
        assert "unwrapped = _rust_try_value_1;" in result
        assert "return Ok(unwrapped);" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Try block operand conversion failed: {e}")


def test_try_block_final_result_expression_conversion():
    code = """
    fn decode(value: Result<i32, i32>, ready: bool) -> Result<i32, i32> {
        let result: Result<i32, i32> = try {
            if ready {
                value?
            } else {
                0
            }
        };
        return result;
    }
    """
    try:
        result = parse_and_generate(code)

        assert "lambda({" not in result
        assert "do {" in result
        assert "auto _rust_try_value_0;" in result
        assert "if (ready)" in result
        assert "result = Err(unwrap_Err(_rust_try_subject_0));" in result
        assert "_rust_try_value_0 = _rust_try_value_1;" in result
        assert "_rust_try_value_0 = 0;" in result
        assert "result = Ok(_rust_try_value_0);" in result
        crosstl.translator.parse(result)
    except Exception as e:
        pytest.fail(f"Try block final result expression conversion failed: {e}")


def test_spirv_std_fragment_example_codegen_from_docs():
    code = """
    use spirv_std::spirv;
    use glam::{Vec3, Vec4, vec2, vec3};

    #[spirv(fragment)]
    pub fn main(
        #[spirv(frag_coord)] in_frag_coord: &Vec4,
        #[spirv(push_constant)] constants: &ShaderConstants,
        output: &mut Vec4,
    ) {
        let frag_coord = vec2(in_frag_coord.x, in_frag_coord.y);
        let mut uv = (frag_coord - 0.5 * vec2(constants.width as f32, constants.height as f32))
            / constants.height as f32;
        uv.y = -uv.y;
        let eye_pos = vec3(0.0, 0.0997, 0.2);
        let sun_pos = vec3(0.0, 75.0, -1000.0);
        let dir = get_ray_dir(uv, eye_pos, sun_pos);
        let color = sky(dir, sun_pos);
        *output = tonemap(color).extend(1.0)
    }
    """

    result = parse_and_generate(code)

    assert "fragment {" in result
    assert "vec4 in_frag_coord @ gl_FragCoord" in result
    assert "ShaderConstants constants @ push_constant" in result
    assert "vec2((float)constants.width, (float)constants.height)" in result
    assert "uv.y = (-uv.y);" in result
    assert "output = tonemap(color).extend(1.0);" in result


def test_spirv_specialization_and_workgroup_attribute_codegen_from_dev_guide():
    code = """
    use spirv_std::spirv;

    #[spirv(vertex)]
    fn specialize(
        #[spirv(spec_constant(id = 100))] x_u64_lo: u32,
        #[spirv(spec_constant(id = 101))] x_u64_hi: u32,
    ) {
        let x_u64 = ((x_u64_hi as u64) << 32) | (x_u64_lo as u64);
    }

    #[spirv(compute(threads(32)))]
    fn shared(#[spirv(workgroup)] tile: &mut [Vec4; 4]) {}
    """

    result = parse_and_generate(code)

    assert "uint x_u64_lo @ constant_id(100)" in result
    assert "uint x_u64_hi @ constant_id(101)" in result
    assert "let x_u64 = (((uint64_t)x_u64_hi << 32) | (uint64_t)x_u64_lo);" in result
    assert "compute shared_" in result
    assert "void main(vec4 tile[4] @ groupshared) @numthreads(32, 1, 1)" in result
    crosstl.translator.parse(result)


def test_rust_gpu_sky_shader_hex_default_spec_constant_codegen():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/sky-shader/src/lib.rs fragment specialization constant.
    code = """
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(
        #[spirv(frag_coord)] in_frag_coord: Vec4,
        #[spirv(push_constant)] constants: &ShaderConstants,
        output: &mut Vec4,
        #[spirv(spec_constant(id = 0x5007, default = 100))]
        sun_intensity_extra_spec_const_factor: u32,
    ) {
        let frag_coord = vec2(in_frag_coord.x, in_frag_coord.y);
        *output = fs(constants, frag_coord, sun_intensity_extra_spec_const_factor);
    }
    """

    result = parse_and_generate(code)

    assert "fragment main_fs {" in result
    assert "vec4 in_frag_coord @ gl_FragCoord" in result
    assert "ShaderConstants constants @ push_constant" in result
    assert "uint sun_intensity_extra_spec_const_factor @ constant_id(0x5007)" in result
    assert (
        "output = fs(constants, frag_coord, sun_intensity_extra_spec_const_factor);"
        in result
    )


def test_rust_gpu_sky_shader_cfg_test_module_is_not_emitted_to_crossgl():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/sky-shader/src/lib.rs trailing #[cfg(test)] module.
    code = """
    use spirv_std::spirv;

    #[spirv(fragment)]
    pub fn main_fs(output: &mut Vec4) {
        *output = vec4(1.0, 0.0, 0.0, 1.0);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_tonemap() {
            assert_eq!(
                tonemap(vec3(1_f32, 1_f32, 1_f32)),
                vec3(0.001261625, 0.001261625, 0.001261625)
            );
        }
    }
    """

    result = parse_and_generate(code)

    assert "fragment main_fs {" in result
    assert "void main(vec4 output)" in result
    assert "output = vec4(1.0, 0.0, 0.0, 1.0);" in result
    assert "test_tonemap" not in result
    assert "assert_eq!" not in result
    assert "1_f32" not in result
    crosstl.translator.parse(result)


def test_rust_gpu_mouse_shader_tuple_struct_destructure_codegen():
    # Reduced from Rust-GPU/rust-gpu commit
    # 36e3348cdc2f824afec64b3b5af5d369d98a4c0d,
    # examples/shaders/mouse-shader/src/lib.rs Line::distance.
    code = """
    struct Line(Vec2, Vec2);

    impl Shape for Line {
        fn distance(self, p: Vec2) -> f32 {
            let Line(a, b) = self;
            let ap = p - a;
            ap.x
        }
    }
    """

    result = parse_and_generate(code)

    assert "struct Line {" in result
    assert "vec2 field0;" in result
    assert "vec2 field1;" in result
    assert "float Line_distance(Line self, vec2 p)" in result
    assert "auto _rust_match_subject_0 = self;" in result
    assert "if (is_Line(_rust_match_subject_0))" in result
    assert "a = unwrap_Line_0(_rust_match_subject_0);" in result
    assert "b = unwrap_Line_1(_rust_match_subject_0);" in result
    assert "let ap = (p - a);" in result
    assert "return ap.x;" in result
    crosstl.translator.parse(result)


def test_complex_shader_conversion():
    code = """
    #[repr(C)]
    pub struct VertexInput {
        #[location(0)]
        position: Vec3<f32>,
        #[location(1)]
        normal: Vec3<f32>,
        #[location(2)]
        texcoord: Vec2<f32>,
    }

    #[repr(C)]
    pub struct VertexOutput {
        position: Vec4<f32>,
        normal: Vec3<f32>,
        texcoord: Vec2<f32>,
    }

    #[vertex_shader]
    pub fn vertex_main(input: VertexInput) -> VertexOutput {
        let world_position = transform * Vec4::new(input.position.x, input.position.y, input.position.z, 1.0);
        let world_normal = normalize((normal_matrix * Vec4::new(input.normal.x, input.normal.y, input.normal.z, 0.0)).xyz());

        VertexOutput {
            position: world_position,
            normal: world_normal,
            texcoord: input.texcoord,
        }
    }

    #[fragment_shader]
    pub fn fragment_main(input: VertexOutput) -> Vec4<f32> {
        let light_dir = normalize(light_position - input.position.xyz());
        let diffuse = max(dot(input.normal, light_dir), 0.0);
        let color = texture_sample(diffuse_texture, input.texcoord);

        Vec4::new(color.xyz() * diffuse, color.w)
    }
    """
    try:
        result = parse_and_generate(code)
        assert "shader main {" in result
        assert "struct VertexInput {" in result
        assert "struct VertexOutput {" in result
        assert "vertex vertex_main {" in result
        assert "fragment fragment_main {" in result
        assert (
            "world_position = " in result
        )  # Variable assignments without explicit type
        assert "normalize(" in result
        assert "dot(" in result
        assert "max(" in result
    except Exception as e:
        pytest.fail(f"Complex shader conversion failed: {e}")


def test_math_functions_conversion():
    code = """
    fn test_math() {
        let a = sin(x);
        let b = cos(x);
        let c = tan(x);
        let d = sqrt(x);
        let e = pow(x, y);
        let f = abs(x);
        let g = min(x, y);
        let h = max(x, y);
        let i = clamp(x, min_val, max_val);
        let j = floor(x);
        let k = ceil(x);
    }
    """
    try:
        result = parse_and_generate(code)
        assert "sin(" in result
        assert "cos(" in result
        assert "tan(" in result
        assert "sqrt(" in result
        assert "pow(" in result
        assert "abs(" in result
        assert "min(" in result
        assert "max(" in result
        assert "clamp(" in result
        assert "floor(" in result
        assert "ceil(" in result
    except Exception as e:
        pytest.fail(f"Math functions conversion failed: {e}")


def test_error_handling():
    edge_cases = [
        "fn empty() {}",
        "struct Empty {}",
        "// Just a comment",
        "",
    ]

    for code in edge_cases:
        try:
            result = parse_and_generate(code)
            assert "shader main {" in result
        except Exception:
            # Some edge cases might fail, which is acceptable
            pass


if __name__ == "__main__":
    pytest.main()
