"""Rust resources are entry-point parameters or typed refusals, never default-valued statics.

The generator wrote every resource except a directly used, explicitly bound texture or sampler as a module-level
``static`` with ``Default::default()`` (or an empty ``Vec``) under a "compile-only placeholder" comment, so every
read returned a default value instead of the bound data. The rust-gpu output binds storage buffers as entry-point
slices, structured buffers included, and textures, storage images and samplers as spirv-std ``Image!`` and
``Sampler`` parameters; what rust-gpu cannot bind is refused by name.
"""

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from tests.test_translator.test_codegen.test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)


def rust(code):
    return generate_code(parse_code(tokenize_code(code)))


def test_textures_used_by_helpers_are_passed_to_them(tmp_path):
    code = rust("""
    shader Threaded {
        sampler2D colorMap @binding(2);
        sampler2D detailMap;
        vec4 detail(vec2 uv) { return texture(detailMap, uv) * 2.0; }
        vec4 shade(vec2 uv) { return detail(uv) + texture(colorMap, uv); }
        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return shade(uv) + texture(colorMap, uv);
            }
        }
    }
    """)
    # Each texture is an entry parameter with its sampler, and reaches the helpers that read it as arguments.
    assert (
        "#[spirv(descriptor_set = 0, binding = 2)] colorMap: &Image!(2D, type=f32, sampled), "
        "#[spirv(descriptor_set = 0, binding = 3)] colorMap_sampler: &Sampler" in code
    )
    assert (
        "pub fn detail(uv: Vec2, detailMap: &Image!(2D, type=f32, sampled), detailMap_sampler: &Sampler)"
        in code
    )
    assert "shade(uv, colorMap, colorMap_sampler, detailMap, detailMap_sampler)" in code
    assert "Default::default()" not in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_stage_local_buffer_object_and_image_are_refused_by_name():
    with pytest.raises(RustLoweringRefused) as refused:
        rust("""
    shader StageLocal {
        compute {
            RWStructuredBuffer<int> values @binding(1);
            image2D target @binding(4);
            void main(uint index, ivec2 pixel) {
                int value = buffer_load(values, index);
                buffer_store(values, index, value + 1);
                imageStore(target, pixel, vec4(1.0));
            }
        }
    }
    """)
    # The structured buffer binds as a storage slice; the formatless image and the plain-value parameters are
    # refused.
    assert refused.value.reasons == (
        "resource: storage image 'target' (image2D) has no format layout qualifier, and rust-gpu storage images "
        "without a format need the StorageImage*WithoutFormat capabilities",
        "stage: compute entry parameter 'index' is not a compute built-in input",
        "stage: compute entry parameter 'pixel' is not a compute built-in input",
    )


def test_stage_local_structured_buffer_binds_as_a_storage_slice():
    code = rust("""
    shader StageLocal {
        compute {
            RWStructuredBuffer<int> values @binding(1);
            void main() {
                uint index = gl_GlobalInvocationID.x;
                int value = buffer_load(values, index);
                buffer_store(values, index, value + 1);
            }
        }
    }
    """)
    assert (
        "#[spirv(storage_buffer, descriptor_set = 0, binding = 1)] values: &mut [i32]"
        in code
    )
    assert "let value: i32 = values[index as usize];" in code
    assert "values[index as usize] = (value + 1);" in code


def test_texture_arrays_bind_as_image_arrays(tmp_path):
    code = rust("""
    shader Arrays {
        sampler2D fixedTextures[2];
        sampler2D bindlessTextures[];
        fragment {
            vec4 main(int index, vec2 uv) @ gl_FragColor {
                return texture(fixedTextures[1], uv) + texture(bindlessTextures[index], uv);
            }
        }
    }
    """)
    # A sized array is a fixed-size array of images, an unsized one a RuntimeArray; each has one sampler.
    assert "fixedTextures: &[Image!(2D, type=f32, sampled); 2]" in code
    assert "bindlessTextures: &RuntimeArray<Image!(2D, type=f32, sampled)>" in code
    assert "fixedTextures_sampler: &Sampler" in code
    assert "bindlessTextures_sampler: &Sampler" in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_resource_referenced_outside_functions_is_refused():
    code = """
    shader Initialized {
        sampler2D colorMap;
        sampler2D aliasMap = colorMap;
        fragment {
            vec4 main(vec2 uv) @ gl_FragColor {
                return texture(aliasMap, uv);
            }
        }
    }
    """

    with pytest.raises(RustLoweringRefused) as refused:
        rust(code)

    assert refused.value.reasons == (
        "resource: 'aliasMap' has an initializer, and a rust-gpu resource binding has none",
        "resource: 'colorMap' is referenced where no rust-gpu entry-point parameter reaches",
        "resource: 'aliasMap' is referenced where no rust-gpu entry-point parameter reaches",
    )


def test_unreferenced_texture_is_left_out_instead_of_a_placeholder(tmp_path):
    code = rust("""
    shader Unused {
        sampler2D unusedMap @binding(7);
        compute {
            layout(local_size_x = 1) in;
            void main() {}
        }
    }
    """)
    # rust-gpu binds a resource only as an entry parameter; one no function reads needs none, and is not written
    # as a defaulted static either.
    assert "unusedMap" not in code
    assert "UNUSED_MAP" not in code
    assert "Default::default()" not in code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_store_to_a_glsl_buffer_block_parameter_is_refused():
    code = """
    shader BlockStore {
        layout(std430, binding = 0) buffer DataBlock {
            float values[4];
        } dataBlock;
        compute {
            void main() {
                dataBlock.values[0] = 1.0;
            }
        }
    }
    """

    with pytest.raises(RustLoweringRefused) as refused:
        rust(code)

    assert "the block is passed by value, so a store would not reach the buffer" in str(
        refused.value
    )
