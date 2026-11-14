"""Rust resources are bound entry parameters threaded through helpers, never default-valued statics.

The generator wrote every resource except a directly used, explicitly bound texture or sampler as a module-level
``static`` with ``Default::default()`` (or an empty ``Vec``) under a "compile-only placeholder" comment, so every
read returned a default value instead of the bound data.
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


def binding(set_index, index, storage=False):
    storage_class = "storage_buffer, " if storage else ""
    return (
        '#[cfg_attr(feature = "crossgl_gpu", spirv('
        f"{storage_class}descriptor_set = {set_index}, binding = {index}))] "
    )


def test_resource_used_by_helpers_is_threaded_to_them_and_bound_at_the_entry(tmp_path):
    generated = rust("""
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

    assert "static COLOR_MAP" not in generated
    assert "static DETAIL_MAP" not in generated
    assert "pub fn detail(uv: Vec2<f32>, detailMap: Texture2D<f32>)" in generated
    assert (
        "pub fn shade(uv: Vec2<f32>, colorMap: Texture2D<f32>, detailMap: Texture2D<f32>)"
        in generated
    )
    assert "shade(uv, colorMap, detailMap)" in generated
    assert "detail(uv, detailMap)" in generated
    assert f"{binding(0, 2)}colorMap: Texture2D<f32>" in generated
    assert f"{binding(0, 3)}detailMap: Texture2D<f32>" in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_stage_local_buffer_and_image_are_bound_entry_parameters(tmp_path):
    generated = rust("""
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

    assert "static VALUES" not in generated
    assert "static TARGET" not in generated
    assert f"{binding(0, 1, storage=True)}values: RWStructuredBuffer<i32>" in generated
    assert f"{binding(0, 4)}target: Image2D<Vec4<f32>>" in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


def test_resource_arrays_are_bound_entry_parameters(tmp_path):
    generated = rust("""
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

    assert "static FIXED_TEXTURES" not in generated
    assert "static BINDLESS_TEXTURES" not in generated
    assert "fixedTextures: [Texture2D<f32>; 2]" in generated
    assert "bindlessTextures: Vec<Texture2D<f32>>" in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


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

    message = str(refused.value)
    assert "resource: 'colorMap' (sampler2D) would be a module-level static" in message
    assert "referenced outside the program's functions" in message
    assert "resource: 'aliasMap' (sampler2D)" in message
    assert "a resource with an initializer cannot be bound" in message


def test_unreferenced_resource_is_declared_without_a_placeholder_comment(tmp_path):
    generated = rust("""
    shader Unused {
        sampler2D unusedMap @binding(7);
        compute {
            void main() {}
        }
    }
    """)

    assert (
        "// CrossGL resource metadata: name=unusedMap kind=texture set=0 binding=7"
        in generated
    )
    assert "compile-only placeholder" not in generated
    assert_generated_rust_smoke_compiles(generated, tmp_path)


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
