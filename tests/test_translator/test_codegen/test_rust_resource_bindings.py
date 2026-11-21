"""Rust resources are entry-point parameters or typed refusals, never default-valued statics.

The generator wrote every resource except a directly used, explicitly bound texture or sampler as a module-level
``static`` with ``Default::default()`` (or an empty ``Vec``) under a "compile-only placeholder" comment, so every
read returned a default value instead of the bound data. The rust-gpu output binds storage buffers as entry-point
slices; textures, images, samplers and HLSL buffer objects are not lowered yet, so each is refused by name.
"""

import pytest

from crosstl.translator.codegen.rust_refusals import RustLoweringRefused
from tests.test_translator.test_codegen.test_rust_codegen import (
    generate_code,
    parse_code,
    tokenize_code,
)


def rust(code):
    return generate_code(parse_code(tokenize_code(code)))


def test_textures_used_by_helpers_are_refused_by_name():
    with pytest.raises(RustLoweringRefused) as refused:
        rust("""
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
    assert refused.value.reasons[:2] == (
        "resource: 'colorMap' (sampler2D) is not lowered to a rust-gpu resource binding",
        "resource: 'detailMap' (sampler2D) is not lowered to a rust-gpu resource binding",
    )


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
    assert refused.value.reasons[:2] == (
        "resource: 'values' (RWStructuredBuffer<int>) is not lowered to a rust-gpu resource binding",
        "resource: 'target' (image2D) is not lowered to a rust-gpu resource binding",
    )


def test_texture_arrays_are_refused_by_name():
    with pytest.raises(RustLoweringRefused) as refused:
        rust("""
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
    assert refused.value.reasons[:2] == (
        "resource: 'fixedTextures' (sampler2D[2]) is not lowered to a rust-gpu resource binding",
        "resource: 'bindlessTextures' (sampler2D[]) is not lowered to a rust-gpu resource binding",
    )


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


def test_unreferenced_texture_is_refused_instead_of_a_placeholder():
    with pytest.raises(RustLoweringRefused) as refused:
        rust("""
    shader Unused {
        sampler2D unusedMap @binding(7);
        compute {
            void main() {}
        }
    }
    """)
    assert refused.value.reasons[:1] == (
        "resource: 'unusedMap' (sampler2D) is not lowered to a rust-gpu resource binding",
    )


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
