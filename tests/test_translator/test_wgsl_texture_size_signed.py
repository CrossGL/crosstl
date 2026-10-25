from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate(texture_type, body):
    source = f"""
    shader Probe {{
        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer ivec3* outp;
            uniform {texture_type} u0;
            void main() {{
                {body}
            }}
        }}
    }}
    """
    return WGSLCodeGen().generate(Parser(Lexer(source).tokens).parse())


def test_texture_size_on_array_texture_adds_layer_count_as_signed():
    generated = generate("sampler2DArray", "ivec3 r = textureSize(u0, 0);")

    assert (
        "var r: vec3<i32> = vec3<i32>(vec3<u32>(textureDimensions(u0, 0), "
        "textureNumLayers(u0)));"
    ) in generated


def test_texture_size_on_cube_array_texture_adds_layer_count():
    generated = generate("samplerCubeArray", "ivec3 r = textureSize(u0, 0);")

    assert (
        "vec3<i32>(vec3<u32>(textureDimensions(u0, 0), textureNumLayers(u0)))"
        in generated
    )


def test_texture_size_on_2d_texture_returns_signed_vector():
    generated = generate("sampler2D", "ivec2 r = textureSize(u0, 0);")

    assert "var r: vec2<i32> = vec2<i32>(textureDimensions(u0, 0));" in generated


def test_texture_size_on_3d_and_1d_textures_returns_signed_values():
    generated_3d = generate("sampler3D", "ivec3 r = textureSize(u0, 0);")
    generated_1d = generate("sampler1D", "int r = textureSize(u0, 0);")

    assert "var r: vec3<i32> = vec3<i32>(textureDimensions(u0, 0));" in generated_3d
    assert "var r: i32 = i32(textureDimensions(u0, 0));" in generated_1d


def test_texture_size_component_access_reads_signed_vector():
    generated = generate("sampler2D", "int w = textureSize(u0, 0).x;")

    assert "var w: i32 = vec2<i32>(textureDimensions(u0, 0)).x;" in generated
