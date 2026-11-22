"""A declaration whose initializer is a vector or matrix of another shape is refused (#4599).

``vec4 m = a;`` with ``vec3 a`` and ``mat3x4 m = a;`` with ``mat4x3 a`` are invalid CrossGL, as in GLSL. The Vulkan
generator used to write an ``OpStore`` of a value whose type is not the pointer's, which ``spirv-val`` rejects, and
the text generators copied the declaration through to compilers that reject it.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.diagnostics import TranslationDiagnostic
from crosstl.translator.semantics import InitializerShapeMismatchError

TARGETS = ["vulkan", "opengl", "directx", "metal", "slang", "wgsl", "cuda"]

COMPUTE = """
shader main {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4 data[];
        void main() {{
            {body}
        }}
    }}
}}
"""

RAY = textwrap.dedent("""
    shader main {{
        struct Payload {{
            vec4 color @ ray_payload_read(caller) @ ray_payload_write(anyhit, closesthit);
        }}
        ray_closest_hit {{
            {helper}
            void main(inout Payload payload @ payload, in BuiltInTriangleIntersectionAttributes attributes @ hit_attribute) {{
                {body}
            }}
        }}
    }}
""")


def translate(tmp_path, source, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


MISMATCHES = [
    ("vec3 a = vec3(1.0); vec4 m = a; data[0] = m;", "vec4", "vec3"),
    ("vec4 a = vec4(1.0); vec2 m = a; data[0] = vec4(m, m);", "vec2", "vec4"),
    ("ivec3 a = ivec3(1); ivec4 m = a; data[0] = vec4(m);", "ivec4", "ivec3"),
    ("vec3 a = vec3(1.0); vec4 m = a * 2.0; data[0] = m;", "vec4", "vec3"),
    ("mat3 a = mat3(1.0); mat4 m = a; data[0] = m[0];", "mat4", "mat3"),
    ("mat4x3 a = mat4x3(1.0); mat3x4 m = a; data[0] = m[0];", "mat3x4", "mat4x3"),
    ("mat3x4 m = mat4x3(1.0); data[0] = m[0];", "mat3x4", "mat4x3"),
    ("vec4 a = vec4(1.0); mat2 m = a; data[0] = vec4(m[0], m[1]);", "mat2", "vec4"),
    ("vec4 m = vec3(1.0); data[0] = m;", "vec4", "vec3"),
    # Matrix arithmetic has the shape GLSL 4.60 section 5.10 gives it.
    (
        "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec4 m = a * v; data[0] = m;",
        "vec4",
        "vec3",
    ),
    (
        "mat4x3 a = mat4x3(1.0); mat3x4 b = mat3x4(1.0); mat4 m = a * b; data[0] = m[0];",
        "mat4",
        "mat3",
    ),
    (
        "mat3x4 a = mat3x4(1.0); vec4 v = vec4(1.0); vec4 m = v * a; data[0] = m;",
        "vec4",
        "vec3",
    ),
    ("mat4x3 a = mat4x3(1.0); mat3x4 m = -a; data[0] = m[0];", "mat3x4", "mat4x3"),
    ("mat4x3 a = mat4x3(1.0); mat3x4 m = 2.0 * a; data[0] = m[0];", "mat3x4", "mat4x3"),
    ("mat4x3 a = mat4x3(1.0); mat3x4 m = a + a; data[0] = m[0];", "mat3x4", "mat4x3"),
    # A matrix product inside further vector arithmetic keeps its shape.
    (
        "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec4 m = -(a * v); data[0] = m;",
        "vec4",
        "vec3",
    ),
    (
        "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec4 m = (a * v) * 2.0; data[0] = m;",
        "vec4",
        "vec3",
    ),
    (
        "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec4 m = a * v + w; data[0] = m;",
        "vec4",
        "vec3",
    ),
    (
        "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec4 m = flag ? a * v : w; data[0] = m;",
        "vec4",
        "vec3",
    ),
]


@pytest.mark.parametrize("body,declared,initializer", MISMATCHES)
@pytest.mark.parametrize("backend", TARGETS)
def test_mismatched_shape_is_refused(tmp_path, backend, body, declared, initializer):
    with pytest.raises(InitializerShapeMismatchError) as caught:
        translate(tmp_path, COMPUTE.format(body=body), backend)
    error = caught.value
    assert isinstance(error, TranslationDiagnostic)
    assert error.code == "project.translate.initializer-shape-mismatch"
    assert error.name == "m"
    assert error.declared_type == declared
    assert error.initializer_type == initializer
    assert f"declaration of 'm' has type '{declared}'" in str(error)


@pytest.mark.parametrize(
    "body,declared,initializer",
    [
        ("mat3x4 m = ObjectToWorld3x4();", "mat3x4", "mat4x3"),
        ("mat3x4 m = ObjectToWorld();", "mat3x4", "mat4x3"),
        ("mat4x3 m = WorldToObject4x3();", "mat4x3", "mat3x4"),
    ],
)
@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx", "slang"])
def test_dxr_transform_of_the_other_shape_is_refused(
    tmp_path, backend, body, declared, initializer
):
    source = RAY.format(helper="", body=body + " payload.color = vec4(m[0][0]);")
    with pytest.raises(InitializerShapeMismatchError) as caught:
        translate(tmp_path, source, backend)
    assert caught.value.declared_type == declared
    assert caught.value.initializer_type == initializer


MATCHING = [
    "vec3 a = vec3(1.0); vec4 m = vec4(a, 1.0); data[0] = m;",
    "vec4 a = vec4(1.0); vec3 m = a.xyz; data[0] = vec4(m, 1.0);",
    "vec4 m = vec4(1.0); vec4 n = m; data[0] = n;",
    "ivec4 a = ivec4(1); vec4 m = vec4(a); data[0] = m;",
    "mat4x3 a = mat4x3(1.0); mat4x3 m = a; data[0] = vec4(m[0], 1.0);",
    "mat4 a = mat4(1.0); vec4 m = a[1]; data[0] = m;",
    "mat4x3 a = mat4x3(1.0); vec3 m = a * vec4(1.0); data[0] = vec4(m, 1.0);",
    "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 m = a * v; data[0] = vec4(m, 1.0);",
    "mat4x3 a = mat4x3(1.0); vec3 v = vec3(1.0); vec4 m = v * a; data[0] = m;",
    "mat3x4 a = mat3x4(1.0); vec3 v = vec3(1.0); vec4 m = a * v; data[0] = m;",
    "mat4x3 a = mat4x3(1.0); mat3x4 b = mat3x4(1.0); mat3 m = a * b; data[0] = vec4(m[0], 1.0);",
    "mat3x4 a = mat3x4(1.0); mat4x3 b = mat4x3(1.0); mat4 m = a * b; data[0] = m[0];",
    "mat4x3 a = mat4x3(1.0); mat4x3 m = -a * 2.0 + a / 2.0; data[0] = vec4(m[0], 1.0);",
    "mat4 a = mat4(1.0); mat4 m = true ? a : a * a; data[0] = m[0];",
    # A matrix product inside further vector arithmetic.
    "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec3 m = -(a * v); data[0] = vec4(m, 1.0);",
    "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec3 m = (a * v) * 2.0; data[0] = vec4(m, 1.0);",
    "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec3 m = a * v + w; data[0] = vec4(m, 1.0);",
    "mat4x3 a = mat4x3(1.0); vec4 v = vec4(1.0); vec3 w = vec3(1.0); bool flag = true; vec3 m = flag ? a * v : w; data[0] = vec4(m, 1.0);",
    # Same shape, other component type: the conversion is not this check's business.
    "ivec2 a = ivec2(1); uvec2 m = a; data[0] = vec4(m, m);",
    "ivec4 a = ivec4(1); bvec4 m = a; data[0] = vec4(m);",
    # A scalar initializer is a separate question and is left alone.
    "float a = 1.0; vec4 m = vec4(a); data[0] = m;",
]


@pytest.mark.parametrize("body", MATCHING)
@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx", "metal"])
def test_matching_shape_still_translates(tmp_path, backend, body):
    assert translate(tmp_path, COMPUTE.format(body=body), backend)


@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx", "slang"])
def test_dxr_transform_of_the_declared_shape_still_translates(tmp_path, backend):
    body = (
        "mat4x3 a = ObjectToWorld3x4(); mat3x4 b = ObjectToWorld4x3(); "
        "payload.color = vec4(a[0][0] + b[0][0]);"
    )
    assert translate(tmp_path, RAY.format(helper="", body=body), backend)


@pytest.mark.parametrize("backend", ["vulkan", "directx"])
def test_function_the_program_defines_under_a_dxr_name_has_its_own_type(
    tmp_path, backend
):
    helper = "mat3x4 ObjectToWorld3x4() { return mat3x4(1.0); }"
    body = "mat3x4 m = ObjectToWorld3x4(); payload.color = vec4(m[0][0]);"
    assert translate(tmp_path, RAY.format(helper=helper, body=body), backend)


def test_texel_read_into_a_narrower_vector_still_translates(tmp_path):
    # A texel read takes the leading components of a narrower destination.
    source = """shader S {
    compute {
        @ format(rg32f) image2D rg;
        @ numthreads(1, 1, 1)
        void main() {
            vec2 t = imageLoad(rg, ivec2(0, 0));
            imageStore(rg, ivec2(1, 1), vec4(t, 0.0, 1.0));
        }
    }
}
"""
    assert translate(tmp_path, source, "opengl")


MODULE = """
shader main {{
    {module}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer vec4 data[];
        {stage}
        void main() {{ data[0] = {use}; }}
    }}
}}
"""


@pytest.mark.parametrize(
    "module,stage,use",
    [
        ("const vec3 B = vec3(1.0); const vec4 m = B;", "", "m"),
        ("vec3 B = vec3(1.0); vec4 m = B;", "", "m"),
        ("const mat4x3 A = mat4x3(1.0); const vec4 m = A * vec4(1.0);", "", "m"),
        ("const vec3 B = vec3(1.0);", "vec4 m = B;", "m"),
    ],
)
@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx", "metal"])
def test_module_and_stage_scope_declarations_are_checked(
    tmp_path, backend, module, stage, use
):
    source = MODULE.format(module=module, stage=stage, use=use)
    with pytest.raises(InitializerShapeMismatchError) as caught:
        translate(tmp_path, source, backend)
    assert caught.value.name == "m"
    assert caught.value.declared_type == "vec4"
    assert caught.value.initializer_type == "vec3"


@pytest.mark.parametrize("backend", ["vulkan", "opengl", "directx", "metal"])
def test_matching_module_scope_declarations_still_translate(tmp_path, backend):
    source = MODULE.format(
        module="const vec3 B = vec3(1.0); const vec4 G = vec4(B, 1.0);",
        stage="",
        use="G",
    )
    assert translate(tmp_path, source, backend)
