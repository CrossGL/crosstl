"""GLSL output puts every atomic counter in atomic counter binding 0 and writes its offset.

GLSL 4.60 section 7.3 guarantees one atomic counter buffer binding (``gl_MaxAtomicCounterBindings`` is 1), and
glslang rejects ``atomic_uint binding is too large`` past it. GLSL output gave counters texture units instead, and
dropped a counter's ``@offset``. See :mod:`crosstl.translator.codegen.glsl_atomic_counters`.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.refusals import OpenGLRefusal

GLSLANG = shutil.which("glslangValidator")

COUNTER_AFTER_TEXTURE = """shader P {
    sampler2D tex;
    fragment {
        uniform atomic_uint hits;
        vec4 main() @gl_FragColor {
            uint n = atomicCounterIncrement(hits);
            return vec4(texture(tex, vec2(0.5)).x, float(n), 0.0, 1.0);
        }
    }
}
"""

SEVERAL_BINDINGS = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* outp;
        sampler2D tex;
        layout(binding = 0, offset = 4) uniform atomic_uint a;
        layout(binding = 2) uniform atomic_uint b;
        uniform atomic_uint c;
        uniform atomic_uint d[2];
        uniform atomic_uint e;
        void main() {
            outp[0] = atomicCounterIncrement(a) + atomicCounterIncrement(b) + atomicCounterIncrement(c)
                + atomicCounterIncrement(d[1]) + atomicCounterIncrement(e) + uint(textureLod(tex, vec2(0.0), 0.0).x);
        }
    }
}
"""


def counters(source, *declarations):
    return (
        "shader P {\n    compute {\n        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
        "        buffer uint* outp;\n"
        + "".join(f"        {declaration}\n" for declaration in declarations)
        + f"        void main() {{ outp[0] = {source}; }}\n    }}\n}}\n"
    )


def translate(tmp_path, source, name="probe.cgl", backend="opengl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def counter_lines(glsl):
    return [line for line in glsl.splitlines() if "atomic_uint" in line]


def assert_glslang_accepts(tmp_path, code, stage):
    if GLSLANG is None:
        return
    path = tmp_path / f"out.{stage}"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_counter_does_not_take_a_texture_unit(tmp_path):
    glsl = translate(tmp_path, COUNTER_AFTER_TEXTURE)

    assert counter_lines(glsl) == ["layout(binding = 0) uniform atomic_uint hits;"]
    assert_glslang_accepts(tmp_path, glsl, "frag")


def test_counters_of_several_bindings_follow_one_another_in_binding_0(tmp_path):
    glsl = translate(tmp_path, SEVERAL_BINDINGS)

    # Binding 0 holds a (4..8), c (8..12), d (12..20) and e (20..24); binding 2 follows from 24.
    assert counter_lines(glsl) == [
        "layout(binding = 0, offset = 4) uniform atomic_uint a;",
        "layout(binding = 0, offset = 24) uniform atomic_uint b;",
        "layout(binding = 0, offset = 8) uniform atomic_uint c;",
        "layout(binding = 0, offset = 12) uniform atomic_uint d[2];",
        "layout(binding = 0, offset = 20) uniform atomic_uint e;",
    ]
    assert_glslang_accepts(tmp_path, glsl, "comp")


def test_binding_records_give_each_counter_its_offset():
    codegen = GLSLCodeGen()
    codegen.generate(crosstl.translator.parse(SEVERAL_BINDINGS))
    records = {
        record["name"]: (record["binding"], record.get("offset"))
        for record in codegen.resource_binding_records()
        if record["namespace"] == "atomic-counter-buffer"
    }

    assert records == {
        "a": (0, 4),
        "b": (0, 24),
        "c": (0, 8),
        "d": (0, 12),
        "e": (0, 20),
    }


def test_glsl_round_trip_keeps_implicit_offsets(tmp_path):
    source = (
        "#version 430\n"
        "layout (binding = 0, offset = 4) uniform atomic_uint;\n"
        "layout (binding = 0) uniform atomic_uint bar, q;\n"
        "layout (binding = 0) uniform atomic_uint baz[2][3];\n"
        "layout (binding = 0, offset = 0x40) uniform atomic_uint z;\n"
        "layout (binding = 0) uniform atomic_uint w;\n"
        "layout (binding = 1) uniform atomic_uint other;\n"
        "void main() { gl_Position = vec4(float(atomicCounterIncrement(bar) + atomicCounter(baz[1][2])"
        " + atomicCounter(q) + atomicCounter(z) + atomicCounter(w) + atomicCounter(other))); }\n"
    )
    glsl = translate(tmp_path, source, name="probe.vert")

    assert counter_lines(glsl) == [
        "layout(binding = 0, offset = 4) uniform atomic_uint bar;",
        "layout(binding = 0, offset = 8) uniform atomic_uint q;",
        "layout(binding = 0, offset = 12) uniform atomic_uint baz[2][3];",
        "layout(binding = 0, offset = 64) uniform atomic_uint z;",
        "layout(binding = 0, offset = 68) uniform atomic_uint w;",
        "layout(binding = 0, offset = 72) uniform atomic_uint other;",
    ]
    assert_glslang_accepts(tmp_path, glsl, "vert")


@pytest.mark.parametrize(
    ("declarations", "message"),
    [
        (
            (
                "layout(binding = 1, offset = 4) uniform atomic_uint a;",
                "layout(binding = 1, offset = 4) uniform atomic_uint b;",
            ),
            "'a' and 'b': they overlap in atomic counter binding 1 at offset 4",
        ),
        (
            (
                "layout(binding = 1) uniform atomic_uint a[2];",
                "layout(binding = 1, offset = 4) uniform atomic_uint b;",
            ),
            "'a' and 'b': they overlap",
        ),
        (
            (
                "layout(binding = 1, offset = 6) uniform atomic_uint a;",
                "uniform atomic_uint b;",
            ),
            "'a' at offset 6: an atomic counter offset is an integer literal that is a multiple of 4",
        ),
    ],
)
def test_unplaceable_counters_are_refused(tmp_path, declarations, message):
    source = counters(
        "atomicCounterIncrement(a[0]) + atomicCounterIncrement(b)", *declarations
    )
    if "a[2]" not in declarations[0]:
        source = source.replace("a[0]", "a")

    with pytest.raises(OpenGLRefusal, match=message):
        translate(tmp_path, source)
