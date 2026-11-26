"""Unqualified module-scope variables that the program writes are private GLSL globals, not uniforms.

CrossGL's unqualified module-scope variable of a plain value type has one copy per invocation (a WGSL
``var<private>``, an HLSL ``static`` global). The GLSL generator wrote it as a ``uniform``, so a store to it was an
assignment glslang refuses (``can't modify a uniform``), and GLSL ES also refused the uniform's initializer. A
global the program writes is now declared without a storage qualifier; one it only reads stays a uniform.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.glsl_written_globals import (
    written_unqualified_global_ids,
)

GLSLANG = shutil.which("glslangValidator")


def translate(tmp_path, source, backend="opengl", name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def assert_compiles(code, stage="comp"):
    if GLSLANG is None:
        return
    result = subprocess.run(
        [GLSLANG, "--stdin", "-S", stage],
        input=code,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def written_names(program):
    """The names :func:`written_unqualified_global_ids` selects; the generator leaves resources among them alone."""
    ids = written_unqualified_global_ids(program)
    return {
        variable.name for variable in program.global_variables if id(variable) in ids
    }


WRITTEN_SCALAR = """
shader WrittenScalar {
    RWStructuredBuffer<float> fin @set(0) @binding(0);
    RWStructuredBuffer<float> outp @set(0) @binding(1);
    float counter = 1.0;
    void bump() {
        counter = ((counter * 2.0) + 1.0);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(uvec3 id @ gl_GlobalInvocationID) {
            uint i = id.x;
            float before = counter;
            counter = fin[i];
            bump();
            outp[i] = ((counter * 10.0) + before);
        }
    }
}
"""

WRITTEN_ARRAY = """
shader WrittenArray {
    RWStructuredBuffer<float> fin @set(0) @binding(0);
    RWStructuredBuffer<float> outp @set(0) @binding(1);
    float table[4];
    void stash(int k, float v) {
        table[k] = v;
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(uvec3 id @ gl_GlobalInvocationID) {
            uint i = id.x;
            stash(int(i % 4u), fin[i]);
            outp[i] = table[int(i % 4u)];
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["opengl", "gles"])
def test_written_global_is_private(tmp_path, backend):
    code = translate(tmp_path, WRITTEN_SCALAR, backend)

    assert "uniform float counter" not in code
    assert "\nfloat counter = 1.0;" in code
    assert_compiles(code)


@pytest.mark.parametrize("backend", ["opengl", "gles"])
def test_global_array_written_through_element_is_private(tmp_path, backend):
    code = translate(tmp_path, WRITTEN_ARRAY, backend)

    assert "uniform float table" not in code
    assert "\nfloat table[4];" in code
    assert_compiles(code)


@pytest.mark.parametrize(
    "statement",
    [
        "counter++;",
        "--counter;",
        "counter += 1.0;",
        "set_value(counter);",
        "float fraction = modf(fin[0], counter);",
    ],
)
def test_each_write_form_makes_the_global_private(statement):
    program = parse(f"""
        shader WriteForms {{
            RWStructuredBuffer<float> fin @set(0) @binding(0);
            float counter;
            float untouched;
            void set_value(out float value) {{
                value = 2.0;
            }}
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {{
                    {statement}
                    fin[0] = counter + untouched;
                }}
            }}
        }}
        """)
    names = written_names(program)

    assert "counter" in names
    assert "untouched" not in names


def test_read_only_global_stays_uniform(tmp_path):
    code = translate(
        tmp_path,
        """
        shader ReadOnly {
            RWStructuredBuffer<float> outp @set(0) @binding(0);
            float gain;
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main(uvec3 id @ gl_GlobalInvocationID) {
                    outp[id.x] = gain;
                }
            }
        }
        """,
    )

    assert "uniform float gain;" in code


def test_write_to_a_shadowing_local_leaves_the_global_uniform():
    program = parse("""
        shader Shadowed {
            RWStructuredBuffer<float> outp @set(0) @binding(0);
            float gain;
            float scaled(float gain) {
                gain = gain * 2.0;
                return gain;
            }
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {
                    float local = gain;
                    outp[0] = scaled(local);
                }
            }
        }
        """)

    assert "gain" not in written_names(program)


def test_qualified_globals_are_not_candidates():
    program = parse("""
        shader Qualified {
            RWStructuredBuffer<float> outp @set(0) @binding(0);
            shared float tile[4];
            compute {
                layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
                void main(uvec3 lid @ gl_LocalInvocationID) {
                    tile[lid.x] = 1.0;
                    outp[lid.x] = tile[lid.x];
                }
            }
        }
        """)

    assert "tile" not in written_names(program)
