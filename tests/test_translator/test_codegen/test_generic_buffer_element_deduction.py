"""Generic functions deduce their type parameter from a ``buffer T*`` element.

``pick(outp[i])`` for ``buffer float* outp`` must specialize ``pick_float``
on every target. Specialization collection used to run before backends
populated their global-variable type maps, so the ``outp[i]`` argument typed
as unknown: Metal emitted a ``pick_float`` call without its declaration,
while CUDA, HIP, Slang, GLSL and SPIR-V refused the program outright.
"""

import textwrap

import pytest

import crosstl

TARGETS = ("opengl", "directx", "metal", "vulkan", "slang", "cuda", "hip")

PICK = textwrap.dedent("""
    shader P {
        buffer float* outp;
        T pick<T>(T x) { return x; }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = pick(outp[i]) + 1.0;
            }
        }
    }
    """)

SEL2 = textwrap.dedent("""
    shader P {
        buffer float* outp;
        T sel2<T>(T a, T b) { return a; }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = sel2(outp[0], outp[1]);
            }
        }
    }
    """)


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("backend", TARGETS)
def test_buffer_element_deduces_scalar_parameter(tmp_path, backend):
    code = _translate(tmp_path, PICK, backend)
    if backend == "vulkan":
        # SPIR-V names nothing: the specialization is one OpFunction called
        # once from main, not a dropped or unresolved generic helper.
        assert code.count("OpFunction ") >= 2
        assert "OpFunctionCall" in code
        return
    assert "pick_float(" in code
    # Declaration plus call site: Metal used to emit only the call.
    assert code.count("pick_float") >= 2
    assert "pick(" not in code


@pytest.mark.parametrize("backend", TARGETS)
def test_two_buffer_elements_deduce_shared_parameter(tmp_path, backend):
    code = _translate(tmp_path, SEL2, backend)
    if backend == "vulkan":
        assert code.count("OpFunction ") >= 2
        assert "OpFunctionCall" in code
        return
    assert "sel2_float(" in code
    assert code.count("sel2_float") >= 2
    assert "sel2(" not in code
