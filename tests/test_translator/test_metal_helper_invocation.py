"""Metal output of CrossGL's ``gl_HelperInvocation`` fragment input.

Metal has no input attribute for helper invocations; a fragment function, or a function it calls, asks with
``simd_is_helper_thread()`` (Metal Shading Language Specification, section 6.9.2). A fragment-input structure member
or entry-point parameter bound to ``@ gl_HelperInvocation`` used to be refused as a stage varying, and a bare read of
the built-in was written verbatim, which the Metal compiler rejects. The Metal compiler checks each translation where
it is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.ast import IdentifierNode
from crosstl.translator.codegen.metal_stage_input_builtins import (
    lower_metal_helper_invocation_reads,
)

_MEMBER = """
shader Probe {
    struct FragmentInput {
        vec2 uv;
        bool helper @ gl_HelperInvocation;
    }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, input.helper ? 1.0 : 0.0, 1.0);
            return output;
        }
    }
}
"""

_ONLY_MEMBER = """
shader Probe {
    struct FragmentInput { bool helper @ gl_HelperInvocation; }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.helper ? 1.0 : 0.0);
            return output;
        }
    }
}
"""

_PARAMETER = """
shader Probe {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main(FragmentInput input, bool helper @ gl_HelperInvocation) {
            FragmentOutput output;
            output.color = vec4(input.uv, helper ? 1.0 : 0.0, 1.0);
            return output;
        }
    }
}
"""

# The built-in is read in the entry point and in a helper function the entry point calls.
_BARE = """
shader Probe {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    float shade(float x) { return gl_HelperInvocation ? 0.0 : x; }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, shade(1.0), gl_HelperInvocation ? 1.0 : 0.0);
            return output;
        }
    }
}
"""

_WRITE = """
shader Probe {
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main() {
            FragmentOutput output;
            gl_HelperInvocation = false;
            output.color = vec4(1.0);
            return output;
        }
    }
}
"""

_GLSL = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = vec4(uv, gl_HelperInvocation ? 1.0 : 0.0, 1.0); }
"""

_HLSL = """
float4 main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(uv, IsHelperLane() ? 1.0 : 0.0, 1.0);
}
"""

_PLAIN = """
shader Probe {
    struct FragmentOutput { vec4 color; }
    fragment {
        FragmentOutput main() { FragmentOutput output; output.color = vec4(1.0); return output; }
    }
}
"""


def _translate(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal")


def _compile(tmp_path, code):
    if not shutil.which("xcrun"):
        return
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(source)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    assert "warning" not in result.stderr, result.stderr + code


def _entry_signature(code):
    match = re.search(r"^fragment \w+ \w+\((.*?)\) \{", code, flags=re.S | re.M)
    assert match is not None, code
    return " ".join(match.group(1).split())


def test_input_member_is_filled_from_the_helper_thread_query(tmp_path):
    code = _translate(tmp_path, _MEMBER)

    assert _entry_signature(code) == "FragmentInput_StageIn input_stage_in [[stage_in]]"
    stage_in = re.search(r"struct FragmentInput_StageIn \{(.*?)\};", code, re.S)
    assert stage_in is not None and "helper" not in stage_in.group(1), code
    assert "input.helper = simd_is_helper_thread();" in code
    assert "gl_HelperInvocation" not in code
    _compile(tmp_path, code)


def test_structure_of_only_the_helper_input_takes_no_stage_input(tmp_path):
    code = _translate(tmp_path, _ONLY_MEMBER)

    assert _entry_signature(code) == ""
    assert "input.helper = simd_is_helper_thread();" in code
    _compile(tmp_path, code)


def test_entry_parameter_becomes_a_local(tmp_path):
    code = _translate(tmp_path, _PARAMETER)

    assert _entry_signature(code) == "FragmentInput input [[stage_in]]"
    assert "bool helper = simd_is_helper_thread();" in code
    _compile(tmp_path, code)


def test_bare_reads_call_the_query_in_entry_points_and_helpers(tmp_path):
    code = _translate(tmp_path, _BARE)

    assert "return simd_is_helper_thread() ? 0.0 : x;" in code
    assert code.count("simd_is_helper_thread()") == 2
    assert "gl_HelperInvocation" not in code
    _compile(tmp_path, code)


@pytest.mark.parametrize("source, suffix", [(_GLSL, ".frag"), (_HLSL, ".hlsl")])
def test_glsl_and_hlsl_helper_reads_reach_metal(tmp_path, source, suffix):
    code = _translate(tmp_path, source, suffix)

    assert "simd_is_helper_thread()" in code
    assert "gl_HelperInvocation" not in code
    _compile(tmp_path, code)


def test_assignment_to_the_built_in_is_refused(tmp_path):
    with pytest.raises(Exception, match="assignment to gl_HelperInvocation") as info:
        _translate(tmp_path, _WRITE)

    assert "metal-lowering-unsupported" in str(info.value)


def test_programs_without_the_built_in_are_returned_unchanged():
    ast = parse(_PLAIN)

    assert lower_metal_helper_invocation_reads(ast, pytest.fail) is ast


def _identifier_names(ast):
    return [node.name for node in ast.walk() if isinstance(node, IdentifierNode)]


def test_rewrite_leaves_the_input_ast_unchanged():
    ast = parse(_BARE)

    rewritten = lower_metal_helper_invocation_reads(ast, pytest.fail)

    assert rewritten is not ast
    assert _identifier_names(ast).count("gl_HelperInvocation") == 2
    assert "gl_HelperInvocation" not in _identifier_names(rewritten)
    assert _identifier_names(rewritten).count("simd_is_helper_thread") == 2
