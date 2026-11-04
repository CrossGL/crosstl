"""Rule passes that look at every stage scan a shared helper function once, and thread it once for all stages.

A module with many entry points (one per template instantiation of a Metal kernel, say) shares its helper
functions between all of them. A pass that walks every helper again for each entry costs entries times module
size, which is what made large imported modules take minutes before a refusal. A helper that reads a built-in
gets one parameter for it, which every stage that calls the helper fills.
"""

import re

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import builtin_input_reads, glsl_mesh_stages

HELPERS = """
    float scale(float x) { float twice = x * 2.0; return twice; }
    float offset(float x) { return scale(x) + 1.0; }
"""


def _program(stage_blocks):
    return Parser(
        Lexer("shader S {" + HELPERS + " ".join(stage_blocks) + "}").tokens
    ).parse()


def _compute_blocks(count):
    return [
        "compute { void main() { float v = offset(float(gl_GlobalInvocationID.x)); } }"
    ] * count


def _helper_body_walks(program, run):
    # Key by identity but keep each body alive, and undo the patch before returning: a counter left installed would
    # wrap the next measurement and count that program's nodes whenever one reuses a freed body's address.
    bodies = {id(function.body): function for function in program.functions}
    walks = {function.name: 0 for function in bodies.values()}
    walk = builtin_input_reads._walk

    def counting(node, seen=None):
        if seen is None and id(node) in bodies:
            walks[bodies[id(node)].name] += 1
        return walk(node, seen)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtin_input_reads, "_walk", counting)
        run(program)
    return walks


def test_builtin_input_pass_walks_shared_helpers_once():
    def run(program):
        builtin_input_reads.pass_builtin_inputs(program, "directx")

    one = _helper_body_walks(_program(_compute_blocks(1)), run)
    many = _helper_body_walks(_program(_compute_blocks(6)), run)

    assert one == many
    assert all(count <= 2 for count in many.values()), many


SHARED_HELPER = """shader S {
    float shade() { return gl_FrontFacing ? 1.0 : 0.0; }
    fragment { vec4 main() @location(0) { return vec4(shade()); } }
    fragment { vec4 main() @location(0) { return vec4(shade() * 2.0); } }
}
"""


def test_builtin_input_pass_threads_a_shared_helper_once_for_every_stage():
    # Both fragment stages call the module-scope helper. The first stage gives it a parameter; the second must pass
    # its own built-in to that parameter rather than call the helper without an argument.
    program = Parser(Lexer(SHARED_HELPER).tokens).parse()

    changed = builtin_input_reads.pass_builtin_inputs(program, "metal")

    shade = next(f for f in program.functions if f.name == "shade")
    assert changed == 2
    assert len(shade.parameters) == 1


@pytest.mark.parametrize(
    "backend, attribute",
    [("metal", r"\[\[front_facing\]\]"), ("directx", r":\s*SV_IsFrontFace")],
)
def test_every_stage_passes_its_built_in_to_a_shared_helper(
    tmp_path, backend, attribute
):
    path = tmp_path / "shared.cgl"
    path.write_text(SHARED_HELPER)

    code = crosstl.translate(str(path), backend=backend, format_output=False)

    assert len(re.findall(r"bool\s+\w+\s*" + attribute, code)) == 2, code
    assert not re.search(r"\bshade\(\s*\)", code), code


def test_mesh_pass_skips_identifier_scans_without_task_payloads(monkeypatch):
    calls = []
    scan = glsl_mesh_stages._identifier_names

    def counting(root):
        calls.append(root)
        return scan(root)

    monkeypatch.setattr(glsl_mesh_stages, "_identifier_names", counting)
    program = _program(_compute_blocks(6))

    assert glsl_mesh_stages._payload_users(program) == {}
    assert calls == []
