"""The DirectX generator walks a program for its function list once per analysis phase, and walks again when a
container that holds functions changes."""

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.codegen import directx_program_emission
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.hlsl_program_functions import HlslProgramFunctions

SAMPLED = """
shader Sampled {
    struct FragmentInput { vec2 uv; }
    struct FragmentOutput { vec4 color; }
    float scale(float v) { return v * 2.0; }
    vec4 shade(sampler2D map, vec2 uv) { return texture(map, uv) * scale(0.5); }
    vec4 shadeShadow(sampler2DShadow map, vec3 uvz) { return vec4(texture(map, uvz)); }
    fragment {
        uniform sampler2D albedoMap;
        uniform sampler2D detailMap;
        uniform sampler2DShadow shadowMap;
        vec4 tint(vec4 c) { return c * 0.5; }
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            vec4 lit = tint(shade(albedoMap, input.uv)) + texture(detailMap, input.uv);
            output.color = lit * shadeShadow(shadowMap, vec3(input.uv, 0.5));
            return output;
        }
    }
}
"""

COMPUTE = """
shader Kernels {
    RWStructuredBuffer<float> values;
    float twice(float v) { return v * 2.0; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        float offset(float v) { return v + 1.0; }
        void main() {
            uint id = gl_GlobalInvocationID.x;
            values[id] = offset(twice(values[id]));
        }
    }
}
"""


def _function(name, body=None):
    return A.FunctionNode(name, "void", [], body or [])


def _program():
    stage = A.StageNode("fragment", _function("main"), local_functions=[_function("t")])
    return A.ShaderNode(
        "P",
        None,
        stages={"fragment": stage},
        functions=[_function("f"), _function("g")],
    )


def _render(source, monkeypatch, keep):
    """Render ``source``, counting whole-program walks made by ``collect_functions``."""
    walks = []
    collect = HLSLCodeGen.collect_functions

    def counting(self, root):
        if isinstance(root, A.ShaderNode):
            kept = self.hlsl_program_functions
            if kept is None or kept.lookup(root) is None:
                walks.append(root)
        return collect(self, root)

    with monkeypatch.context() as patch:
        patch.setattr(HLSLCodeGen, "collect_functions", counting)
        if not keep:
            patch.setattr(HlslProgramFunctions, "lookup", lambda self, root: None)
        generator = HLSLCodeGen()
        code = generator.generate(crosstl.translator.parse(source))
    assert generator.hlsl_program_functions is None
    return code, len(walks)


@pytest.mark.parametrize("source", [SAMPLED, COMPUTE], ids=["sampled", "compute"])
def test_kept_list_renders_the_same_code_with_fewer_walks(source, monkeypatch):
    kept_code, kept_walks = _render(source, monkeypatch, keep=True)
    walked_code, walked_walks = _render(source, monkeypatch, keep=False)
    assert kept_code == walked_code
    assert kept_walks < walked_walks
    assert walked_walks - kept_walks >= 10


def test_kept_list_answers_only_for_its_root_with_a_new_list():
    root = _program()
    functions = HLSLCodeGen().collect_functions(root)
    assert sorted(f.name for f in functions) == ["f", "g", "main", "t"]
    kept = HlslProgramFunctions(root, functions)
    first = kept.lookup(root)
    assert first == functions and first is not functions
    first.append(_function("caller_added"))
    assert kept.lookup(root) == functions
    assert kept.lookup(_program()) is None


@pytest.mark.parametrize(
    "change",
    [
        "append_function",
        "replace_function_list",
        "remove_stage_function",
        "replace_stage",
        "add_stage",
        "add_struct",
    ],
)
def test_a_changed_function_container_makes_the_generator_walk_again(change):
    root = _program()
    generator = HLSLCodeGen()
    generator.hlsl_program_functions = HlslProgramFunctions(
        root, generator.collect_functions(root)
    )
    stage = root.stages["fragment"]
    if change == "append_function":
        root.functions.append(_function("h"))
    elif change == "replace_function_list":
        root.functions = [_function("f"), _function("g")]
    elif change == "remove_stage_function":
        stage.local_functions.pop()
    elif change == "replace_stage":
        root.stages.replace_stage(
            "fragment", 0, A.StageNode("fragment", _function("main"))
        )
    elif change == "add_stage":
        root.stages.append("fragment", A.StageNode("fragment", _function("other")))
    else:
        root.structs.append(A.StructNode("S", []))
    assert generator.hlsl_program_functions.lookup(root) is None
    walked = [
        id(node)
        for node in generator.walk_ast(root)
        if isinstance(node, A.FunctionNode)
    ]
    assert [id(f) for f in generator.collect_functions(root)] == walked


def test_an_error_inside_the_analysis_phase_drops_the_kept_list(monkeypatch):
    class Stop(Exception):
        pass

    def stop(self, ast):
        assert self.hlsl_program_functions is not None
        raise Stop

    monkeypatch.setattr(HLSLCodeGen, "validate_explicit_sampler_role_conflicts", stop)
    generator = HLSLCodeGen()
    with pytest.raises(Stop):
        generator.generate(crosstl.translator.parse(SAMPLED))
    assert generator.hlsl_program_functions is None


def test_emission_runs_without_a_kept_list(monkeypatch):
    generator = HLSLCodeGen()
    seen = []
    license_comments = directx_program_emission.source_license_comments

    def recording(ast, backend):
        seen.append(generator.hlsl_program_functions)
        return license_comments(ast, backend)

    monkeypatch.setattr(directx_program_emission, "source_license_comments", recording)
    generator.generate(crosstl.translator.parse(SAMPLED))
    assert seen == [None]
