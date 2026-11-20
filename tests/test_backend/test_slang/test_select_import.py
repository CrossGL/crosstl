"""Slang's select(c, t, f) imports as CrossGL's select(f, t, c)."""

from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser


def to_crossgl(code):
    ast = SlangParser(SlangLexer(code).tokenize()).parse()
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


def kernel(body, prelude=""):
    return f"""
{prelude}
RWStructuredBuffer<float> data;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    float s = data[id.x];
    {body}
}}
"""


SELECT_BODY = """
bool2 a0 = bool2(s > 0.0, s > 1.0);
int2 a1 = int2(1, 2);
int2 a2 = int2(3, 4);
int2 r = select(a0, a1, a2);
"""

USER_SELECT = (
    "float select(float first, float second, bool flag) "
    "{ if (flag) { return first; } return second; }"
)


def test_select_imports_with_the_condition_last():
    code = to_crossgl(kernel(SELECT_BODY))
    assert "ivec2 r = select(a2, a1, a0);" in code


def test_user_defined_select_is_left_alone():
    code = to_crossgl(kernel("float r = select(1.0, 2.0, s > 0.0);", USER_SELECT))
    assert "float r = select(1.0, 2.0," in code
