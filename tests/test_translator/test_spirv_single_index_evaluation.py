"""SPIR-V evaluates a side-effecting subscript index once.

C, GLSL, HLSL and MSL evaluate each operand once, so ``float a = t[k--]`` decrements ``k`` once. The
SPIR-V emitter used to look at a declaration's initializer as a possible resource alias before
lowering it as a value, and to read an assignment's target back after storing through it; each
visit lowered the index again and repeated its side effect.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

SHADER = """shader P {
    struct S { float x; float y; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        float t[4];
        int k;
        int next() { return k++; }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            k = 0;
            t[0] = 1.0; t[1] = 2.0; t[2] = 3.0; t[3] = 4.0;
            BODY
        }
    }
}
"""

SIDE_EFFECT_CASES = {
    "scalar declaration": "float a = t[k--]; outp[i] = a;",
    "pre-increment declaration": "float a = t[++k]; outp[i] = a;",
    "integer declaration": "int u[4]; u[0] = 0; int a = u[k++]; outp[i] = float(a);",
    "vector declaration": (
        "vec2 w[2]; w[0] = vec2(1.0); vec2 a = w[k++]; outp[i] = a.x;"
    ),
    "struct declaration": "S s[2]; s[0].x = 1.0; S q = s[k++]; outp[i] = q.x;",
    "member declaration": "S s[2]; s[0].x = 1.0; float a = s[k++].x; outp[i] = a;",
    "vector component declaration": (
        "vec4 v = vec4(1.0); float a = v[k++]; outp[i] = a;"
    ),
    "storage buffer declaration": "float a = outp[k++]; outp[i] = a;",
    "call index declaration": "float a = t[next()]; outp[i] = a;",
    "assignment index declaration": "int j; float a = t[(j = k + 1)]; outp[i] = a;",
    "nested index declaration": "int u[4]; u[0] = 1; float a = t[u[k++]]; outp[i] = a;",
    "for initializer": "for (float a = t[k++]; a < 0.0; a += 1.0) { } outp[i] = 1.0;",
    "assignment value": "float a = (t[k++] = 2.0); outp[i] = a;",
    "assignment value statement": "float a; a = (t[k--] = 2.0); outp[i] = a;",
}


def _spirv(tmp_path, body):
    path = tmp_path / "index.cgl"
    path.write_text(SHADER.replace("BODY", body))
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed.
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _main_body(spirv):
    entry = re.search(r'OpEntryPoint GLCompute (%\S+) "main"', spirv).group(1)
    start = spirv.index(f"{entry} = OpFunction ")
    return spirv[start : spirv.index("OpFunctionEnd", start)]


def _id(spirv, name):
    return re.search(rf'OpName (%\S+) "{name}"\n', spirv).group(1)


def _stores_to(spirv, body, name):
    return len(re.findall(rf"OpStore {re.escape(_id(spirv, name))} ", body))


def _validate(tmp_path, spirv):
    if shutil.which("spirv-as") is None or shutil.which("spirv-val") is None:
        return
    assembly = tmp_path / "index.spvasm"
    binary = tmp_path / "index.spv"
    assembly.write_text(spirv)
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "body", SIDE_EFFECT_CASES.values(), ids=list(SIDE_EFFECT_CASES)
)
def test_side_effecting_index_runs_once(tmp_path, body):
    spirv = _spirv(tmp_path, body)
    main = _main_body(spirv)
    # One store for `k = 0`, one for the single evaluation of the index.
    expected = 1 if "next()" in body or "j = k" in body else 2
    assert _stores_to(spirv, main, "k") == expected
    if "next()" in body:
        assert main.count("OpFunctionCall") == 1
    if "j = k" in body:
        assert _stores_to(spirv, main, "j") == 1
    _validate(tmp_path, spirv)


def test_declaration_reads_the_element_the_side_effect_selected(tmp_path):
    spirv = _spirv(tmp_path, "float a = t[k--]; outp[i] = a;")
    main = _main_body(spirv)
    k, t = (re.escape(_id(spirv, name)) for name in ("k", "t"))
    old_k = re.search(rf"(%\S+) = OpLoad \S+ {k}\n\s*%\S+ = OpISub \S+ \1 ", main)
    assert old_k is not None
    chains = re.findall(rf"(%\S+) = OpAccessChain \S+ {t} (%\S+)\n", main)
    # Every access chain the declaration builds indexes with the value `k` had before `--`.
    declaration_chains = [chain for chain, used in chains if used == old_k.group(1)]
    assert declaration_chains
    assert len(chains) == len(declaration_chains) + 4  # t[0] .. t[3] = ...
    assert any(
        re.search(rf"= OpLoad \S+ {re.escape(chain)}\n", main)
        for chain in declaration_chains
    )


def test_storage_buffer_declaration_indexes_with_one_value(tmp_path):
    spirv = _spirv(tmp_path, "float a = outp[k++]; outp[i] = a;")
    main = _main_body(spirv)
    k, outp = (re.escape(_id(spirv, name)) for name in ("k", "outp"))
    old_k = re.search(rf"(%\S+) = OpLoad \S+ {k}\n\s*%\S+ = OpIAdd \S+ \1 ", main)
    assert old_k is not None
    indices = re.findall(rf"OpAccessChain \S+ {outp} \S+ (%\S+)\n", main)
    # The declaration's chains all use the old `k`; the final store indexes with `i`.
    assert set(indices[:-1]) == {old_k.group(1)}
    assert indices[-1] != old_k.group(1)


@pytest.mark.parametrize(
    "body",
    [
        "float a = t[k]; outp[i] = a;",
        "float a = t[k + 1]; outp[i] = a;",
        "S s[2]; s[1].x = 1.0; float a = s[k].x; outp[i] = a;",
    ],
)
def test_pure_index_declarations_still_translate(tmp_path, body):
    spirv = _spirv(tmp_path, body)
    assert _stores_to(spirv, _main_body(spirv), "k") == 1
    _validate(tmp_path, spirv)
