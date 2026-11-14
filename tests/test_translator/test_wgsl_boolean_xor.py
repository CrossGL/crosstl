"""WGSL output writes a boolean exclusive or as inequality, because WGSL defines ``^`` for integers only."""

import shutil
import subprocess

import pytest

import crosstl

PRELUDE = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        int pick(int k) {
            return k & 1;
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
"""
EPILOGUE = """
        }
    }
}
"""


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PRELUDE + body + EPILOGUE)
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def test_scalar_and_vector_xor_become_inequality(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            bool p = iin[i] > 0;
            bool q = iin[i] > 2;
            bool r = p ^ q;
            bvec2 a = bvec2(p, q);
            bvec2 b = bvec2(q, q);
            bvec2 c = a ^ b;
            outp[i] = (r ? 1.0 : 0.0) + (c.y ? 2.0 : 0.0);
""",
    )
    assert "var r: bool = (p != q);" in wgsl
    assert "var c: vec2<bool> = (a != b);" in wgsl
    assert " ^ " not in wgsl


def test_xor_of_comparisons_and_literals(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            bool p = iin[i] > 0;
            bool s = (p ^ true) ^ (iin[i] == 3);
            bool n = !p ^ (iin[i] < 1 && p);
            outp[i] = (s ? 1.0 : 0.0) + (n ? 2.0 : 0.0);
""",
    )
    assert "var s: bool = ((p != true) != (iin[i] == 3));" in wgsl
    assert "var n: bool = (!p != ((iin[i] < 1) && p));" in wgsl


def test_compound_xor_on_scalar_vector_and_element(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            bool p = iin[i] > 0;
            bool flags[2];
            flags[0] = p;
            flags[1] = false;
            p ^= iin[i] > 2;
            flags[1] ^= p;
            bvec3 v = bvec3(true, false, true);
            v ^= bvec3(iin[i] > 1);
            outp[i] = (p ? 1.0 : 0.0) + (flags[1] ? 2.0 : 0.0) + (v.z ? 4.0 : 0.0);
""",
    )
    assert "p = (p != (iin[i] > 2));" in wgsl
    assert "flags[1] = (flags[1] != p);" in wgsl
    assert "v = (v != vec3<bool>((iin[i] > 1)));" in wgsl
    assert "^=" not in wgsl


def test_compound_xor_target_that_calls_a_function_is_evaluated_once(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            bool flags[2];
            flags[0] = iin[i] > 0;
            flags[1] = false;
            flags[pick(iin[i])] ^= true;
            outp[i] = flags[0] ? 1.0 : 0.0;
""",
    )
    assert "let xor_target = &flags[pick(iin[i])];" in wgsl
    assert "*xor_target = (*xor_target != true);" in wgsl
    assert wgsl.count("pick(iin[i])") == 1


def test_compound_xor_in_for_update(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            bool t = false;
            for (int k = 0; k < 3; t ^= true) {
                k++;
            }
            outp[i] = t ? 1.0 : 0.0;
""",
    )
    assert "(k < 3); t = (t != true))" in wgsl


def test_for_update_whose_xor_target_calls_a_function_is_refused(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(PRELUDE + """
            bool flags[2];
            flags[0] = false;
            flags[1] = false;
            for (int k = 0; k < 3; flags[pick(k)] ^= true) {
                k++;
            }
            outp[i] = flags[0] ? 1.0 : 0.0;
""" + EPILOGUE)
    with pytest.raises(ValueError, match="boolean \\^= whose target calls a function"):
        crosstl.translate(str(path), backend="wgsl")


def test_integer_xor_keeps_the_operator(tmp_path):
    wgsl = translate(
        tmp_path,
        """
            int m = iin[i] ^ 5;
            uint u = i ^ 3u;
            m ^= 2;
            outp[i] = float(m) + float(u);
""",
    )
    assert "var m: i32 = (iin[i] ^ 5);" in wgsl
    assert "var u: u32 = (i ^ 3u);" in wgsl
    assert "m ^= 2;" in wgsl


def test_hlsl_boolean_xor_reaches_wgsl_as_inequality(tmp_path):
    path = tmp_path / "kernel.hlsl"
    path.write_text("""RWStructuredBuffer<int> iin : register(u0);
RWStructuredBuffer<float> outp : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool p = iin[id.x] > 0;
    bool q = iin[id.x] > 2;
    bool r = p ^ q;
    r ^= p;
    outp[id.x] = r ? 1.0 : 0.0;
}
""")
    wgsl = crosstl.translate(str(path), backend="wgsl", format_output=False)
    assert "(p != q)" in wgsl
    assert "r = (r != p);" in wgsl
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(wgsl)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
