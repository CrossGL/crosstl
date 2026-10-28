"""HIP output calls device library functions through the overload their arguments select.

HIP declares ``__ffsll`` for ``long long`` and ``unsigned long long``, so an ``int64_t``, ``uint64_t`` or 32-bit
argument is ambiguous; ``norm``/``rnorm`` take ``const double*`` and the float forms are ``normf``/``rnormf``; and
``atomicAddNoRet`` takes a pointer. hipcc rejected the calls as CrossGL wrote them.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.hip_library_calls import ffs64_argument
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused

KERNEL = """shader P {{
    compute {{
        layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
        shared float acc[4];

        void main() {{
            uint i = gl_LocalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def translate_hip(tmp_path, body):
    path = tmp_path / "kernel.cgl"
    path.write_text(KERNEL.format(body=body))
    return crosstl.translate(str(path), backend="hip", format_output=False)


def kernel_body(code):
    return code[code.index("__global__ void compute_main") :]


def test_ffsll_argument_selects_the_unsigned_long_long_overload(tmp_path):
    code = kernel_body(
        translate_hip(
            tmp_path,
            """uint64_t u = uint64_t(i) << 40u;
            int64_t s = int64_t(i) - int64_t(3);
            uint a = __ffsll(u);
            uint b = __ffsll(s);
            uint c = __ffsll(i);
            acc[i] = float(a + b + c);""",
        )
    )

    assert "__ffsll(static_cast<unsigned long long>(u))" in code
    assert "__ffsll(static_cast<unsigned long long>(s))" in code
    assert "__ffsll(static_cast<unsigned long long>(i))" in code
    assert not re.search(r"__ffsll\((?!static_cast<unsigned long long>)", code)


def test_ffsll_keeps_an_argument_that_matches_an_overload():
    assert ffs64_argument("x", "unsigned long long") == "x"
    assert ffs64_argument("x", "long long") == "x"
    assert ffs64_argument("x", "uint64_t") == "static_cast<unsigned long long>(x)"
    assert ffs64_argument("x", None) == "static_cast<unsigned long long>(x)"


def test_find_lsb_of_64_bit_values_selects_one_ffsll_overload(tmp_path):
    code = kernel_body(
        translate_hip(
            tmp_path,
            """uint64_t u = uint64_t(i) << 40u;
            int64_t s = int64_t(i) - int64_t(3);
            int d = findLSB(u);
            int e = findLSB(s);
            int f = findLSB(i);
            acc[i] = float(d + e + f);""",
        )
    )

    assert "(__ffsll(static_cast<unsigned long long>(u)) - 1)" in code
    assert "(__ffsll(static_cast<unsigned long long>(s)) - 1)" in code
    assert "(__ffs(i) - 1)" in code


def test_euclidean_norm_is_named_for_the_element_type(tmp_path):
    code = kernel_body(
        translate_hip(
            tmp_path,
            """float f = float(i);
            float farr[2];
            farr[0] = f;
            farr[1] = 1.0;
            double darr[2];
            darr[0] = 3.0;
            darr[1] = 4.0;
            float n1 = norm(1, &f);
            float n2 = rnorm(2, farr);
            double n3 = norm(2, darr);
            double n4 = rnorm(2, darr);
            acc[i] = n1 + n2 + float(n3 + n4);""",
        )
    )

    assert "normf(1, &f)" in code
    assert "rnormf(2, farr)" in code
    assert re.search(r"[^r]norm\(2, darr\)", code)
    assert re.search(r"[^f]rnorm\(2, darr\)", code)


def test_euclidean_norm_of_other_elements_is_refused(tmp_path):
    with pytest.raises(HipLoweringRefused, match="norm of int elements"):
        translate_hip(
            tmp_path,
            """int iarr[2];
            iarr[0] = 3;
            iarr[1] = 4;
            acc[i] = norm(2, iarr);""",
        )


def test_atomic_add_no_ret_takes_the_address_of_its_target(tmp_path):
    code = kernel_body(
        translate_hip(
            tmp_path,
            """atomicAddNoRet(acc[i], float(i));""",
        )
    )

    assert "atomicAddNoRet(&acc[i], float(i));" in code
    assert "atomicAddNoRet_system" not in code


def test_user_function_named_norm_is_called_as_written(tmp_path):
    path = tmp_path / "kernel.cgl"
    path.write_text("""shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        shared float acc[1];

        float norm(int n, float x) {
            return float(n) * x;
        }

        void main() {
            acc[0] = norm(2, 1.5);
        }
    }
}
""")
    code = crosstl.translate(str(path), backend="hip", format_output=False)

    assert "norm(2, 1.5f)" in code
    assert "normf(" not in kernel_body(code)
