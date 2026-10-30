"""CUDA and HIP output keep the grouping of nested prefix operators."""

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader main {
    buffer float* results;
    buffer int* counters;
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a = results[0];
        int n = counters[0];
        results[1] = -(-a);
        results[2] = +(+a);
        results[3] = -(--a);
        results[4] = +(++a);
        results[5] = -(+a);
        results[6] = -a;
        results[7] = -results[0];
        counters[1] = -(-n);
        counters[2] = ~(~n);
        counters[3] = -(-(-n));
        --n;
        counters[4] = n;
    }
    }
}
"""


@pytest.mark.parametrize("generator", [CudaCodeGen, HipCodeGen])
def test_nested_prefix_signs_keep_their_grouping(generator):
    code = generator().generate(Parser(Lexer(SOURCE).tokens).parse())

    assert "results[1] = -(-a);" in code
    assert "results[2] = +(+a);" in code
    assert "results[3] = -(--a);" in code
    assert "results[4] = +(++a);" in code
    assert "counters[1] = -(-n);" in code
    assert "counters[3] = -(-(-n));" in code
    # Operators that cannot merge, and plain operands, stay as they were.
    assert "results[5] = -+a;" in code
    assert "counters[2] = ~~n;" in code
    assert "results[6] = -a;" in code
    assert "results[7] = -results[0];" in code
    assert "--n;" in code
    # No pre-increment or pre-decrement appears where the source has none.
    for merged in ("= --a", "= ++a", "---a", "+++a", "= --n", "---n"):
        assert merged not in code
