import pytest

from tools.coverage import probes, validators
from tools.coverage.features import opencl
from tools.coverage.model import STRING, Param, Signature, Type

# Excerpts of the pinned opencl-c-base.h, Builtins.def and TokenKinds.def.
BASE = """
/**
 * OpenCL v1.1/1.2/2.0 s6.2.4.2 - as_type operators
 * Reinterprets a data type as another data type of the same size
 */
#define as_char(x) __builtin_astype((x), char)
#define as_char2(x) __builtin_astype((x), char2)
#define as_uint(x) __builtin_astype((x), uint)
#define as_uint16(x) __builtin_astype((x), uint16)
#define as_float(x) __builtin_astype((x), float)
#define as_size_t(x) __builtin_astype((x), size_t)
#ifdef cl_khr_fp16
#define as_half(x) __builtin_astype((x), half)
#define as_half4(x) __builtin_astype((x), half4)
#endif // cl_khr_fp16
#define NAN as_float(INT_MAX)

typedef char char2 __attribute__((ext_vector_type(2)));

#if defined(__OPENCL_CPP_VERSION__) || (__OPENCL_C_VERSION__ >= CL_VERSION_1_2)
// OpenCL v1.2 s6.12.13, v2.0 s6.13.13 - printf

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));
#endif

#ifdef cl_intel_device_side_avc_motion_estimation
int intel_only(int);
#endif
"""
BUILTINS = """
LANGBUILTIN(__builtin_coro_suspend, "cIb", "n", COR_LANG)

// OpenCL v2.0 s6.13.16, s9.17.3.5 - Pipe functions.
// We need the generic prototype, since the packet type could be anything.
LANGBUILTIN(read_pipe, "i.", "tn", OCL_PIPE)
LANGBUILTIN(commit_write_pipe, "v.", "tn", OCL_PIPE)
LANGBUILTIN(work_group_commit_read_pipe, "v.", "tn", OCL_PIPE)
LANGBUILTIN(get_pipe_num_packets, "Ui.", "tn", OCL_PIPE)
LANGBUILTIN(peek_pipe, "i.", "tn", OCL_PIPE)

// OpenCL v2.0 s6.13.17 - Enqueue kernel functions.
// Custom builtin check allows to perform special check of passed block arguments.
LANGBUILTIN(enqueue_kernel, "i.", "tn", OCL_DSE)
LANGBUILTIN(get_kernel_work_group_size, "Ui.", "tn", OCL_DSE)

// OpenCL v2.0 s6.13.9 - Address space qualifier functions.
// FIXME: Pointer parameters of OpenCL builtins should have their address space
// requirement defined.
LANGBUILTIN(to_global, "v*v*", "tn", OCL_GAS)
LANGBUILTIN(to_private, "v*v*", "tn", OCL_GAS)

// OpenCL half load/store builtin
LANGBUILTIN(__builtin_store_half, "vdh*", "n", ALL_OCL_LANGUAGES)
"""
KEYWORDS = """
UNARY_EXPR_OR_TYPE_TRAIT(sizeof, SizeOf, KEYALL)
KEYWORD(__builtin_astype            , KEYOPENCLC | KEYOPENCLCXX)
UNARY_EXPR_OR_TYPE_TRAIT(vec_step, VecStep, KEYOPENCLC | KEYOPENCLCXX | KEYALTIVEC | KEYZVECTOR)
"""


def _features():
    return {f.name: f for f in opencl.compiler(BASE, BUILTINS, KEYWORDS)}


def test_opencl_lists_each_as_type_operator_once_with_its_widths_folded_in():
    features = _features()
    as_types = sorted(n for n, f in features.items() if f.group == "as-type")
    assert as_types == ["as_char", "as_float", "as_half", "as_size_t", "as_uint"]
    assert features["as_uint"].signature == Signature(
        Type("uint"), (Param(Type("float")),)
    )
    assert features["as_float"].signature == Signature(
        Type("float"), (Param(Type("int")),)
    )
    assert features["as_size_t"].signature.result == Type("uint64")


def test_opencl_as_type_probes_reinterpret_a_value_of_the_other_kind():
    # A value conversion in place of the reinterpretation then computes another value.
    assert opencl.reinterpreted("uint") == "float"
    assert opencl.reinterpreted("int64") == "double"
    assert opencl.reinterpreted("uint16") == "half"
    assert opencl.reinterpreted("half") == "int16"
    assert opencl.reinterpreted("double") == "int64"
    # A byte has no floating-point type of its size.
    assert opencl.reinterpreted("int8") == "uint8"
    assert opencl.reinterpreted("uint8") == "int8"


def test_opencl_lists_printf_with_a_format_and_one_variable_argument():
    printf = _features()["printf"]
    assert printf.group == "printf"
    assert printf.signature == Signature(
        Type("int"), (Param(STRING, "const"), Param(Type("float")))
    )
    assert "intel_only" not in _features()


def test_opencl_lists_the_opencl_built_ins_clang_checks_itself_by_section():
    features = _features()
    assert features["read_pipe"].group == "pipe"
    assert features["enqueue_kernel"].group == "enqueue-kernels"
    assert features["to_global"].group == "address-space-qualifier"
    assert features["vec_step"].group == "operators"
    for name in ("__builtin_coro_suspend", "__builtin_store_half", "sizeof"):
        assert name not in features


def test_opencl_lists_a_compiler_built_in_with_no_specified_declaration_as_untested():
    peek = _features()["peek_pipe"]
    assert peek.signature is None
    with pytest.raises(probes.ProbeError, match="SPECIFIED"):
        probes.build(peek, "opencl")


def test_opencl_specified_declarations_name_their_functions():
    for name in opencl.SPECIFIED:
        assert opencl.specified(name) is not None, name
    assert opencl.specified("to_global").params == (Param(Type("int"), "out"),)


def _probe(name):
    return probes.build(_features()[name], "opencl").source


def test_opencl_probe_commits_a_reservation_of_its_kind_on_its_own_pipe():
    source = _probe("commit_write_pipe")
    assert ", write_only pipe int a0) {" in source
    assert "reserve_id_t a1 = reserve_write_pipe(a0, 1u);" in source
    assert "commit_write_pipe(a0, a1);" in source
    assert "a1_pipe" not in source
    group = _probe("work_group_commit_read_pipe")
    assert "reserve_id_t a1 = work_group_reserve_read_pipe(a0, 1u);" in group


def test_opencl_probe_passes_a_block_and_reads_a_converted_pointer():
    assert "enqueue_kernel(a0, CLK_ENQUEUE_FLAGS_NO_WAIT, a2, ^{});" in _probe(
        "enqueue_kernel"
    )
    assert "get_kernel_work_group_size(^{});" in _probe("get_kernel_work_group_size")
    assert "uint r = (uint)(to_private(&a0) != 0);" in _probe("to_private")


def test_opencl_probe_prints_its_float_argument():
    assert 'int r = printf("%f\\n", a1);' in _probe("printf")


def test_opencl_probe_reinterprets_with_as_type():
    source = _probe("as_uint")
    assert "float a0 = fin[i + 0u];" in source
    assert "uint r = as_uint(a0);" in source
    assert _probe("as_half").startswith(
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    )


@pytest.mark.skipif(
    not validators.available("opencl"), reason="clang or DXC is missing"
)
def test_opencl_compiler_built_in_probes_are_valid_opencl_c():
    for name, feature in _features().items():
        if feature.signature is None:
            continue
        verdict = validators.validate("opencl", _probe(name))
        assert verdict.accepted, (name, verdict.log)
