"""The mlx_model marker selects only pure-Python model checks."""

import functools
import types

from crosstl.translator.lexer import Lexer
from demos.integrations.mlx.tests import model_tests, pins
from demos.integrations.mlx.tests.kernels import test_current_quantized_qvm_split_k

HELPERS = {"lexer": Lexer}


def _square(value):
    return value * value


def _pure_model():
    return [_square(value) for value in range(4)]


def _calls_translator():
    return Lexer("x")


def _calls_translator_through_a_helper():
    return _square(2), _calls_translator()


def _finds_translator_in_a_table():
    return HELPERS["lexer"]


def _imports_translator_lazily():
    from crosstl.translator import parser

    return parser


def _runs_a_process():
    import subprocess

    return subprocess.run


def _reads_a_lane_module():
    return pins.pin_mode()


PARTIAL = functools.partial(_calls_translator)


def _uses_a_partial():
    return PARTIAL()


def test_pure_model_reaches_nothing_foreign():
    assert model_tests.foreign_references(_pure_model) == []


def test_walk_finds_translator_calls_imports_and_processes():
    cases = {
        _calls_translator: "crosstl.translator.lexer.Lexer",
        _calls_translator_through_a_helper: "crosstl.translator.lexer.Lexer",
        _finds_translator_in_a_table: "crosstl.translator.lexer.Lexer",
        _imports_translator_lazily: "imports crosstl.translator",
        _runs_a_process: "imports subprocess",
        _uses_a_partial: "crosstl.translator.lexer.Lexer",
    }
    for function, expected in cases.items():
        found = model_tests.foreign_references(function)
        assert any(expected in reference for reference in found), (function, found)


def test_walk_follows_lane_module_attributes():
    # pins only uses the standard library and this demo.
    assert model_tests.foreign_references(_reads_a_lane_module) == []
    module = types.ModuleType("demos.integrations.mlx.synthetic")
    module.translate = _calls_translator

    def reads_synthetic_module():
        return module.translate()

    found = model_tests.foreign_references(reads_synthetic_module)
    assert any("crosstl.translator.lexer.Lexer" in item for item in found), found


def test_model_test_names_follow_the_kernel_conventions():
    model = [
        "test_quantized_qvm_split_k_expectations_reject_defects",
        "test_winograd_reference_rejects_defects",
        "test_sdpa_gqa_checks_reject_defects",
        "test_quantized_qmv_model_matches_dot_products",
        "test_quantized_qvm_opengl_assertions_name_buffer_elements",
        "test_quantized_qvm_split_k_inputs_keep_float32_exact",
        "test_sdpa_gqa_reference_matches_kernel_index_arithmetic",
        "test_sdpa_gqa_bounds_stay_meaningful",
        "test_quantized_qvm_cases_reach_every_path",
        "test_segmented_references_are_exact_and_guarded",
        "test_reduce_cases_follow_mlx_dispatch",
    ]
    other = [
        "test_current_mlx_quantized_qvm_split_k_native_parity",
        "test_current_mlx_quantized_qvm_split_k_contract_is_exact",
        "test_quantized_qvm_split_k_configs_scope_target_options",
        "test_ci_requires_native_quantized_qvm_split_k_parity",
    ]
    assert all(model_tests.MODEL_TEST_NAME.match(name) for name in model)
    assert not any(model_tests.MODEL_TEST_NAME.match(name) for name in other)


def _item(function, params=None):
    callspec = None if params is None else types.SimpleNamespace(params=params)
    return types.SimpleNamespace(obj=function, callspec=callspec)


def test_split_k_model_checks_are_marked_and_its_native_gate_is_not():
    module = test_current_quantized_qvm_split_k
    entry = module.SELECTED_ENTRIES[0]
    mutation = next(iter(module.MUTATIONS))
    marked = [
        (
            module.test_quantized_qvm_split_k_expectations_reject_defects,
            {"entry": entry, "mutation": mutation},
        ),
        (
            module.test_quantized_qvm_split_k_model_matches_partial_products,
            {"entry": entry},
        ),
        (
            module.test_quantized_qvm_split_k_opengl_assertions_name_buffer_elements,
            {"entry": entry},
        ),
    ]
    for function, params in marked:
        assert model_tests.is_model_test(_item(function, params)), function
    native = module.test_current_mlx_quantized_qvm_split_k_native_parity
    assert model_tests.foreign_references(native)


def test_model_checks_that_take_fixtures_are_not_marked():
    def test_example_model_matches_values(tmp_path):
        return tmp_path

    assert model_tests.foreign_references(test_example_model_matches_values) == []
    assert not model_tests.is_model_test(_item(test_example_model_matches_values))
    assert not model_tests.is_model_test(
        _item(test_example_model_matches_values, {"entry": "x"})
    )
    assert model_tests.is_model_test(
        _item(test_example_model_matches_values, {"tmp_path": "x"})
    )
