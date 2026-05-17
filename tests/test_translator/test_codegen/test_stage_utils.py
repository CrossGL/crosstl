from enum import Enum

from crosstl.translator.codegen.stage_utils import (
    compute_local_size,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class DummyStage(Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"


def test_normalize_stage_name_accepts_strings_and_enums():
    assert normalize_stage_name(None) is None
    assert normalize_stage_name("fragment") == "fragment"
    assert normalize_stage_name("ShaderStage.VERTEX") == "vertex"
    assert normalize_stage_name(DummyStage.FRAGMENT) == "fragment"


def test_stage_matches_normalizes_target_and_stage():
    assert stage_matches(None, "vertex")
    assert stage_matches("ShaderStage.VERTEX", DummyStage.VERTEX)
    assert not stage_matches("fragment", DummyStage.VERTEX)


def test_should_emit_qualified_function_keeps_helpers_and_filters_stages():
    assert should_emit_qualified_function("vertex", None)
    assert should_emit_qualified_function("vertex", "inline")
    assert should_emit_qualified_function("vertex", "ShaderStage.VERTEX")
    assert not should_emit_qualified_function("vertex", "fragment")
    assert not should_emit_qualified_function("vertex", "ray_miss")


def test_compute_local_size_uses_common_config_keys():
    assert compute_local_size() == ("1", "1", "1")
    assert compute_local_size({"local_size": (8, 4, 2)}) == ("8", "4", "2")
    assert compute_local_size({"workgroup_size": [16, 2, 1]}) == ("16", "2", "1")
    assert compute_local_size({"numthreads": (3, 5, 7)}) == ("3", "5", "7")
    assert compute_local_size({"local_size_x": 9, "local_size_y": 8}) == (
        "9",
        "8",
        "1",
    )
