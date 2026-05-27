from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import ShaderStage
from crosstl.translator.stage_utils import (
    SHADER_STAGE_BY_NAME,
    SHADER_STAGE_NAMES,
    STAGE_NAME_ALIASES,
    STAGE_QUALIFIER_NAMES,
    normalize_stage_name,
    shader_stage_from_name,
)

CANONICAL_STAGE_CASES = tuple((stage.value, stage.value) for stage in ShaderStage)
ALIAS_STAGE_CASES = tuple(sorted(STAGE_NAME_ALIASES.items()))
STAGE_CASES = CANONICAL_STAGE_CASES + ALIAS_STAGE_CASES
IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def _apply_separator(name, separator):
    return name.replace("_", separator)


def _apply_case(name, case_style):
    if case_style == "upper":
        return name.upper()
    if case_style == "title":
        return "_".join(part.capitalize() for part in name.split("_"))
    return name


def _wrap_stage_name(name, wrapper):
    if wrapper == "single":
        return f"'{name}'"
    if wrapper == "double":
        return f'"{name}"'
    if wrapper == "enum":
        return f"ShaderStage.{name}"
    return name


@settings(max_examples=50, deadline=None)
@given(
    stage_case=st.sampled_from(STAGE_CASES),
    separator=st.sampled_from(["_", "-", " "]),
    case_style=st.sampled_from(["lower", "upper", "title"]),
    wrapper=st.sampled_from(["plain", "single", "double", "enum"]),
)
def test_generated_stage_aliases_normalize_across_backend_spellings(
    stage_case,
    separator,
    case_style,
    wrapper,
):
    alias, canonical = stage_case
    variant = _wrap_stage_name(
        _apply_case(_apply_separator(alias, separator), case_style),
        wrapper,
    )

    assert normalize_stage_name(variant) == canonical
    assert shader_stage_from_name(variant) is ShaderStage(canonical)


@settings(max_examples=25, deadline=None)
@given(stage=st.sampled_from(tuple(ShaderStage)))
def test_generated_shader_stage_enums_roundtrip_through_shared_lookup(stage):
    assert normalize_stage_name(stage) == stage.value
    assert shader_stage_from_name(stage) is stage
    assert SHADER_STAGE_BY_NAME[stage.value] is stage
    assert stage.value in SHADER_STAGE_NAMES
    assert stage.value in STAGE_QUALIFIER_NAMES


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    separator=st.sampled_from(["_", "-", " "]),
    wrapper=st.sampled_from(["plain", "single", "double", "enum"]),
)
def test_generated_unknown_stage_names_are_sanitized_but_not_canonicalized(
    suffix,
    separator,
    wrapper,
):
    unknown_name = f"custom_{suffix}_stage"
    variant = _wrap_stage_name(_apply_separator(unknown_name, separator), wrapper)

    assert normalize_stage_name(variant) == unknown_name
    assert shader_stage_from_name(variant) is None


def test_stage_alias_table_only_targets_canonical_shader_stage_names():
    assert SHADER_STAGE_NAMES == frozenset(stage.value for stage in ShaderStage)
    assert STAGE_QUALIFIER_NAMES == SHADER_STAGE_NAMES
    assert SHADER_STAGE_BY_NAME == {stage.value: stage for stage in ShaderStage}
    assert set(STAGE_NAME_ALIASES.values()) <= SHADER_STAGE_NAMES
    assert not set(STAGE_NAME_ALIASES) & SHADER_STAGE_NAMES
