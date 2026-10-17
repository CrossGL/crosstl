"""Coverage and descriptor selection for batched native binary checks."""

from dataclasses import replace

import pytest

from crosstl.project import load_project_config
from demos.integrations.mlx.tests.kernels import (
    test_current_additive,
    test_current_division,
    test_current_extrema,
    test_current_floating_binary,
    test_current_multiplication,
)
from demos.integrations.mlx.tests.kernels.floating_binary_runtime import (
    BINARY_OPENGL_WORKLOADS,
    MLX_BINARY_SOURCE,
    _batch_workloads,
    _binary_load_units,
    _dispatch_shape,
    _project_config,
)


def test_batches_preserve_every_case_and_source_profile(tmp_path):
    native_test = (
        test_current_floating_binary.test_current_floating_binary_native_parity
    )
    (parameters,) = (
        mark.args[1]
        for mark in native_test.pytestmark
        if mark.name == "parametrize" and mark.args[0] == "profile"
    )
    assert parameters == (
        "division",
        "multiplication",
        "half",
        "comparison",
        "additive",
    )
    all_cases = [
        case
        for fixture in (
            test_current_extrema,
            test_current_additive,
            test_current_division,
        )
        for case in fixture._cases(("float16", "bfloat16", "float32"))
    ] + list(test_current_multiplication._cases())
    batches = [
        list(test_current_floating_binary._cases(profile)) for profile in parameters
    ]
    flattened = [case for batch in batches for case in batch]
    assert [len(batch) for batch in batches] == [2, 2, 6, 4, 4]
    assert sorted(flattened, key=lambda c: c.entry) == sorted(
        all_cases, key=lambda c: c.entry
    )
    assert len(flattened) == len({case.entry for case in flattened}) == 18
    assert sum(len(case.pairs) for case in flattened) == 124688
    assert sum(len(case.expected) - len(case.pairs) for case in flattened) == 144
    for cases in batches:
        workloads = _batch_workloads(cases)
        entries = [case.entry for case in cases]
        path = tmp_path / "crosstl.toml"
        profile = cases[0].provenance.get("binary32MultiplicationProfile")
        path.write_text(
            _project_config(
                workloads[0],
                entry_points=entries,
                binary32_multiplication_profile=profile,
            )
        )
        batch = load_project_config(tmp_path, path)
        assert tuple(batch.entry_points[MLX_BINARY_SOURCE]) == tuple(entries)
        assert dict(batch.entry_workgroup_size_rules[MLX_BINARY_SOURCE]) == {
            "*": ("1", "1", "1")
        }
        for workload in workloads:
            path.write_text(
                _project_config(workload, binary32_multiplication_profile=profile)
            )
            single = load_project_config(tmp_path, path)
            assert single.source_options == batch.source_options
            assert single.index_range_assertions == batch.index_range_assertions


def test_multiplication_profile_is_explicit_and_limited_to_real_float_products(
    tmp_path,
):
    selected = []
    for workload in BINARY_OPENGL_WORKLOADS:
        assert "binary32_multiplication_profile" not in _project_config(workload)
        enabled = workload.operator_type == "Multiply" and workload.input_type in {
            "float",
            "bfloat16_t",
        }
        if not enabled:
            with pytest.raises(AssertionError):
                _project_config(workload, binary32_multiplication_profile="rne-flush")
            continue
        path = tmp_path / "crosstl.toml"
        path.write_text(
            _project_config(workload, binary32_multiplication_profile="rne-flush")
        )
        config = load_project_config(tmp_path, path)
        assert dict(config.source_options["metal"]) == {
            "max_template_specializations": 64,
            "max_template_materialization_work": 4096,
            "binary32_multiplication_profile": "rne-flush",
        }
        selected.append((workload.shape, workload.input_type))
    assert len(selected) == len(set(selected)) == 36
    assert len({shape for shape, _ in selected}) == 18


@pytest.mark.parametrize(
    "option,value,types,count",
    (
        (
            "binary32_power_operand_profile",
            "flush-subnormals",
            {"float", "bfloat16_t"},
            36,
        ),
        ("binary32_power_accuracy_profile", "portable-finite", {"float"}, 18),
    ),
)
def test_power_operand_profile_is_explicit_and_limited_to_characterized_types(
    tmp_path, option, value, types, count
):
    selected = []
    for workload in BINARY_OPENGL_WORKLOADS:
        assert option not in _project_config(workload)
        enabled = workload.operator_type == "Power" and workload.input_type in types
        if not enabled:
            with pytest.raises(AssertionError):
                _project_config(workload, **{option: value})
            continue
        path = tmp_path / "crosstl.toml"
        path.write_text(
            _project_config(
                workload,
                workgroup_width=2,
                **{option: value},
            )
        )
        config = load_project_config(tmp_path, path)
        assert config.source_options["metal"][option] == value
        assert config.entry_workgroup_size_rules[MLX_BINARY_SOURCE][
            workload.entry_point
        ] == ("2", "1", "1")
        selected.append((workload.shape, workload.input_type))
    assert len(selected) == len(set(selected)) == count
    assert len({shape for shape, _ in selected}) == 18


@pytest.mark.parametrize(
    "entries",
    ([], ["vv_Powerfloat32", "vv_Addfloat32"], ["vv_Powerfloat16"], ["unknown"]),
)
@pytest.mark.parametrize(
    "option,value",
    (
        ("binary32_power_operand_profile", "flush-subnormals"),
        ("binary32_power_accuracy_profile", "portable-finite"),
    ),
)
def test_power_operand_profile_rejects_unrelated_or_missing_entries(
    entries, option, value
):
    workload = next(
        w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == "vv_Powerfloat32"
    )
    with pytest.raises(AssertionError):
        _project_config(
            workload,
            entry_points=entries,
            **{option: value},
        )


@pytest.mark.parametrize("profile", ("preserve-subnormals", "rne-flush", "", False))
@pytest.mark.parametrize(
    "option", ("binary32_power_operand_profile", "binary32_power_accuracy_profile")
)
def test_power_operand_profile_rejects_uncharacterized_profile(profile, option):
    workload = next(
        w for w in BINARY_OPENGL_WORKLOADS if w.entry_point == "vv_Powerfloat32"
    )
    with pytest.raises(AssertionError):
        _project_config(workload, **{option: profile})


@pytest.mark.parametrize("width", (1, 2, 32))
def test_binary_dispatch_shape_preserves_exact_thread_count(width):
    assert _dispatch_shape(65536, width) == {
        "workgroupCount": [65536 // width, 1, 1],
        "workgroupSize": [width, 1, 1],
    }


@pytest.mark.parametrize(
    "count,width",
    ((0, 1), (True, 1), (1, 2), (3, 2), (4, 0), (4, True), (4, 1.5), (2048, 2048)),
)
def test_binary_dispatch_shape_rejects_empty_invalid_or_partial_groups(count, width):
    with pytest.raises(AssertionError):
        _dispatch_shape(count, width)


@pytest.mark.parametrize("profile", ("rne-gradual", "flush", "", False))
def test_multiplication_batch_rejects_uncharacterized_profile(profile):
    workload = _batch_workloads(list(test_current_multiplication._cases(("float32",))))[
        0
    ]
    with pytest.raises(AssertionError):
        _project_config(workload, binary32_multiplication_profile=profile)


@pytest.mark.parametrize(
    "entries",
    (
        [],
        ["vv_Multiplyfloat32", "vv_Addfloat32"],
        ["vv_Multiplyfloat32", "vv_Multiplyfloat16"],
        ["vv_Multiplycomplex64"],
        ["unknown"],
    ),
)
def test_multiplication_batch_rejects_unrelated_or_missing_entries(entries):
    workload = _batch_workloads(list(test_current_multiplication._cases(("float32",))))[
        0
    ]
    with pytest.raises(AssertionError):
        _project_config(
            workload,
            entry_points=entries,
            binary32_multiplication_profile="rne-flush",
        )


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_mixed_half_batch_preserves_operation_specific_comparisons(target):
    cases = list(test_current_floating_binary._cases("half"))
    assert {case.operation for case in cases} == {
        "Minimum",
        "Maximum",
        "Add",
        "Subtract",
        "Divide",
        "Multiply",
    }
    guard = test_current_extrema._guard("float16")

    def encoded(words):
        return test_current_extrema._payload("float16", target, words)["values"]

    for case in cases:
        compare = test_current_floating_binary._compare_for(case)
        selection = case.operation in ("Minimum", "Maximum")
        assert compare is (
            test_current_extrema._compare_native
            if selection
            else test_current_additive._check_words
        )
        expected = encoded([0x7E01] + [guard] * 8)
        actual = encoded([0x7E02] + [guard] * 8)
        if selection:
            with pytest.raises(AssertionError):
                compare(actual, expected, "float16", target)
        else:
            assert compare(actual, expected, "float16", target) == {
                "nanPayloadDifferences": 1,
                "finiteMismatchCount": 0,
            }
        for actual, expected in (
            ([0x8000] + [guard] * 8, [0] + [guard] * 8),
            ([1] + [guard] * 8, [0] + [guard] * 8),
            ([0x7C00] + [guard] * 8, [0x7E01] + [guard] * 8),
            ([0] + [guard] * 7 + [0], [0] + [guard] * 8),
        ):
            with pytest.raises(AssertionError):
                compare(encoded(actual), encoded(expected), "float16", target)


def test_unknown_operation_has_no_default_comparison():
    case = next(test_current_multiplication._cases())
    with pytest.raises(ValueError, match="No comparison policy"):
        test_current_floating_binary._compare_for(replace(case, operation="Unknown"))


@pytest.mark.parametrize("failure", ("empty", "duplicate", "mixed-profile"))
def test_batch_rejects_incompatible_cases(failure):
    cases = list(test_current_extrema._cases(("float16",)))
    if failure == "empty":
        cases = []
    elif failure == "duplicate":
        cases.append(cases[0])
    else:
        cases.extend(test_current_extrema._cases(("float32",)))
    with pytest.raises(AssertionError):
        _batch_workloads(cases)


@pytest.mark.parametrize(
    "failure",
    (
        None,
        "missing",
        "duplicate",
        "duplicate-id",
        "unexpected",
        "wrong-source",
        "wrong-target",
        "blocked",
        "failed",
    ),
)
def test_batch_selects_exact_ready_load_units(failure):
    entries = ["vv_Minimumfloat16", "vv_Maximumfloat16"]
    units = [
        {
            "id": str(index),
            "entryPoint": {"source": entry},
            "target": "opengl",
            "source": MLX_BINARY_SOURCE,
            "validation": {"loadReady": True},
        }
        for index, entry in enumerate(reversed(entries))
    ]
    if failure == "missing":
        units.pop()
    elif failure == "duplicate":
        units[0] = units[1]
    elif failure == "duplicate-id":
        units[0]["id"] = units[1]["id"]
    elif failure == "unexpected":
        units[0]["entryPoint"]["source"] = "vv_Addfloat16"
    elif failure == "wrong-source":
        units[0]["source"] = "other.metal"
    elif failure == "wrong-target":
        units[0]["target"] = "directx"
    elif failure == "blocked":
        units[0]["validation"]["loadReady"] = False
    loader = {"success": failure != "failed", "loadUnits": units}
    if failure:
        with pytest.raises(AssertionError):
            _binary_load_units(loader, entries, "opengl")
    else:
        selected = _binary_load_units(loader, entries, "opengl")
        assert [selected[entry]["id"] for entry in entries] == ["1", "0"]
