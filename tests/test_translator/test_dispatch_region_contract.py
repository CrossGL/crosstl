"""Artifact launch obligations survive descriptor loss and late request edits."""

import copy
import itertools
import json
from dataclasses import replace

import pytest

from crosstl.formatter import CodeFormatter, ShaderLanguage
from crosstl.project import (
    build_native_loader_dispatch_request,
    generate_native_loader_execution_abi,
    translate_project,
)
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_loader_abi import NativeLoaderABIError
from crosstl.project.native_loader_dispatch import NativeLoaderDispatchError
from crosstl.project.runtime_verification import (
    NativeRuntimeParityAdapter,
    RuntimeAdapterSetupError,
    RuntimeDispatchGeometry,
    RuntimeExecutionState,
)
from crosstl.translator.artifact_contracts import DISPATCH_REGION_PREFIX
from crosstl.translator.dispatch_region_contract import (
    check_dispatch_region_geometry,
    parse_dispatch_region_header,
    with_dispatch_region_header,
)
from crosstl.translator.dispatch_regions import plan_dispatch_regions
from tests.test_translator.test_artifact_contracts import HEADERS
from tests.test_translator.test_dispatch_region_project import _case, _package, _request
from tests.test_translator.test_metadata_index_bounds import (
    GUARD,
    SOURCE,
    _index_config,
)

REGION = plan_dispatch_regions([37], [32])[-1]
INDEX_REGION = plan_dispatch_regions((1, 4, 2), (1, 1, 1))[0]
REGION_INDEX_SOURCE = SOURCE.replace(
    "device uint* result [[buffer(3)]]",
    "device uint* result [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]]",
).replace(
    "result[0] = values[offset + index];",
    "ulong at = ulong(group.y) + ulong(groups.y) * ulong(group.z); "
    "values += (at / offset) * index; result[group.y + groups.y * group.z] = values[at % offset];",
)


def _region_index_config(
    root, target, *, region=INDEX_REGION, source=REGION_INDEX_SOURCE
):
    config = _index_config(
        root,
        target=target,
        source=source,
        conditions=(
            dict(entryPoint="read_value", parameter="index", minimum=1, maximum=16),
            dict(entryPoint="read_value", parameter="offset", minimum=1, maximum=2),
        ),
    )
    if target != "metal" and region is not None:
        options = dict(config.source_options)
        options["metal"] = dict(
            options["metal"],
            target_options={target: {"dispatch_region": region.to_json()}},
        )
        config = replace(config, source_options=options)
    return config


def _region_index_request(descriptor, package, index, offset):
    values = [17 + position * 13 for position in range(128)]
    expected = [values[(n // offset) * index + n % offset] for n in range(8)] + [
        GUARD
    ] * 8
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name, parameter = binding["name"], origin["parameter"]
        words = {
            "values": values,
            "index": [index],
            "offset": [offset],
            "result": [GUARD] * 16,
        }[parameter]
        inputs[name] = dict(
            dtype=binding["scalarLayout"]["elementType"],
            shape=[len(words)],
            values=words,
        )
        if parameter == "result":
            outputs[name] = dict(dtype="uint32", shape=[16], values=expected)
        else:
            observed.append(name)
    return (
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {
                "workgroupCount": list(INDEX_REGION.workgroup_count),
                "workgroupSize": [1, 1, 1],
            },
            observe_inputs=observed,
        ),
        outputs,
    )


def test_region_and_metadata_bounds_compose_for_wide_indices(tmp_path):
    report = translate_project(
        _region_index_config(tmp_path, "opengl"), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize(
    "region", [None, plan_dispatch_regions((1, 65535, 65535), (1, 1, 1))[0]]
)
def test_missing_or_overflowing_region_cannot_authorize_narrowing(tmp_path, region):
    report = translate_project(
        _region_index_config(tmp_path, "opengl", region=region), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report


@pytest.mark.parametrize(
    "mutation",
    [
        "group.y = 4294967295u;",
        "group = uint3(0u, 4294967295u, 0u);",
        "groups.y = 4294967295u;",
    ],
)
def test_dispatch_parameter_mutation_invalidates_original_bounds(tmp_path, mutation):
    source = REGION_INDEX_SOURCE.replace("ulong at =", mutation + " ulong at =")
    report = translate_project(
        _region_index_config(tmp_path, "opengl", source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report


@pytest.mark.parametrize("method", ["generate", "generate_program", "generate_stage"])
def test_codegen_region_bounds_do_not_survive_removed_contract(tmp_path, method):
    from crosstl.translator.codegen.GLSL_codegen import OpenGLIndexTypeError
    from crosstl.translator.dispatch_region_lowering import specialize_dispatch_region
    from tests.test_translator.test_metadata_index_bounds import _generator

    source = REGION_INDEX_SOURCE.replace(
        "values += (at / offset) * index;", "values += at;"
    ).replace("values[at % offset]", "values[0]")
    generator, ast, _, _ = _generator(tmp_path, source)
    specialized = specialize_dispatch_region(ast, INDEX_REGION)
    args = (specialized, "compute") if method == "generate_stage" else (specialized,)
    artifact = getattr(generator, method)(*args)
    assert parse_dispatch_region_header(artifact) == INDEX_REGION
    assert getattr(generator, method)(*args) == artifact
    specialized.annotations.pop("dispatch_region")
    with pytest.raises(OpenGLIndexTypeError):
        getattr(generator, method)(*args)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("mutation", ["removed", "changed"])
def test_descriptor_cannot_erase_artifact_region(tmp_path, target, mutation):
    descriptor, package, _ = _package(tmp_path, target, REGION)
    source = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert parse_dispatch_region_header(source) == REGION
    altered = copy.deepcopy(descriptor)
    if mutation == "removed":
        altered["provenance"].pop("dispatchRegion")
        altered["provenance"].pop("dispatchRegionProgram")
    else:
        # A valid but different region cannot inherit this artifact's identity.
        altered["provenance"]["dispatchRegion"] = plan_dispatch_regions([5], [5])[
            0
        ].to_json()
    inputs, initial, expected = _case((37, 1, 1), (32, 1, 1))
    with pytest.raises(
        NativeLoaderDispatchError, match="dispatch-region-contract-invalid"
    ):
        _request(altered, package, REGION, inputs, initial, expected)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_final_native_geometry_is_checked_against_snapshot(tmp_path, target):
    descriptor, package, _ = _package(tmp_path, target, REGION)
    inputs, initial, expected = _case((37, 1, 1), (32, 1, 1))
    request = _request(descriptor, package, REGION, inputs, initial, expected)
    geometry = replace(
        request.execution_plan.dispatch,
        workgroup_count=(2, 1, 1),
        global_size=(10, 1, 1),
        grid_size=(10, 1, 1),
    )
    plan = replace(request.execution_plan, dispatch=geometry)
    state = RuntimeExecutionState(
        request=replace(request, execution_plan=plan), plan=plan
    )
    adapter = NativeRuntimeParityAdapter(validate=False)
    adapter.target = target
    try:
        with pytest.raises(
            RuntimeAdapterSetupError, match="specialized physical region"
        ) as failure:
            adapter.prepare_buffers(state)
        assert failure.value.details["reasonKind"] == "dispatch-region-contract-invalid"
        assert "nativeRuntimeDispatch" not in state.details
    finally:
        for directory in state.temporary_directories:
            directory.cleanup()


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_region_c_execution_wrapper_is_explicitly_unsupported(tmp_path, target):
    descriptor, _, _ = _package(tmp_path, target, REGION)
    with pytest.raises(
        NativeLoaderABIError, match="execution-dispatch-region-unsupported"
    ):
        generate_native_loader_execution_abi(descriptor)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "late", "schema", "bool", "missing", "duplicate-key", "geometry"],
)
def test_malformed_region_header_is_rejected(mutation):
    header = with_dispatch_region_header("", REGION)
    if mutation == "duplicate":
        source = header + header
    elif mutation == "late":
        source = "\n" + header
    elif mutation == "schema":
        source = header.replace('"schemaVersion":1', '"schemaVersion":2')
    elif mutation == "bool":
        source = header.replace('"schemaVersion":1', '"schemaVersion":true')
    elif mutation == "missing":
        source = DISPATCH_REGION_PREFIX + json.dumps({"schemaVersion": 1})
    elif mutation == "duplicate-key":
        source = header.replace(
            '"schemaVersion":1', '"schemaVersion":1,"schemaVersion":1'
        )
    else:
        source = header.replace('"workgroupCount":[1,1,1]', '"workgroupCount":[2,1,1]')
    with pytest.raises(ValueError):
        parse_dispatch_region_header(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("workgroup_count", (2, 1, 1)),
        ("workgroup_count", (True, 1, 1)),
        ("workgroup_size", (32, 1, 1)),
        ("global_size", (10, 1, 1)),
        ("grid_size", (10, 1, 1)),
        ("thread_grid_size", (5, 1, 1)),
    ],
)
def test_region_geometry_rejects_inconsistent_extents(field, value):
    geometry = RuntimeDispatchGeometry(workgroup_count=(1,), workgroup_size=(5,))
    check_dispatch_region_geometry(REGION, geometry)
    with pytest.raises(ValueError):
        check_dispatch_region_geometry(REGION, replace(geometry, **{field: value}))


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_reflection_checks_region_header_shape(tmp_path, target):
    descriptor, package, _ = _package(tmp_path, target, REGION)
    path = package / descriptor["artifact"]["packagePath"]
    assert reflect_target_host_interface(path, target=target)["status"] == "ready"
    source = path.read_text()
    path.write_text(
        source.replace('"workgroupSize":[5,1,1]', '"workgroupSize":[4,1,1]')
    )
    assert reflect_target_host_interface(path, target=target)["status"] == "failed"


@pytest.mark.parametrize("language", [ShaderLanguage.HLSL, ShaderLanguage.GLSL])
@pytest.mark.parametrize("order", [(3,), *itertools.permutations(range(4))])
def test_formatter_retains_region_with_other_artifact_contracts(language, order):
    formatter = CodeFormatter()
    if not formatter.has_clang:
        pytest.skip("clang-format is not installed")
    headers = (*HEADERS, with_dispatch_region_header("", REGION))
    source = "".join(headers[index] for index in order) + "void main() {}\n"
    formatted = formatter.format_code(source, language)
    assert parse_dispatch_region_header(formatted) == REGION
    assert formatted.startswith("".join(headers[index] for index in order))
