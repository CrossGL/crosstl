"""Explicit binary32 rounding for software cooperative matrix products."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_runtime_artifact_manifest,
    translate_project,
    validate_project_report,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXCooperativeMatrixUnsupportedError,
)
from tests.test_translator.test_directx_cooperative_matrix import (
    MATRIX,
    PRECISE_OPERANDS,
    REQUIRE_ENV,
    _case,
    _codegen,
    _coordinates,
    _project_request,
    _source,
)
from tests.test_translator.test_directx_float_atomics import _compile
from tests.test_translator.test_directx_software_reductions import _execute_words
from tests.test_translator.test_fused_math import _oracle
from tests.test_translator.test_loop_updates import _execute

PROFILES = ("rne-gradual", "rne-flush", "fma-rne-gradual", "fma-rne-flush")


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("shape", [(32, 4), (16, 8)])
def test_matrix_rounding_compiles_integer_intermediates(tmp_path, profile, shape):
    generated = _codegen(cooperative_matrix_mma_profile=profile).generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    assert "SoftFloat" in generated
    assert "uint2 result = asuint(accumulator);" in generated
    assert "return asfloat(result);" in generated
    assert "precise float term" not in generated
    assert generated.count("GroupMemoryBarrierWithGroupSync();") == 2
    assert generated.count("[loop]") == 2
    for element in range(2):
        prefix = f"term{element}"
        assert (
            f"for (uint {prefix}Index = 0u; {prefix}Index < 8u; ++{prefix}Index)"
            in generated
        )
        assert f"uint {prefix}A = row{element} * 8u + {prefix}Index;" in generated
        assert f"uint {prefix}B = {prefix}Index * 8u + column{element};" in generated
        assert f"ownerLanes[{prefix}A]][ownerElements[{prefix}A]]" in generated
        assert f"ownerLanes[{prefix}B]][ownerElements[{prefix}B]]" in generated
    if profile.startswith("fma-"):
        assert generated.count("uint term") == 10
        assert "term0Left, term0Right, result[0]," in generated
        assert "(term0Left ^ term0Right)" not in generated
    else:
        assert generated.count("uint term") == 12
        assert "(term0Left ^ term0Right) & 0x80000000u" in generated
        assert "result[0], 0x3f800000u, term0," in generated
    listing = tmp_path / "translated.ll"
    _compile(generated, tmp_path, flags=("-Fc", str(listing)))
    assembly = listing.read_text()
    assert "llvm.loop.unroll.disable" in assembly
    assert len(re.findall(r"br .*?!llvm.loop", assembly)) >= 4
    assert len(re.findall(r"call void @dx.op.barrier", assembly)) == 4
    assert not re.search(r"= f(?:add|mul|sub|div)\b", assembly)


@pytest.mark.parametrize("profile", [True, 32, "", "nearest", [], {}])
def test_matrix_rounding_rejects_invalid_profiles(profile):
    with pytest.raises(ValueError, match="cooperative_matrix_mma_profile"):
        _codegen(cooperative_matrix_mma_profile=profile)


def test_matrix_rounding_default_and_generation_state():
    ast = parse(_source())
    assert _codegen().generate(ast) == _codegen(
        cooperative_matrix_mma_profile=None
    ).generate(ast)
    generator = _codegen(cooperative_matrix_mma_profile="rne-gradual")
    first = generator.generate(ast)
    assert first == generator.generate(ast)
    generator.set_cooperative_matrix_mma_profile("rne-flush")
    assert first != generator.generate(ast)
    generator.set_software_subgroup_width(None)
    plain = generator.generate(parse("shader Plain { void helper() {} }"))
    assert "__crossgl_software_matrix" not in plain
    assert "SoftFloat" not in plain


@pytest.mark.parametrize("name", ["asuint", "asfloat"])
def test_matrix_rounding_rejects_captured_bitcasts(name):
    result_type, argument_type = (
        ("float", "uint") if name == "asfloat" else ("uint", "float")
    )
    source = _source(
        helpers=f"{result_type} {name}({argument_type} value) {{ return {result_type}(0); }}"
    )
    with pytest.raises(DirectXCooperativeMatrixUnsupportedError) as error:
        _codegen(cooperative_matrix_mma_profile="rne-gradual").generate(parse(source))
    assert error.value.reason == "rounding-builtin-shadowed"


def test_matrix_rounding_helpers_avoid_user_names(tmp_path):
    source = _source(
        helpers="uint __crossgl_software_matrix_round_bits(uint value) { return value; }"
    )
    generated = _codegen(cooperative_matrix_mma_profile="rne-gradual").generate(
        parse(source)
    )
    assert "uint __crossgl_software_matrix_round_bits_1(" in generated
    _compile(generated, tmp_path)


@pytest.mark.parametrize("profile", PROFILES)
def test_project_matrix_rounding_package_records_profile(tmp_path, profile):
    _, request, _ = _project_request(
        tmp_path, "directx", rounding=True, mma_profile=profile
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert (
        report["artifacts"][0]["provenance"]["cooperativeMatrixMmaProfile"] == profile
    )
    assert request.artifact["provenance"]["cooperativeMatrixMmaProfile"] == profile
    assert "uint2 result = asuint(accumulator);" in request.artifact_path.read_text()
    _compile(request.artifact_path.read_text(), tmp_path)
    path = tmp_path / "report.json"
    assert validate_project_report(path)["success"]
    assert build_runtime_artifact_manifest(path)["success"]
    for invalid in (
        None,
        "native",
        "rne-flush" if profile == "rne-gradual" else "rne-gradual",
        1,
        [],
    ):
        provenance = report["artifacts"][0]["provenance"]
        if invalid is None:
            provenance.pop("cooperativeMatrixMmaProfile", None)
        else:
            provenance["cooperativeMatrixMmaProfile"] = invalid
        path.write_text(json.dumps(report))
        result = validate_project_report(path)
        assert not result["success"]
        assert "cooperativeMatrixMmaProfile" in json.dumps(result["diagnostics"])


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_matrix_rounding_participates_in_dispatch_region_identity(
    tmp_path, monkeypatch, target
):
    from crosstl.project import pipeline
    from crosstl.translator.dispatch_regions import plan_dispatch_regions
    from tests.test_translator.test_dispatch_region_project import _config

    settings = []
    build = pipeline.build_dispatch_region_program

    def record(**kwargs):
        settings.append(kwargs["settings"])
        return build(**kwargs)

    monkeypatch.setattr(pipeline, "build_dispatch_region_program", record)
    hashes = []
    region = plan_dispatch_regions([32], [32])[0]
    for profile in (None, *PROFILES):
        config = _config(tmp_path / str(profile), target, region)
        config.source_options["metal"]["target_options"][target][
            "cooperative_matrix_mma_profile"
        ] = profile
        report = translate_project(config, format_output=False).to_json()
        assert not report["diagnostics"], report["diagnostics"]
        hashes.append(
            report["artifacts"][0]["provenance"]["dispatchRegionProgram"][
                "settingsHash"
            ]
        )
    assert "cooperativeMatrixMmaProfile" not in settings[0]
    assert [s["cooperativeMatrixMmaProfile"] for s in settings[1:]] == list(PROFILES)
    assert len(set(hashes)) == len(PROFILES) + 1


@pytest.mark.parametrize("profile", [True, 32, "", "native", [], {}])
@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_project_matrix_rounding_rejects_invalid_profiles(tmp_path, profile, target):
    (tmp_path / "plain.cgl").write_text("shader Plain { void helper() {} }")
    if isinstance(profile, (list, dict)):
        with pytest.raises(ValueError, match="strings, integers, or booleans"):
            ProjectConfig(
                root=tmp_path,
                targets=(target,),
                source_options={"crossgl": {"cooperative_matrix_mma_profile": profile}},
            )
        return
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            source_options={"crossgl": {"cooperative_matrix_mma_profile": profile}},
        ),
        format_output=False,
    ).to_json()
    assert report["diagnostics"]
    assert any(
        "cooperative_matrix_mma_profile must be" in row["message"]
        for row in report["diagnostics"]
    )
    assert not any(row["status"] == "translated" for row in report["artifacts"])


def _multiply_words(left, right, accumulator, profile):
    result = [row[:] for row in accumulator]
    flush = profile.endswith("-flush")
    for row in range(8):
        for column in range(8):
            for inner in range(8):
                x, y = left[row][inner], right[inner][column]
                if profile.startswith("fma-"):
                    result[row][column] = _oracle(x, y, result[row][column], flush)
                else:
                    product = _oracle(x, y, (x ^ y) & 0x80000000, flush)
                    result[row][column] = _oracle(
                        result[row][column], 0x3F800000, product, flush
                    )
    return result


def _profile_case(profile, *, rounding=True, products=2):
    assert products in (1, 2)
    words, _ = _case(rounding=rounding)
    expected = []
    for subgroup in range(12):
        matrices = [[[0] * 8 for _ in range(8)] for _ in range(3)]
        for lane in range(32):
            for operand, matrix in enumerate(matrices):
                for element, (row, column) in enumerate(_coordinates(lane)):
                    matrix[row][column] = words[
                        (subgroup * 32 + lane) * 6 + operand * 2 + element
                    ]
        left, right, accumulator = matrices
        result = _multiply_words(left, right, accumulator, profile)
        if products == 2:
            result = _multiply_words(right, left, result, profile)
        expected.extend(
            result[row][column]
            for lane in range(32)
            for row, column in _coordinates(lane)
        )
    return words, expected


def _plain_product_case(profile):
    """``(left * right) * accumulator`` with each product into a +0.0 matrix."""
    words, _ = _case(rounding=True)
    zero = [[0] * 8 for _ in range(8)]
    expected = []
    for subgroup in range(12):
        matrices = [[[0] * 8 for _ in range(8)] for _ in range(3)]
        for lane in range(32):
            for operand, matrix in enumerate(matrices):
                for element, (row, column) in enumerate(_coordinates(lane)):
                    matrix[row][column] = words[
                        (subgroup * 32 + lane) * 6 + operand * 2 + element
                    ]
        left, right, accumulator = matrices
        product = _multiply_words(left, right, zero, profile)
        result = _multiply_words(product, accumulator, zero, profile)
        expected.extend(
            result[row][column]
            for lane in range(32)
            for row, column in _coordinates(lane)
        )
    return words, expected


def test_plain_product_oracle_differs_from_accumulation():
    _, plain = _plain_product_case("fma-rne-gradual")
    _, accumulated = _profile_case("fma-rne-gradual")
    _, separately_rounded = _plain_product_case("rne-gradual")
    assert len(plain) == len(accumulated) == 12 * 64
    assert plain != accumulated
    # Fusing each inner product changes some of the rounded plain products.
    assert any(a != b for a, b in zip(plain, separately_rounded))


def _edge_case(profile):
    triples = [
        PRECISE_OPERANDS,
        (0x00800000, 0x3F000000, 0),
        (1, 0x3F800000, 0),
        (0x007FFFFF, 0x3F800000, 0),
        (0x00800000, 0x3F7FFFFF, 0),
        (0x7F7FFFFF, 0x40000000, 0),
        (0x7F800000, 0, 0),
        (0xFF800001, 0x3F800000, 0),
        (0x80000000, 0x3F800000, 0x80000000),
        (0xBF800000, 0x3F800000, 0x3F800000),
        (0x3F800000, 0x33800000, 0x3F800000),
        (0x3F800001, 0x33800000, 0x3F800000),
    ]
    words, expected = [], []
    for a, b, c in triples:
        zero = (a ^ b) & 0x80000000
        left = [[a] + [zero] * 7 for _ in range(8)]
        right = [[b] * 8] + [[0x3F800000] * 8 for _ in range(7)]
        accumulator = [[c] * 8 for _ in range(8)]
        result = _multiply_words(left, right, accumulator, profile)
        for lane in range(32):
            coordinates = _coordinates(lane)
            words.extend(
                matrix[row][column]
                for matrix in (left, right, accumulator)
                for row, column in coordinates
            )
            expected.extend(result[row][column] for row, column in coordinates)
    return words, expected


def test_matrix_rounding_edge_oracle_distinguishes_profiles_and_contraction():
    words, gradual = _edge_case("rne-gradual")
    other_words, flush = _edge_case("rne-flush")
    assert words == other_words and len(words) == 2304
    assert len(gradual) == len(flush) == 768
    assert gradual[:64] == flush[:64] == [0] * 64
    assert _oracle(*PRECISE_OPERANDS) == 0xA8800000
    assert gradual[64:128] == [0x00400000] * 64
    assert flush[64:128] == [0] * 64
    assert gradual[8 * 64 : 9 * 64] == [0x80000000] * 64
    assert 0x7F800000 in gradual and 0x7FC00000 in gradual
    assert gradual[10 * 64 : 11 * 64] == [0x3F800000] * 64
    assert gradual[11 * 64 :] == [0x3F800001] * 64
    _, fused = _edge_case("fma-rne-gradual")
    assert fused[:64] == [0xA8800000] * 64
    assert _profile_case("rne-gradual") == _case(rounding=True)
    _, separately_rounded = _profile_case("rne-gradual")
    _, fused = _profile_case("fma-rne-gradual")
    assert sum(a != b for a, b in zip(separately_rounded, fused)) == 321


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("inputs", ["edges", "quarter", "rounding"])
def test_matrix_rounding_native_edges(tmp_path, monkeypatch, profile, inputs):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX matrix execution")
    assert sys.platform == "win32", "DirectX matrix execution requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    body = f"""{MATRIX} result = cooperative_matrix_multiply_accumulate(left, right, accumulator);
        outputWords[index * 2u] = asuint(cooperative_matrix_element(result, 0));
        outputWords[index * 2u + 1u] = asuint(cooperative_matrix_element(result, 1));"""
    generated = _codegen(cooperative_matrix_mma_profile=profile).generate_stage(
        parse(_source(body)), "compute"
    )
    words, expected = (
        _edge_case(profile)
        if inputs == "edges"
        else _profile_case(profile, rounding=inputs == "rounding", products=1)
    )
    assert (
        _execute_words(
            tmp_path,
            generated,
            words,
            expected,
            workgroup_size=(32, 4, 1),
            buffer_byte_offset=16,
        )
        == expected
    )


def test_matrix_rounding_keeps_required_native_ci():
    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/demo-project-testing.yml").read_text()
    )
    step = next(
        s
        for s in workflow["jobs"]["portable-host"]["steps"]
        if s.get("id") == "collective-helpers"
    )
    assert "tests/test_translator/test_directx_matrix_rounding.py" in step["run"]
    assert "tests/test_translator/test_directx_cooperative_matrix.py" in step["run"]
    assert step["env"][REQUIRE_ENV] == "${{ runner.os == 'Windows' && '1' || '0' }}"
    assert not step.get("continue-on-error")
    # Run the expensive control before the batch can exhaust its remaining time.
    command = step["run"]
    assert command.index(
        "tests/test_translator/test_directx_matrix_rounding.py"
    ) < command.index("tests/test_translator/test_static_resource_aggregates.py")
    assert "if" not in step


@pytest.mark.parametrize("target", ["metal", "vulkan"])
def test_project_matrix_rounding_rejects_other_targets(tmp_path, target):
    (tmp_path / "plain.cgl").write_text("shader Plain { void helper() {} }")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            targets=(target,),
            source_options={
                "crossgl": {"cooperative_matrix_mma_profile": "rne-gradual"}
            },
        ),
        format_output=False,
    ).to_json()
    assert report["diagnostics"]
    assert any("only by the DirectX" in row["message"] for row in report["diagnostics"])
    assert not any(row["status"] == "translated" for row in report["artifacts"])


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("shape,rounding", [((32, 4, 1), False), ((16, 8, 1), True)])
def test_matrix_rounding_native_words(tmp_path, monkeypatch, profile, shape, rounding):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX matrix execution")
    assert sys.platform == "win32", "DirectX matrix execution requires Windows"
    monkeypatch.setenv("CROSTL_REQUIRE_DIRECTX_FLOAT_ATOMICS", "1")
    generated = _codegen(cooperative_matrix_mma_profile=profile).generate_stage(
        parse(_source(width=shape[0], height=shape[1])), "compute"
    )
    words, expected = _profile_case(profile, rounding=rounding)
    actual = _execute_words(
        tmp_path,
        generated,
        words,
        expected,
        workgroup_size=shape,
        buffer_byte_offset=16,
    )
    assert actual == expected


@pytest.mark.parametrize("profile", PROFILES)
def test_project_matrix_rounding_native_package(tmp_path, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX matrix execution")
    assert sys.platform == "win32", "DirectX matrix execution requires Windows"
    _, request, expected = _project_request(
        tmp_path,
        "directx",
        rounding=True,
        mma_profile=profile,
        word_case=_profile_case(profile),
    )
    _execute(request, expected, tmp_path)


def test_original_metal_matrix_matches_fused_reference(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for matrix execution"
        )
    if sys.platform != "darwin":
        pytest.skip("The original Metal matrix control requires macOS")
    source, request, expected = _project_request(
        tmp_path, "metal", rounding=True, word_case=_profile_case("fma-rne-gradual")
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="matrix_products",
    )


@pytest.mark.parametrize("profile", ["fma-rne-gradual", "rne-gradual"])
def test_project_matrix_plain_products_native_package(tmp_path, profile):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for DirectX matrix execution")
    assert sys.platform == "win32", "DirectX matrix execution requires Windows"
    _, request, expected = _project_request(
        tmp_path,
        "directx",
        mma_profile=profile,
        word_case=_plain_product_case(profile),
        plain_products=True,
    )
    _execute(request, expected, tmp_path)


def test_original_metal_plain_products_match_fused_reference(tmp_path):
    if os.environ.get("CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_COOPERATIVE_MATRIX_RUNTIME=1 for matrix execution"
        )
    if sys.platform != "darwin":
        pytest.skip("The original Metal matrix control requires macOS")
    source, request, expected = _project_request(
        tmp_path,
        "metal",
        word_case=_plain_product_case("fma-rne-gradual"),
        plain_products=True,
    )
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="matrix_products",
    )
