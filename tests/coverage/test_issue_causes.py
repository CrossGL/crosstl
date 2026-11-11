"""How wrong results are grouped into one bug issue per cause."""

import pytest

from tools.coverage import causes, issues

from .issue_cases import backend_row, coverage_report, feature_entry, issues_after


def test_causes_keep_the_digits_of_names():
    first = causes.cause(
        {"detail": "probe.hlsl:3:9: error: undeclared identifier 'packSnorm2x16'"}
    )
    second = causes.cause(
        {"detail": "probe.hlsl:4:9: error: undeclared identifier 'packSnorm4x8'"}
    )
    assert first == "probe.hlsl:N:N: error: undeclared identifier 'packSnorm2x16'"
    assert first != second


def test_causes_keep_the_digits_of_template_arguments():
    # Slang writes HLSL's float3x4 as matrix<float,3,4>; transposed shapes are different causes, as in HLSL.
    detail = (
        "probe.slang:{}:5: error[E30019]: type mismatch in expression: expected an expression of "
        "type 'matrix<float,{},{}>', got 'matrix<float,{},{}>'"
    )
    three_by_four = causes.cause({"detail": detail.format(12, 3, 4, 4, 3)})
    four_by_three = causes.cause({"detail": detail.format(14, 4, 3, 3, 4)})
    assert three_by_four == detail.format("N", 3, 4, 4, 3).replace(":5:", ":N:")
    assert three_by_four != four_by_three
    assert causes.cause({"detail": "probe.cu(7): error: too many arguments (3)"}) == (
        "probe.cu(N): error: too many arguments (N)"
    )


def test_titles_drop_the_glslang_location():
    detail = "ERROR: probe.frag:2: 'location' : SPIR-V requires location for user input/output"
    row = backend_row(
        "glsl", "output", [feature_entry("textureSize", "wrong", detail=detail)]
    )
    [bug] = [
        a for a in issues.plan(coverage_report(row), []) if a.key.startswith("wrong/")
    ]
    assert bug.title == (
        "GLSL output: 'location' : SPIR-V requires location for user input/output"
    )


def test_causes_keep_an_operator_spelling():
    detail = "error: cannot index into a value of type `[]`"
    feature = {"name": "vector_extract_dynamic", "crossgl": "[]", "detail": detail}
    assert causes.cause(feature) == detail


def test_causes_name_what_an_expression_spelling_calls():
    # Slang's getRealtimeClockLow is CrossGL's clockRealtime2x32EXT().x, which fails as getRealtimeClock does.
    detail = (
        "probe.slang:9:14: error[E30015]: undefined identifier 'clockRealtime2x32EXT'"
    )
    low = {"name": "getRealtimeClockLow", "detail": detail}
    full = {"name": "getRealtimeClock", "detail": detail}
    assert (
        causes.cause(low, "slang")
        == causes.cause(full, "slang")
        == ("probe.slang:N:N: error[E30015]: undefined identifier '<feature>'")
    )


def test_features_whose_diagnostics_name_them_share_a_cause():
    row = backend_row(
        "wgsl",
        "output",
        [
            feature_entry(
                "firstLeadingBit",
                "wrong",
                detail="error: no definition in scope for identifier: `findMSB`",
            )
            | {"crossgl": "findMSB"},
            feature_entry(
                "countOneBits",
                "wrong",
                detail="error: no definition in scope for identifier: `bitCount`",
            )
            | {"crossgl": "bitCount"},
            feature_entry(
                "RayQuery.Abort",
                "wrong",
                detail="ValueError: RayQuery.Abort receiver must be RayQuery",
            ),
            feature_entry(
                "RayQuery.Proceed",
                "wrong",
                detail="ValueError: RayQuery.Proceed receiver must be RayQuery",
            ),
            feature_entry(
                "max",
                "wrong",
                detail="error: no definition in scope for identifier: `max_value`",
            ),
        ],
    )
    bugs = [
        a for a in issues.plan(coverage_report(row), []) if a.key.startswith("wrong/")
    ]
    assert [a.title for a in bugs] == [
        "WGSL output: ValueError: <feature> receiver must be RayQuery",
        "WGSL output: no definition in scope for identifier: `<feature>`",
        "WGSL output: no definition in scope for identifier: `max_value`",
    ]
    assert "Affected features: `countOneBits`, `firstLeadingBit`." in bugs[1].body
    # The prefix of an input row's diagnostic is not the name of a feature called ``round``.
    detail = 'after the round trip: probe.cu(3): error: identifier "f64" is undefined'
    assert causes.cause({"name": "round", "detail": detail}) == (
        'after the round trip: probe.cu(N): error: identifier "f64" is undefined'
    )
    assert causes.cause(
        {"name": "Vec3::round", "crossgl": "round", "detail": detail}
    ) == ('after the round trip: probe.cu(N): error: identifier "f64" is undefined')


def test_a_shared_diagnostic_that_names_a_feature_stays_one_cause():
    detail = "ValueError: Rust geometry stage requires maxvertexcount attribute"
    features = [
        feature_entry(name, "wrong", category, detail)
        for name, category in (
            ("emit_vertex", "function"),
            ("end_primitive", "function"),
            ("geometry", "stage"),
        )
    ]
    row = backend_row("rust", "output", features)
    assert set(causes.causes_of(row).values()) == {detail}
    # A diagnostic that names every feature failing with it still leaves the name out.
    overload = (
        'error: no instance of overloaded function "{}" matches the argument list'
    )
    spelled = [
        feature_entry("modf", "wrong", detail=overload.format("modf")),
        feature_entry("modff", "wrong", detail=overload.format("modf")) | {
            "crossgl": "modf"
        },
        feature_entry("frexp", "wrong", detail=overload.format("frexp")),
    ]
    assert set(causes.causes_of(backend_row("cuda", "output", spelled)).values()) == {
        overload.format(causes.FEATURE)
    }
    # The issue filed before the stage failed the same way takes it in rather than a second issue opening.
    row["features"] = features[:2]
    state = issues_after(issues.plan(coverage_report(row), []))
    (bug,) = (i for i in state if i.key.startswith("wrong/"))
    row["features"] = features
    actions = [
        a
        for a in issues.plan(coverage_report(row), state)
        if a.key.startswith("wrong/")
    ]
    assert [(a.kind, a.number) for a in actions] == [("edit", bug.number)]
    assert "Affected features: `emit_vertex`, `end_primitive`, `geometry`." in (
        actions[0].body
    )


def test_a_method_shares_a_cause_under_the_name_crossgl_gives_it():
    detail = "after the round trip: probe.hlsl:{}:5: error: use of undeclared identifier '{}'"
    row = backend_row(
        "hlsl",
        "input",
        [
            feature_entry(
                f"RWByteAddressBuffer.{name}",
                "wrong",
                "method",
                detail.format(line, spelled),
            )
            | {"crossgl": name}
            for name, spelled, line in (
                ("InterlockedAnd", "atomicAnd", 11),
                ("InterlockedMax", "atomicMax", 12),
            )
        ],
    )
    bugs = [
        a for a in issues.plan(coverage_report(row), []) if a.key.startswith("wrong/")
    ]
    assert [a.title for a in bugs] == [
        "HLSL input: use of undeclared identifier '<feature>'"
    ]
    # Without the language the spelling is not known, so each diagnostic keeps it.
    assert causes.cause(row["features"][0]).endswith("'atomicAnd'")


@pytest.mark.parametrize(
    "language, detail, spelled",
    [
        # OpenCL C features have no CrossGL spelling of their own, since nothing is generated for OpenCL C.
        (
            "opencl",
            "ValueError: DirectX typed buffer atomic '{}' original argument must be scalar int, "
            "got memory_order_relaxed",
            {
                "atomic_fetch_add_explicit": "atomicAdd",
                "atomic_exchange_explicit": "atomicExchange",
                "atomic_xchg": "atomicExchange",
                "atom_cmpxchg": "atomicCompSwap",
            },
        ),
        (
            "slang",
            "after the round trip: error[E30015]: undefined identifier '{}'",
            {"fwidth_coarse": "fwidthCoarse", "fwidth_fine": "fwidthFine"},
        ),
        # Names CrossGL has for a function that are no GLSL function: a wave operation and a built-in value.
        (
            "slang",
            "error[E30015]: undefined identifier '{}'",
            {
                "WavePrefixMax": "WavePrefixExclusiveMax",
                "WaveGetNumWaves": "gl_NumSubgroups",
            },
        ),
    ],
    ids=["opencl", "slang", "slang-other-names"],
)
def test_a_function_shares_a_cause_under_the_crossgl_name_of_its_language(
    language, detail, spelled
):
    row = backend_row(
        language,
        "input",
        [
            feature_entry(name, "wrong", detail=detail.format(crossgl))
            for name, crossgl in spelled.items()
        ],
    )
    assert set(causes.causes_of(row).values()) == {detail.format(causes.FEATURE)}
