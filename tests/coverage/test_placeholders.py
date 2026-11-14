import pytest

from tests.coverage.issue_cases import backend_row, coverage_report, feature_entry
from tools.coverage import causes, issues, placeholders, runner, validators
from tools.coverage.model import Feature, Param, Signature, Type

# What the HIP importer writes for a device timer it cannot import, and the rust-gpu generator for a resource.
HIP_TIMER = "(/* HIP device timer clock64() not directly supported in CrossGL */ 0)"
RUST_RESOURCE = (
    "// CrossGL Rust limitation: resource tex is emitted as a compile-only placeholder static, not a rust-gpu "
    "resource binding"
)


@pytest.mark.parametrize(
    "source, expected",
    [
        (f"var r: i64 = {HIP_TIMER};", HIP_TIMER[1:-3]),
        (f"{RUST_RESOURCE}\nstatic tex: u32 = 0;", RUST_RESOURCE),
        (
            "OpNop\n; WARNING: dropped OpReadClockKHR\nOpReturn",
            "; WARNING: dropped OpReadClockKHR",
        ),
        ("float r = /* unsupported\n   wave op */ 0.0;", "/* unsupported wave op */"),
        (
            "x = 0  # using default value for the lane count",
            "# using default value for the lane count",
        ),
    ],
)
def test_a_comment_that_marks_a_loss_is_a_placeholder(source, expected):
    assert placeholders.placeholder(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "#version 450\n#extension GL_KHR_shader_subgroup_basic : require\nvoid main() {}",
        "#pragma warning(disable: 3557)\n// Kernel: k\nuint unsupported_count = 0u;",
        "#[spirv(compute(threads(64)))]\npub fn main() {}",
        'int r = 0; // warnings are errors in this build\nconst char* s = "unsupported";',
    ],
)
def test_ordinary_comments_and_names_are_not_placeholders(source):
    assert placeholders.placeholder(source) == ""


def _accept_everything(monkeypatch, translations):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )
    monkeypatch.setattr(
        runner, "_native_rejection", lambda feature, language: (feature, "")
    )
    monkeypatch.setattr(runner, "parse_crossgl", lambda source: None)
    monkeypatch.setattr(
        runner, "translate", lambda source, suffix, backend, _: translations[backend]
    )


SIN = Signature(Type("float"), (Param(Type("float")),))


def test_a_round_trip_through_a_placeholder_import_is_wrong(monkeypatch):
    # The CrossGL import keeps the comment; the HIP written back from it has only the zero, which hipcc accepts.
    _accept_everything(monkeypatch, {"cgl": f"r = {HIP_TIMER};", "hip": "r = 0;"})
    feature = Feature("hip", "function", "sin", SIN, crossgl="sin")
    result = runner.run_input(feature, "hip")
    assert result.outcome == "wrong"
    assert result.detail == placeholders.detail(
        HIP_TIMER[1:-3], "after the round trip: "
    )


def test_a_translation_with_a_placeholder_is_wrong(monkeypatch):
    _accept_everything(monkeypatch, {"rust": f"{RUST_RESOURCE}\nstatic tex: u32 = 0;"})
    feature = Feature("rust", "function", "sin", SIN, crossgl="sin")
    result = runner.run_output(feature, "rust")
    assert (result.outcome, result.detail) == (
        "wrong",
        placeholders.detail(RUST_RESOURCE),
    )


def test_placeholders_in_one_row_are_one_bug_that_says_what_failed():
    comments = {
        "clock64": HIP_TIMER[1:-3],
        "__ballot": (
            "/* hip warp intrinsic __ballot(a0) not directly supported in CrossGL */"
        ),
    }
    row = backend_row(
        "hip",
        "input",
        [
            feature_entry(
                name,
                "wrong",
                detail=placeholders.detail(text, "after the round trip: "),
            )
            for name, text in comments.items()
        ],
    )
    [(cause, features)] = causes.grouped(row)
    assert cause == "after the round trip: " + placeholders.PLACEHOLDER
    assert len(features) == 2
    bug = issues.plan(coverage_report(row), [])[1]
    assert (
        "wrote a placeholder for the feature instead of raising a typed diagnostic"
        in bug.body
    )
    assert comments["__ballot"] in bug.body
