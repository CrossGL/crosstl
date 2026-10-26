"""Tests of the crash-fuzzing harness: generator validity, classification, mutation and the baseline ratchet."""

from __future__ import annotations

import json
import linecache
import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, seed, settings
from hypothesis import strategies as st

from crosstl.translator.diagnostics import TranslationDiagnostic
from tests.conformance.fuzz import campaign, mutate, outcomes, worker
from tests.conformance.fuzz.preprocessor import units
from tests.conformance.fuzz.printers import PRINTERS, GLSLPrinter
from tests.conformance.fuzz.program import (
    FLOAT_TO_INTEGER_RANGE,
    Config,
    Construct,
    Convert,
    Literal,
    programs,
)
from tests.conformance.fuzz.spirv import SpirvPrinter
from tests.conformance.fuzz.targets import IMPORTERS
from tests.conformance.process import run
from tests.conformance.toolchains import tool

FEW = settings(
    max_examples=10,
    database=None,
    deadline=None,
    derandomize=True,
    suppress_health_check=list(HealthCheck),
)


def _compile(tool_name, arguments, text, suffix, then=None):
    """Compile ``text`` with a native tool; ``then`` (tool name, arguments) checks the output in turn."""
    steps = [(tool_name, arguments)] + ([then] if then else [])
    tools = [tool(name) for name, _ in steps]
    for found in tools:
        if not found.available:
            pytest.skip(found.reason)
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / f"k{suffix}"
        source.write_text(text, encoding="utf-8")
        output = Path(scratch) / "k.out"
        for found, (_, step_arguments) in zip(tools, steps):
            filled = [a.format(src=source, out=output) for a in step_arguments]
            result = run([*found.command, *filled], timeout=120)
            log = result.log or ""
            # DXC can exit 0 after printing errors, so look at the log as well.
            assert (
                result.returncode == 0 and "error" not in log.lower()
            ), f"{log}\n{text}"


# -- generated programs are valid in every language -----------------------------------------------------------
@FEW
@given(programs())
def test_programs_compile_as_metal(program):
    _compile(
        "metal",
        ["-c", "{src}", "-o", "{out}"],
        PRINTERS["metal"]().print(program),
        ".metal",
    )


@FEW
@given(programs())
def test_programs_compile_as_hlsl(program):
    arguments = ["-T", "cs_6_0", "-E", "main", "-HV", "2021", "{src}", "-Fo", "{out}"]
    _compile("dxc", arguments, PRINTERS["directx"]().print(program), ".hlsl")


@FEW
@given(programs(), st.sampled_from(("450", "310 es")))
def test_programs_compile_as_glsl(program, version):
    text = GLSLPrinter(version).print(program)
    _compile("glslangValidator", ["-V", "{src}", "-o", "{out}"], text, ".comp")


@FEW
@given(programs())
def test_programs_assemble_and_validate_as_spirv(program):
    _compile(
        "spirv-as",
        ["--target-env", "vulkan1.1", "{src}", "-o", "{out}"],
        SpirvPrinter().print(program),
        ".spvasm",
        then=("spirv-val", ["--target-env", "vulkan1.1", "{out}"]),
    )


@FEW
@given(programs(Config(vectors=False)))
def test_scalar_programs_compile_as_opencl(program):
    text = PRINTERS["opencl"]().print(program)
    _compile(
        "clang", ["-x", "cl", "-cl-std=CL1.2", "-fsyntax-only", "{src}"], text, ".cl"
    )


@FEW
@given(programs())
def test_printing_is_deterministic(program):
    for name, printer in PRINTERS.items():
        if name in ("cuda", "hip", "opencl"):
            continue
        assert printer().print(program) == printer().print(program)
    assert SpirvPrinter().print(program) == SpirvPrinter().print(program)


def _nodes(node):
    yield node
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _nodes(item)
    elif hasattr(node, "__dataclass_fields__"):
        for name in node.__dataclass_fields__:
            yield from _nodes(getattr(node, name))


@given(programs())
def test_float_literal_conversions_to_integers_are_defined(program):
    # DXC folds an out-of-range float-to-int constant conversion to undef (#3109).
    for node in _nodes(program):
        if (
            not isinstance(node, Convert)
            or node.type.kind not in FLOAT_TO_INTEGER_RANGE
        ):
            continue
        low, high = FLOAT_TO_INTEGER_RANGE[node.type.kind]
        operand = node.operand
        literals = operand.args if isinstance(operand, Construct) else (operand,)
        for literal in literals:
            if isinstance(literal, Literal) and literal.type.kind == "float":
                assert low <= literal.value <= high


def test_scalar_printers_refuse_vectors():
    from tests.conformance.fuzz.program import Type

    with pytest.raises(ValueError):
        PRINTERS["cuda"]().type(Type("float", 3))


# -- outcome classification -----------------------------------------------------------------------------------
FAKE_MODULE = """\
def explicit():
    raise ValueError("deliberate")

def implicit():
    return int("not a number")

def subscript():
    return [][0]

def explicit_key():
    raise KeyError("k")

def multi_line():
    raise SyntaxError(
        "spans lines"
    )

def recurse(n):
    return recurse(n + 1)
"""


@pytest.fixture(scope="module")
def fake():
    filename = os.path.join(outcomes.PACKAGE_ROOT, "_fuzz_classification_fixture.py")
    lines = FAKE_MODULE.splitlines(keepends=True)
    linecache.cache[filename] = (len(FAKE_MODULE), None, lines, filename)
    namespace = {}
    exec(compile(FAKE_MODULE, filename, "exec"), namespace)
    yield namespace
    linecache.cache.pop(filename, None)
    outcomes._raise_lines.cache_clear()


def _raised(function, *args):
    try:
        function(*args)
    except BaseException as error:  # noqa: BLE001
        return error
    raise AssertionError("expected an exception")


@pytest.mark.parametrize(
    "name, status",
    [
        ("explicit", "rejected"),
        ("multi_line", "rejected"),
        ("implicit", "crash"),
        ("subscript", "crash"),
        ("explicit_key", "crash"),
    ],
)
def test_classification(fake, name, status):
    assert outcomes.classify(_raised(fake[name])) == status


def test_typed_diagnostics_are_refusals_and_success_is_accepted():
    assert outcomes.classify(TranslationDiagnostic("no")) == "refused"
    assert outcomes.classify(None) == "accepted"


def test_exceptions_raised_outside_crosstl_are_crashes():
    assert outcomes.classify(_raised(int, "x")) == "crash"


def test_recursion_is_attributed_to_the_repeating_function(fake):
    error = _raised(fake["recurse"], 0)
    assert outcomes.classify(error) == "crash"
    key = outcomes.signature("metal", "cgl", "crash", error)
    assert (
        key
        == "metal|crash|RecursionError|crosstl/_fuzz_classification_fixture.py:recurse"
    )


def test_signatures_name_the_route_and_omit_line_numbers(fake):
    error = _raised(fake["subscript"])
    assert outcomes.signature("cgl", "metal", "crash", error) == (
        "cgl->metal|crash|IndexError|crosstl/_fuzz_classification_fixture.py:subscript"
    )
    assert outcomes.signature("opengl", "cgl", "timeout") == "opengl|timeout"


# -- mutation -------------------------------------------------------------------------------------------------
@FEW
@given(st.text(max_size=200))
def test_tokens_cover_the_text(text):
    assert "".join(mutate.tokens(text)) == text


@FEW
@given(mutate.mutated(st.just("int main() { return 0; }"), "metal"))
def test_mutants_are_text(text):
    assert isinstance(text, str)


# -- the worker -----------------------------------------------------------------------------------------------
@settings(FEW, max_examples=40)
@given(units())
def test_preprocessor_never_crashes(unit):
    job = {"kind": "preprocess", "importer": "preprocessor", "backend": "preprocess"}
    outcome = worker.run_job({**job, "unit": unit.to_json()}, None)
    assert outcome["status"] in ("accepted", "refused"), outcome


def test_worker_classifies_a_translation(tmp_path):
    job = {
        "kind": "translate",
        "importer": "directx",
        "extension": ".hlsl",
        "backend": "cgl",
        "source": "[numthreads(1, 1, 1)] void main() {}\n",
    }
    assert worker.run_job(job, str(tmp_path))["status"] == "accepted"


def test_every_importer_has_a_strategy():
    for name, target in IMPORTERS.items():

        @seed(0)
        @settings(FEW, max_examples=3)
        @given(target.strategy())
        def draw(case):
            kind, extension, payload = case
            assert kind in ("valid", "mutant", "seed")
            assert extension.startswith(".")
            assert target.job(case, target.backends[0])["importer"] == name

        draw()


# -- the baseline ratchet -------------------------------------------------------------------------------------
class FakeWorker:
    def __init__(self, outcomes_by_source):
        self.outcomes = outcomes_by_source

    def translate(self, job):
        return dict(self.outcomes.get(job.get("source"), {"status": "accepted"}))


def _baseline(tmp_path, entries):
    path = tmp_path / "baseline.json"
    (tmp_path / "reproducers").mkdir()
    crashes = {}
    for key, source in entries.items():
        name = f"metal-{len(crashes)}.metal"
        (tmp_path / "reproducers" / name).write_text(source, encoding="utf-8")
        crashes[key] = {
            "importer": "metal",
            "backend": "cgl",
            "extension": ".metal",
            "input": "valid",
            "reproducer": name,
            "message": "",
        }
    path.write_text(
        json.dumps({"schemaVersion": 1, "crashes": crashes}), encoding="utf-8"
    )
    return path


def test_replay_reports_reproducers_that_no_longer_crash(tmp_path):
    path = _baseline(
        tmp_path,
        {
            "metal|crash|IndexError|a.py:f": "still",
            "metal|crash|KeyError|b.py:g": "gone",
        },
    )
    fake_worker = FakeWorker(
        {"still": {"status": "crash", "signature": "metal|crash|IndexError|a.py:f"}}
    )
    baseline = campaign.load_baseline(path)
    fixed = campaign.replay(
        baseline, fake_worker, ["metal"], campaign.reproducers(path)
    )
    assert fixed == {"metal|crash|KeyError|b.py:g": "accepted"}


def test_updating_the_baseline_adds_findings_and_drops_fixed_entries(tmp_path):
    path = _baseline(tmp_path, {"metal|crash|KeyError|b.py:g": "gone"})
    baseline = campaign.load_baseline(path)
    finding = {
        "text": "kernel",
        "importer": "metal",
        "backend": "cgl",
        "extension": ".metal",
        "input": "mutant",
        "message": "boom",
    }
    campaign.write_baseline(
        path,
        baseline,
        {"metal|crash|IndexError|a.py:f": finding},
        {"metal|crash|KeyError|b.py:g": "accepted"},
    )
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert list(updated["crashes"]) == ["metal|crash|IndexError|a.py:f"]
    name = updated["crashes"]["metal|crash|IndexError|a.py:f"]["reproducer"]
    assert (tmp_path / "reproducers" / name).read_text(encoding="utf-8") == "kernel"
    assert sorted(p.name for p in (tmp_path / "reproducers").iterdir()) == [name]


def test_a_campaign_records_new_signatures_once_and_shrinks_them():
    crash = {
        "status": "crash",
        "signature": "opencl|crash|IndexError|x.py:f",
        "message": "m",
    }

    class Crashy:
        def translate(self, job):
            return (
                dict(crash) if "fuzz_main" in job["source"] else {"status": "accepted"}
            )

    result = campaign.Campaign(
        IMPORTERS["opencl"],
        Crashy(),
        known=(),
        seconds=None,
        examples=30,
        seed_value=3,
        shrink_seconds=30,
        backends=None,
    ).run()
    assert list(result.findings) == ["opencl|crash|IndexError|x.py:f"]
    assert result.generated == 30
    assert sum(result.crashes.values()) >= 1


def test_known_signatures_do_not_fail_a_campaign():
    crash = {"status": "crash", "signature": "opencl|crash|IndexError|x.py:f"}

    class Crashy:
        def translate(self, job):
            return dict(crash)

    result = campaign.Campaign(
        IMPORTERS["opencl"],
        Crashy(),
        known=("opencl|crash|IndexError|x.py:f",),
        seconds=None,
        examples=10,
        seed_value=0,
        shrink_seconds=30,
        backends=None,
    ).run()
    assert result.findings == {}
    assert result.crashes["opencl|crash|IndexError|x.py:f"] == 10


def test_the_committed_baseline_is_well_formed():
    baseline = campaign.load_baseline(campaign.BASELINE)
    directory = campaign.reproducers(campaign.BASELINE)
    for key, entry in baseline["crashes"].items():
        route = (
            entry["importer"]
            if entry["backend"] == "cgl"
            else f"{entry['importer']}->{entry['backend']}"
        )
        assert key.startswith(route + "|"), key
        assert (directory / entry["reproducer"]).is_file(), key
        assert entry["importer"] in IMPORTERS
    if directory.is_dir():
        listed = {entry["reproducer"] for entry in baseline["crashes"].values()}
        assert {p.name for p in directory.iterdir()} == listed
