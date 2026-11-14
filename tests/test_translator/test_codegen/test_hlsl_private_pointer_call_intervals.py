"""DirectX private pointer helpers whose integer parameters their calls bound."""

import pytest

import crosstl.translator
from crosstl.translator.codegen import hlsl_private_pointer_call_intervals as intervals
from crosstl.translator.codegen.directx_codegen import (
    DirectXPrivatePointerParameterError,
    HLSLCodeGen,
)
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_warnings_clean_if_available,
)

# Moves the view by 4 * i for i < count / 4: offsets 0, 4, 12, ... so a count
# of at most 8 stays within an 8-element backing array.
HELPER = """
        int runtime_count() {
            return 5;
        }

        void write_tail(thread float* values, int count) {
            for (int i = 0; i < count / 4; ++i) {
                values += 4 * i;
                values[3] = 1.0;
            }
        }
"""


def _shader(calls):
    return f"""
    shader CallBoundedPrivatePointerRebase {{
        {HELPER}

        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {{
                float backing[8];
                int n = runtime_count();
                {calls}
            }}
        }}
    }}
    """


@pytest.mark.parametrize(
    "calls",
    [
        "write_tail(backing, 8);",
        "write_tail(backing, clamp(n, 0, 8));",
        "write_tail(backing, min(max(n, 0), 8));",
        "int bounded = clamp(n - 3, 0, 8); write_tail(backing, bounded);",
        "write_tail(backing, 8); write_tail(backing, clamp(n, 1, 7));",
    ],
    ids=["literal", "clamp", "min-max", "clamped-local", "two-calls"],
)
def test_hlsl_private_pointer_rebase_uses_call_bounded_count(calls, tmp_path):
    generated = HLSLCodeGen().generate(crosstl.translator.parse(_shader(calls)))

    assert "void write_tail(inout float values[8], int values_base, int count)" in (
        generated
    )
    assert "values_base += int((4 * i));" in generated
    assert "values[(values_base + 3)] = 1.0;" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


@pytest.mark.parametrize(
    "calls",
    [
        "write_tail(backing, n);",
        "write_tail(backing, 8); write_tail(backing, n);",
        "write_tail(backing, max(n, 0));",
        # The analysis skips the dead call, so not every call is recorded.
        "if (false) { write_tail(backing, n); } write_tail(backing, 8);",
    ],
    ids=["unbounded", "one-call-unbounded", "no-upper-bound", "unrecorded-call"],
)
def test_hlsl_private_pointer_rebase_rejects_count_a_call_leaves_open(calls):
    with pytest.raises(DirectXPrivatePointerParameterError) as excinfo:
        HLSLCodeGen().generate(crosstl.translator.parse(_shader(calls)))

    assert excinfo.value.reason == "pointer-parameter-rebinding"
    assert excinfo.value.function_name == "write_tail"
    assert "statically enumerable iterations" in str(excinfo.value)


def test_hlsl_private_pointer_rebase_rejects_count_past_the_backing():
    # count / 4 reaches 4: the view moves to 0, 4, 12 and 24.
    shader = _shader("write_tail(backing, clamp(n, 0, 16));")

    with pytest.raises(DirectXPrivatePointerParameterError) as excinfo:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert excinfo.value.function_name == "write_tail"
    assert excinfo.value.reason == "view-out-of-bounds"


def test_hlsl_private_pointer_rebase_seed_respects_reassigned_count():
    shader = f"""
    shader ReassignedCount {{
        int runtime_count() {{
            return 5;
        }}

        void write_tail(thread float* values, int count) {{
            count = runtime_count();
            for (int i = 0; i < count / 4; ++i) {{
                values += 4 * i;
                values[3] = 1.0;
            }}
        }}

        compute {{
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            void main() {{
                float backing[8];
                write_tail(backing, 8);
            }}
        }}
    }}
    """

    with pytest.raises(DirectXPrivatePointerParameterError) as excinfo:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "statically enumerable iterations" in str(excinfo.value)


def _interval(table):
    return lambda expression: table.get(expression)


def _no_call(expression):
    return None


@pytest.mark.parametrize(
    ("name", "arguments", "table", "expected"),
    [
        ("min", ("a", "b"), {"a": (-3, 9), "b": (2, 4)}, (-3, 4)),
        ("max", ("a", "b"), {"a": (-3, 9), "b": (2, 4)}, (2, 9)),
        ("clamp", ("x", "lo", "hi"), {"lo": (0, 0), "hi": (8, 8)}, (0, 8)),
        (
            "clamp",
            ("x", "lo", "hi"),
            {"x": (3, 20), "lo": (0, 1), "hi": (8, 8)},
            (3, 8),
        ),
        # Undefined in MSL and HLSL when lo > hi; min(max(x, lo), hi) is hi.
        ("clamp", ("x", "lo", "hi"), {"lo": (5, 5), "hi": (2, 2)}, (2, 2)),
        ("min", ("x", "b"), {"b": (2, 4)}, None),
        ("max", ("x", "b"), {"b": (2, 4)}, None),
        ("clamp", ("x", "lo", "hi"), {"lo": (0, 0)}, None),
        ("abs", ("a",), {"a": (1, 2)}, None),
        ("min", ("a",), {"a": (1, 2)}, None),
    ],
)
def test_builtin_interval_bounds_each_side(name, arguments, table, expected):
    assert (
        intervals.builtin_interval(name, list(arguments), _interval(table), _no_call)
        == expected
    )


def test_builtin_interval_combines_one_sided_bounds():
    calls = {"inner": ("max", ["x", "zero"]), "outer": ("min", ["inner", "eight"])}
    table = {"zero": (0, 0), "eight": (8, 8)}

    def interval_of(expression):
        if expression in calls:
            return intervals.builtin_interval(
                *calls[expression], interval_of, calls.get
            )
        return table.get(expression)

    assert interval_of("inner") is None
    assert intervals.builtin_interval(*calls["outer"], interval_of, calls.get) == (0, 8)


def test_call_argument_intervals_drop_values_outside_the_parameter_type():
    slots = ((1, "count", intervals.SIGNED_PARAMETER_RANGES["int16_t"]),)
    table = {"small": (0, 8), "large": (0, 1 << 20)}

    assert intervals.call_argument_intervals(
        slots, ["view", "small"], _interval(table)
    ) == {"count": (0, 8)}
    assert intervals.call_argument_intervals(
        slots, ["view", "large"], _interval(table)
    ) == {"count": None}
    assert intervals.call_argument_intervals(slots, ["view"], _interval(table)) == {
        "count": None
    }


def test_joined_parameter_seeds_need_every_call_site():
    first, second, third = object(), object(), object()
    view_calls = {
        "main": [
            {"callee": "tail", "node": first, "argumentIntervals": {"n": (0, 4)}},
            {"callee": "tail", "node": second, "argumentIntervals": {"n": (2, 8)}},
            {"callee": "head", "node": third, "argumentIntervals": {"n": None}},
        ]
    }
    sites = {"tail": {id(first), id(second)}, "head": {id(third)}}

    assert intervals.joined_parameter_seeds(view_calls, sites) == {
        "tail": {"n": (0, 8)}
    }
    # A call the analysis did not record leaves the helper unseeded.
    sites["tail"].add(id(object()))
    assert intervals.joined_parameter_seeds(view_calls, sites) == {}


def test_analyze_with_call_seeds_falls_back_to_the_unseeded_pass():
    passes = []

    class State:
        recorded = []

        @classmethod
        def snapshot(cls):
            return list(cls.recorded)

        @classmethod
        def restore(cls, snapshot):
            cls.recorded = list(snapshot)

    def run_pass(seeds):
        passes.append(seeds)
        State.recorded.append(dict(seeds))
        if seeds:
            raise ValueError("seeded pass fails")
        return "unseeded"

    result = intervals.analyze_with_call_seeds(
        run_pass, lambda _: {"tail": {"n": (0, 8)}}, State, 4, (ValueError,)
    )

    assert result == "unseeded"
    assert passes == [{}, {"tail": {"n": (0, 8)}}, {}]
    assert State.recorded == [{}]
