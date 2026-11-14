"""Constrained helper calls reuse the call-site type context between appends."""

import re

from crosstl.backend.Metal.preprocessor import MetalPreprocessor

CONSTRAINT = (
    "template <typename T, metal::enable_if_t<metal::is_integral_v<T>, bool> = true>"
)

SOURCE = f"""{CONSTRAINT}
T inner(T value) {{ return value + T(1); }}
{CONSTRAINT}
T outer(T value) {{ T widened = value; return inner(widened); }}
kernel void run(device int* ints [[buffer(0)]],
                device uint* uints [[buffer(1)]],
                device short* shorts [[buffer(2)]]) {{
    int i = ints[0];
    uint u = uints[0];
    short s = shorts[0];
    ints[1] = outer(i);
    uints[1] = outer(u);
    shorts[1] = outer(s);
}}
"""


def _materialize_counting_contexts(monkeypatch):
    builds = []
    original = MetalPreprocessor._reachability_type_context

    def counting(self, code, excluded_spans, structs):
        builds.append(len(code))
        return original(self, code, excluded_spans, structs)

    monkeypatch.setattr(MetalPreprocessor, "_reachability_type_context", counting)
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            SOURCE
        )
    )
    return output, builds


def test_nested_calls_resolve_locals_of_appended_bodies(monkeypatch):
    output, _builds = _materialize_counting_contexts(monkeypatch)

    for scalar in ("int", "uint", "short"):
        assert re.search(rf"\b{scalar} outer_{scalar}\({scalar} value\)", output)
        assert re.search(rf"\b{scalar} inner_{scalar}\({scalar} value\)", output)
        # `widened` is a local of the appended outer body, so its type is only
        # known to a context that covers that body.
        assert f"{scalar} widened = value; return inner_{scalar}(widened);" in output
    assert "ints[1] = outer_int(i);" in output
    assert "uints[1] = outer_uint(u);" in output
    assert "shorts[1] = outer_short(s);" in output


def test_context_is_rebuilt_per_generation_not_per_append(monkeypatch):
    _output, builds = _materialize_counting_contexts(monkeypatch)

    # One context for the source and one once the first nested call inside the
    # three appended outer bodies is reached. The inner bodies call nothing.
    assert len(builds) == 2
    assert builds[0] < builds[1]
