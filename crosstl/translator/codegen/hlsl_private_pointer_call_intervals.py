"""Bound the integer parameters of DirectX private pointer helpers by their calls.

HLSL has no pointers into private arrays, so a helper's ``thread T*``
parameter becomes an array and a by-value base index, and ``p += n`` moves
only the base. The private pointer analysis must prove every view offset
such a move reaches stays inside the backing array. In a helper like::

    float tail(const thread float* x, int n) {
        for (int i = 0; i < n / 4; i++) { x += 4 * i; ... }
    }

that depends on ``n``, which only its callers bound, for instance with
``clamp(count - base, 0, 8)``.

The analysis therefore records, at every call of a helper that moves a
private pointer parameter, the interval of each argument bound to a signed
integer parameter. When every call of such a helper in the program was
recorded, each with a bounded argument, the hull of those intervals seeds
the parameter in the helper's own analysis, and the program is analyzed
again until the seeds stop changing. A call the analysis skips, such as one
in a branch it proves dead, leaves the helper unseeded, and a seeded pass
that fails falls back to the unseeded one: seeds only ever add proofs.

Integer ``min``, ``max`` and ``clamp`` calls are bounded by their arguments,
so a clamped value is bounded even when the value it clamps is not.
"""

# Signed parameter types a seed may describe, with their value ranges.
SIGNED_PARAMETER_RANGES = {
    "int": (-(1 << 31), (1 << 31) - 1),
    "int16_t": (-(1 << 15), (1 << 15) - 1),
    "int64_t": (-(1 << 63), (1 << 63) - 1),
}


def builtin_interval(function_name, arguments, interval_of, call_of):
    """The interval of an integer ``min``, ``max`` or ``clamp`` call, or ``None``.

    ``interval_of`` maps an expression to its ``(low, high)`` interval or
    ``None``; ``call_of`` maps it to ``(function name, arguments)`` when it
    is a call. Each builtin is monotone in every argument, and one bounded
    argument already bounds one side: ``max(x, 0)`` is at least 0 and
    ``min(x, 8)`` at most 8 whatever ``x`` is, so ``min(max(x, 0), 8)`` and
    ``clamp(x, 0, 8)`` lie in ``[0, 8]``.
    """
    low, high = _builtin_bounds(function_name, arguments, interval_of, call_of)
    if low is None or high is None:
        return None
    return low, high


BOUNDED_BUILTINS = {"min": 2, "max": 2, "clamp": 3}


def _bounds(expression, interval_of, call_of):
    interval = interval_of(expression)
    if interval is not None:
        return interval
    call = call_of(expression)
    if call is None:
        return None, None
    return _builtin_bounds(*call, interval_of, call_of)


def _builtin_bounds(function_name, arguments, interval_of, call_of):
    """``(low, high)`` of a builtin call; either bound may be ``None``."""
    if BOUNDED_BUILTINS.get(function_name) != len(arguments):
        return None, None
    bounds = [_bounds(argument, interval_of, call_of) for argument in arguments]
    if function_name == "min":
        return _min_bounds(bounds)
    if function_name == "max":
        return _max_bounds(bounds)
    # clamp(x, lo, hi) is min(max(x, lo), hi).
    return _min_bounds([_max_bounds(bounds[:2]), bounds[2]])


def _min_bounds(bounds):
    """At least the smallest lower bound, at most any known upper bound."""
    lows = [low for low, _ in bounds]
    highs = [high for _, high in bounds if high is not None]
    return (None if None in lows else min(lows)), (min(highs) if highs else None)


def _max_bounds(bounds):
    """At least any known lower bound, at most the largest upper bound."""
    lows = [low for low, _ in bounds if low is not None]
    highs = [high for _, high in bounds]
    return (max(lows) if lows else None), (None if None in highs else max(highs))


def signed_parameter_slots(function, private_indices, mapped_type_of):
    """``(index, name, range)`` of a function's signed integer scalar parameters.

    ``private_indices`` are the positions of its private pointer parameters,
    which are never scalars; ``mapped_type_of`` maps a parameter to its HLSL
    type name.
    """
    slots = []
    parameters = getattr(function, "parameters", None) or []
    for index, parameter in enumerate(parameters):
        if index in private_indices:
            continue
        value_range = SIGNED_PARAMETER_RANGES.get(mapped_type_of(parameter))
        name = getattr(parameter, "name", None)
        if value_range is not None and name:
            slots.append((index, name, value_range))
    return tuple(slots)


def call_argument_intervals(slots, arguments, interval_of):
    """``{parameter name: interval or None}`` for one call's signed arguments.

    An interval outside the parameter's range is dropped: the conversion
    could change the value.
    """
    intervals = {}
    for index, name, (minimum, maximum) in slots:
        argument = arguments[index] if index < len(arguments) else None
        interval = interval_of(argument) if argument is not None else None
        if (
            interval is not None
            and not minimum <= interval[0] <= interval[1] <= maximum
        ):
            interval = None
        intervals[name] = interval
    return intervals


def joined_parameter_seeds(view_calls, call_sites):
    """Seeds for helpers whose every call site was recorded with bounded intervals.

    ``view_calls`` maps each caller to its recorded calls; ``call_sites``
    maps each helper that moves a private pointer view to the ids of every
    call node that names it anywhere in the program. Only those helpers are
    seeded: elsewhere an unknown index already falls back to the whole
    backing array.
    """
    recorded = {}
    for caller in sorted(view_calls):
        for call in view_calls[caller]:
            entry = recorded.setdefault(call["callee"], ({}, set()))
            entry[1].add(id(call["node"]))
            for name, interval in (call.get("argumentIntervals") or {}).items():
                entry[0].setdefault(name, []).append(interval)
    seeds = {}
    for callee in sorted(recorded):
        intervals, nodes = recorded[callee]
        if callee not in call_sites or nodes != call_sites[callee]:
            continue
        for name in sorted(intervals):
            values = intervals[name]
            if any(value is None for value in values):
                continue
            seeds.setdefault(callee, {})[name] = (
                min(value[0] for value in values),
                max(value[1] for value in values),
            )
    return seeds


def analyze_with_call_seeds(run_pass, seeds_from, state, iterations, errors):
    """Run the private pointer analysis to a fixed point of call-site seeds.

    ``run_pass(seeds)`` analyzes the program with ``seeds`` and returns its
    result; ``seeds_from(result)`` derives the next seeds. ``state`` gives
    ``snapshot()`` and ``restore(snapshot)`` for what a pass records, so a
    pass whose seeds change, or that fails, leaves no trace. Without a fixed
    point within ``iterations`` passes, or when a seeded pass raises one of
    ``errors``, the result of the unseeded analysis is returned.
    """
    initial = state.snapshot()
    seeds = {}
    for _ in range(iterations):
        try:
            result = run_pass(seeds)
        except errors:
            if not seeds:
                raise
            break
        next_seeds = seeds_from(result)
        if next_seeds == seeds:
            return result
        state.restore(initial)
        seeds = next_seeds
    state.restore(initial)
    return run_pass({})
