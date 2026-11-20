"""The marker a family method returns when none of its cases applied.

Long methods that try a run of ``if ...: return ...`` cases are split into family methods, one per run. A family
method returns ``UNHANDLED`` when it fell through every case, and its caller then tries the next family, exactly as
the original method went on to its next case. ``None`` cannot serve as the marker, because a case may return it.
"""


class _Unhandled:
    __slots__ = ()

    def __repr__(self):
        return "UNHANDLED"

    def __reduce__(self):
        return "UNHANDLED"


UNHANDLED = _Unhandled()
