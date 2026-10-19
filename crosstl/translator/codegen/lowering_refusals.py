"""Collected refusals for constructs an emitter cannot lower to its target.

An emitter that meets a construct it cannot express must not write a comment and carry on with a neutral value, an
omitted store or a dropped call: the comment is invisible to every consumer of the output, so the artifact compiles
and runs with different results. The emitter records the reason here instead and keeps going, so that one run
reports every construct that needs work, and :meth:`LoweringRefusals.raise_pending` stops the translation once the
output is complete. No output is returned while a refusal is pending.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = [
    "LoweringRefused",
    "LoweringRefusalMixin",
    "LoweringRefusals",
    "active_refusals",
    "refuse_unsupported",
]

_ACTIVE = ContextVar("crosstl_lowering_refusals", default=None)
_BY_TARGET = {}

# A loss marker that reached the rendered program: ``/* unsupported <target> <reason> */``. Every emitter site that
# used to write one now records a refusal instead; this pattern lets :meth:`LoweringRefusals.run` refuse any output
# that still carries one, whichever code path wrote it.
_OUTPUT_MARKER = re.compile(r"/\*\s*unsupported\s+(?P<reason>.*?)\s*\*/", re.S)


class LoweringRefused(TranslationDiagnostic):
    """One or more constructs in a program have no faithful lowering to the target.

    Subclasses set ``target`` (the backend name), ``label`` (the target's name in messages) and ``code``.
    ``reasons`` lists every refusal in emission order, one entry per distinct site.
    """

    target_name = "target"
    label = "The target"
    code = f"{CODE_PREFIX}lowering-unsupported"
    # Backend names the shared helpers use for this target besides ``label``.
    aliases = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in (cls.label, *cls.aliases):
            _BY_TARGET.setdefault(name, cls)

    def __init__(self, reasons):
        self.reasons = tuple(reasons)
        self.missing_capabilities = (f"{self.target_name}.lowering",)
        count = len(self.reasons)
        noun = "construct" if count == 1 else "constructs"
        listed = "; ".join(self.reasons)
        super().__init__(
            f"{self.label} cannot lower {count} {noun}: {listed}",
            target=self.target_name,
        )


class LoweringRefusals:
    """The refusal reasons recorded while one program is generated."""

    def __init__(self, error_type):
        self.error_type = error_type
        self.reasons = []
        self.depth = 0

    def reset(self):
        self.reasons = []

    def record(self, reason):
        """Record ``reason`` once; a construct generated twice is still one refusal."""
        text = " ".join(str(reason).split())
        if text and text not in self.reasons:
            self.reasons.append(text)

    def raise_pending(self, cause=None):
        """Raise the recorded refusals; a later failure ``cause`` is listed last so that neither is hidden."""
        if not self.reasons:
            return
        reasons = list(self.reasons)
        if cause is not None:
            reasons.append(f"generation then failed: {cause}")
        error = self.error_type(reasons)
        self.reasons = []
        if cause is not None:
            raise error from cause
        raise error

    @contextmanager
    def active(self):
        """Make this sink the one :func:`refuse_unsupported` records into, for code run outside :meth:`run`."""
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def record_output_markers(self, output):
        """Refuse every loss marker left in a rendered program; the program is never returned with one."""
        if isinstance(output, str):
            for match in _OUTPUT_MARKER.finditer(output):
                self.record(match.group("reason"))

    def run(self, generate, *args, **kwargs):
        """Call ``generate`` and raise the refusals it recorded; nested calls report through the outermost one.

        A typed error (a :class:`TranslationDiagnostic` or an error carrying ``project_diagnostic_code``) propagates
        unchanged. Any other exception may be the consequence of a placeholder left by an earlier refusal, so the
        refusals are reported with the exception as their cause and its message as their last reason. The rendered
        output is checked for loss markers before it is returned.
        """
        outermost = self.depth == 0
        if outermost:
            self.reset()
        self.depth += 1
        token = _ACTIVE.set(self)
        try:
            result = generate(*args, **kwargs)
        except Exception as error:
            if outermost and _is_typed(error):
                # A typed refusal already stops the translation with its own code.
                self.reset()
                raise
            if outermost:
                self.raise_pending(cause=error)
            raise
        finally:
            _ACTIVE.reset(token)
            self.depth -= 1
        if outermost:
            self.record_output_markers(result)
            self.raise_pending()
        return result


def active_refusals():
    """The sink of the generation in progress, or ``None`` outside one."""
    return _ACTIVE.get()


def refuse_unsupported(target, category, detail, placeholder):
    """Refuse a construct found by a helper shared between emitters and return ``placeholder`` in its place.

    The reason ``<category>: <detail>`` goes to the sink of the generation in progress, which raises the emitter's
    typed refusal once the program is rendered; the placeholder only lets generation reach the other refusals. A
    call outside any generation has no output to protect and is refused at once.
    """
    reason = f"{category}: {detail}"
    sink = _ACTIVE.get()
    if sink is None:
        raise _BY_TARGET.get(target, LoweringRefused)([reason])
    sink.record(reason)
    return placeholder


class LoweringRefusalMixin:
    """``refuse_lowering`` for an emitter whose generation runs through ``self.refusals.run``.

    Subclasses set ``LOWERING_REFUSAL_TYPE`` to their :class:`LoweringRefused` subclass. The sink is created on first
    use, so emitters built without ``__init__`` (as some tests do) still refuse.
    """

    LOWERING_REFUSAL_TYPE = LoweringRefused

    @property
    def refusals(self):
        sink = self.__dict__.get("_lowering_refusals")
        if sink is None:
            sink = LoweringRefusals(self.LOWERING_REFUSAL_TYPE)
            self.__dict__["_lowering_refusals"] = sink
        return sink

    def refuse_lowering(self, reason, placeholder=""):
        """Record a construct with no faithful lowering and return ``placeholder`` in its place.

        The placeholder only lets generation reach the other refusals in the program; no program is returned while
        a refusal is pending.
        """
        self.refusals.record(reason)
        return placeholder


def _is_typed(error):
    return isinstance(error, TranslationDiagnostic) or bool(
        getattr(error, "project_diagnostic_code", None)
    )
