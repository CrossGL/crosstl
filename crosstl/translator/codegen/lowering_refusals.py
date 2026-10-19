"""Collected refusals for constructs an emitter cannot lower to its target.

An emitter that meets a construct it cannot express must not write a comment and carry on with a neutral value, an
omitted store or a dropped call: the comment is invisible to every consumer of the output, so the artifact compiles
and runs with different results. The emitter records the reason here instead and keeps going, so that one run
reports every construct that needs work, and :meth:`LoweringRefusals.raise_pending` stops the translation once the
output is complete. No output is returned while a refusal is pending.
"""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["LoweringRefused", "LoweringRefusals"]


class LoweringRefused(TranslationDiagnostic):
    """One or more constructs in a program have no faithful lowering to the target.

    Subclasses set ``target`` (the backend name), ``label`` (the target's name in messages) and ``code``.
    ``reasons`` lists every refusal in emission order, one entry per distinct site.
    """

    target_name = "target"
    label = "The target"
    code = f"{CODE_PREFIX}lowering-unsupported"

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

    def run(self, generate, *args, **kwargs):
        """Call ``generate`` and raise the refusals it recorded; nested calls report through the outermost one.

        A typed error (a :class:`TranslationDiagnostic` or an error carrying ``project_diagnostic_code``) propagates
        unchanged. Any other exception may be the consequence of a placeholder left by an earlier refusal, so the
        refusals are reported with the exception as their cause and its message as their last reason.
        """
        outermost = self.depth == 0
        if outermost:
            self.reset()
        self.depth += 1
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
            self.depth -= 1
        if outermost:
            self.raise_pending()
        return result


def _is_typed(error):
    return isinstance(error, TranslationDiagnostic) or bool(
        getattr(error, "project_diagnostic_code", None)
    )
