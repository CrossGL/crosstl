"""Typed refusal for HLSL ``select`` calls whose overload cannot be resolved from the source."""


class HLSLSelectOverloadError(ValueError):
    """Reject select calls whose source overload ownership is unresolved."""

    project_diagnostic_code = "project.translate.hlsl-select-overload-unresolved"
    missing_capabilities = ("hlsl.select-overload-resolution",)

    def __init__(self):
        super().__init__(
            "Cannot resolve a three-argument HLSL select call in the presence "
            "of a source-defined select overload"
        )
        self.reason = "select-overload-ownership-unresolved"
