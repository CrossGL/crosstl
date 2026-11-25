"""Typed diagnostic that the Metal generator raises when Metal has no equivalent target representation."""


class UnsupportedMetalFeatureError(ValueError):
    """Raised when Metal has no equivalent target representation."""

    project_diagnostic_code = "project.translate.unsupported-feature"

    def __init__(
        self,
        feature,
        message,
        *,
        missing_capabilities=(),
        operation=None,
        reason=None,
        source_location=None,
    ):
        super().__init__(message)
        self.feature = feature
        self.missing_capabilities = tuple(missing_capabilities)
        self.operation = operation
        self.reason = reason
        self.source_location = source_location
