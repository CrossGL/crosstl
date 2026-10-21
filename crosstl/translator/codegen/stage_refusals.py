"""Typed refusals for shader stages, and stage interfaces, a target cannot declare."""

from __future__ import annotations

from ..diagnostics import CODE_PREFIX, TranslationDiagnostic


class StageUnsupportedError(TranslationDiagnostic):
    """Shader stages the target language has no entry point for."""

    def __init__(self, target, label, stages, reason):
        self.stages = tuple(stages)
        self.missing_capabilities = tuple(
            f"{target}.stage.{stage}" for stage in self.stages
        )
        super().__init__(
            f"{label} target does not support shader stage(s): {', '.join(self.stages)}: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-stage-unsupported",
        )


class RayStageInterfaceUnsupportedError(TranslationDiagnostic):
    """A ray tracing entry whose payload, hit attributes or callable data the emitter cannot pass as a parameter."""

    def __init__(self, target, stage, role, detail):
        self.missing_capabilities = (
            f"{target}.ray-{role.replace('_', '-')}-parameter",
        )
        super().__init__(
            f"DirectX {stage} stage requires a {role} parameter: {detail}",
            target=target,
            code=f"{CODE_PREFIX}{target}-ray-stage-interface-unsupported",
        )
