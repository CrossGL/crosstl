"""Value, type, state and error classes of the SPIR-V generator."""

import re
from dataclasses import dataclass
from typing import Optional


class SpirvType:
    """Represents a SPIR-V type with storage class information."""

    def __init__(self, base_type: str, storage_class: Optional[str] = None):
        """Store the base type name and optional storage class."""
        self.base_type = base_type
        self.storage_class = storage_class

    def __str__(self) -> str:
        """Return a readable type label for debug output."""
        if self.storage_class:
            return f"{self.base_type} ({self.storage_class})"
        return self.base_type


class SpirvId:
    """Represents a SPIR-V ID with its associated type."""

    def __init__(
        self, id_value: int, spirv_type: SpirvType, name: Optional[str] = None
    ):
        """Store the numeric result id, type metadata, and optional name."""
        self.id = id_value
        self.type = spirv_type
        self.name = name

    def __str__(self) -> str:
        """Return a readable SPIR-V id label for debug output."""
        if self.name:
            return f"%{self.id} ({self.name}: {self.type})"
        return f"%{self.id} ({self.type})"


@dataclass(frozen=True)
class PointerOffsetState:
    """Mutable offset cell and immutable root for one logical pointer value."""

    root_pointer: SpirvId
    offset_variable: Optional[SpirvId]
    domain: str
    reset_marker: Optional[str] = None


@dataclass(frozen=True)
class InlineReturnContext:
    """Return flag and optional result storage for one inline call."""

    return_type: SpirvId
    returned_pointer: SpirvId
    result_pointer: Optional[SpirvId]
    merge_stack_depth: int


class UnsupportedSPIRVFeatureError(ValueError):
    """Raised when SPIR-V codegen sees a feature it cannot safely lower."""

    project_diagnostic_code = "project.translate.unsupported-feature"

    def __init__(
        self,
        feature: str,
        message: str,
        *,
        missing_capabilities=(),
        source_location=None,
    ):
        super().__init__(message)
        self.feature = feature
        self.missing_capabilities = tuple(missing_capabilities)
        self.source_location = source_location


CALLEE_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*")
INDIRECT_CALL_DETAIL = (
    "Vulkan SPIR-V has no function pointers, so a call through a function table "
    "or other callee expression cannot be lowered"
)


class SPIRVBuiltinLoweringError(UnsupportedSPIRVFeatureError):
    """Raised when a call has no SPIR-V lowering that preserves its meaning.

    Covers calls to names that are neither declared functions nor supported
    built-ins, and built-in calls whose operand count or types have no defined
    lowering. Emitting a default value instead would be silently wrong output.
    """

    project_diagnostic_code = "project.translate.vulkan-builtin-unsupported"

    def __init__(
        self,
        function_name: str,
        detail: str,
        *,
        reason: str = "invalid-operands",
        source_location=None,
    ):
        self.function_name = function_name
        self.detail = detail
        self.reason = reason
        super().__init__(
            f"builtin:{function_name}",
            f"Cannot lower call to '{function_name}' to SPIR-V: {detail}",
            missing_capabilities=(f"spirv.builtin.{function_name}",),
            source_location=source_location,
        )


class SPIRVAtomicFenceLoweringError(UnsupportedSPIRVFeatureError):
    """Raised when Vulkan SPIR-V cannot preserve an atomic fence contract."""

    project_diagnostic_code = "project.translate.vulkan-atomic-fence-unsupported"
    missing_capabilities = ("spirv.atomic-thread-fence-contract-lowering",)

    def __init__(
        self,
        reason: str,
        *,
        memory_flags=None,
        memory_order=None,
        thread_scope=None,
        source_location=None,
    ):
        self.reason = reason
        self.memory_flags = memory_flags
        self.memory_order = memory_order
        self.thread_scope = thread_scope
        contract = (
            f"flags={memory_flags or '<missing>'}, "
            f"order={memory_order or '<missing>'}, "
            f"scope={thread_scope or '<missing>'}"
        )
        super().__init__(
            "atomic-thread-fence-contract",
            "Cannot lower CrossGL atomicThreadFence to Vulkan SPIR-V without "
            f"changing its semantics ({contract}): {reason.replace('-', ' ')}",
            missing_capabilities=self.missing_capabilities,
            source_location=source_location,
        )
