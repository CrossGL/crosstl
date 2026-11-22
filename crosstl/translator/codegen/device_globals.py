"""Shared checks for mutable program-scope device-memory declarations."""

from ..ast import PointerType, ReferenceType


def is_mutable_device_global(node):
    """Whether ``node`` is a mutable device-qualified value, not a pointer."""
    qualifiers = {
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", []) or []
    }
    return (
        "device" in qualifiers
        and getattr(node, "is_mutable", True)
        and not isinstance(
            getattr(node, "var_type", None), (PointerType, ReferenceType)
        )
    )


def device_global_refusal_reason(node, target):
    """Describe why a target cannot faithfully lower ``node``."""
    name = getattr(node, "name", getattr(node, "variable_name", "<unnamed>"))
    return (
        "device-global declaration: "
        f"'{name}' uses mutable program-scope device memory, "
        f"which {target} cannot represent"
    )
