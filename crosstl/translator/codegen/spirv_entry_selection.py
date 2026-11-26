"""One entry point of a CrossGL program as its own SPIR-V module.

``crosstl.translate(path, "vulkan", entry_point="name")`` and the project pipeline's entry-scoped artifacts ask for a
module that holds only the named entry. :meth:`SpirvEntrySelectionMixin.entry_scoped_ast` returns a copy of the
program with that entry's stage alone, without the other stage-qualified functions and without the declarations
that belong to another kernel (the ``entry_argument`` annotation that CUDA and OpenCL C input give the kernel
parameters they hoist). The generator then writes the copy as it writes any program, so the module has one
``OpEntryPoint`` and keeps the bindings the source gave its resources.
"""

from __future__ import annotations

from copy import deepcopy

from ..ast import StageMap
from ..stage_utils import STAGE_QUALIFIER_NAMES
from .metal_argument_tables import ENTRY_ARGUMENT_ANNOTATION
from .stage_utils import function_stage_name

__all__ = ["SpirvEntrySelectionMixin", "VulkanEntryPointSelectionError"]


class VulkanEntryPointSelectionError(ValueError):
    """Raised when a requested standalone SPIR-V entry cannot be selected."""

    project_diagnostic_code = "project.translate.vulkan-entry-point-unavailable"
    missing_capabilities = ("artifact.entry-point-selection",)

    def __init__(
        self,
        message,
        *,
        entry_point=None,
        available_entry_points=None,
        reason=None,
    ):
        super().__init__(message)
        self.entry_point = entry_point
        self.available_entry_points = tuple(available_entry_points or ())
        self.reason = reason


def _is_stage_function(function):
    return function_stage_name(function) in STAGE_QUALIFIER_NAMES


def _entry_names(ast):
    """``(kind, position, name)`` for each entry: a stage of ``ast.stages`` or a stage-qualified function."""
    entries = []
    for position, (_stage_type, stage) in enumerate(
        (getattr(ast, "stages", None) or {}).items()
    ):
        name = getattr(getattr(stage, "entry_point", None), "name", None)
        if isinstance(name, str):
            entries.append(("stage", position, name))
    for position, function in enumerate(getattr(ast, "functions", None) or []):
        if _is_stage_function(function) and isinstance(
            getattr(function, "name", None), str
        ):
            entries.append(("function", position, function.name))
    return entries


def _other_kernel_declaration(node, selected, entry_names):
    get = getattr(node, "get_annotation", None)
    kernel = get(ENTRY_ARGUMENT_ANNOTATION) if callable(get) else None
    return isinstance(kernel, str) and kernel != selected and kernel in entry_names


class SpirvEntrySelectionMixin:
    """``entry_scoped_ast`` and ``generate_entry`` for the SPIR-V generator."""

    def generate_entry(self, ast, entry_point):
        """Generate a SPIR-V module whose only entry point is ``entry_point``."""
        return self.generate(self.entry_scoped_ast(ast, entry_point))

    def entry_scoped_ast(self, ast, entry_point):
        """A copy of ``ast`` that keeps the entry named ``entry_point`` and drops the program's other entries."""
        if not isinstance(entry_point, str) or not entry_point.strip():
            raise VulkanEntryPointSelectionError(
                "SPIR-V entry-point selection requires a non-empty source entry name",
                entry_point=entry_point,
                reason="invalid-name",
            )
        entry_point = entry_point.strip()
        entries = _entry_names(ast)
        available = sorted({name for _kind, _position, name in entries})
        matches = [entry for entry in entries if entry[2] == entry_point]
        if not matches:
            raise VulkanEntryPointSelectionError(
                f"SPIR-V source entry point '{entry_point}' was not found",
                entry_point=entry_point,
                available_entry_points=available,
                reason="not-found",
            )
        if len(matches) != 1:
            raise VulkanEntryPointSelectionError(
                f"SPIR-V source entry point '{entry_point}' is ambiguous",
                entry_point=entry_point,
                available_entry_points=available,
                reason="ambiguous",
            )
        kind, selected_position, _name = matches[0]
        scoped = deepcopy(ast)
        stages = StageMap()
        for position, (stage_type, stage) in enumerate(
            (getattr(scoped, "stages", None) or {}).items()
        ):
            if kind == "stage" and position == selected_position:
                stages.append(stage_type, stage)
        scoped.stages = stages
        scoped.functions = [
            function
            for position, function in enumerate(
                getattr(scoped, "functions", None) or []
            )
            if not _is_stage_function(function)
            or (kind == "function" and position == selected_position)
        ]
        entry_names = set(available)
        for field in ("global_variables", "cbuffers"):
            container = getattr(scoped, field, None)
            if isinstance(container, list):
                container[:] = [
                    node
                    for node in container
                    if not _other_kernel_declaration(node, entry_point, entry_names)
                ]
        return scoped
