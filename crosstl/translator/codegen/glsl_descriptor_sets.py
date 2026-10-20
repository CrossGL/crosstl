"""OpenGL binding points for resources that carry Vulkan descriptor sets or HLSL register spaces.

OpenGL has no descriptor sets: every resource class has one flat array of binding points (uniform-buffer and
shader-storage binding points, texture units, image units; GLSL 4.60 section 4.4.5 and 4.4.6, OpenGL 4.6 section
7.6.2), and implementations expose few of them (``GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS`` is at least 80). So
``layout(set = S, binding = B)`` must become a small ``binding`` that is unique within its class.

The plan is computed once per program, from every declaration of every stage, so the stages of one program agree.
Within each class, resources of set 0 keep their ``binding``, and resources of the other sets keep theirs when it is
still free (the mapping SPIRV-Cross and glslang's OpenGL mode use, which hosts already assume); the ones that find
it taken move, in (set, binding) order, to the lowest free binding points.

Declarations that share one (set, binding) in one class alias the same descriptor and keep sharing a binding.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Tuple

__all__ = ["GLSLDescriptorSetBindingsMixin", "plan_descriptor_set_bindings"]

PlanEntry = Tuple[str, int, int, int, str]  # namespace, set, binding, count, name


def plan_descriptor_set_bindings(
    entries: Iterable[PlanEntry],
) -> dict[tuple[str, int, int], int]:
    """Map ``(name, set, binding)`` of every entry to its OpenGL binding point."""
    by_namespace: dict[str, dict[tuple[int, int], tuple[int, list[str]]]] = {}
    for namespace, descriptor_set, binding, count, name in entries:
        groups = by_namespace.setdefault(namespace, {})
        width, names = groups.get((descriptor_set, binding), (1, []))
        groups[(descriptor_set, binding)] = (max(width, count or 1), names + [name])
    plan: dict[tuple[str, int, int], int] = {}
    for namespace in sorted(by_namespace):
        groups = by_namespace[namespace]
        assigned = _assign_namespace(groups)
        for key, (_, names) in groups.items():
            for name in names:
                plan[(name, key[0], key[1])] = assigned[key]
    return plan


def _assign_namespace(groups):
    """Set-0 resources keep their binding; the others claim theirs if free, else the lowest free range."""
    order = sorted(groups)
    claimed: list[tuple[int, int]] = []
    assigned = {}
    for key in order:
        if key[0] == 0:
            claimed.append((key[1], key[1] + groups[key][0] - 1))
            assigned[key] = key[1]
    pending = []
    for key in order:
        if key[0] == 0:
            continue
        start, width = key[1], groups[key][0]
        if _free(claimed, start, width):
            claimed.append((start, start + width - 1))
            assigned[key] = start
        else:
            pending.append(key)
    for key in pending:
        width = groups[key][0]
        start = 0
        while not _free(claimed, start, width):
            start += 1
        claimed.append((start, start + width - 1))
        assigned[key] = start
    return assigned


def _free(claimed, start, width):
    end = start + width - 1
    return all(end < used_start or used_end < start for used_start, used_end in claimed)


class GLSLDescriptorSetBindingsMixin:
    """Computes the per-program plan and answers the binding of one declaration."""

    opengl_descriptor_set_plan: dict[tuple[str, int, int], int] | None = None
    opengl_descriptor_set_ast = None

    @contextmanager
    def opengl_descriptor_set_plan_for(self, ast):
        """Plan bindings over ``ast`` for the duration of the block, unless an outer program already chose one.

        An entry-scoped rendering keeps the plan of the whole program it was cut from.
        """
        if self.opengl_descriptor_set_ast is not None:
            yield
            return
        self.opengl_descriptor_set_ast = ast
        try:
            yield
        finally:
            self.opengl_descriptor_set_ast = None
            self.opengl_descriptor_set_plan = None

    def prepare_opengl_descriptor_set_plan(self):
        """Compute the plan once the rendering knows its structs (uniform struct blocks are classified by them)."""
        ast = self.opengl_descriptor_set_ast
        self.opengl_descriptor_set_plan = None
        if ast is None:
            return
        from .stage_utils import collect_stage_local_structs

        own_structs = self.structs_by_name
        self.structs_by_name = dict(own_structs)
        for node in list(
            getattr(ast, "structs", []) or []
        ) + collect_stage_local_structs(ast):
            name = getattr(node, "name", None)
            if (
                name
                and name not in self.structs_by_name
                and self.is_struct_declaration_node(node)
            ):
                self.structs_by_name[name] = node
        try:
            entries = self.opengl_descriptor_set_plan_entries(ast)
        finally:
            self.structs_by_name = own_structs
        self.opengl_descriptor_set_plan = plan_descriptor_set_bindings(entries)

    def opengl_descriptor_set_plan_entries(self, ast) -> list[PlanEntry]:
        from .stage_utils import (
            collect_stage_local_cbuffers,
            collect_stage_local_variables,
        )

        entries: list[PlanEntry] = []
        cbuffers = list(
            getattr(ast, "cbuffers", []) or []
        ) + collect_stage_local_cbuffers(ast)
        for node in cbuffers:
            entry = self.opengl_descriptor_set_entry(node, "uniform buffer binding", 1)
            if entry is not None:
                entries.append(entry)
        variables = list(getattr(ast, "global_variables", []) or [])
        variables += collect_stage_local_variables(
            ast, None, self.is_stage_local_resource_variable
        )
        for node in variables:
            try:
                classified = self.global_resource_binding_namespace(node)
            except (
                Exception
            ):  # noqa: BLE001 - declarations the emitter later refuses take no binding point
                continue
            if classified is None:
                continue
            namespace, count, _ = classified
            entry = self.opengl_descriptor_set_entry(node, namespace, count)
            if entry is not None:
                entries.append(entry)
        return entries

    def opengl_descriptor_set_entry(self, node, namespace, count) -> PlanEntry | None:
        name = self.resource_node_name(node)
        if not name:
            return None
        try:
            descriptor_set = self.resource_descriptor_set_index(node)
            choices = self.explicit_resource_binding_choices(node, flatten_sets=False)
        except ValueError:
            return None
        if not choices:
            return None
        return namespace, descriptor_set, choices[0][2], max(count or 1, 1), name

    def opengl_set_binding(self, node, descriptor_set, binding):
        """The OpenGL binding point of ``binding`` in ``descriptor_set`` for the declaration ``node``."""
        plan = self.opengl_descriptor_set_plan
        if plan is not None:
            planned = plan.get((self.resource_node_name(node), descriptor_set, binding))
            if planned is not None:
                return planned
        return binding

    def resource_descriptor_set_index(self, node):
        """The ``set``/``space`` of ``node`` (an attribute or ``register(x, spaceN)``), 0 when it has none."""
        for attr in getattr(node, "attributes", None) or []:
            if str(getattr(attr, "name", "") or "").lower() not in {"set", "space"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            value = self.binding_index_value(arguments[0]) if arguments else None
            if value is None:
                raise ValueError(
                    f"invalid descriptor set of '{self.resource_node_name(node)}'"
                )
            return value
        return self.register_space_index(node)

    def register_space_index(self, node):
        """``N`` of HLSL ``register(x, spaceN)``, 0 without a space argument."""
        for attr in getattr(node, "attributes", None) or []:
            if str(getattr(attr, "name", "") or "").lower() != "register":
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) < 2:
                continue
            value = self.binding_index_value(arguments[1], ("space",))
            if value is None:
                raise ValueError(
                    self.unsupported_resource_space_message(
                        node, "register", self.attribute_value_to_string(arguments[1])
                    )
                )
            return value
        return 0
