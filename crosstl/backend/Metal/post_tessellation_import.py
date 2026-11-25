"""Metal post-tessellation vertex functions imported as CrossGL tessellation evaluation stages.

A Metal post-tessellation vertex function is a ``vertex`` function marked ``[[patch(triangle, N)]]`` or
``[[patch(quad, N)]]``. It reads its patch's control points through a ``patch_control_point<T>`` member of its
``[[stage_in]]`` structure, the per-patch values through the other members of that structure, the position in the
patch through ``[[position_in_patch]]`` and the patch's index through ``[[patch_id]]`` (Metal Shading Language
Specification 4.1, section 5.1.5, "Post-Tessellation Vertex Functions", and section 5.2.3.2). The importer printed
such a function as a CrossGL vertex stage that kept ``patch_control_point``, ``@patch`` and ``@position_in_patch``
verbatim, which no target reads, and refused a subscript of the control points.

CrossGL spells the stage as the HLSL importer writes a domain shader, which every target with tessellation lowers::

    tessellation_evaluation {
        @domain("tri")
        vec4 probe(Patch patch, vec3 coord @gl_TessCoord, OutputPatch<ControlPoint, 3> patch_points) { ... }
    }

:meth:`MetalPostTessellationImport.plan_metal_post_tessellation_functions` rewrites each such function before the
program is printed:

* ``[[patch(triangle, N)]]`` becomes ``@domain("tri")`` and ``[[patch(quad, N)]]`` ``@domain("quad")``, written
  before the function: an attribute after the parameter list is the return value's semantic;
* the ``patch_control_point<T>`` member leaves the stage-input structure, and the function takes the control points
  as an ``OutputPatch<T, N>`` parameter instead. ``patch.points[i]`` reads that parameter and ``patch.points.size()``
  is ``N``. The rest of the stage-input structure stays the patch-constant parameter, its members keeping the
  ``@location`` of their ``[[attribute(n)]]``; a structure with no other member is no longer a parameter or a
  declaration;
* ``[[position_in_patch]]`` becomes ``gl_TessCoord`` and ``[[patch_id]]`` ``gl_PrimitiveID``.

A function whose patch size is not an integer literal while it reads the control points, a structure with more than
one control-point member, and a stage-input structure that another declaration also uses are refused through
:meth:`refuse_import`, as is any other use of the control-point member.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .MetalAst import AttributeNode, MemberAccessNode, MethodCallNode, VariableNode

__all__ = ["MetalPostTessellationImport"]

_PATCH_TYPES = {"triangle": "tri", "quad": "quad"}
_CONTROL_POINTS = re.compile(r"(?:metal\s*::\s*)?patch_control_point\s*<\s*(.+?)\s*>")


@dataclass
class _PostTessellationFunction:
    """One ``[[patch]]`` vertex function and the parameter that replaces its control-point member."""

    domain: str = ""
    points: VariableNode | None = None
    point_type: str | None = None
    count: int | None = None
    replaced: set = field(default_factory=set)


def _attribute(node, name):
    return next(
        (
            attribute
            for attribute in getattr(node, "attributes", None) or []
            if isinstance(attribute, AttributeNode) and attribute.name == name
        ),
        None,
    )


def _children(node):
    """``(owner, key, child)`` for every child slot of a Metal AST node, lists included."""
    if isinstance(node, list):
        for index, child in enumerate(node):
            yield node, index, child
        return
    for key, child in list(vars(node).items()):
        if key in {"source_location", "parent"}:
            continue
        yield node, key, child


def _replace(owner, key, value):
    if isinstance(owner, list):
        owner[key] = value
    else:
        setattr(owner, key, value)


def _uses_type(node, type_name, skip, seen):
    """Whether a declaration under ``node`` other than ``skip`` has the type ``type_name``."""
    if node is None or isinstance(node, (str, int, float, bool)) or id(node) in seen:
        return False
    seen.add(id(node))
    if node is not skip and isinstance(node, VariableNode):
        if re.search(rf"\b{re.escape(type_name)}\b", str(getattr(node, "vtype", ""))):
            return True
    if not isinstance(node, list) and not hasattr(node, "__dict__"):
        return False
    return any(
        _uses_type(child, type_name, skip, seen)
        for _owner, _key, child in _children(node)
    )


class MetalPostTessellationImport:
    """Rewrites ``[[patch]]`` vertex functions into the CrossGL tessellation evaluation form."""

    def plan_metal_post_tessellation_functions(self, ast):
        """Rewrite every ``[[patch]]`` vertex function of ``ast`` in place (module docstring)."""
        self.metal_post_tessellation_functions = {}
        structs = {
            struct.name: struct
            for struct in getattr(ast, "structs", None) or []
            if getattr(struct, "name", None)
        }
        for function in getattr(ast, "functions", None) or []:
            patch = _attribute(function, "patch")
            if getattr(function, "qualifier", None) != "vertex" or patch is None:
                continue
            plan = self.plan_metal_post_tessellation_function(
                ast, function, patch, structs
            )
            if plan is not None:
                self.metal_post_tessellation_functions[id(function)] = plan

    def plan_metal_post_tessellation_function(self, ast, function, patch, structs):
        arguments = [str(argument).strip() for argument in patch.args]
        domain = _PATCH_TYPES.get(arguments[0] if arguments else "")
        if domain is None:
            return self.refuse_import(
                f"Metal post-tessellation function '{function.name}' has patch type "
                f"'{arguments[0] if arguments else ''}', and Metal patches are triangle or quad",
                None,
            )
        plan = _PostTessellationFunction(domain=domain)
        if len(arguments) > 1:
            plan.count = int(arguments[1]) if arguments[1].isdigit() else None
        function.attributes = [
            attribute for attribute in function.attributes if attribute is not patch
        ]
        for parameter in function.params:
            for attribute in getattr(parameter, "attributes", None) or []:
                if attribute.name == "position_in_patch":
                    attribute.name = "gl_TessCoord"
                elif attribute.name == "patch_id":
                    attribute.name = "gl_PrimitiveID"
        stage_in = next(
            (p for p in function.params if _attribute(p, "stage_in") is not None), None
        )
        struct = structs.get(str(getattr(stage_in, "vtype", "")).strip())
        members = [
            member
            for member in getattr(struct, "members", None) or []
            if _CONTROL_POINTS.fullmatch(str(getattr(member, "vtype", "")).strip())
        ]
        if not members:
            return plan
        if len(members) > 1:
            return self.refuse_import(
                f"Metal post-tessellation structure '{struct.name}' has more than one patch_control_point member",
                None,
            )
        if _uses_type(ast, struct.name, stage_in, set()):
            return self.refuse_import(
                f"Metal post-tessellation structure '{struct.name}' is also used outside the [[stage_in]] "
                f"parameter of '{function.name}', and CrossGL passes its control points separately",
                None,
            )
        member = members[0]
        plan.point_type = _CONTROL_POINTS.fullmatch(str(member.vtype).strip()).group(1)
        struct.members = [m for m in struct.members if m is not member]
        name = self.reserve_generated_identifier(f"{stage_in.name}_{member.name}")
        plan.points = VariableNode(
            plan.point_type, name, array_sizes=[str(plan.count or 1)]
        )
        if not self.rewrite_metal_patch_points(function, stage_in, member, plan):
            return None
        # The patch size is kept in the OutputPatch type even when the body never reads a control point.
        kept = [stage_in] if struct.members else []
        if not struct.members:
            ast.structs = [
                declared for declared in ast.structs if declared is not struct
            ]
        if plan.count is not None:
            kept.append(plan.points)
        index = function.params.index(stage_in)
        function.params[index : index + 1] = kept
        return plan

    def rewrite_metal_patch_points(self, function, stage_in, member, plan):
        """Point every ``patch.points`` of the body at the control-point parameter; False once refused."""
        refused = []

        def is_points(node):
            return (
                isinstance(node, MemberAccessNode)
                and node.member == member.name
                and isinstance(node.object, VariableNode)
                and node.object.name == stage_in.name
            )

        def visit(node, seen):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if id(node) in seen or not (
                isinstance(node, list) or hasattr(node, "__dict__")
            ):
                return
            seen.add(id(node))
            for owner, key, child in _children(node):
                if isinstance(child, MethodCallNode) and is_points(child.object):
                    if child.method != "size" or child.args or plan.count is None:
                        refused.append(f"{member.name}.{child.method}()")
                        continue
                    _replace(owner, key, str(plan.count))
                elif is_points(child):
                    if plan.count is None:
                        refused.append(f"{member.name} without a literal patch size")
                        continue
                    _replace(owner, key, VariableNode("", plan.points.name))
                else:
                    visit(child, seen)

        visit(function.body, set())
        if refused:
            self.refuse_import(
                f"Metal post-tessellation function '{function.name}' uses its control points as "
                f"{refused[0]}, which has no CrossGL OutputPatch form",
                None,
            )
            return False
        return True

    def is_metal_post_tessellation_function(self, function):
        return id(function) in getattr(self, "metal_post_tessellation_functions", {})

    def metal_post_tessellation_function_annotation(self, function, indent):
        """The ``@domain("tri")`` line written before a post-tessellation function, else an empty string."""
        plan = getattr(self, "metal_post_tessellation_functions", {}).get(id(function))
        if plan is None:
            return ""
        return "    " * indent + f'@domain("{plan.domain}")\n'

    def metal_post_tessellation_parameter_declaration(self, var):
        """``OutputPatch<T, N> name`` for the parameter that replaces a control-point member, else None."""
        for plan in getattr(self, "metal_post_tessellation_functions", {}).values():
            if var is plan.points:
                return (
                    f"OutputPatch<{self.map_type(plan.point_type)}, {plan.count}> "
                    f"{self.render_identifier(var.name)}"
                )
        return None
