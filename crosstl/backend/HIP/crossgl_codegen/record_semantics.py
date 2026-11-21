"""Operators and record semantics in the HIP to CrossGL converter: constructors, layout, external ABI, storage and unions."""

import re

from crosstl.translator.refusals import HipSourceRefusal

from ...assignment_operands import grouped_operand
from ...thread_hierarchy import global_invocation_axis
from ..HipAst import (
    ArrayAccessNode,
    CastNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    FunctionNode,
    InitializerListNode,
    MemberAccessNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class HipRecordSemanticMixin:
    """Record semantic members of ``HipToCrossGLConverter``."""

    def visit_AssignmentNode(self, node):
        self.validate_record_assignment(node)
        left = self.visit_lvalue_expression(node.left)
        operator = getattr(node, "operator", "=")
        runtime_status = (
            self.format_hip_runtime_status_expression(node.right)
            if operator == "="
            else None
        )
        if runtime_status is not None:
            comments, value = runtime_status
            for comment in comments:
                self.emit(comment)
            self.clear_lvalue_metadata_source(node.left)
            self.emit(f"{left} = {value};")
            return

        right = self.visit(node.right)
        self.clear_lvalue_metadata_source(node.left)
        return f"{left} {operator} {right}"

    def visit_BinaryOpNode(self, node):
        self.validate_record_operator_operands(node.op, (node.left, node.right))
        axis = global_invocation_axis(node)
        if axis is not None:
            return f"gl_GlobalInvocationID.{axis}"
        left = grouped_operand(self, node.left)
        right = grouped_operand(self, node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        self.validate_record_operator_operands(node.op, (node.operand,))
        mutates_operand = node.op in {"++", "--"} or (
            isinstance(node.op, str) and node.op.endswith("_POST")
        )
        if node.op == "&" or mutates_operand:
            operand = self.visit_lvalue_expression(node.operand)
        else:
            operand = grouped_operand(self, node.operand)
        if mutates_operand:
            self.clear_lvalue_metadata_source(node.operand)
        if isinstance(node.op, str) and node.op.endswith("_POST"):
            return f"({operand}{node.op[:-5]})"
        elif hasattr(node, "postfix") and node.postfix:
            return f"({operand}{node.op})"
        else:
            return f"({node.op}{operand})"

    def record_definition_from_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        type_name = self.strip_type_qualifiers(type_name)
        for _ in range(16):
            resolved = self.resolve_concrete_type_alias(type_name)
            type_name = self.strip_elaborated_record_type_keyword(resolved).strip()
            if self.has_array_suffix(type_name):
                type_name = type_name.split("[", 1)[0].strip()
                continue
            base_name, template_args = self.parse_cpp_template(type_name)
            if base_name in {"array", "ptr"} and template_args:
                type_name = template_args[0]
                continue
            base_type, pointer_depth = self.split_pointer_declarators(type_name)
            if pointer_depth:
                type_name = base_type
                continue
            break
        definition = self.record_definitions.get(type_name)
        if definition is not None:
            if self.is_generated_matrix_helper_struct(definition):
                return None
            return definition
        matches = {
            id(node): node
            for node in self.record_definitions.values()
            if node is not None
            and not self.is_generated_matrix_helper_struct(node)
            and getattr(node, "name", None)
            and self.convert_hip_record_name_to_crossgl(node.name) == type_name
        }
        if len(matches) == 1:
            return next(iter(matches.values()))
        return None

    def record_value_definition_from_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        compact = type_name.replace(" ", "")
        if "*" in compact or "&" in compact or compact.startswith("ptr<"):
            return None
        return self.record_definition_from_type(type_name)

    def record_display_name(self, node):
        if node is None:
            return "unknown"
        qualified_name = self.record_qualified_names.get(id(node))
        if qualified_name:
            return qualified_name
        name = getattr(node, "name", None)
        if name:
            return str(name)
        if getattr(node, "is_union", False):
            return "anonymous union"
        if getattr(node, "is_class", False):
            return "anonymous class"
        return "anonymous struct"

    def raise_record_semantic(self, issue, context):
        feature, record = issue
        raise self.RECORD_SEMANTIC_ERROR(
            feature, self.record_display_name(record), context
        )

    def record_base_definitions(self, node):
        definitions = []
        for base in getattr(node, "base_classes", []) or []:
            base_name = base.get("type", "") if isinstance(base, dict) else str(base)
            definition = self.resolve_record_base_definition(node, base_name)
            if definition is not None:
                definitions.append(definition)
        return definitions

    def record_own_constructors(self, node):
        name = getattr(node, "name", None)
        return [
            member
            for member in getattr(node, "members", []) or []
            if isinstance(member, FunctionNode)
            and (
                getattr(member, "is_record_constructor", False)
                or (
                    getattr(member, "return_type", None) == ""
                    and getattr(member, "name", None) == name
                )
            )
        ]

    def record_inherited_constructor_markers(self, node):
        return [
            member
            for member in getattr(node, "members", []) or []
            if isinstance(member, FunctionNode)
            and getattr(member, "is_inherited_constructor", False)
        ]

    def record_direct_constructor_issue(self, node):
        if self.record_own_constructors(node):
            return ("record constructor use", node)
        if self.record_inherited_constructor_markers(node):
            return ("inherited record constructor use", node)
        return None

    def record_destructor_issue(self, node, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        name = getattr(node, "name", None)
        for member in getattr(node, "members", []) or []:
            if not isinstance(member, FunctionNode):
                continue
            if (
                getattr(member, "is_record_destructor", False)
                or getattr(member, "name", None) == f"~{name}"
            ):
                return ("record destructor use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_destructor_issue(base, visiting)
            if issue is not None:
                return issue
        for member in self.record_storage_members(node, include_bases=False):
            definition = self.record_value_definition_from_type(
                getattr(member, "vtype", "")
            )
            issue = self.record_destructor_issue(definition, visiting)
            if issue is not None:
                return issue
        return None

    def record_default_construction_issue(self, node, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        for constructor in self.record_own_constructors(node):
            if not (getattr(constructor, "params", None) or []):
                return ("record default constructor use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_default_construction_issue(base, visiting)
            if issue is not None:
                return issue
        for member in self.record_storage_members(node, include_bases=False):
            if getattr(member, "value", None) is not None:
                return (f"default member initializer '{member.name}'", node)
            definition = self.record_value_definition_from_type(
                getattr(member, "vtype", "")
            )
            issue = self.record_default_construction_issue(definition, visiting)
            if issue is not None:
                return issue
        return None

    def record_copy_constructor_issue(self, node, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        for constructor in self.record_own_constructors(node):
            params = getattr(constructor, "params", None) or []
            if len(params) != 1:
                continue
            raw_type = (
                params[0].get("type", "")
                if isinstance(params[0], dict)
                else getattr(params[0], "vtype", "")
            )
            if self.record_definition_from_type(raw_type) is node:
                return ("record copy/move constructor use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_copy_constructor_issue(base, visiting)
            if issue is not None:
                return issue
        for member in self.record_storage_members(node, include_bases=False):
            definition = self.record_value_definition_from_type(
                getattr(member, "vtype", "")
            )
            issue = self.record_copy_constructor_issue(definition, visiting)
            if issue is not None:
                return issue
        return None

    def record_assignment_operator_issue(self, node, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        for member in getattr(node, "members", []) or []:
            if isinstance(member, FunctionNode) and getattr(member, "name", None) in {
                "operator=",
                f"{getattr(node, 'name', '')}::operator=",
            }:
                return ("record assignment operator use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_assignment_operator_issue(base, visiting)
            if issue is not None:
                return issue
        for member in self.record_storage_members(node, include_bases=False):
            definition = self.record_value_definition_from_type(
                getattr(member, "vtype", "")
            )
            issue = self.record_assignment_operator_issue(definition, visiting)
            if issue is not None:
                return issue
        return None

    def record_conversion_operator_issue(self, node, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        if any(
            isinstance(member, FunctionNode)
            and getattr(member, "is_conversion_operator", False)
            for member in getattr(node, "members", []) or []
        ):
            return ("record conversion operator use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_conversion_operator_issue(base, visiting)
            if issue is not None:
                return issue
        return None

    def record_layout_aliases(self, node, visiting=None):
        if node is None:
            return {}
        visiting = set(visiting or ())
        if id(node) in visiting:
            return {}
        visiting.add(id(node))
        aliases = {}
        for base in self.record_base_definitions(node):
            aliases.update(self.record_layout_aliases(base, visiting))
        for member in getattr(node, "members", []) or []:
            if isinstance(member, TypeAliasNode):
                aliases[member.name] = member.alias_type
        return aliases

    def resolve_record_layout_member_type(self, owner, type_name):
        if not isinstance(type_name, str):
            return type_name
        aliases = self.record_layout_aliases(owner)
        text = self.resolve_record_member_type_alias(type_name)
        resolving = set()
        while text not in resolving:
            resolving.add(text)
            changed = False
            for name in sorted(aliases, key=lambda value: (-len(value), value)):
                pattern = re.compile(
                    rf"(?<![A-Za-z0-9_:]){re.escape(name)}(?![A-Za-z0-9_:])"
                )
                replacement = str(aliases[name])
                updated = pattern.sub(lambda _match: replacement, text)
                if updated != text:
                    text = updated
                    changed = True
            if not changed:
                return text
        raise HipSourceRefusal(f"recursive HIP record type alias: {type_name}")

    def record_member_layout_issue(self, owner, member):
        if not isinstance(member, VariableNode) or "static" in set(
            getattr(member, "qualifiers", []) or []
        ):
            return None
        if hasattr(member, "bitfield_width"):
            name = member.name or "<unnamed>"
            return (f"record bitfield member '{name}' width", owner)
        member_type = self.resolve_record_layout_member_type(
            owner, getattr(member, "vtype", "")
        )
        if "&" in str(member_type):
            name = member.name or "<unnamed>"
            return (f"record reference member '{name}' aliasing", owner)
        return None

    def record_layout_issue(self, node, visiting=None):
        if node is None:
            return None
        visiting = set(visiting or ())
        if id(node) in visiting:
            return None
        visiting.add(id(node))
        for base in self.record_base_definitions(node):
            issue = self.record_layout_issue(base, visiting)
            if issue is not None:
                return issue
        for member in getattr(node, "members", []) or []:
            issue = self.record_member_layout_issue(node, member)
            if issue is not None:
                return issue
            if not isinstance(member, VariableNode) or "static" in set(
                getattr(member, "qualifiers", []) or []
            ):
                continue
            member_type = self.resolve_record_layout_member_type(
                node, getattr(member, "vtype", "")
            )
            nested = self.record_value_definition_from_type(member_type)
            issue = self.record_layout_issue(nested, visiting)
            if issue is not None:
                return issue
        return None

    def validate_record_layout_use(self, definition, context):
        issue = self.record_layout_issue(definition)
        if issue is not None:
            self.raise_record_semantic(issue, context)

    def resolve_reference_type_aliases(self, type_name):
        """Resolve aliases without erasing cv/reference declarator markers."""
        if not isinstance(type_name, str):
            return type_name
        return self.resolve_record_member_type_alias(type_name)

    def is_record_reference_type(self, type_name, definition=None):
        resolved_type = self.resolve_reference_type_aliases(type_name)
        if definition is None:
            definition = self.record_definition_from_type(resolved_type)
        return definition is not None and bool(re.search(r"&{1,2}", resolved_type))

    def is_nonconst_record_reference_type(self, type_name, definition=None):
        resolved_type = self.resolve_reference_type_aliases(type_name)
        return self.is_record_reference_type(resolved_type, definition) and not bool(
            re.search(r"(?:^|\s)const(?:\s|$)", resolved_type)
        )

    def record_layout_attribute_issue(self, attributes):
        for attribute in attributes or []:
            compact = re.sub(r"\s+", "", str(attribute)).lower()
            if compact.startswith(("alignas(", "__align__(", "pragma_pack(")):
                return str(attribute)
            if compact.startswith("__attribute__(") and (
                "packed" in compact or "aligned" in compact
            ):
                return str(attribute)
            if compact.startswith("__declspec(") and (
                "align" in compact or "pack" in compact
            ):
                return str(attribute)
        return None

    def record_member_uses_bool_storage(self, owner, member_type):
        type_name = self.resolve_record_layout_member_type(owner, member_type)
        type_name = self.resolve_reference_type_aliases(type_name).strip()
        while self.has_array_suffix(type_name):
            type_name = type_name.split("[", 1)[0].strip()
        type_name = self.strip_elaborated_record_type_keyword(
            self.strip_type_qualifiers(type_name)
        ).strip()
        base_name, template_args = self.parse_cpp_template(type_name)
        if base_name in {"array", "std::array"} and template_args:
            return self.record_member_uses_bool_storage(owner, template_args[0])
        return type_name == "bool"

    def record_member_narrow_external_storage_type(
        self, owner, member_type, resolving=None
    ):
        """Return the source narrow type whose external ABI HLSL cannot retain."""
        resolving = set(resolving or ())
        type_name = self.resolve_record_layout_member_type(owner, member_type)
        type_name = self.resolve_reference_type_aliases(type_name).strip()
        while self.has_array_suffix(type_name):
            type_name = type_name.split("[", 1)[0].strip()
        type_name = self.strip_elaborated_record_type_keyword(
            self.strip_type_qualifiers(type_name)
        ).strip()

        base_name, template_args = self.parse_cpp_template(type_name)
        if base_name in {"array", "std::array"} and template_args:
            return self.record_member_narrow_external_storage_type(
                owner, template_args[0], resolving
            )
        if re.fullmatch(r"vec[1-4]", base_name) and len(template_args) == 1:
            return self.record_member_narrow_external_storage_type(
                owner, template_args[0], resolving
            )

        type_name = self.canonical_cpp_scalar_type(type_name)
        qualified_enum_name = (
            f"{self.record_display_name(owner)}::{type_name}" if owner else type_name
        )
        underlying = self.enum_underlying_types.get(qualified_enum_name)
        if underlying is None:
            underlying = self.enum_underlying_types.get(type_name.lstrip(":"))
        if underlying is not None and type_name not in resolving:
            issue = self.record_member_narrow_external_storage_type(
                owner, underlying, {*resolving, type_name}
            )
            return type_name if issue is not None else None

        if type_name in self.HIP_EXTERNAL_RECORD_NARROW_INTEGER_TYPES:
            return type_name
        if re.fullmatch(
            r"(?:char|uchar|short|ushort|i8|u8|i16|u16|int8|uint8|int16|uint16)(?:_t)?[1-4]",
            type_name,
        ):
            return type_name
        return None

    def record_external_abi_issue(self, node, visiting=None):
        if node is None:
            return None
        visiting = set(visiting or ())
        if id(node) in visiting:
            return None
        visiting.add(id(node))

        attribute = self.record_layout_attribute_issue(getattr(node, "attributes", []))
        if attribute is not None:
            return (f"external record layout attribute '{attribute}'", node)
        if any(
            isinstance(member, FunctionNode)
            and "virtual" in set(getattr(member, "qualifiers", []) or [])
            for member in getattr(node, "members", []) or []
        ):
            return ("external polymorphic record virtual member/vptr layout", node)
        for base in self.record_base_definitions(node):
            issue = self.record_external_abi_issue(base, visiting)
            if issue is not None:
                return issue
        for member in self.record_storage_members(node, include_bases=False):
            attribute = self.record_layout_attribute_issue(
                getattr(member, "attributes", [])
            )
            if attribute is not None:
                return (
                    f"external record member layout attribute '{attribute}'",
                    node,
                )
            member_type = getattr(member, "vtype", "")
            if self.record_member_uses_bool_storage(node, member_type):
                return (
                    f"external record bool storage member '{member.name}' ABI",
                    node,
                )
            narrow_type = self.record_member_narrow_external_storage_type(
                node, member_type
            )
            if narrow_type is not None:
                return (
                    f"external record narrow integer storage member "
                    f"'{member.name}' type '{narrow_type}' ABI",
                    node,
                )
            nested = self.record_value_definition_from_type(
                self.resolve_record_layout_member_type(node, member_type)
            )
            issue = self.record_external_abi_issue(nested, visiting)
            if issue is not None:
                return issue
        return None

    def validate_record_kernel_parameter_semantics(self, raw_type, definition, context):
        if self.is_nonconst_record_reference_type(raw_type, definition):
            self.raise_record_semantic(
                ("non-const kernel record reference writeback", definition), context
            )
        issue = self.record_external_abi_issue(definition)
        if issue is not None:
            self.raise_record_semantic(issue, context)

    def record_storage_members(self, node, include_bases=True, visiting=None):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return []
        visiting.add(id(node))
        members = []
        if include_bases:
            for base in self.record_base_definitions(node):
                members.extend(self.record_storage_members(base, True, visiting))
        members.extend(
            member
            for member in getattr(node, "members", []) or []
            if isinstance(member, VariableNode)
            and "static" not in set(getattr(member, "qualifiers", []) or [])
        )
        return members

    def record_contains_tracked_union_storage(self, node, visiting=None):
        """Whether record ``node`` holds a union whose active member the importer tracks.

        Such a union is written as separate members (see :meth:`union_tracks_active_member`), so a copy, a call
        argument or an assignment of the record would carry storage of members that were never written.
        """
        if node is None:
            return False
        if self.union_tracks_active_member(node):
            return True
        visiting = set(visiting or ())
        if id(node) in visiting:
            return False
        visiting.add(id(node))
        for member in self.record_storage_members(node, include_bases=False):
            member_type = self.resolve_record_member_type_alias(
                getattr(member, "vtype", "")
            )
            nested = self.record_definition_from_type(member_type)
            if self.record_contains_tracked_union_storage(nested, visiting):
                return True
        return any(
            self.record_contains_tracked_union_storage(base, visiting)
            for base in self.record_base_definitions(node)
        )

    def validate_record_aggregate_initializer(self, node, initializer, context):
        members = self.record_storage_members(node)
        elements = list(getattr(initializer, "elements", []) or [])
        if elements and self.record_base_definitions(node):
            self.raise_record_semantic(
                ("base subobject aggregate initialization", node), context
            )
        designated = {
            target: element.value
            for element in elements
            if isinstance(element, DesignatedInitializerNode)
            for kind, target in getattr(element, "designators", [])[:1]
            if kind == "member"
        }
        positional = [
            element
            for element in elements
            if not isinstance(element, DesignatedInitializerNode)
        ]
        for index, member in enumerate(members):
            supplied = designated.get(member.name)
            if supplied is None and index < len(positional):
                supplied = positional[index]
            if supplied is None:
                if getattr(member, "value", None) is not None:
                    self.raise_record_semantic(
                        (f"default member initializer '{member.name}'", node), context
                    )
                nested = self.record_value_definition_from_type(
                    getattr(member, "vtype", "")
                )
                issue = self.record_default_construction_issue(nested)
                if issue is not None:
                    self.raise_record_semantic(issue, context)
                continue
            nested = self.record_value_definition_from_type(
                getattr(member, "vtype", "")
            )
            if nested is not None and isinstance(supplied, InitializerListNode):
                issue = self.record_direct_constructor_issue(nested)
                if issue is not None:
                    self.raise_record_semantic(issue, context)
                self.validate_record_aggregate_initializer(nested, supplied, context)

    def validate_record_variable_initialization(self, variable, definition):
        if definition is None:
            return
        context = f"variable '{getattr(variable, 'name', '')}'"
        self.validate_record_layout_use(definition, context)
        issue = self.record_destructor_issue(definition)
        if issue is not None:
            self.raise_record_semantic(issue, context)
        value = getattr(variable, "value", None)
        if value is None:
            issue = self.record_default_construction_issue(definition)
            if issue is not None:
                self.raise_record_semantic(issue, context)
            return
        if isinstance(value, InitializerListNode):
            issue = self.record_direct_constructor_issue(definition)
            if issue is not None:
                self.raise_record_semantic(issue, context)
            self.validate_record_aggregate_initializer(definition, value, context)
            return
        if isinstance(value, FunctionCallNode):
            constructed = self.record_definition_from_type(
                value.name if isinstance(value.name, str) else ""
            )
            if constructed is definition:
                issue = self.record_direct_constructor_issue(definition)
                if issue is not None:
                    self.raise_record_semantic(issue, context)
        source = self.infer_record_expression_definition(value)
        if source is definition:
            if self.record_contains_tracked_union_storage(definition):
                self.raise_record_semantic(
                    ("union-containing copy construction active storage", definition),
                    context,
                )
            issue = self.record_copy_constructor_issue(definition)
            if issue is not None:
                self.raise_record_semantic(issue, context)
            return
        if source is not None and self.record_is_derived_from(source, definition):
            self.raise_record_semantic(
                ("derived-to-base value conversion", source), context
            )
        if source is not definition:
            issue = self.record_direct_constructor_issue(definition)
            if issue is not None:
                self.raise_record_semantic(issue, context)

    def record_storage_member_info(self, node, member_name, visiting=None):
        if node is None:
            return None
        visiting = set(visiting or ())
        if id(node) in visiting:
            return None
        visiting.add(id(node))
        for member in getattr(node, "members", []) or []:
            if isinstance(member, VariableNode) and member.name == member_name:
                return node, member
        for member in getattr(node, "members", []) or []:
            if not isinstance(member, StructNode) or getattr(member, "name", None):
                continue
            found = self.record_storage_member_info(member, member_name, visiting)
            if found is not None:
                return found
        for base in self.record_base_definitions(node):
            found = self.record_storage_member_info(base, member_name, visiting)
            if found is not None:
                return found
        return None

    def record_storage_member(self, node, member_name, visiting=None):
        found = self.record_storage_member_info(node, member_name, visiting)
        return found[1] if found is not None else None

    def record_storage_object_path(self, expression):
        if isinstance(expression, CastNode):
            return self.record_storage_object_path(expression.expression)
        if isinstance(expression, UnaryOpNode):
            path = self.record_storage_object_path(expression.operand)
            if path is None:
                return None
            if expression.op in {"*", "&"}:
                return f"{expression.op}{path}"
            return path
        if isinstance(expression, MemberAccessNode):
            path = self.record_storage_object_path(expression.object)
            if path is None:
                return None
            operator = "->" if getattr(expression, "is_pointer", False) else "."
            return f"{path}{operator}{expression.member}"
        if isinstance(expression, ArrayAccessNode):
            path = self.record_storage_object_path(expression.array)
            index = self.evaluate_enum_constant_expression(expression.index, {})
            if path is None or index is None:
                return None
            return f"{path}[{index}]"
        if isinstance(expression, str):
            return self.lookup_union_object_identity(expression)
        return None

    def union_state_key(self, object_path, node):
        return object_path, id(node)

    def lookup_union_active_member(self, key):
        for scope in reversed(self.union_active_member_scopes):
            if key in scope:
                return scope[key]
        return None

    def register_union_active_member(self, key, member_name):
        for scope in reversed(self.union_active_member_scopes):
            if key in scope:
                scope[key] = member_name
                return
        self.union_active_member_scopes[-1][key] = member_name

    def register_union_variable_initializer(self, name, definition, value):
        if definition is None or not self.union_tracks_active_member(definition):
            return
        if not isinstance(value, InitializerListNode) or not value.elements:
            return
        members = self.record_storage_members(definition, include_bases=False)
        if members:
            active_member = members[0].name
            first = value.elements[0]
            if isinstance(first, DesignatedInitializerNode):
                for kind, target in getattr(first, "designators", [])[:1]:
                    if kind in {"field", "member"}:
                        active_member = target
            object_path = self.record_storage_object_path(name)
            self.register_union_active_member(
                self.union_state_key(object_path, definition), active_member
            )

    def validate_union_member_access(self, expression):
        definition = self.infer_record_expression_definition(expression.object)
        found = self.record_storage_member_info(definition, expression.member)
        if found is None:
            return
        owner, _member = found
        if not self.union_tracks_active_member(owner):
            return
        root_name = self.get_lvalue_metadata_root_name(expression.object)
        if root_name is not None and self.is_union_indirect_object(root_name):
            self.raise_record_semantic(
                ("union indirect object access with unprovable active storage", owner),
                f"object '{root_name}'",
            )
        object_path = self.record_storage_object_path(expression.object)
        if object_path is None:
            self.raise_record_semantic(
                ("union member access with unprovable active storage", owner),
                f"member '{expression.member}'",
            )
        key = self.union_state_key(object_path, owner)
        active = self.lookup_union_active_member(key)
        if active is None:
            self.register_union_active_member(key, expression.member)
            return
        if active != expression.member:
            display_path = re.sub(r"#\d+", "", object_path)
            self.raise_record_semantic(
                (
                    f"union overlapping storage access from '{active}' to "
                    f"'{expression.member}'",
                    owner,
                ),
                f"object '{display_path}'",
            )

    def format_record_static_member_access(self, expression):
        definition = self.infer_record_expression_definition(expression.object)
        found = self.record_storage_member_info(definition, expression.member)
        if found is None:
            return None
        owner, member = found
        if "static" not in set(getattr(member, "qualifiers", []) or []):
            return None
        owner_names = [
            self.record_qualified_names.get(id(owner)),
            self.convert_hip_record_name_to_crossgl(getattr(owner, "name", "")),
        ]
        for owner_name in owner_names:
            if not owner_name:
                continue
            candidate = f"{owner_name}::{expression.member}"
            value = self.resolve_enum_value_alias(candidate)
            if value is not None:
                return str(value)
            if candidate in self.unsupported_record_static_constants:
                self.raise_record_semantic(
                    (
                        f"static record state '{candidate}' cannot be represented "
                        "exactly",
                        owner,
                    ),
                    "member access",
                )
        self.raise_record_semantic(
            (
                f"static record state '{expression.member}' cannot be represented "
                "exactly",
                owner,
            ),
            "member access",
        )
