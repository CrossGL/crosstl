"""Program, struct, enum and nested-record declarations in the HIP to CrossGL converter."""

import hashlib

from crosstl.translator.refusals import HipSourceRefusal

from ..HipAst import (
    BinaryOpNode,
    EnumNode,
    StructNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class HipRecordMixin:
    """Record members of ``HipToCrossGLConverter``."""

    def visit_PreprocessorNode(self, node):
        content = self.format_preprocessor_content(node.content)
        if node.directive == "include":
            if "hip_runtime.h" in content:
                self.emit("// HIP runtime functionality built-in")
            elif "hip/hip_runtime_api.h" in content:
                self.emit("// HIP runtime API functionality built-in")
            else:
                self.emit(f"// include {content}".strip())
        elif content:
            self.emit(f"// {node.directive} {content}")
        else:
            self.emit(f"// {node.directive}")

    def is_crossgl_scalar_type(self, type_name):
        return str(type_name).strip() in self.CROSSL_SCALAR_TYPES

    def visit_StructNode(self, node):
        output_name = (
            self.convert_hip_record_name_to_crossgl(node.name) if node.name else None
        )
        if output_name:
            qualified_name = self.record_qualified_names.get(id(node))
            if qualified_name is None:
                qualified_name = "::".join(
                    [*self.record_name_stack, str(getattr(node, "name", output_name))]
                )
        else:
            qualified_name = None
        if (
            output_name
            and qualified_name != output_name
            and output_name in self.record_definitions
            and self.record_definitions[output_name] is None
        ):
            raise HipSourceRefusal(
                f"Conflicting HIP nested record output name '{output_name}'"
            )
        self.record_name_stack.append(qualified_name or "anonymous")
        nested_records, nested_enums, members = self.materialize_nested_record_members(
            node,
            output_name or "anonymous",
        )
        (
            inherited_aliases,
            inherited_members,
            inherited_enums,
            inherited_records,
            inherited_constants,
        ) = self.collect_inherited_record_members(node)
        own_aliases = [
            member for member in members if isinstance(member, TypeAliasNode)
        ]
        aliases = [*inherited_aliases, *own_aliases]
        storage_members = [*inherited_members, *members]
        storage_names = [
            member.name
            for member in storage_members
            if isinstance(member, VariableNode)
            and member.name
            and not hasattr(member, "bitfield_width")
            and "static" not in set(getattr(member, "qualifiers", []) or [])
        ]
        if len(storage_names) != len(set(storage_names)):
            raise HipSourceRefusal(
                f"HIP inherited record '{qualified_name or output_name}' has "
                "colliding storage member names"
            )
        self.push_type_alias_scope()
        try:
            for alias in aliases:
                self.register_type_alias(alias.name, alias.alias_type)
            if output_name:
                enclosing_scope = self.type_alias_scopes[-2]
                global_scope = self.type_alias_scopes[0]
                if qualified_name != output_name:
                    global_scope[qualified_name] = output_name
                for alias in aliases:
                    resolved = self.resolve_record_member_type_alias(alias.alias_type)
                    enclosing_scope[f"{output_name}::{alias.name}"] = resolved
                    if qualified_name:
                        global_scope[f"{qualified_name}::{alias.name}"] = resolved
                for inherited_record in inherited_records:
                    inherited_name = getattr(inherited_record, "name", None)
                    if inherited_name and qualified_name:
                        global_scope[f"{qualified_name}::{inherited_name}"] = (
                            self.convert_hip_record_name_to_crossgl(inherited_name)
                        )
            own_constants = [
                member
                for member in members
                if isinstance(member, VariableNode)
                and "static" in set(getattr(member, "qualifiers", []) or [])
            ]
            constant_values = self.register_record_static_constants(
                [*inherited_constants, *own_constants],
                qualified_name or output_name,
            )
            for enum_node in [*inherited_enums, *nested_enums]:
                self.register_nested_enum(
                    enum_node,
                    qualified_name or output_name,
                    constant_values,
                )
            for nested_record in nested_records:
                self.visit_StructNode(nested_record)

            materialized = self.clone_ast_node(node)
            materialized.name = output_name
            materialized.members = storage_members
            union = None
            if getattr(materialized, "is_union", False):
                if materialized.name:
                    union = self.hip_union_layout_record(node, materialized.name)
                if union is None:
                    # A union without the layout contract: the importer tracks its active member instead.
                    name = materialized.name or "anonymous"
                    self.emit(
                        f"// {self.SOURCE_LANGUAGE_NAME} union {name} represented as "
                        "struct-like layout; overlapping storage is not modeled"
                    )
                if not materialized.name:
                    return

            self.emit(f"struct {materialized.name} {{")
            self.indent_level += 1
            layouts = getattr(self, "emitted_buffer_layouts", None)
            if layouts is not None:
                layouts.struct(
                    materialized.name,
                    getattr(node, "attributes", None),
                    getattr(node, "is_union", False),
                )
            for member in materialized.members:
                if not isinstance(member, VariableNode) or "static" in set(
                    getattr(member, "qualifiers", []) or []
                ):
                    continue
                if self.record_member_layout_issue(node, member) is not None:
                    if union is not None:
                        union.member(
                            "",
                            member.name,
                            None,
                            member,
                            self.hip_union_member_issue(node, member),
                        )
                    continue
                member_type = self.convert_hip_type_to_crossgl(
                    self.resolve_record_member_type_alias(
                        getattr(member, "vtype", "int")
                    )
                )
                if layouts is not None:
                    # HIP stores a three-component vector member at its scalar alignment; see buffer_layout_import.
                    layouts.member(
                        materialized.name, member_type, member.name, len(self.output)
                    )
                if union is not None:
                    union.member(
                        member_type,
                        member.name,
                        len(self.output),
                        member,
                        self.hip_union_member_issue(node, member),
                    )
                self.emit(f"{member_type} {member.name};")
            self.indent_level -= 1
            self.emit("};")
        finally:
            self.pop_type_alias_scope()
            self.record_name_stack.pop()

    def collect_inherited_record_members(self, node, visiting=None):
        visiting = set(visiting or ())
        marker = id(node)
        if marker in visiting:
            raise HipSourceRefusal("HIP record inheritance cycle is unsupported")
        visiting.add(marker)
        aliases = []
        members = []
        enums = []
        records = []
        constants = []
        for base in getattr(node, "base_classes", []) or []:
            if isinstance(base, dict):
                base_name = base.get("type", "")
                is_virtual = bool(base.get("is_virtual"))
            else:
                base_name = str(base)
                is_virtual = False
            if is_virtual:
                raise HipSourceRefusal(
                    f"HIP virtual record inheritance is unsupported: {base_name}"
                )
            base_node = self.resolve_record_base_definition(node, base_name)
            if base_node is None:
                raise HipSourceRefusal(
                    f"HIP record '{getattr(node, 'name', '')}' has unresolved "
                    f"base '{base_name}'"
                )
            (
                base_aliases,
                base_members,
                base_enums,
                base_records,
                base_constants,
            ) = self.collect_inherited_record_members(base_node, visiting)
            aliases.extend(base_aliases)
            members.extend(base_members)
            enums.extend(base_enums)
            records.extend(base_records)
            constants.extend(base_constants)
            local_enums = [
                member
                for member in getattr(base_node, "members", []) or []
                if isinstance(member, EnumNode)
            ]
            for member in getattr(base_node, "members", []) or []:
                if isinstance(member, TypeAliasNode):
                    aliases.append(member)
                elif isinstance(member, EnumNode):
                    enums.append(member)
                elif isinstance(member, StructNode):
                    records.append(member)
                elif isinstance(member, VariableNode) and "static" in set(
                    getattr(member, "qualifiers", []) or []
                ):
                    constants.append(member)
                elif isinstance(member, VariableNode):
                    inherited_member = self.clone_ast_node(member)
                    for enum_node in local_enums:
                        prefix = self.nested_enum_declarator_prefix(enum_node)
                        if not self.nested_record_declarator_matches(
                            inherited_member, prefix
                        ):
                            continue
                        underlying = (
                            getattr(enum_node, "underlying_type", None) or "int"
                        )
                        inherited_member.vtype = (
                            self.rewrite_nested_record_declarator_type(
                                getattr(inherited_member, "vtype", ""),
                                prefix,
                                underlying,
                            )
                        )
                        break
                    members.append(inherited_member)
        return aliases, members, enums, records, constants

    def resolve_record_base_definition(self, node, base_name):
        base_name = self.strip_elaborated_record_type_keyword(str(base_name).strip())
        base_name = base_name.lstrip(":")
        if not base_name:
            return None
        qualified_name = self.record_qualified_names.get(id(node), "")
        owners = qualified_name.split("::")[:-1] if qualified_name else []
        candidates = []
        if "::" not in base_name:
            for size in range(len(owners), 0, -1):
                candidates.append("::".join([*owners[:size], base_name]))
        candidates.append(base_name)
        for candidate in candidates:
            definition = self.record_definitions.get(candidate)
            if definition is not None:
                return definition
        return None

    def materialize_nested_record_members(self, node, parent_name):
        pending = list(getattr(node, "members", []) or [])
        nested_records = []
        nested_enums = []
        members = []
        index = 0
        while index < len(pending):
            member = pending[index]
            if isinstance(member, EnumNode):
                prefix = self.nested_enum_declarator_prefix(member)
                declarations = []
                next_index = index + 1
                while next_index < len(pending):
                    declaration = pending[next_index]
                    if not isinstance(declaration, VariableNode):
                        break
                    if not self.nested_record_declarator_matches(declaration, prefix):
                        break
                    declarations.append(declaration)
                    next_index += 1
                underlying = getattr(member, "underlying_type", None) or "int"
                for declaration in declarations:
                    rewritten = self.clone_ast_node(declaration)
                    rewritten.vtype = self.rewrite_nested_record_declarator_type(
                        getattr(declaration, "vtype", ""),
                        prefix,
                        underlying,
                    )
                    members.append(rewritten)
                nested_enums.append(member)
                index = next_index
                continue
            if not isinstance(member, StructNode):
                members.append(member)
                index += 1
                continue

            prefix = self.nested_record_declarator_prefix(member)
            declarations = []
            next_index = index + 1
            while next_index < len(pending):
                declaration = pending[next_index]
                if not isinstance(declaration, VariableNode):
                    break
                if not self.nested_record_declarator_matches(declaration, prefix):
                    break
                declarations.append(declaration)
                next_index += 1

            if not getattr(member, "name", None) and not declarations:
                pending[index : index + 1] = list(getattr(member, "members", []) or [])
                continue

            nested_name = getattr(member, "name", None)
            if nested_name:
                nested_name = self.convert_hip_record_name_to_crossgl(nested_name)
            else:
                nested_name = self.anonymous_nested_record_name(
                    parent_name,
                    member,
                    declarations,
                )

            materialized = self.clone_ast_node(member)
            materialized.name = nested_name
            nested_records.append(materialized)
            for declaration in declarations:
                rewritten = self.clone_ast_node(declaration)
                rewritten.vtype = self.rewrite_nested_record_declarator_type(
                    getattr(declaration, "vtype", ""),
                    prefix,
                    nested_name,
                )
                members.append(rewritten)
            index = next_index

        return nested_records, nested_enums, members

    def nested_enum_declarator_prefix(self, node):
        name = getattr(node, "name", None) or "<anonymous>"
        return f"enum {name}"

    def register_record_static_constants(self, constants, owner_name):
        values = {}
        if not owner_name:
            return values
        for constant in constants:
            qualifiers = set(getattr(constant, "qualifiers", []) or [])
            qualified_name = f"{owner_name}::{constant.name}"
            if not qualifiers.intersection({"const", "constexpr"}):
                self.unsupported_record_static_constants.add(qualified_name)
                continue
            value = self.evaluate_enum_constant_expression(
                getattr(constant, "value", None), values
            )
            if value is None:
                self.unsupported_record_static_constants.add(qualified_name)
                continue
            values[constant.name] = value
            self.enum_value_aliases[qualified_name] = str(value)
            self.unsupported_record_static_constants.discard(qualified_name)
        return values

    def register_nested_enum(self, node, owner_name, enclosing_values=None):
        underlying = getattr(node, "underlying_type", None) or "int"
        enum_name = getattr(node, "name", None)
        if enum_name:
            self.enum_underlying_types.setdefault(enum_name, underlying)
            self.register_type_alias(enum_name, underlying)
            if owner_name:
                qualified_name = f"{owner_name}::{enum_name}"
                self.enum_underlying_types[qualified_name] = underlying
                self.type_alias_scopes[0][qualified_name] = underlying
        next_value = 0
        local_values = dict(enclosing_values or {})
        members = getattr(node, "members", None) or getattr(node, "variants", [])
        for member in members:
            if isinstance(member, tuple):
                member_name, member_value = member
            else:
                member_name = getattr(member, "name", str(member))
                member_value = getattr(member, "value", None)
            if member_value is None:
                if next_value is None:
                    raise HipSourceRefusal(
                        f"HIP enum '{owner_name}' has an implicit value after an "
                        "unsupported initializer"
                    )
                value = next_value
            else:
                value = self.evaluate_enum_constant_expression(
                    member_value, local_values
                )
                if value is None:
                    raise HipSourceRefusal(
                        f"HIP enum '{owner_name}' initializer for '{member_name}' "
                        "cannot be represented exactly"
                    )
            if getattr(node, "is_scoped", False) and enum_name:
                alias = f"{owner_name}::{enum_name}::{member_name}"
            else:
                alias = f"{owner_name}::{member_name}"
            self.enum_value_aliases[alias] = str(value)
            local_values[member_name] = value
            if enum_name:
                local_values[f"{enum_name}::{member_name}"] = value
            next_value = value + 1

    def evaluate_enum_constant_expression(self, value, local_values):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            normalized = self.normalize_cpp_numeric_literal(value).strip()
            if normalized in local_values:
                return local_values[normalized]
            resolved = self.resolve_enum_value_alias(normalized)
            if resolved is not None:
                try:
                    return int(str(resolved).rstrip("uU"), 0)
                except ValueError:
                    return None
            try:
                return int(normalized.rstrip("uU"), 0)
            except ValueError:
                return None
        if isinstance(value, UnaryOpNode):
            operand = self.evaluate_enum_constant_expression(
                value.operand, local_values
            )
            if operand is None:
                return None
            operations = {
                "+": lambda item: item,
                "-": lambda item: -item,
                "~": lambda item: ~item,
                "!": lambda item: int(not item),
            }
            operation = operations.get(value.op)
            return operation(operand) if operation is not None else None
        if isinstance(value, BinaryOpNode):
            left = self.evaluate_enum_constant_expression(value.left, local_values)
            right = self.evaluate_enum_constant_expression(value.right, local_values)
            if left is None or right is None:
                return None
            operations = {
                "+": lambda lhs, rhs: lhs + rhs,
                "-": lambda lhs, rhs: lhs - rhs,
                "*": lambda lhs, rhs: lhs * rhs,
                "/": self.divide_enum_constant,
                "%": self.modulo_enum_constant,
                "<<": lambda lhs, rhs: lhs << rhs,
                ">>": lambda lhs, rhs: lhs >> rhs,
                "&": lambda lhs, rhs: lhs & rhs,
                "|": lambda lhs, rhs: lhs | rhs,
                "^": lambda lhs, rhs: lhs ^ rhs,
                "&&": lambda lhs, rhs: int(bool(lhs) and bool(rhs)),
                "||": lambda lhs, rhs: int(bool(lhs) or bool(rhs)),
            }
            operation = operations.get(value.op)
            if operation is None:
                return None
            try:
                return operation(left, right)
            except (ArithmeticError, ValueError):
                return None
        return None

    def divide_enum_constant(self, left, right):
        if right == 0:
            raise ZeroDivisionError
        quotient = abs(left) // abs(right)
        return -quotient if (left < 0) != (right < 0) else quotient

    def modulo_enum_constant(self, left, right):
        return left - self.divide_enum_constant(left, right) * right

    def resolve_enum_value_alias(self, name, resolving=None):
        lookup_name = self.canonical_record_static_member_name(name)
        if (
            isinstance(lookup_name, str)
            and lookup_name in self.unsupported_record_static_constants
        ):
            owner_name = lookup_name.rsplit("::", 1)[0]
            definition = self.record_definitions.get(owner_name)
            if definition is None:
                definition = self.record_definition_from_type(owner_name)
            if definition is not None:
                self.raise_record_semantic(
                    (
                        f"static record state '{name}' cannot be represented exactly",
                        definition,
                    ),
                    "qualified member access",
                )
            raise HipSourceRefusal(
                f"HIP static record constant '{name}' cannot be represented exactly"
            )
        if (
            not isinstance(lookup_name, str)
            or lookup_name not in self.enum_value_aliases
        ):
            return None
        resolving = set(resolving or ())
        if lookup_name in resolving:
            raise HipSourceRefusal(f"recursive HIP enum value alias: {lookup_name}")
        resolving.add(lookup_name)
        value = self.enum_value_aliases[lookup_name]
        nested = self.resolve_enum_value_alias(value, resolving)
        return (
            nested if nested is not None else self.normalize_cpp_numeric_literal(value)
        )

    def nested_record_declarator_prefix(self, node):
        if getattr(node, "is_union", False):
            keyword = "union"
        elif getattr(node, "is_class", False):
            keyword = "class"
        else:
            keyword = "struct"
        name = getattr(node, "name", None) or "<anonymous>"
        return f"{keyword} {name}"

    def nested_record_declarator_matches(self, declaration, prefix):
        return self.nested_record_type_matches(
            getattr(declaration, "vtype", ""),
            prefix,
        )

    def nested_record_type_matches(self, raw_type, prefix):
        normalized = " ".join(str(raw_type).split())
        return normalized == prefix or any(
            normalized.startswith(prefix + suffix) for suffix in (" ", "*", "[")
        )

    def rewrite_nested_record_declarator_type(
        self,
        raw_type,
        prefix,
        nested_name,
    ):
        normalized = " ".join(str(raw_type).split())
        if not self.nested_record_type_matches(normalized, prefix):
            raise HipSourceRefusal(
                f"nested record declarator '{normalized}' does not match '{prefix}'"
            )
        return nested_name + normalized[len(prefix) :]

    def anonymous_nested_record_name(self, parent_name, node, declarations):
        if getattr(node, "is_union", False):
            keyword = "union"
        elif getattr(node, "is_class", False):
            keyword = "class"
        else:
            keyword = "struct"
        declarator_name = getattr(declarations[0], "name", "member")
        signature = [parent_name, declarator_name, keyword]
        for member in getattr(node, "members", []) or []:
            signature.extend(
                (
                    type(member).__name__,
                    str(getattr(member, "name", "")),
                    str(getattr(member, "vtype", "")),
                )
            )
        digest = hashlib.sha256("\0".join(signature).encode("utf-8")).hexdigest()[:12]
        parent = self.sanitize_crossgl_type_identifier(parent_name)
        declarator = self.sanitize_crossgl_type_identifier(declarator_name)
        return f"CrossGLAnonymous_{parent}_{declarator}_{keyword}_{digest}"

    def clone_ast_node(self, node):
        cloned = node.__class__.__new__(node.__class__)
        cloned.__dict__ = dict(getattr(node, "__dict__", {}))
        return cloned
