"""Type aliases, alias templates and struct member aliases."""

import re

from ....translator.codegen.array_utils import evaluate_literal_int_expression
from ..aliased_member_owners import resolve_member_alias_type
from ..MetalAst import (
    CallableTypeAliasNode,
    FunctionNode,
    StructNode,
    TypeAliasNode,
)
from ..partial_struct_specializations import (
    UndecidedPartialSpecialization,
    partial_specialization_for_owner,
)
from ..preprocessor import MetalPreprocessor
from ..template_text import substitute_template_identifiers
from ..using_declared_library_names import using_declared_alias_targets
from .support import (
    MetalAliasTemplateResolutionError,
    MetalScalarAliasResolutionError,
    MetalStructAliasResolutionError,
)


class MetalAliasTemplateMixin:
    """Members of ``MetalToCrossGLConverter``: type aliases, alias templates and struct member aliases."""

    def iter_ast_children(self, node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, dict):
            yield from node.values()
            return
        if isinstance(node, (list, tuple, set)):
            yield from node
            return
        yield from getattr(node, "__dict__", {}).values()

    @staticmethod
    def is_template_alias_declaration(alias):
        return bool(getattr(alias, "is_template_alias", False))

    @staticmethod
    def alias_source_offset(node):
        location = (
            getattr(node, "source_location", None)
            or getattr(node, "declaration_source_location", None)
            or {}
        )
        return location.get("offset") if isinstance(location, dict) else None

    @staticmethod
    def normalize_qualified_type_name(name):
        return re.sub(r"\s*::\s*", "::", str(name or "").strip())

    def prepare_alias_template_resolution(self, ast, typedefs):
        declarations = {}
        plain_declarations = {}
        for alias in typedefs or []:
            if not isinstance(alias, TypeAliasNode) or not getattr(alias, "name", None):
                continue
            qualified_name = self.normalize_qualified_type_name(
                getattr(alias, "qualified_name", None) or alias.name
            )
            destination = (
                declarations
                if self.is_template_alias_declaration(alias)
                else plain_declarations
            )
            destination.setdefault(qualified_name, []).append(alias)

        self.alias_template_declarations = declarations
        self.alias_template_plain_declarations = plain_declarations
        self.scalar_alias_bindings = {}
        self.scalar_alias_names = {
            declaration.name
            for declarations in plain_declarations.values()
            for declaration in declarations
        }
        self.parameter_type_contexts = {
            id(parameter): function
            for function in getattr(ast, "functions", ()) or ()
            for parameter in getattr(function, "params", ()) or ()
        }
        self.type_alias_enum_declarations = {}
        for declaration in getattr(ast, "enums", ()) or ():
            name = getattr(declaration, "qualified_name", None) or declaration.name
            if name:
                self.type_alias_enum_declarations.setdefault(name, []).append(
                    declaration
                )
        self.alias_template_cache = {}
        self.alias_template_resolution_stack = []
        self.alias_template_structs = [
            node
            for node in getattr(ast, "structs", []) or []
            if isinstance(node, StructNode) and getattr(node, "name", None)
        ]
        self.alias_template_structs_by_qualified_name = {}
        for struct_node in self.alias_template_structs:
            qualified_name = self.normalize_qualified_type_name(
                getattr(struct_node, "qualified_name", None) or struct_node.name
            )
            self.alias_template_structs_by_qualified_name.setdefault(
                qualified_name, []
            ).append(struct_node)
        self.alias_template_using_namespaces = list(
            getattr(ast, "using_namespace_directives", []) or []
        )
        # The Metal standard library declares ``metal`` and its nested ``fast`` and ``precise`` namespaces.
        self.alias_template_namespaces = {
            "metal",
            "metal::fast",
            "metal::precise",
        } | set(getattr(ast, "declared_namespaces", ()) or ())
        self.alias_template_using_declarations = list(
            getattr(ast, "using_declarations", []) or []
        )
        self.struct_member_alias_resolution_stack = []
        self.current_type_resolution_context = None

    def alias_resolution_namespace(self):
        context = self.current_type_resolution_context
        return self.normalize_qualified_type_name(
            getattr(context, "namespace", "") if context is not None else ""
        )

    def scalar_alias_declarations(self, name, context=None):
        if str(name).rsplit("::", 1)[-1] not in self.scalar_alias_names:
            return []
        if not re.fullmatch(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", str(name)):
            return []
        previous = self.current_type_resolution_context
        if context is not None:
            self.current_type_resolution_context = context
        try:
            for tier in self.alias_lookup_name_tiers(name):
                declarations = [
                    declaration
                    for candidate in tier
                    for declaration in self.alias_template_plain_declarations.get(
                        candidate, ()
                    )
                    if self.declaration_visible_at_current_offset(declaration)
                ]
                if not declarations and any(
                    self.declaration_visible_at_current_offset(declaration)
                    for candidate in tier
                    for declaration in (
                        *self.alias_template_structs_by_qualified_name.get(
                            candidate, ()
                        ),
                        *self.type_alias_enum_declarations.get(candidate, ()),
                        *self.alias_template_declarations.get(candidate, ()),
                    )
                ):
                    return []
                if declarations:
                    owners = {
                        getattr(declaration, "qualified_name", declaration.name)
                        for declaration in declarations
                    }
                    if len(owners) != 1:
                        raise MetalScalarAliasResolutionError(
                            str(name),
                            "lookup is ambiguous in the declaration context",
                            source_location=getattr(
                                self.current_type_resolution_context,
                                "source_location",
                                None,
                            )
                            or getattr(
                                self.current_type_resolution_context,
                                "declaration_source_location",
                                None,
                            ),
                        )
                    return declarations
            return []
        finally:
            self.current_type_resolution_context = previous

    def scalar_alias_binding(self, name, *, context=None, stack=()):
        if str(name).rsplit("::", 1)[-1] not in self.scalar_alias_names:
            return None
        if context is None and str(name) in self.alias_resolution_shadow_names():
            return None
        declarations = self.scalar_alias_declarations(name, context)
        if not declarations:
            return None
        bindings = [
            self.scalar_alias_declaration_binding(alias, stack)
            for alias in declarations
        ]
        concrete = [binding for binding in bindings if binding is not None]
        if concrete and (len(concrete) != len(bindings) or len(set(concrete)) != 1):
            raise MetalScalarAliasResolutionError(
                str(name),
                "visible declarations define conflicting concrete alias targets",
                source_location=getattr(declarations[-1], "source_location", None),
            )
        return concrete[0] if concrete else None

    def scalar_alias_needs_inline(self, alias):
        if getattr(alias, "namespace", ""):
            return True
        return any(
            name.rsplit("::", 1)[-1] == alias.name
            for name in (
                *self.alias_template_structs_by_qualified_name,
                *self.type_alias_enum_declarations,
                *self.alias_template_declarations,
            )
        )

    def scalar_alias_declaration_binding(self, alias, stack=()):
        key = id(alias)
        if key in self.scalar_alias_bindings:
            return self.scalar_alias_bindings[key]
        if alias in stack:
            raise MetalScalarAliasResolutionError(
                alias.name,
                "the alias dependency chain is recursive",
                source_location=getattr(alias, "source_location", None),
                dependency_chain=[node.name for node in (*stack, alias)],
            )
        target = str(alias.alias_type).strip()
        if (
            getattr(alias, "array_sizes", None)
            or getattr(alias, "declarator_type_suffix", "")
            or any(token in target for token in "*&[]")
            or self.is_template_alias_declaration(alias)
            or getattr(alias, "is_function_type", False)
        ):
            return None
        qualifiers = set(getattr(alias, "qualifiers", ()) or ())
        parent = self.scalar_alias_binding(target, context=alias, stack=(*stack, alias))
        if parent is not None:
            target, inherited = parent
            qualifiers.update(inherited)
        mapped = self.type_map.get(target)
        if mapped is None or self.crossgl_typedef_source_type(mapped) is None:
            return None
        if qualifiers - {"const", "volatile"}:
            return None
        binding = (target, tuple(sorted(qualifiers)))
        self.scalar_alias_bindings[key] = binding
        return binding

    def alias_resolution_offset(self):
        return self.alias_source_offset(self.current_type_resolution_context)

    def alias_resolution_shadow_names(self):
        names = set(getattr(self, "local_type_alias_names", set()))
        context = self.current_type_resolution_context
        for _kind, name in getattr(context, "template_parameters", None) or []:
            if name:
                names.add(name)
        return names

    @staticmethod
    def namespace_lookup_scopes(namespace):
        parts = [part for part in str(namespace or "").split("::") if part]
        return ["::".join(parts[:index]) for index in range(len(parts), -1, -1)]

    def visible_using_namespace_targets(self, scope, *, use_offset=None):
        if use_offset is None:
            use_offset = self.alias_resolution_offset()
        context = self.current_type_resolution_context
        context_labels = set()
        if isinstance(context, FunctionNode):
            context_labels.add(f"function {context.name}")
        elif isinstance(context, StructNode):
            aggregate_kind = getattr(context, "aggregate_kind", "struct")
            context_labels.add(f"{aggregate_kind} {context.name}")
        targets = []
        for directive in self.alias_template_using_namespaces:
            if self.normalize_qualified_type_name(directive.get("namespace")) != scope:
                continue
            declaration_context = directive.get("declaration_context")
            if declaration_context and declaration_context not in context_labels:
                continue
            location = directive.get("source_location") or {}
            directive_offset = (
                location.get("offset") if isinstance(location, dict) else None
            )
            if (
                use_offset is not None
                and directive_offset is not None
                and directive_offset >= use_offset
            ):
                continue
            target = self.normalize_qualified_type_name(directive.get("target"))
            if not target:
                continue
            target_candidates = []
            for parent_scope in self.namespace_lookup_scopes(scope):
                candidate = f"{parent_scope}::{target}" if parent_scope else target
                if candidate not in target_candidates:
                    target_candidates.append(candidate)
            declared_names, declared_scopes = self.alias_template_declared_names()
            resolved = next(
                (
                    candidate
                    for candidate in target_candidates
                    if candidate in declared_names or candidate in declared_scopes
                ),
                target_candidates[-1],
            )
            if resolved not in targets:
                targets.append(resolved)
        return targets

    def alias_template_declared_names(self):
        """Return the declared alias and struct names and their ``::`` scopes.

        A scope is every prefix ``p`` of a declared name such that the name starts
        with ``p + "::"``. The declaration tables are built once per program, so
        the sets are cached against the table objects and their sizes.
        """
        tables = (
            self.alias_template_declarations,
            self.alias_template_plain_declarations,
            self.alias_template_structs_by_qualified_name,
        )
        sizes = tuple(len(table) for table in tables)
        cached = getattr(self, "_alias_template_declared_names", None)
        if (
            cached is not None
            and cached[1] == sizes
            and all(table is previous for table, previous in zip(tables, cached[0]))
        ):
            return cached[2], cached[3]
        names = set()
        scopes = set()
        for table in tables:
            for name in table:
                names.add(name)
                separator = name.find("::")
                while separator != -1:
                    scopes.add(name[:separator])
                    separator = name.find("::", separator + 1)
        self._alias_template_declared_names = (tables, sizes, names, scopes)
        return names, scopes

    def alias_lookup_name_tiers(self, name):
        raw_name = self.normalize_qualified_type_name(name)
        globally_qualified = raw_name.startswith("::")
        raw_name = raw_name.lstrip(":")
        if globally_qualified:
            return [[raw_name]]

        namespace = self.alias_resolution_namespace()
        if "::" in raw_name:
            tiers = []
            for scope in self.namespace_lookup_scopes(namespace):
                candidate = f"{scope}::{raw_name}" if scope else raw_name
                if [candidate] not in tiers:
                    tiers.append([candidate])
            return tiers

        tiers = []
        for scope in self.namespace_lookup_scopes(namespace):
            direct = f"{scope}::{raw_name}" if scope else raw_name
            tiers.append([direct])
            # ``using c10::metal::opmath_t;`` makes the name a member of the scope it appears in
            # ([namespace.udecl]/1), ahead of the namespaces a using-directive nominates.
            declared = using_declared_alias_targets(
                self.alias_template_using_declarations,
                scope,
                raw_name,
                self.normalize_qualified_type_name,
            )
            if declared:
                tiers.append(declared)
            imported = [
                f"{target}::{raw_name}"
                for target in self.visible_using_namespace_targets(scope)
            ]
            if imported:
                tiers.append(imported)
        return tiers

    def declaration_visible_at_current_offset(self, declaration):
        if self.metal_later_declarations_visible:
            return True
        use_offset = self.alias_resolution_offset()
        declaration_offset = self.alias_source_offset(declaration)
        return (
            use_offset is None
            or declaration_offset is None
            or declaration_offset <= use_offset
        )

    def alias_template_declaration_for_name(self, name):
        raw_name = self.normalize_qualified_type_name(name).lstrip(":")
        if "::" not in raw_name and raw_name in self.alias_resolution_shadow_names():
            return None

        for tier in self.alias_lookup_name_tiers(name):
            aliases = [
                declaration
                for candidate in tier
                for declaration in self.alias_template_declarations.get(candidate, [])
                if self.declaration_visible_at_current_offset(declaration)
            ]
            plain = [
                declaration
                for candidate in tier
                for declaration in self.alias_template_plain_declarations.get(
                    candidate, []
                )
                if self.declaration_visible_at_current_offset(declaration)
            ]
            if plain and not aliases:
                return None
            if not aliases:
                continue
            signature = self.normalize_qualified_type_name(name)
            if len(aliases) != 1 or plain:
                representative = aliases[0]
                raise MetalAliasTemplateResolutionError(
                    representative.name,
                    signature,
                    "lookup is ambiguous in the declaration context",
                    source_location=getattr(representative, "source_location", None),
                )
            return aliases[0]
        return None

    @staticmethod
    def matching_type_angle(text, angle_start):
        depth = 0
        for index in range(angle_start, len(text)):
            if text[index] == "<":
                depth += 1
            elif text[index] == ">":
                depth -= 1
                if depth == 0:
                    return index
        return None

    def alias_template_instance_parts(self, type_text):
        text = str(type_text or "").strip()
        angle_start = text.find("<")
        if angle_start <= 0:
            return None
        angle_end = self.matching_type_angle(text, angle_start)
        if angle_end is None or text[angle_end + 1 :].strip():
            return None
        name = text[:angle_start].strip()
        if not re.fullmatch(r"(?:::)?[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*", name):
            return None
        inner = text[angle_start + 1 : angle_end]
        arguments = self.split_generic_arguments(inner) if inner.strip() else []
        return name, arguments

    @staticmethod
    def dependent_alias_type_parts(type_text):
        text = str(type_text or "").strip()
        depth = 0
        split_at = None
        index = 0
        while index + 1 < len(text):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">" and depth > 0:
                depth -= 1
            elif char == ":" and text[index + 1] == ":" and depth == 0:
                split_at = index
                index += 1
            index += 1
        if split_at is None:
            return None
        owner = text[:split_at].strip()
        member = text[split_at + 2 :].strip()
        if not owner or not re.fullmatch(r"[A-Za-z_]\w*", member):
            return None
        return owner, member

    def alias_template_error(
        self,
        declaration,
        signature,
        reason,
        *,
        parameter_pack=None,
        dependency_chain=(),
        required_work_items=None,
    ):
        return MetalAliasTemplateResolutionError(
            getattr(declaration, "name", str(signature).split("<", 1)[0]),
            signature,
            reason,
            source_location=(
                getattr(declaration, "template_source_location", None)
                or getattr(declaration, "source_location", None)
            ),
            dependency_chain=dependency_chain,
            parameter_pack=parameter_pack,
            limit=(
                self.max_template_specializations
                if required_work_items is not None
                else None
            ),
            limit_source=(
                self.template_specialization_limit_source
                if required_work_items is not None
                else None
            ),
            required_work_items=required_work_items,
        )

    def unresolved_alias_argument_parameter(
        self, argument, *, include_symbolic_parameters
    ):
        text = str(argument).strip()
        expansion = re.search(r"\b([A-Za-z_]\w*)\s*\.\.\.", text)
        if expansion:
            return expansion.group(1)
        if not include_symbolic_parameters:
            return None
        for name in self.alias_resolution_shadow_names():
            if re.search(rf"\b{re.escape(name)}\b", text):
                return name
        return None

    @staticmethod
    def replace_alias_template_identifiers(text, bindings):
        if not bindings:
            return str(text)
        return substitute_template_identifiers(text, bindings)

    def bind_alias_template_parameters(self, declaration, arguments, signature):
        parameters = list(getattr(declaration, "template_parameters", None) or [])
        defaults = dict(getattr(declaration, "template_parameter_defaults", None) or {})
        pack_indexes = [
            index
            for index, (kind, _name) in enumerate(parameters)
            if str(kind).endswith("...")
        ]
        if len(pack_indexes) > 1 or (
            pack_indexes and pack_indexes[0] != len(parameters) - 1
        ):
            pack_name = parameters[pack_indexes[0]][1] if pack_indexes else None
            raise self.alias_template_error(
                declaration,
                signature,
                "the parameter-pack declaration must be unique and final",
                parameter_pack=pack_name,
            )

        concrete_arguments = [str(argument).strip() for argument in arguments]
        for argument in concrete_arguments:
            unresolved = self.unresolved_alias_argument_parameter(
                argument,
                include_symbolic_parameters=bool(pack_indexes),
            )
            if unresolved is not None:
                raise self.alias_template_error(
                    declaration,
                    signature,
                    f"argument '{argument}' contains unresolved pack or template "
                    f"parameter '{unresolved}'",
                    parameter_pack=unresolved,
                )

        pack_index = pack_indexes[0] if pack_indexes else None
        fixed_count = pack_index if pack_index is not None else len(parameters)
        scalar_bindings = {}
        for index, (_kind, name) in enumerate(parameters[:fixed_count]):
            if index < len(concrete_arguments):
                scalar_bindings[name] = concrete_arguments[index]
                continue
            default = defaults.get(name)
            if default is None:
                raise self.alias_template_error(
                    declaration,
                    signature,
                    f"required template argument '{name}' is missing",
                )
            scalar_bindings[name] = self.replace_alias_template_identifiers(
                default, scalar_bindings
            ).strip()

        if pack_index is None and len(concrete_arguments) > len(parameters):
            raise self.alias_template_error(
                declaration,
                signature,
                "too many concrete template arguments were supplied",
            )

        pack_bindings = {}
        if pack_index is not None:
            pack_name = parameters[pack_index][1]
            pack_bindings[pack_name] = tuple(concrete_arguments[fixed_count:])
        elif len(concrete_arguments) < len(parameters):
            # Missing parameters without defaults were rejected above.
            pass
        return scalar_bindings, pack_bindings

    def expand_alias_template_target(
        self,
        declaration,
        signature,
        target,
        scalar_bindings,
        pack_bindings,
    ):
        text = self.replace_alias_template_identifiers(target, scalar_bindings).strip()

        def expand_fragment(fragment, active_packs):
            fragment = str(fragment).strip()
            direct_pack = next(
                (
                    name
                    for name in active_packs
                    if re.fullmatch(rf"{re.escape(name)}\s*\.\.\.", fragment)
                ),
                None,
            )
            if direct_pack is not None:
                values = active_packs[direct_pack]
                if len(values) == 1:
                    return values[0]
                raise self.alias_template_error(
                    declaration,
                    signature,
                    "the alias target expands a pack where exactly one type is "
                    "required",
                    parameter_pack=direct_pack,
                )

            angle_start = fragment.find("<")
            if angle_start != -1:
                angle_end = self.matching_type_angle(fragment, angle_start)
                if angle_end is None:
                    raise self.alias_template_error(
                        declaration,
                        signature,
                        "the alias target has an unterminated template argument list",
                    )
                prefix = fragment[:angle_start].strip()
                inner = fragment[angle_start + 1 : angle_end]
                suffix = fragment[angle_end + 1 :].strip()
                arguments = self.split_generic_arguments(inner) if inner.strip() else []
                expanded_arguments = []
                for argument in arguments:
                    stripped = argument.strip()
                    if stripped.endswith("..."):
                        pattern = stripped[:-3].strip()
                        referenced_packs = [
                            name
                            for name in active_packs
                            if re.search(rf"\b{re.escape(name)}\b", pattern)
                        ]
                        if not referenced_packs:
                            raise self.alias_template_error(
                                declaration,
                                signature,
                                f"pack expansion '{stripped}' has no bound pack",
                            )
                        lengths = {len(active_packs[name]) for name in referenced_packs}
                        if len(lengths) != 1:
                            raise self.alias_template_error(
                                declaration,
                                signature,
                                "simultaneously expanded packs have different lengths",
                                parameter_pack=referenced_packs[0],
                            )
                        remaining_packs = {
                            name: values
                            for name, values in active_packs.items()
                            if name not in referenced_packs
                        }
                        for pack_index in range(next(iter(lengths))):
                            item_bindings = {
                                name: active_packs[name][pack_index]
                                for name in referenced_packs
                            }
                            expanded_pattern = self.replace_alias_template_identifiers(
                                pattern, item_bindings
                            )
                            expanded_arguments.append(
                                expand_fragment(expanded_pattern, remaining_packs)
                            )
                        continue
                    expanded_arguments.append(expand_fragment(stripped, active_packs))
                rendered = f"{prefix}<{', '.join(expanded_arguments)}>"
                if suffix:
                    rendered += suffix
                return rendered

            expansion = re.search(r"\b([A-Za-z_]\w*)\s*\.\.\.", fragment)
            if expansion:
                raise self.alias_template_error(
                    declaration,
                    signature,
                    f"pack '{expansion.group(1)}' could not be expanded in the "
                    "alias target",
                    parameter_pack=expansion.group(1),
                )
            for pack_name in active_packs:
                if re.search(rf"\b{re.escape(pack_name)}\b", fragment):
                    raise self.alias_template_error(
                        declaration,
                        signature,
                        f"pack '{pack_name}' is referenced without expansion",
                        parameter_pack=pack_name,
                    )
            return fragment

        return expand_fragment(text, pack_bindings)

    @staticmethod
    def canonical_alias_argument(argument):
        return re.sub(r"\s+", "", str(argument).strip())

    def charge_alias_template_materialization(self, declaration, signature):
        next_count = self.materialized_template_specialization_count + 1
        if next_count > self.max_template_specializations:
            raise self.alias_template_error(
                declaration,
                signature,
                "the configured template materialization budget was exceeded "
                f"({next_count} required, limit {self.max_template_specializations} "
                f"from {self.template_specialization_limit_source})",
                required_work_items=next_count,
            )
        self.materialized_template_specialization_count = next_count

    def resolve_alias_template_declaration(self, declaration, arguments):
        qualified_name = self.normalize_qualified_type_name(
            getattr(declaration, "qualified_name", None) or declaration.name
        )
        normalized_arguments = tuple(
            self.canonical_alias_argument(argument) for argument in arguments
        )
        signature = f"{qualified_name}<{', '.join(arguments)}>"
        cache_key = (id(declaration), normalized_arguments)
        if cache_key in self.alias_template_cache:
            return self.alias_template_cache[cache_key]
        if cache_key in self.alias_template_resolution_stack:
            cycle_start = self.alias_template_resolution_stack.index(cache_key)
            cycle_keys = self.alias_template_resolution_stack[cycle_start:] + [
                cache_key
            ]
            signatures = [
                self.alias_template_resolution_signature(key) for key in cycle_keys
            ]
            raise self.alias_template_error(
                declaration,
                signature,
                "the alias-template dependency chain is recursive",
                dependency_chain=signatures,
            )

        scalar_bindings, pack_bindings = self.bind_alias_template_parameters(
            declaration, arguments, signature
        )
        if isinstance(declaration, CallableTypeAliasNode) or getattr(
            declaration, "is_function_type", False
        ):
            raise self.alias_template_error(
                declaration,
                signature,
                "callable alias-template targets cannot be represented as types",
            )

        self.charge_alias_template_materialization(declaration, signature)
        self.alias_template_resolution_stack.append(cache_key)
        previous_context = self.current_type_resolution_context
        self.current_type_resolution_context = declaration
        try:
            expanded = self.expand_alias_template_target(
                declaration,
                signature,
                declaration.alias_type,
                scalar_bindings,
                pack_bindings,
            )
            if getattr(declaration, "alias_type_globally_qualified", False):
                # Look the target up from the global namespace, then drop the ``::`` as the parser does.
                expanded = f"::{str(expanded).lstrip(':')}"
            resolved = self.materialize_alias_template_type(expanded, required=True)
            if str(resolved).startswith("::"):
                resolved = str(resolved)[2:]
        finally:
            self.current_type_resolution_context = previous_context
            self.alias_template_resolution_stack.pop()
        self.alias_template_cache[cache_key] = resolved
        return resolved

    def alias_template_resolution_signature(self, cache_key):
        declaration_id, arguments = cache_key
        declaration = next(
            (
                candidate
                for candidates in self.alias_template_declarations.values()
                for candidate in candidates
                if id(candidate) == declaration_id
            ),
            None,
        )
        name = (
            self.normalize_qualified_type_name(
                getattr(declaration, "qualified_name", None) or declaration.name
            )
            if declaration is not None
            else "<alias>"
        )
        return f"{name}<{', '.join(arguments)}>"

    def canonical_struct_template_argument(self, argument, *, context=None):
        text = str(argument).strip()
        instance = self.alias_template_instance_parts(text)
        if instance is not None:
            name, arguments = instance
            return (
                f"{name}<"
                + ",".join(
                    self.canonical_struct_template_argument(arg, context=context)
                    for arg in arguments
                )
                + ">"
            )
        binding = self.scalar_alias_binding(text, context=context)
        if binding is not None:
            text = " ".join([*binding[1], binding[0]])
        elif context is None:
            text = self.resolve_local_type_aliases(text)
        return self.canonical_alias_argument(text)

    def struct_templates_for_dependent_owner(self, owner):
        instance = self.alias_template_instance_parts(owner)
        owner_name, arguments = instance if instance is not None else (owner, [])
        canonical_arguments = tuple(
            self.canonical_struct_template_argument(argument) for argument in arguments
        )
        for tier in self.alias_lookup_name_tiers(owner_name):
            exact = []
            primary = []
            for candidate_name in tier:
                for (
                    qualified_name,
                    nodes,
                ) in self.alias_template_structs_by_qualified_name.items():
                    candidate_instance = self.alias_template_instance_parts(
                        qualified_name
                    )
                    if candidate_instance is not None:
                        specialized_name, specialized_arguments = candidate_instance
                        if specialized_name == candidate_name:
                            for node in nodes:
                                if not self.declaration_visible_at_current_offset(node):
                                    continue
                                previous_context = self.current_type_resolution_context
                                self.current_type_resolution_context = node
                                try:
                                    identity = tuple(
                                        self.canonical_struct_template_argument(
                                            argument, context=node
                                        )
                                        for argument in specialized_arguments
                                    )
                                finally:
                                    self.current_type_resolution_context = (
                                        previous_context
                                    )
                                if identity == canonical_arguments:
                                    exact.append(node)
                        continue
                    if qualified_name == candidate_name:
                        primary.extend(
                            node
                            for node in nodes
                            if self.declaration_visible_at_current_offset(node)
                        )
            matches = exact or primary
            if not matches:
                continue
            if not exact:
                # [temp.spec.partial.match]: the primary applies only when no partial specialization does.
                specialization = partial_specialization_for_owner(
                    self, owner, tier, arguments, primary
                )
                if specialization is not None:
                    node, bound_arguments = specialization
                    return (
                        [node],
                        bound_arguments,
                        not getattr(node, "template_parameters", None),
                    )
            return matches, arguments, bool(exact)
        return None

    def struct_member_alias_error(
        self,
        owner,
        member_name,
        reason,
        candidates,
        aliases=(),
        candidate_identities=(),
    ):
        representative = next(iter(aliases), None) or next(iter(candidates), None)
        candidate_locations = tuple(
            (
                getattr(aliases[index], "source_location", None)
                if index < len(aliases)
                else None
            )
            or getattr(candidate, "declaration_source_location", None)
            for index, candidate in enumerate(candidates)
        )
        return MetalStructAliasResolutionError(
            owner,
            member_name,
            reason,
            candidate_identities=(
                tuple(candidate_identities)
                or tuple(
                    self.normalize_qualified_type_name(
                        getattr(candidate, "qualified_name", None) or candidate.name
                    )
                    for candidate in candidates
                )
            ),
            candidate_locations=candidate_locations,
            source_location=(
                getattr(representative, "source_location", None)
                or getattr(representative, "declaration_source_location", None)
            ),
        )

    def struct_alias_integral_bindings(self, struct_node, scalar_bindings):
        members, duplicates = self.struct_compile_time_static_members(struct_node)
        values = {}
        for name, value in scalar_bindings.items():
            resolved = evaluate_literal_int_expression(value, values)
            if resolved is not None:
                values[name] = int(resolved)

        pending = dict(members)
        owner_names = self.struct_static_constant_declaration_names(struct_node)
        while pending:
            progress = False
            for name, member in list(pending.items()):
                member_type = self.normalized_metal_type(
                    self.resolve_type_alias(getattr(member, "vtype", None))
                )
                if member_type not in self.constexpr_integral_type_names():
                    continue
                value = evaluate_literal_int_expression(
                    getattr(member, "default_value", None), values
                )
                if value is None:
                    continue
                value = int(value)
                values[name] = value
                for owner_name in owner_names:
                    values[f"{owner_name}::{name}"] = value
                del pending[name]
                progress = True
            if not progress:
                break
        return values, duplicates, set(pending)

    def sibling_member_alias_target(
        self, struct_node, local_target, target, binding_context, member_stack
    ):
        # `typedef typename Frag_t::frag_type frag_type;` names a member of the
        # struct that the sibling alias `Frag_t` denotes; follow it when that
        # struct is concrete and visible, and keep `target` otherwise.
        head, _separator, rest = local_target.partition("::")
        if not rest or not re.fullmatch(r"[A-Za-z_]\w*", head):
            return target
        scalar_bindings, pack_bindings, signature = binding_context
        chained = self.struct_member_alias_candidate(
            struct_node,
            head,
            scalar_bindings,
            pack_bindings,
            signature,
            member_stack=member_stack,
        )
        if chained is None:
            return target
        owner = self.normalized_metal_type(self.resolve_type_alias(chained[0]))
        if owner not in self.struct_name_map:
            return target
        resolved = self.resolve_concrete_struct_member_aliases(owner, f"::{rest}")
        return target if resolved is None else resolved

    def struct_member_alias_candidate(
        self,
        struct_node,
        member_name,
        scalar_bindings,
        pack_bindings,
        signature,
        member_stack=(),
    ):
        aliases = [
            alias
            for alias in getattr(struct_node, "type_aliases", None) or []
            if getattr(alias, "name", None) == member_name
        ]
        if len(aliases) != 1:
            return None
        alias = aliases[0]
        if member_name in member_stack:
            chain = [
                f"{getattr(struct_node, 'qualified_name', struct_node.name)}::{name}"
                for name in (*member_stack, member_name)
            ]
            raise self.alias_template_error(
                alias,
                signature,
                "the dependent member-alias chain is recursive",
                dependency_chain=chain,
            )
        if self.is_template_alias_declaration(alias):
            raise self.alias_template_error(
                alias,
                signature,
                "a dependent member alias template requires explicit arguments",
            )

        target = self.expand_alias_template_target(
            alias,
            signature,
            alias.alias_type,
            scalar_bindings,
            pack_bindings,
        )
        local_target = self.normalize_qualified_type_name(target)
        if re.fullmatch(r"[A-Za-z_]\w*", local_target):
            chained = self.struct_member_alias_candidate(
                struct_node,
                local_target,
                scalar_bindings,
                pack_bindings,
                signature,
                member_stack=(*member_stack, member_name),
            )
            if chained is not None:
                target, nested_qualifiers, _nested_alias = chained
            else:
                nested_qualifiers = ()
        else:
            nested_qualifiers = ()
            target = self.sibling_member_alias_target(
                struct_node,
                local_target,
                target,
                (scalar_bindings, pack_bindings, signature),
                (*member_stack, member_name),
            )

        target += "".join(
            "[]" if size is None else f"[{self.format_array_extent(size)}]"
            for size in getattr(alias, "array_sizes", None) or []
        )
        integer_bindings, duplicate_constants, unresolved_constants = (
            self.struct_alias_integral_bindings(struct_node, scalar_bindings)
        )
        referenced_names = set(re.findall(r"\b[A-Za-z_]\w*\b", target))
        ambiguous_dependencies = referenced_names & duplicate_constants
        unresolved_dependencies = referenced_names & unresolved_constants
        if ambiguous_dependencies or unresolved_dependencies:
            dependencies = sorted(ambiguous_dependencies | unresolved_dependencies)
            raise MetalStructAliasResolutionError(
                self.normalize_qualified_type_name(
                    getattr(struct_node, "qualified_name", None) or struct_node.name
                ),
                member_name,
                "the alias target has unresolved integral dependencies: "
                + ", ".join(dependencies),
                candidate_identities=(str(alias.alias_type),),
                candidate_locations=(getattr(alias, "source_location", None),),
                source_location=getattr(alias, "source_location", None),
            )
        target = self.replace_alias_template_identifiers(target, integer_bindings)
        previous_context = self.current_type_resolution_context
        self.current_type_resolution_context = alias
        try:
            target = self.materialize_alias_template_type(target, required=True)
            target = self.resolve_type_alias(target)
        finally:
            self.current_type_resolution_context = previous_context
        qualifiers = tuple(
            qualifier
            for qualifier in self.metal_source_overload_type_qualifiers
            if qualifier
            in {
                *(str(value).lower() for value in nested_qualifiers),
                *(
                    str(value).lower()
                    for value in getattr(alias, "qualifiers", ()) or ()
                ),
            }
        )
        return target, qualifiers, alias

    def canonical_struct_member_alias_identity(self, target, qualifiers):
        mapped = self.map_type(target)
        canonical_type = re.sub(r"\s+", "", str(mapped))
        return tuple(qualifiers), canonical_type

    def resolve_dependent_alias_type(self, owner, member, required):
        contract = self.resolve_dependent_alias_contract(owner, member, required)
        return contract[0] if contract is not None else None

    def resolve_dependent_alias_contract(self, owner, member, required):
        resolved_owner = self.materialize_alias_template_type(owner, required=required)
        try:
            match = self.struct_templates_for_dependent_owner(resolved_owner)
        except UndecidedPartialSpecialization as undecided:
            if not required:
                return None
            raise MetalStructAliasResolutionError(
                resolved_owner,
                member,
                undecided.reason,
                source_location=getattr(
                    self.current_type_resolution_context, "source_location", None
                ),
            ) from undecided
        if match is None:
            return None
        struct_nodes, arguments, exact_specialization = match
        signature = f"{resolved_owner}::{member}"
        owner_identities = {
            self.normalize_qualified_type_name(
                getattr(struct_node, "qualified_name", None) or struct_node.name
            )
            for struct_node in struct_nodes
        }
        if len(owner_identities) != 1:
            raise self.struct_member_alias_error(
                resolved_owner,
                member,
                "multiple visible declarations have distinct qualified owners",
                struct_nodes,
            )
        qualified_owner = next(iter(owner_identities))
        alias_candidates = [
            [
                alias
                for alias in getattr(struct_node, "type_aliases", None) or []
                if getattr(alias, "name", None) == member
            ]
            for struct_node in struct_nodes
        ]
        if not any(alias_candidates):
            if not required:
                return None
            raise self.struct_member_alias_error(
                qualified_owner,
                member,
                "the owning declaration does not define a concrete alias",
                struct_nodes,
            )
        resolution_key = (qualified_owner, member)
        if resolution_key in self.struct_member_alias_resolution_stack:
            cycle = self.struct_member_alias_resolution_stack[
                self.struct_member_alias_resolution_stack.index(resolution_key) :
            ] + [resolution_key]
            raise MetalStructAliasResolutionError(
                qualified_owner,
                member,
                "the struct member-alias chain is recursive",
                candidate_identities=tuple(
                    f"{candidate_owner}::{candidate_member}"
                    for candidate_owner, candidate_member in cycle
                ),
                source_location=getattr(
                    struct_nodes[0], "declaration_source_location", None
                ),
            )

        self.struct_member_alias_resolution_stack.append(resolution_key)
        try:
            resolved_candidates = []
            for struct_node in struct_nodes:
                if exact_specialization:
                    scalar_bindings, pack_bindings = {}, {}
                else:
                    scalar_bindings, pack_bindings = (
                        self.bind_alias_template_parameters(
                            struct_node, arguments, signature
                        )
                    )
                candidate = self.struct_member_alias_candidate(
                    struct_node,
                    member,
                    scalar_bindings,
                    pack_bindings,
                    signature,
                )
                if candidate is None:
                    raise self.struct_member_alias_error(
                        qualified_owner,
                        member,
                        "a visible declaration does not define exactly one concrete alias",
                        struct_nodes,
                    )
                target, qualifiers, alias = candidate
                identity = self.canonical_struct_member_alias_identity(
                    target, qualifiers
                )
                resolved_candidates.append((identity, target, alias))
        finally:
            self.struct_member_alias_resolution_stack.pop()

        identities = {identity for identity, _target, _alias in resolved_candidates}
        if len(identities) != 1:
            aliases = [alias for _identity, _target, alias in resolved_candidates]
            candidate_identities = tuple(
                f"{qualified_owner}::{member}="
                f"{self.canonical_alias_argument(target)}"
                + (f" [{', '.join(identity[0])}]" if identity[0] else "")
                for identity, target, _alias in resolved_candidates
            )
            raise self.struct_member_alias_error(
                qualified_owner,
                member,
                "visible declarations define conflicting concrete alias targets",
                struct_nodes,
                aliases,
                candidate_identities=candidate_identities,
            )
        identity, target, _alias = resolved_candidates[0]
        return target, identity[0]

    def metal_standard_remove_cv_alias_visible(self, name):
        raw_name = self.normalize_qualified_type_name(name)
        if raw_name in {"metal::remove_cv_t", "::metal::remove_cv_t"}:
            return True
        if raw_name != "remove_cv_t":
            return False
        if raw_name in self.alias_resolution_shadow_names():
            return False

        for tier in self.alias_lookup_name_tiers(raw_name):
            for candidate in tier:
                declarations = [
                    *self.alias_template_declarations.get(candidate, []),
                    *self.alias_template_plain_declarations.get(candidate, []),
                    *self.alias_template_structs_by_qualified_name.get(candidate, []),
                ]
                if any(
                    self.declaration_visible_at_current_offset(declaration)
                    for declaration in declarations
                ):
                    return False

        namespace = self.alias_resolution_namespace()
        if namespace == "metal" or namespace.startswith("metal::"):
            return True
        return any(
            target == "metal" or target.startswith("metal::")
            for scope in self.namespace_lookup_scopes(namespace)
            for target in self.visible_using_namespace_targets(scope)
        )

    def materialize_alias_template_type(self, metal_type, *, required=False):
        original = str(metal_type or "").strip()
        if not original or ("<" not in original and ":" not in original):
            # Neither a dependent ``Owner::member`` type nor an alias-template instance.
            return original

        candidate = re.sub(r"^typename\s+", "", original).strip()
        suffix = ""
        while candidate.endswith(("*", "&")):
            suffix = candidate[-1] + suffix
            candidate = candidate[:-1].strip()

        dependent = self.dependent_alias_type_parts(candidate)
        if dependent is not None:
            owner, member = dependent
            if self.normalize_qualified_type_name(owner).lstrip(":") in getattr(
                self, "alias_template_namespaces", ()
            ):
                # ``ns::member`` names a member of a namespace, which is a type or a value, never a member of a
                # dependent type ([basic.lookup.qual]/1), so it is left for ordinary name lookup.
                return original
            resolved = resolve_member_alias_type(self, owner, member, required)
            if resolved is not None:
                return f"{resolved}{suffix}"
            if required and self.alias_template_resolution_stack:
                active_key = self.alias_template_resolution_stack[-1]
                active_declaration = next(
                    (
                        declaration
                        for declarations in self.alias_template_declarations.values()
                        for declaration in declarations
                        if id(declaration) == active_key[0]
                    ),
                    None,
                )
                active_signature = self.alias_template_resolution_signature(active_key)
                raise self.alias_template_error(
                    active_declaration,
                    active_signature,
                    f"dependent target '{candidate}' has no concrete backing "
                    "struct member alias",
                )
            return original

        instance = self.alias_template_instance_parts(candidate)
        if instance is None:
            return original
        name, arguments = instance
        declaration = self.alias_template_declaration_for_name(name)
        if declaration is not None:
            return f"{self.resolve_alias_template_declaration(declaration, arguments)}{suffix}"

        if (
            self.resolve_standard_remove_cv_aliases
            and self.metal_standard_remove_cv_alias_visible(name)
            and len(arguments) == 1
        ):
            argument = self.materialize_alias_template_type(
                arguments[0],
                required=required,
            )
            resolved = MetalPreprocessor()._remove_top_level_cv_qualifiers(argument)
            if resolved is not None:
                return f"{resolved}{suffix}"

        resolved_arguments = [
            self.materialize_alias_template_type(argument, required=required)
            for argument in arguments
        ]
        if resolved_arguments != arguments:
            return f"{name}<{', '.join(resolved_arguments)}>{suffix}"
        return original
