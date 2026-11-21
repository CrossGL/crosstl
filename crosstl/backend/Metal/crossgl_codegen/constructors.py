"""Constructor contracts, overload resolution, initializers and constructor factory functions."""

import re

from ..constant_constructor_folding import (
    fold_constant_constructor_call,
    in_constant_initialization,
)
from ..MetalAst import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    DesignatedInitializerNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    InitializerListNode,
    LambdaNode,
    MemberAccessNode,
    PostfixOpNode,
    ReturnNode,
    StructNode,
    VariableNode,
    VectorConstructorNode,
)
from ..preprocessor import MetalTemplateSpecializationError
from ..template_text import substitute_template_identifiers
from .support import MetalConstructorContractError


class MetalConstructorMixin:
    """Members of ``MetalToCrossGLConverter``: constructor contracts, overload resolution, initializers and constructor factory functions."""

    def prepare_metal_constructor_contracts(self, structs):
        self.constructor_contracts_by_owner = {}
        self.constructor_factory_names = {}
        self.pending_constructor_factories = []
        for struct_node in structs or []:
            if not isinstance(struct_node, StructNode):
                continue
            constructors = [
                constructor
                for constructor in getattr(struct_node, "constructors", []) or []
                if isinstance(constructor, ConstructorNode)
            ]
            if not constructors:
                continue
            owner = self.normalized_metal_type(
                self.resolve_type_alias(getattr(struct_node, "name", ""))
            )
            if owner:
                self.constructor_contracts_by_owner[owner] = (
                    struct_node,
                    constructors,
                )
            qualified = self.normalized_metal_type(
                getattr(struct_node, "qualified_name", "")
            )
            if qualified and qualified != owner:
                self.constructor_contracts_by_owner[qualified] = (
                    struct_node,
                    constructors,
                )

    def metal_constructor_contract(self, type_name):
        resolved_alias = self.resolve_type_alias(
            self.resolve_local_type_aliases(type_name)
        )
        conditional_alias = str(resolved_alias).strip()
        if conditional_alias.startswith("metal::"):
            conditional_alias = conditional_alias[len("metal::") :]
        conditional_type = self.resolve_conditional_type(
            conditional_alias,
            require_concrete=True,
        )
        if conditional_type is not None:
            resolved_alias = conditional_type
        if (
            self.metal_pointer_pointee_type_once(resolved_alias) is not None
            or self.reference_element_type(resolved_alias) is not None
            or self.metal_array_type_parts(resolved_alias) is not None
        ):
            return None
        resolved = self.normalized_metal_type(resolved_alias)
        contract = self.constructor_contracts_by_owner.get(resolved)
        if contract is None and "::" in resolved:
            contract = self.constructor_contracts_by_owner.get(
                resolved.rsplit("::", 1)[-1]
            )
        return contract

    @staticmethod
    def constructor_required_parameter_count(constructor):
        return sum(
            getattr(parameter, "default_value", None) is None
            for parameter in getattr(constructor, "params", []) or []
        )

    def has_declared_copy_or_move_constructor(self, owner, constructors):
        owner_identity = self.metal_source_overload_type_identity(owner)
        for constructor in constructors:
            parameters = list(getattr(constructor, "params", []) or [])
            if not parameters:
                continue
            if self.constructor_required_parameter_count(constructor) > 1:
                continue
            parameter_identity = self.metal_source_overload_type_identity(
                self.metal_declaration_expression_type(parameters[0])
            )
            if parameter_identity == owner_identity:
                return True
        return False

    def uses_implicit_copy_constructor(self, owner, arguments):
        contract = self.metal_constructor_contract(owner)
        if contract is None or len(arguments) != 1:
            return False
        _struct_node, constructors = contract
        if self.has_declared_copy_or_move_constructor(owner, constructors):
            return False
        argument_type = self.metal_source_overload_value_type(
            self.expression_metal_type(arguments[0])
        )
        return self.metal_source_overload_type_identity(
            argument_type
        ) == self.metal_source_overload_type_identity(owner)

    def constructor_candidate_signature(self, constructor, type_bindings=None):
        type_bindings = dict(type_bindings or {})
        if type_bindings:
            self.template_type_bindings.append(type_bindings)
        try:
            parameters = []
            for parameter in getattr(constructor, "params", []) or []:
                parameter_type = (
                    self.metal_source_overload_parameter_type(parameter) or "<unknown>"
                )
                qualifiers = self.metal_declaration_type_qualifiers(parameter)
                default = (
                    " = ..."
                    if getattr(parameter, "default_value", None) is not None
                    else ""
                )
                parameters.append(
                    f"{' '.join((*qualifiers, parameter_type))}{default}".strip()
                )
        finally:
            if type_bindings:
                self.template_type_bindings.pop()
        owner = getattr(constructor, "owner_name", None) or constructor.name
        template_parameters = getattr(constructor, "template_parameters", []) or []
        template_suffix = ""
        if template_parameters:
            template_suffix = (
                "<"
                + ", ".join(name for _kind, name in template_parameters if name)
                + ">"
            )
        receiver = self.constructor_receiver_address_space(constructor)
        receiver_suffix = f" {receiver}" if receiver != "thread" else ""
        return f"{owner}{template_suffix}({', '.join(parameters)}){receiver_suffix}"

    def constructor_receiver_address_space(self, constructor):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(constructor, "qualifiers", []) or []
        }
        selected = qualifiers & self.metal_source_overload_address_spaces
        return next(iter(selected), "thread")

    def current_constructor_receiver_address_space(self):
        if getattr(self.current_function, "is_metal_constructor_factory", False):
            return getattr(
                self.current_function,
                "constructor_receiver_address_space",
                "thread",
            )
        return "thread"

    def declaration_constructor_receiver_address_space(self, declaration):
        qualifiers = set(self.metal_declaration_type_qualifiers(declaration))
        selected = qualifiers & self.metal_source_overload_address_spaces
        return next(iter(selected), "thread")

    def infer_constructor_template_bindings(self, constructor, arguments):
        entries = [
            entry
            for entry in getattr(constructor, "template_parameters", []) or []
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[1]
        ]
        if not entries:
            return {}, {}, None
        if any(str(kind).endswith("...") for kind, _name in entries):
            return {}, {}, "variadic constructor templates are not representable"

        kinds = {name: kind for kind, name in entries}
        value_bindings = {}
        type_bindings = {}
        for parameter, argument in zip(constructor.params, arguments):
            actual = self.metal_source_overload_value_type(
                self.expression_metal_type(argument)
            )
            if actual is None:
                return {}, {}, "one or more argument types could not be inferred"
            pattern = self.metal_declaration_expression_type(parameter)
            if not self.infer_constructor_template_type_pattern(
                pattern,
                actual,
                kinds,
                type_bindings,
                value_bindings,
            ):
                return (
                    {},
                    {},
                    f"parameter type '{pattern}' does not match inferred type '{actual}'",
                )

        defaults = getattr(constructor, "template_parameter_defaults", {}) or {}
        all_bindings = {**type_bindings, **value_bindings}
        for kind, name in entries:
            target = value_bindings if kind == "value" else type_bindings
            if name in target:
                continue
            default = defaults.get(name)
            if default is None:
                return {}, {}, f"template parameter '{name}' could not be inferred"
            resolved = substitute_template_identifiers(default, all_bindings).strip()
            if not resolved:
                return {}, {}, f"template parameter '{name}' has an empty default"
            target[name] = resolved
            all_bindings[name] = resolved
        return value_bindings, type_bindings, None

    def infer_constructor_template_type_pattern(
        self,
        pattern,
        actual,
        kinds,
        type_bindings,
        value_bindings,
    ):
        pattern = self.substitute_template_type_text(pattern)
        pattern = self.resolve_local_type_aliases(pattern)
        pattern = re.sub(r"\s+", " ", str(pattern).strip())
        actual = re.sub(r"\s+", " ", str(actual).strip())

        def strip_value_qualifiers(text):
            text = re.sub(
                r"^(?:(?:const|volatile|thread|threadgroup|device|constant)\s+)+",
                "",
                text,
            )
            while text.endswith("&"):
                text = text[:-1].rstrip()
            return text

        pattern = strip_value_qualifiers(pattern)
        actual = strip_value_qualifiers(actual)
        if pattern in kinds:
            target = value_bindings if kinds[pattern] == "value" else type_bindings
            previous = target.get(pattern)
            if previous is None:
                target[pattern] = actual
                return True
            return self.normalized_metal_type(previous) == self.normalized_metal_type(
                actual
            )

        pattern_pointer_depth = len(pattern) - len(pattern.rstrip("*"))
        actual_pointer_depth = len(actual) - len(actual.rstrip("*"))
        if pattern_pointer_depth or actual_pointer_depth:
            if pattern_pointer_depth != actual_pointer_depth:
                return False
            return self.infer_constructor_template_type_pattern(
                pattern[:-pattern_pointer_depth].rstrip(),
                actual[:-actual_pointer_depth].rstrip(),
                kinds,
                type_bindings,
                value_bindings,
            )

        pattern_base, pattern_args = self.generic_type_parts(pattern)
        actual_base, actual_args = self.generic_type_parts(actual)
        if pattern_base and pattern_args:
            if not actual_base or len(pattern_args) != len(actual_args):
                return False
            if self.normalized_metal_type(pattern_base) != self.normalized_metal_type(
                actual_base
            ):
                return False
            return all(
                self.infer_constructor_template_type_pattern(
                    pattern_arg,
                    actual_arg,
                    kinds,
                    type_bindings,
                    value_bindings,
                )
                for pattern_arg, actual_arg in zip(pattern_args, actual_args)
            )

        unresolved = set(re.findall(r"\b[A-Za-z_]\w*\b", pattern)) & set(kinds)
        if unresolved:
            return False
        return self.metal_source_overload_type_identity(
            pattern
        ) == self.metal_source_overload_type_identity(actual)

    def resolve_metal_constructor(
        self,
        owner,
        arguments,
        source_location=None,
        receiver_address_space=None,
        allow_explicit=True,
    ):
        contract = self.metal_constructor_contract(owner)
        if contract is None:
            return None
        _struct_node, all_constructors = contract
        receiver_address_space = (
            receiver_address_space or self.current_constructor_receiver_address_space()
        )
        constructors = [
            constructor
            for constructor in all_constructors
            if self.constructor_receiver_address_space(constructor)
            == receiver_address_space
            and (
                allow_explicit
                or "explicit"
                not in {
                    str(qualifier).lower()
                    for qualifier in getattr(constructor, "qualifiers", []) or []
                }
            )
        ]
        if not constructors:
            reason = (
                "no non-explicit constructor is callable for contextual "
                f"conversion in receiver address space '{receiver_address_space}'"
                if not allow_explicit
                else "no constructor is callable for receiver address space "
                f"'{receiver_address_space}'"
            )
            raise MetalConstructorContractError(
                owner,
                (),
                [
                    self.constructor_candidate_signature(constructor)
                    for constructor in all_constructors
                ],
                reason,
                source_location,
            )
        argument_types = [
            self.metal_source_overload_value_type(self.expression_metal_type(argument))
            for argument in arguments
        ]
        diagnostic_types = [item or "<unknown>" for item in argument_types]
        signatures = [
            self.constructor_candidate_signature(constructor)
            for constructor in constructors
        ]
        arity_candidates = [
            constructor
            for constructor in constructors
            if self.constructor_required_parameter_count(constructor)
            <= len(arguments)
            <= len(getattr(constructor, "params", []) or [])
        ]
        if not arity_candidates:
            raise MetalConstructorContractError(
                owner,
                diagnostic_types,
                signatures,
                "no constructor accepts the supplied argument count",
                source_location,
            )

        if any(argument_type is None for argument_type in argument_types):
            if len(arity_candidates) == 1 and not getattr(
                arity_candidates[0], "template_parameters", None
            ):
                return arity_candidates[0], {}, {}
            raise MetalConstructorContractError(
                owner,
                diagnostic_types,
                [
                    self.constructor_candidate_signature(constructor)
                    for constructor in arity_candidates
                ],
                "one or more argument types could not be inferred and the "
                "arity-matched constructor is not unique",
                source_location,
            )

        ranked = []
        inference_reasons = []
        for constructor in arity_candidates:
            value_bindings, type_bindings, reason = (
                self.infer_constructor_template_bindings(constructor, arguments)
            )
            if reason is not None:
                inference_reasons.append(reason)
                continue
            if value_bindings:
                self.template_value_bindings.append(value_bindings)
            if type_bindings:
                self.template_type_bindings.append(type_bindings)
            try:
                ranks = []
                for argument, argument_type, parameter in zip(
                    arguments, argument_types, constructor.params
                ):
                    if argument_type is None:
                        break
                    rank = self.metal_source_overload_argument_match_rank(
                        argument, argument_type, parameter
                    )
                    if rank is None:
                        break
                    ranks.append(rank)
                else:
                    ranked.append(
                        (tuple(ranks), constructor, value_bindings, type_bindings)
                    )
            finally:
                if type_bindings:
                    self.template_type_bindings.pop()
                if value_bindings:
                    self.template_value_bindings.pop()

        if not ranked:
            reason = (
                inference_reasons[0]
                if inference_reasons
                else "no source-compatible constructor matches the inferred types"
            )
            raise MetalConstructorContractError(
                owner,
                diagnostic_types,
                signatures,
                reason,
                source_location,
            )

        def dominates(left, right):
            return all(a >= b for a, b in zip(left, right)) and any(
                a > b for a, b in zip(left, right)
            )

        winners = [
            entry
            for entry in ranked
            if not any(
                other is not entry and dominates(other[0], entry[0]) for other in ranked
            )
        ]
        if len(winners) > 1:
            non_template = [
                entry
                for entry in winners
                if not getattr(entry[1], "template_parameters", None)
            ]
            if len(non_template) == 1:
                winners = non_template
        if len(winners) != 1:
            raise MetalConstructorContractError(
                owner,
                diagnostic_types,
                [
                    self.constructor_candidate_signature(entry[1], entry[3])
                    for entry in winners
                ],
                "multiple source-compatible constructors remain after type matching",
                source_location,
            )
        return winners[0][1:]

    def reserve_constructor_factory(
        self, struct_node, constructor, value_bindings, type_bindings, source_location
    ):
        binding_key = tuple(
            (
                kind,
                (value_bindings[name] if kind == "value" else type_bindings[name]),
            )
            for kind, name in getattr(constructor, "template_parameters", []) or []
            if name
        )
        key = (id(constructor), binding_key)
        existing = self.constructor_factory_names.get(key)
        if existing is not None:
            return existing

        if binding_key:
            next_count = self.materialized_template_specialization_count + 1
            if next_count > self.max_template_specializations:
                requested = self.constructor_candidate_signature(
                    constructor, type_bindings
                )
                raise MetalTemplateSpecializationError(
                    "Metal template specialization limit exceeded while "
                    f"materializing constructor '{requested}'; {next_count} unique "
                    f"concrete signatures requested, limit "
                    f"{self.max_template_specializations} from "
                    f"{self.template_specialization_limit_source}.",
                    limit=self.max_template_specializations,
                    limit_source=self.template_specialization_limit_source,
                    unique_specialization_count=next_count,
                    requested_signature=requested,
                    source_location=source_location,
                )
            self.materialized_template_specialization_count = next_count

        constructors = self.metal_constructor_contract(struct_node.name)[1]
        ordinal = constructors.index(constructor) + 1
        suffix = "_".join(
            re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
            for _kind, value in binding_key
        )
        base = self.sanitize_identifier(
            f"crosstl_ctor_{self.map_struct_name(struct_node.name)}_{ordinal}"
            + (f"_{suffix}" if suffix else "")
        )
        name = base
        collision = 2
        while name in self.existing_function_names:
            name = f"{base}_{collision}"
            collision += 1
        self.existing_function_names.add(name)
        self.user_function_names.add(name)
        self.constructor_factory_names[key] = name
        self.pending_constructor_factories.append(
            (
                struct_node,
                constructor,
                dict(value_bindings),
                dict(type_bindings),
                name,
            )
        )
        return name

    def generate_explicit_constructor_call(
        self,
        owner,
        arguments,
        is_main=False,
        source_location=None,
        receiver_address_space=None,
        allow_explicit=True,
    ):
        if self.uses_implicit_copy_constructor(owner, arguments):
            return self.generate_expression(arguments[0], is_main)
        selected = self.resolve_metal_constructor(
            owner,
            arguments,
            source_location,
            receiver_address_space,
            allow_explicit,
        )
        if selected is None:
            return None
        constructor, value_bindings, type_bindings = selected
        struct_node = self.metal_constructor_contract(owner)[0]
        if getattr(constructor, "declaration_kind", "definition") in {
            "delete",
            "declaration",
        }:
            raise MetalConstructorContractError(
                owner,
                [
                    self.metal_source_overload_value_type(
                        self.expression_metal_type(argument)
                    )
                    or "<unknown>"
                    for argument in arguments
                ],
                [self.constructor_candidate_signature(constructor, type_bindings)],
                f"the selected constructor is {constructor.declaration_kind}",
                source_location or getattr(constructor, "source_location", None),
            )
        completed_arguments = list(arguments)
        for parameter in constructor.params[len(completed_arguments) :]:
            default = getattr(parameter, "default_value", None)
            if default is None:
                raise MetalConstructorContractError(
                    owner,
                    [],
                    [self.constructor_candidate_signature(constructor, type_bindings)],
                    "a selected omitted parameter has no default expression",
                    source_location,
                )
            completed_arguments.append(default)
        folded = None
        if in_constant_initialization(self):
            folded = self.fold_constant_constructor_initializer(
                struct_node,
                constructor,
                completed_arguments,
                value_bindings,
                type_bindings,
                is_main,
            )
        if folded is not None:
            return folded
        helper = self.reserve_constructor_factory(
            struct_node,
            constructor,
            value_bindings,
            type_bindings,
            source_location,
        )
        if value_bindings:
            self.template_value_bindings.append(value_bindings)
        if type_bindings:
            self.template_type_bindings.append(type_bindings)
        try:
            rendered = ", ".join(
                self.generate_expression(argument, is_main)
                for argument in completed_arguments
            )
        finally:
            if type_bindings:
                self.template_type_bindings.pop()
            if value_bindings:
                self.template_value_bindings.pop()
        return f"{helper}({rendered})"

    def fold_constant_constructor_initializer(
        self,
        struct_node,
        constructor,
        arguments,
        value_bindings,
        type_bindings,
        is_main=False,
    ):
        if value_bindings:
            self.template_value_bindings.append(value_bindings)
        if type_bindings:
            self.template_type_bindings.append(type_bindings)
        try:
            return fold_constant_constructor_call(
                self,
                struct_node,
                constructor,
                arguments,
                [self.generate_expression(argument, is_main) for argument in arguments],
            )
        finally:
            if type_bindings:
                self.template_type_bindings.pop()
            if value_bindings:
                self.template_value_bindings.pop()

    def generate_default_constructor_initializer(self, declaration, is_main=False):
        owner = self.metal_declaration_expression_type(declaration)
        contract = self.metal_constructor_contract(owner)
        if contract is None:
            return None
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(declaration, "qualifiers", []) or []
        }
        if "extern" in qualifiers:
            return None
        if self.variable_has_array_initializer_shape(declaration):
            raise MetalConstructorContractError(
                owner,
                (),
                [
                    self.constructor_candidate_signature(constructor)
                    for constructor in contract[1]
                ],
                "array element construction cannot be represented by one "
                "aggregate assignment",
                getattr(declaration, "source_location", None),
            )
        receiver_address_space = self.declaration_constructor_receiver_address_space(
            declaration
        )
        if receiver_address_space == "threadgroup":
            self.validate_threadgroup_default_construction(
                owner, getattr(declaration, "source_location", None)
            )
            return None
        return self.generate_explicit_constructor_call(
            owner,
            [],
            is_main,
            getattr(declaration, "source_location", None),
            receiver_address_space,
        )

    def validate_threadgroup_default_construction(self, owner, source_location=None):
        """Check a ``threadgroup`` declaration without lowering its construction.

        Metal folds the default constructor of a threadgroup variable into a
        static initializer of threadgroup memory, which the device does not
        apply: the storage holds unspecified values until a thread writes it.
        Lowering the construction to assignments would make every thread
        store into the whole allocation without a barrier, racing with the
        first writes of the other threads. The declaration therefore carries
        no runtime initialization; the type must still be default
        constructible in threadgroup memory, as Metal requires.
        """
        selected = self.resolve_metal_constructor(
            owner, [], source_location, "threadgroup"
        )
        if selected is None:
            return
        constructor, _value_bindings, type_bindings = selected
        kind = getattr(constructor, "declaration_kind", "definition")
        if kind in {"delete", "declaration"}:
            raise MetalConstructorContractError(
                owner,
                [],
                [self.constructor_candidate_signature(constructor, type_bindings)],
                f"the selected constructor is {kind}",
                source_location or getattr(constructor, "source_location", None),
            )

    def local_constructor_array_contract(self, declaration, source_location=None):
        element_type = self.effective_metal_variable_type(declaration)
        dimensions = list(getattr(declaration, "array_sizes", None) or [])
        if not dimensions:
            return None

        contract = self.metal_constructor_contract(element_type)
        if contract is None:
            return None
        candidates = [
            self.constructor_candidate_signature(constructor)
            for constructor in contract[1]
        ]
        location = source_location or getattr(declaration, "source_location", None)
        qualifiers = {
            str(qualifier).lower()
            for qualifier in self.effective_declaration_qualifiers(declaration)
        }
        if getattr(declaration, "is_const", False):
            qualifiers.add("const")
        if "extern" in qualifiers:
            return None
        unsupported_qualifiers = qualifiers & {
            "const",
            "constant",
            "constexpr",
            "constinit",
            "static",
        }
        if unsupported_qualifiers:
            rendered_qualifiers = ", ".join(sorted(unsupported_qualifiers))
            raise MetalConstructorContractError(
                element_type,
                (),
                candidates,
                "local constructor arrays with storage or immutability qualifier(s) "
                f"{rendered_qualifiers} cannot be lowered through runtime "
                "element assignments",
                location,
            )
        if len(dimensions) != 1:
            raise MetalConstructorContractError(
                element_type,
                (),
                candidates,
                "only one-dimensional local constructor arrays can be lowered",
                location,
            )

        extent = dimensions[0]
        if extent is None:
            raise MetalConstructorContractError(
                element_type,
                (),
                candidates,
                "an unsized local constructor array has no finite construction bound",
                location,
            )
        rendered_extent = self.format_array_extent(extent)
        extent_value = self.evaluate_concrete_array_extent(rendered_extent)
        if not isinstance(extent_value, int) or isinstance(extent_value, bool):
            raise MetalConstructorContractError(
                element_type,
                (),
                candidates,
                "the local constructor array extent is not a concrete integral "
                "constant",
                location,
            )
        if extent_value <= 0:
            raise MetalConstructorContractError(
                element_type,
                (),
                candidates,
                "the local constructor array extent must be positive",
                location,
            )
        return {
            "element_type": element_type,
            "extent_value": extent_value,
            "candidates": candidates,
            "receiver_address_space": (
                self.declaration_constructor_receiver_address_space(declaration)
            ),
        }

    def generate_constructor_array_element_initializer(
        self,
        element_type,
        expression,
        is_main,
        receiver_address_space,
    ):
        if isinstance(expression, DesignatedInitializerNode):
            contract = self.metal_constructor_contract(element_type)
            raise MetalConstructorContractError(
                element_type,
                (),
                [
                    self.constructor_candidate_signature(constructor)
                    for constructor in contract[1]
                ],
                "designated array elements cannot select an explicit constructor",
                getattr(expression, "source_location", None),
            )

        if isinstance(expression, InitializerListNode):
            return self.generate_initializer_value(
                expression,
                is_main,
                element_type,
                receiver_address_space=receiver_address_space,
                copy_initialize_lvalue=True,
            )

        expression_owner = None
        if isinstance(expression, VectorConstructorNode):
            expression_owner = expression.type_name
        elif isinstance(expression, FunctionCallNode):
            expression_owner = expression.name
        elif isinstance(expression, CastNode):
            expression_owner = expression.target_type
        if expression_owner is not None and (
            self.metal_source_overload_type_identity(expression_owner)
            == self.metal_source_overload_type_identity(element_type)
        ):
            return self.generate_initializer_value(
                expression,
                is_main,
                element_type,
                receiver_address_space=receiver_address_space,
                copy_initialize_lvalue=True,
            )

        expression_type = self.metal_source_overload_value_type(
            self.expression_metal_type(expression)
        )
        if expression_type is not None and self.metal_source_overload_type_identity(
            expression_type
        ) == self.metal_source_overload_type_identity(element_type):
            return self.generate_initializer_value(
                expression,
                is_main,
                element_type,
                receiver_address_space=receiver_address_space,
                copy_initialize_lvalue=True,
            )
        return self.generate_explicit_constructor_call(
            element_type,
            [expression],
            is_main,
            getattr(expression, "source_location", None),
            receiver_address_space,
        )

    def generate_local_constructor_array_declaration(
        self,
        declaration,
        initializer,
        is_main=False,
        indent=0,
    ):
        location = getattr(initializer, "source_location", None) or getattr(
            declaration, "source_location", None
        )
        info = self.local_constructor_array_contract(declaration, location)
        if info is None:
            return None
        if initializer is None:
            elements = []
        elif isinstance(initializer, InitializerListNode):
            elements = list(initializer.elements)
        else:
            raise MetalConstructorContractError(
                info["element_type"],
                (),
                info["candidates"],
                "a local constructor array requires a brace initializer list",
                location,
            )
        if len(elements) > info["extent_value"]:
            raise MetalConstructorContractError(
                info["element_type"],
                (),
                info["candidates"],
                "the initializer list contains more elements than the array extent",
                location,
            )

        declaration_text = self.format_decl(declaration, include_semantic=False)
        if info["receiver_address_space"] == "threadgroup":
            if elements:
                raise MetalConstructorContractError(
                    info["element_type"],
                    (),
                    info["candidates"],
                    "a threadgroup array initializer is never applied on the "
                    "device, and element assignments by every thread would race",
                    location,
                )
            self.validate_threadgroup_default_construction(
                info["element_type"], location
            )
            return f"{declaration_text};"
        name = self.render_identifier(declaration.name)
        lines = [f"{declaration_text};"]
        for index, element in enumerate(elements):
            value = self.generate_constructor_array_element_initializer(
                info["element_type"],
                element,
                is_main,
                info["receiver_address_space"],
            )
            lines.append(f"{name}[{index}] = {value};")

        if len(elements) < info["extent_value"]:
            index_name = self.reserve_generated_identifier("_crosstl_constructor_index")
            default_value = self.generate_explicit_constructor_call(
                info["element_type"],
                [],
                is_main,
                location,
                info["receiver_address_space"],
            )
            lines.extend(
                [
                    f"for (int {index_name} = {len(elements)}; "
                    f"{index_name} < {info['extent_value']}; {index_name}++) {{",
                    f"    {name}[{index_name}] = {default_value};",
                    "}",
                ]
            )

        indentation = "    " * indent
        return ("\n" + indentation).join(lines)

    def validate_global_constructor_initialization(self, declaration, *, initialized):
        owner = self.metal_declaration_expression_type(declaration)
        contract = self.metal_constructor_contract(owner)
        if contract is None:
            return
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(declaration, "qualifiers", []) or []
        }
        if "extern" in qualifiers and not initialized:
            return
        raise MetalConstructorContractError(
            owner,
            (),
            [
                self.constructor_candidate_signature(constructor)
                for constructor in contract[1]
            ],
            "global object construction cannot be represented by a "
            "target-independent runtime factory",
            getattr(declaration, "source_location", None),
        )

    def constructor_runtime_members(self, struct_node):
        return [
            member
            for member in getattr(struct_node, "members", []) or []
            if isinstance(member, VariableNode)
            and "static"
            not in {
                str(qualifier).lower()
                for qualifier in getattr(member, "qualifiers", []) or []
            }
        ]

    def constructor_member_array_contract(
        self,
        struct_node,
        constructor,
        member,
        initializer,
    ):
        dimensions = list(getattr(member, "array_sizes", None) or [])
        if not dimensions:
            return None

        owner = struct_node.name
        candidates = [self.constructor_candidate_signature(constructor)]
        location = (
            getattr(initializer, "source_location", None)
            if initializer is not None
            else getattr(member, "source_location", None)
        ) or getattr(constructor, "source_location", None)
        element_type = self.effective_metal_variable_type(member)
        qualifiers = {
            str(item).lower() for item in self.effective_declaration_qualifiers(member)
        }
        if "constant" in qualifiers:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' uses constant address-space storage",
                location,
            )
        if self.reference_element_type(element_type) is not None:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' has a reference element type",
                location,
            )
        if len(dimensions) != 1:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' is multidimensional",
                location,
            )

        extent = dimensions[0]
        if extent is None:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' has no finite extent",
                location,
            )
        rendered_extent = self.format_array_extent(extent)
        extent_value = self.evaluate_concrete_array_extent(rendered_extent)
        if not isinstance(extent_value, int) or isinstance(extent_value, bool):
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' extent is not a concrete "
                "integral constant",
                location,
            )
        if extent_value <= 0:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' extent must be positive",
                location,
            )

        initialized = initializer is not None
        if initializer is not None:
            elements = list(initializer.arguments)
        else:
            default_value = getattr(member, "default_value", None)
            if default_value is None:
                elements = []
                initialized = self.metal_constructor_contract(element_type) is not None
            elif isinstance(default_value, InitializerListNode):
                elements = list(default_value.elements)
                initialized = True
            else:
                raise MetalConstructorContractError(
                    owner,
                    (),
                    candidates,
                    f"member array '{member.name}' requires a brace initializer "
                    "list",
                    location,
                )

        if any(isinstance(element, DesignatedInitializerNode) for element in elements):
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' uses a designated array initializer",
                location,
            )
        if len(elements) > extent_value:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                f"member array '{member.name}' initializer contains more "
                "elements than its extent",
                location,
            )

        default_element = None
        if initialized and len(elements) < extent_value:
            default_element = self.constructor_member_array_default_element(
                owner,
                candidates,
                member.name,
                element_type,
                location,
            )
        return {
            "element_type": element_type,
            "elements": elements,
            "extent": extent_value,
            "initialized": initialized,
            "default_element": default_element,
            "location": location,
        }

    def constructor_member_array_default_element(
        self,
        owner,
        candidates,
        member_name,
        element_type,
        source_location,
    ):
        constructor_contract = self.metal_constructor_contract(element_type)
        if constructor_contract is not None:
            value = VectorConstructorNode(element_type, [])
        else:
            scalar_type = self.metal_scalar_arithmetic_type_info(element_type)
            vector_type = self.metal_vector_component_parts(element_type)
            matrix_type = self.metal_matrix_type_parts(element_type)
            wide_vector = self.wide_vector_type_info(element_type, source_location)
            if scalar_type is not None:
                value = (
                    False
                    if self.normalized_metal_type(self.resolve_type_alias(element_type))
                    == "bool"
                    else 0
                )
            elif vector_type is not None or matrix_type is not None or wide_vector:
                component_type = (
                    vector_type[0]
                    if vector_type is not None
                    else matrix_type[0] if matrix_type is not None else element_type
                )
                zero = (
                    False
                    if self.normalized_metal_type(
                        self.resolve_type_alias(component_type)
                    )
                    == "bool"
                    else 0
                )
                value = VectorConstructorNode(element_type, [zero])
            else:
                raise MetalConstructorContractError(
                    owner,
                    (),
                    candidates,
                    f"member array '{member_name}' has omitted elements whose "
                    f"value initialization for '{element_type}' is not "
                    "representable",
                    source_location,
                )
        if isinstance(value, VectorConstructorNode):
            value.source_location = source_location
        return value

    def validate_constructor_factory(self, struct_node, constructor):
        owner = struct_node.name
        candidates = [self.constructor_candidate_signature(constructor)]
        location = getattr(constructor, "source_location", None)
        if getattr(struct_node, "aggregate_kind", "struct") == "union":
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                "union constructors are not representable",
                location,
            )
        if getattr(struct_node, "base_types", None):
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                "base-class construction is not representable by a data-only target struct",
                location,
            )
        if constructor.body is None and constructor.declaration_kind != "default":
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                "the selected constructor has no executable definition",
                location,
            )
        if constructor.declaration_kind == "default" and constructor.params:
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                "defaulted copy/move construction requires field-wise source "
                "object semantics",
                location,
            )

        members = {
            member.name: member
            for member in self.constructor_runtime_members(struct_node)
        }
        initializers_by_member = {}
        for initializer in constructor.initializers:
            target = str(initializer.target).strip()
            if target not in members:
                raise MetalConstructorContractError(
                    owner,
                    (),
                    candidates,
                    f"initializer target '{target}' is a base or delegating constructor",
                    getattr(initializer, "source_location", None) or location,
                )
            if target in initializers_by_member:
                raise MetalConstructorContractError(
                    owner,
                    (),
                    candidates,
                    f"member '{target}' is initialized more than once",
                    getattr(initializer, "source_location", None) or location,
                )
            initializers_by_member[target] = initializer

        for member in members.values():
            member_type = self.effective_metal_variable_type(member)
            requires_assignment = (
                member.name in initializers_by_member
                or getattr(member, "default_value", None) is not None
                or self.metal_constructor_contract(member_type) is not None
            )
            if not requires_assignment:
                continue
            if getattr(member, "array_sizes", None):
                self.constructor_member_array_contract(
                    struct_node,
                    constructor,
                    member,
                    initializers_by_member.get(member.name),
                )
                continue
            qualifiers = {
                str(item).lower() for item in getattr(member, "qualifiers", []) or []
            }
            # On a pointer member, ``constant`` names the pointee's address
            # space; the member itself is an assignable pointer value.
            if (
                "constant" in qualifiers
                and self.pointer_element_type(member_type) is None
            ) or self.reference_element_type(member_type) is not None:
                raise MetalConstructorContractError(
                    owner,
                    (),
                    candidates,
                    f"member '{member.name}' cannot be initialized by portable "
                    "assignment",
                    getattr(
                        initializers_by_member.get(member.name),
                        "source_location",
                        None,
                    )
                    or getattr(member, "source_location", None)
                    or location,
                )

        def invalid_return(node):
            if isinstance(node, LambdaNode):
                return False
            if isinstance(node, ReturnNode) and node.value is not None:
                return True
            return any(invalid_return(child) for child in self.iter_ast_children(node))

        if any(invalid_return(statement) for statement in constructor.body or []):
            raise MetalConstructorContractError(
                owner,
                (),
                candidates,
                "a constructor body cannot return a value",
                location,
            )

    def constructor_scope_identifier_names(self, struct_node, constructor):
        used = {getattr(parameter, "name", None) for parameter in constructor.params}
        used.update(
            member.name for member in self.constructor_runtime_members(struct_node)
        )

        def collect_local_names(node):
            if isinstance(node, VariableNode) and getattr(node, "vtype", None):
                used.add(getattr(node, "name", None))
            for child in self.iter_ast_children(node):
                collect_local_names(child)

        collect_local_names(constructor.body or [])
        used.discard(None)
        return used

    def reserve_constructor_scope_identifier(self, used, base):
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        return name

    def constructor_result_variable_name(self, struct_node, constructor):
        used = self.constructor_scope_identifier_names(struct_node, constructor)
        base = "crosstl_ctor_value"
        return self.reserve_constructor_scope_identifier(used, base)

    def build_constructor_factory_function(self, struct_node, constructor, name):
        self.validate_constructor_factory(struct_node, constructor)
        result_name = self.constructor_result_variable_name(struct_node, constructor)
        used_identifiers = self.constructor_scope_identifier_names(
            struct_node, constructor
        )
        used_identifiers.add(result_name)
        result = VariableNode(struct_node.name, result_name)
        result.is_metal_constructor_storage = True
        statements = [result]
        initializer_map = {
            str(initializer.target).strip(): initializer
            for initializer in constructor.initializers
        }
        for member in self.constructor_runtime_members(struct_node):
            initializer = initializer_map.get(member.name)
            array_contract = self.constructor_member_array_contract(
                struct_node,
                constructor,
                member,
                initializer,
            )
            if array_contract is not None:
                if not array_contract["initialized"]:
                    continue
                member_target = MemberAccessNode(
                    VariableNode("", result_name), member.name
                )
                for index, element in enumerate(array_contract["elements"]):
                    target = ArrayAccessNode(member_target, index)
                    value = self.constructor_member_array_element(
                        array_contract["element_type"], member.vtype, element
                    )
                    assignment = AssignmentNode(target, value)
                    assignment.source_location = (
                        getattr(element, "source_location", None)
                        or array_contract["location"]
                    )
                    statements.append(assignment)

                default_element = array_contract["default_element"]
                if default_element is not None:
                    index_name = self.reserve_constructor_scope_identifier(
                        used_identifiers,
                        f"{result_name}_{self.sanitize_identifier(member.name)}_index",
                    )
                    index = VariableNode("", index_name)
                    target = ArrayAccessNode(member_target, index)
                    assignment = AssignmentNode(target, default_element)
                    assignment.source_location = array_contract["location"]
                    loop = ForNode(
                        AssignmentNode(
                            VariableNode("int", index_name),
                            len(array_contract["elements"]),
                        ),
                        BinaryOpNode(index, "<", array_contract["extent"]),
                        PostfixOpNode(index, "++"),
                        [assignment],
                    )
                    loop.source_location = array_contract["location"]
                    statements.append(loop)
                continue
            if initializer is not None:
                member_type = self.resolve_type_alias(
                    self.metal_declaration_expression_type(member)
                )
                if (
                    self.pointer_element_type(member_type) is not None
                    and len(initializer.arguments) == 1
                ):
                    value = initializer.arguments[0]
                else:
                    value = VectorConstructorNode(
                        member.vtype, list(initializer.arguments)
                    )
                value.source_location = initializer.source_location
            else:
                value = getattr(member, "default_value", None)
                if value is None and self.metal_constructor_contract(member.vtype):
                    value = VectorConstructorNode(member.vtype, [])
                    value.source_location = getattr(member, "source_location", None)
            if value is None:
                continue
            target = MemberAccessNode(VariableNode("", result_name), member.name)
            assignment = AssignmentNode(target, value)
            assignment.source_location = (
                getattr(initializer, "source_location", None)
                if initializer is not None
                else getattr(member, "source_location", None)
            )
            statements.append(assignment)
        statements.extend(list(constructor.body or []))
        statements.append(ReturnNode(VariableNode("", result_name)))
        factory = FunctionNode(
            struct_node.name,
            name,
            constructor.params,
            statements,
        )
        factory.is_metal_constructor_factory = True
        factory.constructor_owner = struct_node.name
        factory.constructor_result_name = result_name
        factory.constructor_receiver_address_space = (
            self.constructor_receiver_address_space(constructor)
        )
        factory.constructor_member_types = {
            member.name: (
                self.struct_member_types.get(struct_node.name, {}).get(member.name)
                or self.effective_metal_variable_type(member)
            )
            for member in self.constructor_runtime_members(struct_node)
        }
        factory.source_location = getattr(constructor, "source_location", None)
        return factory

    def generate_pending_constructor_factories(self):
        generated = []
        index = 0
        while index < len(self.pending_constructor_factories):
            (
                struct_node,
                constructor,
                value_bindings,
                type_bindings,
                name,
            ) = self.pending_constructor_factories[index]
            index += 1
            factory = self.build_constructor_factory_function(
                struct_node, constructor, name
            )
            generated.append(
                self.generate_function(
                    factory,
                    template_value_bindings=value_bindings,
                    template_type_bindings=type_bindings,
                    output_name=name,
                )
            )
        return "".join(generated)
