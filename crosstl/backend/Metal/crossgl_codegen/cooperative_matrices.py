"""Cooperative matrix types and fragments."""

import re

from ....translator.cooperative_matrix import get_cooperative_matrix_fragment_mapping
from ..MetalAst import (
    ArrayAccessNode,
    CastNode,
    MethodCallNode,
)
from ..opaque_matrix_contracts import complete_opaque_matrix_contracts
from .support import MetalCooperativeMatrixFragmentLoweringError


class MetalCooperativeMatrixMixin:
    """Members of ``MetalToCrossGLConverter``: cooperative matrix types and fragments."""

    def metal_cooperative_matrix_type_parts(self, metal_type):
        candidate = self.metal_source_overload_value_type(metal_type)
        if candidate is None:
            return None
        candidate = self.substitute_template_value_text(candidate)
        candidate = self.normalized_metal_type(candidate)
        base_name, generic_args = self.generic_type_parts(candidate)
        if not base_name or base_name.rsplit("::", 1)[-1] != "simdgroup_matrix":
            return None
        return generic_args if len(generic_args) == 3 else None

    def map_metal_cooperative_matrix_type(
        self,
        metal_type,
        *,
        fragment_layout=None,
        subgroup_size=None,
        elements_per_lane=None,
        fragment_provenance=None,
        fragment_mapping=None,
        fragment_mapping_provenance=None,
    ):
        generic_args = self.metal_cooperative_matrix_type_parts(metal_type)
        if generic_args is None:
            return None
        element_type, rows, cols = generic_args
        mapped_args = [
            self.map_type(element_type),
            self.map_generic_type_argument(rows),
            self.map_generic_type_argument(cols),
            "subgroup",
            "unspecified",
            "unspecified",
        ]
        fragment_contract = [
            fragment_layout or "unspecified",
            str(subgroup_size) if subgroup_size is not None else "unspecified",
            (
                str(elements_per_lane)
                if elements_per_lane is not None
                else "unspecified"
            ),
            fragment_provenance or "unspecified",
            fragment_mapping or "unspecified",
            fragment_mapping_provenance or "unspecified",
        ]
        while fragment_contract and fragment_contract[-1] == "unspecified":
            fragment_contract.pop()
        return f"CooperativeMatrix<{','.join([*mapped_args, *fragment_contract])}>"

    def register_cooperative_matrix_fragment_type_contract(
        self, metal_type, complete_type
    ):
        legacy_type = self.map_metal_cooperative_matrix_type(metal_type)
        if legacy_type is None:
            return None
        previous_type = self.cooperative_matrix_fragment_type_replacements.get(
            legacy_type
        )
        if previous_type is not None and previous_type != complete_type:
            return previous_type
        self.cooperative_matrix_fragment_type_replacements[legacy_type] = complete_type
        return None

    def propagate_cooperative_matrix_fragment_type_contracts(self, code):
        replacements = self.cooperative_matrix_fragment_type_replacements
        complete_opaque_matrix_contracts(
            code,
            replacements,
            self.cooperative_matrix_fragment_mapping,
            self.cooperative_matrix_fragment_mapping_provenance,
        )
        if not replacements:
            return code
        legacy_types = sorted(replacements, key=lambda value: (-len(value), value))
        pattern = re.compile("|".join(re.escape(value) for value in legacy_types))
        return pattern.sub(lambda match: replacements[match.group(0)], code)

    def is_metal_cooperative_matrix_expression(self, expression):
        return (
            self.metal_cooperative_matrix_type_parts(
                self.expression_metal_type(expression)
            )
            is not None
        )

    def metal_cooperative_matrix_element_access(self, expression):
        if not isinstance(expression, ArrayAccessNode):
            return None
        thread_elements = expression.array
        if not (
            isinstance(thread_elements, MethodCallNode)
            and thread_elements.method == "thread_elements"
            and not thread_elements.args
            and self.is_metal_cooperative_matrix_expression(thread_elements.object)
        ):
            return None
        return thread_elements.object, expression.index

    # Metal executes SIMD-group matrices on 32-lane SIMD-groups; each lane's
    # thread_elements() is a vec<T, rows * columns / 32>.
    METAL_COOPERATIVE_MATRIX_SIMD_WIDTH = 32

    def register_cooperative_matrix_element_contract(self, matrix_expression):
        """Complete the lane-fragment contract an indexed thread_elements() uses.

        ``m.thread_elements()[i]`` reads one element of the same whole-lane
        view that ``reinterpret_cast<thread vec<T, N>&>(m.thread_elements())``
        transfers, so with a configured lane-coordinate mapping the matrix
        type takes the same ``metal_thread_elements`` contract. Without a
        mapping the type keeps its unspecified fragment contract.
        """
        if self.cooperative_matrix_fragment_mapping is None:
            return
        matrix_type = self.metal_source_overload_value_type(
            self.expression_metal_type(matrix_expression)
        )
        if matrix_type is None:
            return
        matrix_type = self.substitute_template_value_text(matrix_type)
        parts = self.metal_cooperative_matrix_type_parts(matrix_type)
        if parts is None:
            return
        _element_type, rows_text, cols_text = parts
        rows = self.concrete_cooperative_matrix_dimension(rows_text)
        cols = self.concrete_cooperative_matrix_dimension(cols_text)
        width = self.METAL_COOPERATIVE_MATRIX_SIMD_WIDTH
        if rows is None or cols is None or (rows * cols) % width:
            return
        elements_per_lane = rows * cols // width
        if (
            get_cooperative_matrix_fragment_mapping(
                self.cooperative_matrix_fragment_mapping,
                rows,
                cols,
                width,
                elements_per_lane,
            )
            is None
        ):
            return
        contract_key = self.normalized_metal_type(matrix_type)
        contract = (width, elements_per_lane)
        previous = self.cooperative_matrix_fragment_type_contracts.get(contract_key)
        if previous is not None and previous != contract:
            return
        complete_type = self.map_metal_cooperative_matrix_type(
            matrix_type,
            fragment_layout="metal_thread_elements",
            subgroup_size=width,
            elements_per_lane=elements_per_lane,
            fragment_provenance="metal_thread_elements_reference_view",
            fragment_mapping=self.cooperative_matrix_fragment_mapping,
            fragment_mapping_provenance=(
                self.cooperative_matrix_fragment_mapping_provenance
            ),
        )
        if complete_type is None:
            return
        self.cooperative_matrix_fragment_type_contracts[contract_key] = contract
        self.register_cooperative_matrix_fragment_type_contract(
            matrix_type, complete_type
        )

    def metal_cooperative_matrix_fragment_cast(self, expression):
        if not isinstance(expression, CastNode):
            return None
        thread_elements = expression.expression
        if not (
            isinstance(thread_elements, MethodCallNode)
            and thread_elements.method == "thread_elements"
            and not thread_elements.args
        ):
            return None
        matrix_expression = thread_elements.object
        matrix_type = self.expression_metal_type(matrix_expression)
        if self.metal_cooperative_matrix_type_parts(matrix_type) is None:
            return None
        return matrix_expression, matrix_type

    def cooperative_matrix_fragment_cast_qualifiers(self, expression):
        qualifiers = [
            str(qualifier).lower()
            for qualifier in getattr(expression, "qualifiers", ()) or ()
        ]
        target_type = str(getattr(expression, "target_type", "") or "").strip()
        while target_type.endswith(("*", "&")):
            target_type = target_type[:-1].strip()

        seen = set()
        while target_type in self.type_aliases and target_type not in seen:
            seen.add(target_type)
            qualifiers.extend(
                str(qualifier).lower()
                for qualifier in self.type_alias_qualifiers.get(target_type, ())
            )
            target_type = str(self.type_aliases[target_type]).strip()
            while target_type.endswith(("*", "&")):
                target_type = target_type[:-1].strip()
        return tuple(dict.fromkeys(qualifiers))

    def cooperative_matrix_fragment_source_location(
        self, expression, matrix_expression
    ):
        return (
            getattr(expression, "source_location", None)
            or getattr(expression.expression, "source_location", None)
            or getattr(matrix_expression, "source_location", None)
            or getattr(self.current_function, "source_location", None)
            or getattr(self.current_function, "declaration_source_location", None)
        )

    def concrete_cooperative_matrix_dimension(self, expression):
        rendered = self.substitute_template_value_text(str(expression)).strip()
        value = self.evaluate_value_template_constant_expression(rendered)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    def cooperative_matrix_fragment_contract(
        self,
        expression,
        direction,
        *,
        assigned_value=None,
        assignment_operator="=",
    ):
        cast = self.metal_cooperative_matrix_fragment_cast(expression)
        if cast is None:
            return None

        matrix_expression, source_matrix_type = cast
        source_location = self.cooperative_matrix_fragment_source_location(
            expression, matrix_expression
        )
        qualifiers = self.cooperative_matrix_fragment_cast_qualifiers(expression)
        resolved_fragment_type = self.resolve_local_type_aliases(expression.target_type)
        resolved_fragment_type = self.substitute_template_type_text(
            resolved_fragment_type
        )
        resolved_fragment_type = self.resolve_type_alias(resolved_fragment_type)
        resolved_fragment_type = self.substitute_template_value_text(
            resolved_fragment_type
        )
        resolved_fragment_type = re.sub(
            r"\s+", " ", str(resolved_fragment_type).strip()
        )
        matrix_type = self.metal_source_overload_value_type(source_matrix_type)
        if matrix_type is not None:
            matrix_type = self.substitute_template_value_text(matrix_type)

        def reject(reason):
            raise MetalCooperativeMatrixFragmentLoweringError(
                matrix_type or source_matrix_type or "<unknown>",
                resolved_fragment_type or expression.target_type or "<unknown>",
                direction,
                reason,
                source_location,
                qualifiers,
            )

        if not resolved_fragment_type.endswith("&"):
            reject("the cast target is not an lvalue reference")
        fragment_value_type = resolved_fragment_type[:-1].strip()
        if fragment_value_type.endswith(("*", "&")):
            reject("pointer and nested-reference fragment aliases are not supported")

        address_spaces = set(qualifiers) & self.metal_source_overload_address_spaces
        if address_spaces and address_spaces != {"thread"}:
            reject("thread_elements storage requires a thread-address-space reference")
        unsupported_qualifiers = set(qualifiers) - {"const", "thread"}
        if unsupported_qualifiers:
            reject(
                "the cast uses unsupported fragment qualifiers: "
                + ", ".join(sorted(unsupported_qualifiers))
            )
        if direction == "write" and "const" in qualifiers:
            reject("a const fragment reference cannot receive a whole-fragment write")
        if direction == "write" and assignment_operator != "=":
            reject(
                f"assignment operator '{assignment_operator}' has no whole-fragment "
                "transfer contract"
            )

        matrix_parts = self.metal_cooperative_matrix_type_parts(matrix_type)
        if matrix_parts is None:
            reject("the matrix type is not a concrete simdgroup_matrix specialization")
        matrix_element_type, rows_text, cols_text = matrix_parts
        rows = self.concrete_cooperative_matrix_dimension(rows_text)
        cols = self.concrete_cooperative_matrix_dimension(cols_text)
        if rows is None or cols is None:
            reject(
                "the cooperative-matrix row or column count remains dependent "
                f"('{rows_text}' x '{cols_text}')"
            )

        fragment_parts = self.metal_vector_component_parts(fragment_value_type)
        if fragment_parts is None:
            reject("the fragment reference does not name a Metal vector type")
        fragment_element_type, fragment_width, fragment_width_text = fragment_parts
        if fragment_width is None:
            reject(f"the fragment width remains dependent ('{fragment_width_text}')")
        if fragment_width < 2:
            reject("the fragment vector width must be at least two lanes")

        element_count = rows * cols
        if element_count % fragment_width:
            reject(
                f"matrix shape {rows}x{cols} cannot be distributed into "
                f"{fragment_width} elements per lane with an integral SIMD-group "
                "width"
            )
        required_subgroup_size = element_count // fragment_width
        if (
            self.cooperative_matrix_fragment_mapping is not None
            and get_cooperative_matrix_fragment_mapping(
                self.cooperative_matrix_fragment_mapping,
                rows,
                cols,
                required_subgroup_size,
                fragment_width,
            )
            is None
        ):
            reject(
                "configured cooperative-matrix fragment mapping "
                f"'{self.cooperative_matrix_fragment_mapping}' has no registered "
                f"contract for {rows}x{cols}, "
                f"subgroup_size={required_subgroup_size}, and "
                f"elements_per_lane={fragment_width}"
            )
        matrix_contract_key = self.normalized_metal_type(matrix_type)
        source_contract = (required_subgroup_size, fragment_width)
        previous_contract = self.cooperative_matrix_fragment_type_contracts.get(
            matrix_contract_key
        )
        if previous_contract is not None and previous_contract != source_contract:
            reject(
                "the same matrix specialization has conflicting whole-fragment "
                "contracts: "
                f"subgroup_size={previous_contract[0]}, "
                f"elements_per_lane={previous_contract[1]} and "
                f"subgroup_size={required_subgroup_size}, "
                f"elements_per_lane={fragment_width}"
            )
        self.cooperative_matrix_fragment_type_contracts[matrix_contract_key] = (
            source_contract
        )

        matrix_component = self.metal_source_overload_type_identity(matrix_element_type)
        fragment_component = self.metal_source_overload_type_identity(
            fragment_element_type
        )
        if matrix_component is None or matrix_component[0] != "scalar":
            reject(
                f"matrix component type '{matrix_element_type}' is not a concrete "
                "scalar"
            )
        if fragment_component is None or fragment_component[0] != "scalar":
            reject(
                f"fragment component type '{fragment_element_type}' is not a "
                "concrete scalar"
            )
        if matrix_component != fragment_component:
            reject(
                f"fragment component type '{fragment_element_type}' does not match "
                f"matrix component type '{matrix_element_type}'"
            )

        fragment_identity = self.metal_source_overload_type_identity(
            fragment_value_type
        )
        if direction == "write":
            assigned_type = self.expression_metal_type(assigned_value)
            assigned_identity = self.metal_source_overload_type_identity(assigned_type)
            if assigned_identity is None:
                reject("the assigned fragment expression type cannot be inferred")
            if assigned_identity != fragment_identity:
                reject(
                    f"assigned fragment type '{assigned_type}' does not match "
                    f"cast fragment type '{fragment_value_type}'"
                )
            matrix_qualifiers = set(
                self.expression_metal_type_qualifiers(matrix_expression)
            )
            if matrix_qualifiers & {"const", "constant"}:
                reject("the matrix expression is read-only")

        mapped_matrix_type = self.map_metal_cooperative_matrix_type(
            matrix_type,
            fragment_layout="metal_thread_elements",
            subgroup_size=required_subgroup_size,
            elements_per_lane=fragment_width,
            fragment_provenance="metal_thread_elements_reference_view",
            fragment_mapping=self.cooperative_matrix_fragment_mapping,
            fragment_mapping_provenance=(
                self.cooperative_matrix_fragment_mapping_provenance
            ),
        )
        if mapped_matrix_type is None:
            reject("the matrix type cannot retain its inferred fragment contract")
        conflicting_mapped_type = (
            self.register_cooperative_matrix_fragment_type_contract(
                matrix_type, mapped_matrix_type
            )
        )
        if conflicting_mapped_type is not None:
            reject(
                "the same matrix specialization has conflicting generated "
                "whole-fragment type contracts"
            )
        mapped_fragment_type = self.map_type(fragment_value_type)
        return {
            "matrix_expression": matrix_expression,
            "matrix_type": matrix_type,
            "matrix_element_type": self.map_type(matrix_element_type),
            "fragment_type": fragment_value_type,
            "mapped_matrix_type": mapped_matrix_type,
            "mapped_fragment_type": mapped_fragment_type,
            "fragment_width": fragment_width,
            "subgroup_size": required_subgroup_size,
            "rows": rows,
            "cols": cols,
            "wide_fragment": fragment_width > 4,
            "direction": direction,
        }

    def cooperative_matrix_fragment_helper_name(self, contract):
        key = (
            contract["direction"],
            contract["mapped_matrix_type"],
            contract["mapped_fragment_type"],
            contract["fragment_width"],
        )
        existing = self.cooperative_matrix_fragment_helpers.get(key)
        if existing is not None:
            return existing["name"]

        component = re.sub(
            r"[^A-Za-z0-9_]+", "_", contract["matrix_element_type"]
        ).strip("_")
        base_name = self.sanitize_identifier(
            "_crosstl_metal_cooperative_matrix_fragment_"
            f"{contract['direction']}_{component}_{contract['rows']}_"
            f"{contract['cols']}_{contract['fragment_width']}"
        )
        reserved_names = {
            self.sanitize_identifier(name) for name in self.user_function_names
        } | self.wide_vector_reserved_names
        name = base_name
        suffix = 2
        while (
            name in reserved_names
            or name in self.cooperative_matrix_fragment_helper_names
        ):
            name = f"{base_name}_{suffix}"
            suffix += 1

        helper = dict(contract)
        helper["name"] = name
        self.cooperative_matrix_fragment_helpers[key] = helper
        self.cooperative_matrix_fragment_helper_names.add(name)
        self.wide_vector_reserved_names.add(name)
        return name

    def generate_cooperative_matrix_fragment_read(self, expression, is_main=False):
        contract = self.cooperative_matrix_fragment_contract(expression, "read")
        if contract is None:
            return None
        helper = self.cooperative_matrix_fragment_helper_name(contract)
        matrix = self.generate_expression(contract["matrix_expression"], is_main)
        return f"{helper}({matrix})"

    def generate_cooperative_matrix_fragment_assignment(self, node, is_main=False):
        contract = self.cooperative_matrix_fragment_contract(
            node.left,
            "write",
            assigned_value=node.right,
            assignment_operator=node.operator,
        )
        if contract is None:
            return None
        helper = self.cooperative_matrix_fragment_helper_name(contract)
        fragment = self.generate_initializer_value(
            node.right,
            is_main,
            contract["fragment_type"],
        )
        matrix = self.generate_expression(contract["matrix_expression"], is_main)
        return f"{helper}({fragment}, {matrix})"

    @staticmethod
    def cooperative_matrix_fragment_component(contract, value_name, index):
        if contract["wide_fragment"]:
            return f"{value_name}.lanes[{index}]"
        return f"{value_name}.{'xyzw'[index]}"

    def generate_cooperative_matrix_fragment_support_code(self, indent=0):
        if not self.cooperative_matrix_fragment_helpers:
            return ""

        pad = "    " * indent
        body_pad = "    " * (indent + 1)
        code = f"{pad}// Whole-lane Metal cooperative-matrix fragment transfers\n"
        for contract in sorted(
            self.cooperative_matrix_fragment_helpers.values(),
            key=lambda item: item["name"],
        ):
            name = contract["name"]
            matrix_type = contract["mapped_matrix_type"]
            fragment_type = contract["mapped_fragment_type"]
            code += f"{pad}@metal_static\n"
            if contract["direction"] == "read":
                code += f"{pad}{fragment_type} {name}" f"(in {matrix_type} matrix) {{\n"
                code += f"{body_pad}{fragment_type} _fragment_value;\n"
                for index in range(contract["fragment_width"]):
                    component = self.cooperative_matrix_fragment_component(
                        contract, "_fragment_value", index
                    )
                    code += (
                        f"{body_pad}{component} = "
                        f"cooperative_matrix_element(matrix, {index});\n"
                    )
                code += f"{body_pad}return _fragment_value;\n"
            else:
                code += (
                    f"{pad}void {name}(in {fragment_type} _fragment_value, "
                    f"inout {matrix_type} matrix) {{\n"
                )
                for index in range(contract["fragment_width"]):
                    component = self.cooperative_matrix_fragment_component(
                        contract, "_fragment_value", index
                    )
                    code += (
                        f"{body_pad}cooperative_matrix_element(matrix, {index}) = "
                        f"{component};\n"
                    )
            code += f"{pad}}}\n\n"
        return code

    def generate_cooperative_matrix_function_call(self, expression, is_main):
        function_name = str(expression.name).rsplit("::", 1)[-1]
        base_name, template_arguments = self.generic_type_parts(function_name)
        base_name = base_name or function_name
        if base_name in self.user_function_names:
            return None
        if base_name == "make_filled_simdgroup_matrix":
            # make_filled_simdgroup_matrix<T, Cols, Rows = Cols>(T value): the
            # declaration or assignment it initializes carries the contract.
            if len(expression.args) != 1 or len(template_arguments) not in {2, 3}:
                return None
            element_type = self.map_type(template_arguments[0])
            value = self.generate_expression(expression.args[0], is_main)
            return f"cooperative_matrix_fill({element_type}({value}))"
        if base_name == "simdgroup_matrix" and len(expression.args) == 1:
            # simdgroup_matrix<T, Cols, Rows>(T value) is the diagonal
            # constructor: value on the diagonal and zero elsewhere.
            if len(template_arguments) not in {2, 3}:
                return None
            element_type = self.map_type(template_arguments[0])
            value = self.generate_expression(expression.args[0], is_main)
            return f"cooperative_matrix_diagonal({element_type}({value}))"
        if function_name == "simdgroup_multiply":
            # simdgroup_multiply(d, a, b) stores a * b into d; it returns void.
            if len(expression.args) != 3:
                return None
            destination, left, right = (
                self.generate_expression(argument, is_main)
                for argument in expression.args
            )
            return f"{destination} = cooperative_matrix_multiply({left}, {right})"
        operation = {
            "simdgroup_load": "cooperative_matrix_load",
            "simdgroup_store": "cooperative_matrix_store",
            "simdgroup_multiply_accumulate": "cooperative_matrix_multiply_accumulate",
        }.get(function_name)
        if operation is None:
            return None
        source_arguments = list(expression.args)
        if function_name == "simdgroup_store" and len(source_arguments) >= 2:
            source_arguments = [
                source_arguments[1],
                source_arguments[0],
                *source_arguments[2:],
            ]
        arguments = ", ".join(
            self.generate_expression(argument, is_main) for argument in source_arguments
        )
        return f"{operation}({arguments})"
