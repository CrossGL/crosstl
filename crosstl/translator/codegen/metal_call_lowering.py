"""Call lowering in the Metal generator: mesh output accumulators and writes, mesh dispatch, structured buffer calls and parameters, value array returns, visible and intersection function table types, resource dimension descriptors and parameter resource bindings."""

import re

from ...backend.common_ast import ReturnNode as BackendReturnNode
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerReinterpretNode,
    PointerType,
    ReturnNode,
    VariableNode,
)
from ..refusals import MetalRefusal
from .array_utils import split_array_type_suffix
from .buffer_length import BufferLengthUnsupportedError
from .generic_function_utils import (
    generic_function_call_name,
    generic_function_specialization_named,
)
from .image_access_contracts import (
    is_integer_coordinate_type_name,
    texture_resource_dimension_descriptor,
    texture_resource_offset_dimension_key,
)
from .metal_array_copies import (
    generate_metal_array_initializer,
    generate_metal_value_array_literal,
)
from .metal_buffer_scalars import is_metal_buffer_scalar_type_name
from .metal_colour_attachment_inputs import is_metal_colour_attachment_input_parameter
from .metal_restrict import metal_restrict_suffix
from .metal_texel_buffers import TEXEL_BUFFER_TYPES
from .metal_threadgroup_parameters import metal_threadgroup_parameter_attribute


class MetalCallLoweringMixin:
    """Call lowering members of ``MetalCodeGen``."""

    def collect_metal_mesh_output_accumulators(self, func, reserved_parameter_names):
        if not self.current_metal_mesh_output_config:
            return {}

        accumulators = {}
        reserved_names = set(reserved_parameter_names or ())
        reserved_names.add(self.current_metal_mesh_output_parameter)
        reserved_names.update(self.metal_function_local_variable_names(func))

        for node in self.metal_render_walks.body_nodes_of_type(func, AssignmentNode):
            target = getattr(node, "target", getattr(node, "left", None))
            target_info = self.metal_mesh_output_assignment_target(target)
            if target_info is None or target_info["member"] is None:
                continue
            if target_info["role"] not in {"vertices", "primitives"}:
                continue

            element_type = target_info["output"]["element_type"]
            member_types = self.struct_member_types.get(element_type, {})
            operator = getattr(node, "operator", "=")
            if operator == "=" and list(member_types) == [target_info["member"]]:
                continue

            key = self.metal_mesh_output_accumulator_key(target_info)
            if key in accumulators:
                continue

            temp_base = (
                f"_crossglMesh{target_info['role'].title()}"
                f"_{target_info['name']}"
                f"_{self.metal_identifier_suffix(key[2])}"
            )
            temp_name = self.unique_metal_generated_name(temp_base, reserved_names)
            reserved_names.add(temp_name)
            accumulators[key] = {
                "name": temp_name,
                "type": element_type,
            }
        return accumulators

    def generate_metal_mesh_output_accumulator_declarations(self):
        lines = []
        for accumulator in self.current_metal_mesh_output_accumulators.values():
            lines.append(f"    {accumulator['type']} {accumulator['name']} = {{}};\n")
        return "".join(lines)

    def metal_mesh_output_accumulator(self, target_info):
        return self.current_metal_mesh_output_accumulators.get(
            self.metal_mesh_output_accumulator_key(target_info)
        )

    def metal_mesh_output_accumulator_key(self, target_info):
        index_expr = target_info.get("index")
        index_text = self.safe_expression_to_string(index_expr) if index_expr else "0"
        return (target_info["role"], target_info["name"], index_text)

    def metal_identifier_suffix(self, value):
        suffix = "".join(
            ch if (ch.isalnum() or ch == "_") else "_" for ch in str(value)
        ).strip("_")
        if not suffix:
            suffix = "value"
        if suffix[0].isdigit():
            suffix = f"i_{suffix}"
        return suffix

    def metal_mesh_output_assignment_target(self, target):
        mesh_output = self.current_metal_mesh_output_config
        if not mesh_output:
            return None
        outputs = mesh_output.get("output_parameters", {})

        member = None
        access = target
        if isinstance(target, MemberAccessNode):
            member = str(target.member)
            access = getattr(target, "object", None)

        if not isinstance(access, ArrayAccessNode):
            return None
        array_expr = getattr(access, "array", None)
        parameter_name = self.expression_name(array_expr)
        output = outputs.get(parameter_name)
        if output is None:
            return None

        return {
            "name": parameter_name,
            "role": output["role"],
            "output": output,
            "index": getattr(access, "index", None),
            "member": member,
        }

    def unsupported_metal_mesh_output_assignment_diagnostic(self, target_info, reason):
        name = target_info.get("name", "<unknown>")
        role = target_info.get("role", "output")
        return self.refuse_lowering(
            f"mesh output assignment: {role} output '{name}' - {reason}", "(void)0"
        )

    def metal_mesh_vertex_output_value(self, expr, mesh_output):
        value = self.generate_expression(expr)
        return self.metal_mesh_vertex_output_value_from_rendered(
            expr, value, mesh_output
        )

    def metal_mesh_vertex_output_value_from_rendered(self, expr, value, mesh_output):
        value_type = self.expression_result_type(expr)
        mapped_type = self.map_type(value_type) if value_type else None
        vertex_type = mesh_output["vertex_type"]

        if mapped_type == vertex_type:
            return value
        if mapped_type == "float3":
            return f"{vertex_type}{{float4({value}, 1.0)}}"
        return f"{vertex_type}{{{value}}}"

    def metal_mesh_primitive_output_call(
        self, output_parameter, index_expr, value_expr, mesh_output
    ):
        primitive_type = mesh_output.get("primitive_type")
        if not primitive_type or primitive_type == "void":
            return None

        value_type = self.expression_result_type(value_expr)
        mapped_type = self.map_type(value_type) if value_type else None
        if mapped_type != primitive_type:
            return None

        index = self.generate_expression(index_expr)
        value = self.generate_expression(value_expr)
        return f"{output_parameter}.set_primitive({index}, {value})"

    def metal_mesh_index_output_write(
        self, output_parameter, func_name, index_expr, value_expr
    ):
        index = self.generate_expression(index_expr)
        value = self.generate_expression(value_expr)
        vector_width = self.metal_mesh_index_vector_width(value_expr)
        if vector_width is None:
            return f"{output_parameter}.set_index({index}, {value})"

        base_index = index
        if func_name == "SetPrimitive":
            base_index = f"({index}) * {vector_width}"

        if vector_width in {2, 4}:
            return (
                f"{output_parameter}.set_indices("
                f"{base_index}, uchar{vector_width}({value}))"
            )

        if vector_width == 3:
            return "\n".join(
                f"{output_parameter}.set_index("
                f"{self.metal_mesh_index_component_expression(base_index, offset)}, "
                f"{value}.{component})"
                for offset, component in enumerate(("x", "y", "z"))
            )

        return f"{output_parameter}.set_index({index}, {value})"

    def metal_mesh_index_vector_width(self, expr):
        value_type = self.expression_result_type(expr)
        if not value_type:
            return None
        mapped_type = self.map_type(value_type)
        if (
            len(mapped_type) < 2
            or mapped_type[-1] not in {"2", "3", "4"}
            or mapped_type[:-1] not in {"bool", "int", "uint"}
        ):
            return None
        return int(mapped_type[-1])

    def metal_mesh_index_component_expression(self, base_index, offset):
        if offset == 0:
            return base_index
        return f"({base_index}) + {offset}"

    def generate_mesh_op_expression(self, expr):
        arguments = self.normalized_metal_intrinsic_args(expr.arguments)
        if (
            expr.operation == "SetMeshOutputCounts"
            and self.current_metal_mesh_output_parameter
            and len(arguments) >= 2
        ):
            primitive_count = self.generate_expression(arguments[1])
            return (
                f"{self.current_metal_mesh_output_parameter}"
                f".set_primitive_count({primitive_count})"
            )
        if expr.operation == "DispatchMesh":
            if (
                len(arguments) == 4
                and self.current_metal_mesh_payload_parameter is None
            ):
                return self.refuse_lowering(
                    "mesh dispatch: DispatchMesh payload argument requires an "
                    "object_data payload context",
                    "(void)0",
                )
            if self.current_metal_mesh_grid_properties_parameter is None:
                return self.refuse_lowering(
                    "mesh dispatch: DispatchMesh requires mesh_grid_properties context",
                    "(void)0",
                )
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 4
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments[:3])
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments[:3])
                )
            payload_assignment = self.metal_dispatch_mesh_payload_assignment(
                arguments[3]
            )
            # DispatchMesh implies GroupMemoryBarrierWithGroupSync: the payload
            # the mesh grid reads holds every thread's groupshared writes.
            return "\n".join(
                [
                    "threadgroup_barrier(mem_flags::mem_threadgroup)",
                    payload_assignment,
                    grid_assignment,
                ]
            )
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 3
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments)
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments)
                )
            return grid_assignment
        if (
            expr.operation == "DispatchMesh"
            and self.current_metal_mesh_grid_properties_parameter
            and len(arguments) == 1
        ):
            grid_assignment = self.metal_dispatch_mesh_grid_assignment(arguments)
            if grid_assignment is None:
                return self.unsupported_metal_mesh_dispatch(
                    self.metal_dispatch_mesh_grid_argument_reason(arguments)
                )
            return grid_assignment
        return None

    def metal_dispatch_mesh_grid_assignment(self, grid_args):
        reason = self.metal_dispatch_mesh_grid_argument_reason(grid_args)
        if reason is not None:
            return None
        if len(grid_args) == 1:
            grid = self.generate_expression(grid_args[0])
        else:
            grid = "uint3({})".format(
                ", ".join(self.generate_expression(argument) for argument in grid_args)
            )
        return (
            f"{self.current_metal_mesh_grid_properties_parameter}"
            f".set_threadgroups_per_grid({grid})"
        )

    def metal_dispatch_mesh_grid_argument_reason(self, grid_args):
        if len(grid_args) == 1:
            grid_type = self.expression_result_type(grid_args[0])
            if self.map_type(grid_type) == "uint3":
                return None
            type_label = self.type_name_string(grid_type) or "unknown"
            return (
                "DispatchMesh grid argument must be a uint3-compatible vector"
                f"; got '{type_label}'"
            )

        for index, argument in enumerate(grid_args, start=1):
            argument_type = self.expression_result_type(argument)
            if self.is_scalar_integer_type(argument_type):
                continue
            type_label = self.type_name_string(argument_type) or "unknown"
            return (
                f"DispatchMesh grid component argument {index} must be a "
                f"scalar integer; got '{type_label}'"
            )
        return None

    def unsupported_metal_mesh_dispatch(self, reason):
        return self.refuse_lowering(f"mesh dispatch: {reason}", "(void)0")

    def metal_dispatch_mesh_payload_assignment(self, payload_expr):
        if self.current_metal_mesh_payload_parameter is None:
            return self.refuse_lowering(
                "mesh payload dispatch: DispatchMesh payload argument requires an "
                "object_data payload parameter",
                "(void)0",
            )

        payload_type = self.expression_result_type(payload_expr)
        if payload_type is not None:
            mapped_payload_type = self.metal_mesh_payload_value_type(payload_type)
            if (
                self.current_metal_mesh_payload_type is not None
                and mapped_payload_type != self.current_metal_mesh_payload_type
            ):
                return self.refuse_lowering(
                    f"mesh payload dispatch: payload argument type {mapped_payload_type} "
                    f"does not match {self.current_metal_mesh_payload_type}",
                    "(void)0",
                )

        payload_source_diagnostic = self.metal_dispatch_mesh_payload_source_diagnostic(
            payload_expr
        )
        if payload_source_diagnostic is not None:
            return payload_source_diagnostic

        payload_value = self.generate_expression(payload_expr)
        return f"{self.current_metal_mesh_payload_parameter} = {payload_value}"

    def metal_dispatch_mesh_payload_source_diagnostic(self, payload_expr):
        payload_name = self.assignment_target_root_name(payload_expr)
        if not payload_name:
            return self.refuse_lowering(
                "mesh payload dispatch: payload argument must be a threadgroup lvalue",
                "(void)0",
            )

        payload_type = self.expression_result_type(payload_expr)
        if self.metal_type_is_pointer_like(payload_type):
            display_name = (
                self.assignment_target_display_name(payload_expr) or payload_name
            )
            return self.refuse_lowering(
                f"mesh payload dispatch: payload argument '{display_name}' has "
                "pointer or array type; DispatchMesh payload requires a threadgroup "
                "value lvalue",
                "(void)0",
            )

        address_space = self.argument_address_space(payload_expr)
        if address_space == "threadgroup":
            return None

        source_address_space = address_space or "unknown"
        display_name = (
            self.assignment_target_access_display_name(payload_expr) or payload_name
        )
        return self.refuse_lowering(
            f"mesh payload dispatch: payload argument '{display_name}' uses "
            f"{source_address_space} address space; DispatchMesh payload requires a "
            "threadgroup lvalue",
            "(void)0",
        )

    def generate_buffer_call(self, func_name, args):
        if func_name == "buffer_load" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            return f"{buffer}[{index}]"
        if func_name == "buffer_store" and len(args) >= 3:
            buffer = self.generate_expression(args[0])
            index = self.generate_expression(args[1])
            value = self.generate_expression(args[2])
            return f"{buffer}[{index}] = {value}"
        if func_name == "buffer_append" and len(args) >= 2:
            buffer = self.generate_expression(args[0])
            value = self.generate_expression(args[1])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return self.refuse_lowering(
                    "buffer append: requires append/consume counter buffer", "(void)0"
                )
            index = f"atomic_fetch_add_explicit({counter}, 1u, memory_order_relaxed)"
            return f"{buffer}[{index}] = {value}"
        if func_name == "buffer_consume" and args:
            buffer = self.generate_expression(args[0])
            counter = self.structured_buffer_counter_reference(args[0])
            if counter is None:
                return self.refuse_lowering(
                    "buffer consume: requires append/consume counter buffer", "0"
                )
            index = (
                f"(atomic_fetch_sub_explicit({counter}, 1u, "
                "memory_order_relaxed) - 1u)"
            )
            return f"{buffer}[{index}]"
        if func_name == "buffer_dimensions" and args:
            length_expr = self.structured_buffer_length_reference(args[0])
            if len(args) > 2 and length_expr is not None:
                # semantics.buffer_strides answers the stride of a laid-out buffer; one left here holds
                # MSL-laid-out elements, whose stride is their size.
                buffer_type = str(
                    self.resource_base_type(self.expression_result_type(args[0]))
                )
                if "<" not in buffer_type or not buffer_type.endswith(">"):
                    raise BufferLengthUnsupportedError(
                        "metal",
                        f"buffer_dimensions({self.generate_expression(args[0])}) has a stride output "
                        "and no known element type",
                    )
                element = self.structured_buffer_element_type(buffer_type)
                return (
                    f"{self.generate_expression(args[1])} = {length_expr}, "
                    f"{self.generate_expression(args[2])} = uint(sizeof({element}))"
                )
            if length_expr is None:
                raise BufferLengthUnsupportedError(
                    "metal",
                    f"buffer_dimensions({self.generate_expression(args[0])}) needs a "
                    "length sidecar buffer, and this buffer has none",
                )
            if len(args) >= 2:
                target = self.generate_expression(args[1])
                if len(args) < 3:
                    return f"{target} = {length_expr}"
                # GetDimensions(count, stride): the element stride is the size of the element type the buffer
                # is declared with, which carries the source layout; sizeof does not read the element.
                buffer = self.generate_expression(args[0])
                stride = self.generate_expression(args[2])
                return (
                    f"{target} = {length_expr}, "
                    f"{stride} = uint(sizeof({buffer}[0]))"
                )
            return length_expr
        return None

    def structured_buffer_length_reference(self, buffer_arg):
        length = self.structured_buffer_length_data_argument(buffer_arg)
        if length is None:
            return None
        return f"{length}[0]"

    def structured_buffer_length_data_argument(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            length_parameter = self.current_structured_buffer_length_parameters.get(
                array_name
            )
            if length_parameter is not None:
                index = self.generate_expression(index_expr)
                return f"{length_parameter}[{index}]"
            if self.structured_buffer_requires_length(array_name):
                index = self.generate_expression(index_expr)
                length_name = self.structured_buffer_length_parameter_name(array_name)
                return f"{length_name}[{index}]"

        arg_name = self.expression_name(arg)
        length_parameter = self.current_structured_buffer_length_parameters.get(
            arg_name
        )
        if length_parameter is not None:
            return length_parameter

        if self.structured_buffer_requires_length(arg_name):
            return self.structured_buffer_length_parameter_name(arg_name)
        return None

    def structured_buffer_counter_reference(self, buffer_arg):
        counter = self.structured_buffer_counter_data_argument(buffer_arg)
        if counter is None:
            return None
        return counter

    def structured_buffer_counter_data_argument(self, arg):
        if isinstance(arg, ArrayAccessNode) or (
            hasattr(arg, "__class__") and "ArrayAccess" in str(arg.__class__)
        ):
            array_expr = getattr(arg, "array", getattr(arg, "array_expr", None))
            index_expr = getattr(arg, "index", getattr(arg, "index_expr", None))
            array_name = self.expression_name(array_expr)
            counter_parameter = self.current_structured_buffer_counter_parameters.get(
                array_name
            )
            if counter_parameter is not None:
                index = self.generate_expression(index_expr)
                return f"{counter_parameter}[{index}]"
            if self.global_structured_buffer_requires_counter(array_name):
                index = self.generate_expression(index_expr)
                counter_name = self.structured_buffer_counter_parameter_name(array_name)
                return f"{counter_name}[{index}]"

        arg_name = self.expression_name(arg)
        counter_parameter = self.current_structured_buffer_counter_parameters.get(
            arg_name
        )
        if counter_parameter is not None:
            return counter_parameter

        if self.global_structured_buffer_requires_counter(arg_name):
            return self.structured_buffer_counter_parameter_name(arg_name)
        return None

    def default_sampler_expression(self):
        return "sampler(mag_filter::linear, min_filter::linear)"

    def sampler_variable_names(self):
        return {
            sampler_variable.name for sampler_variable, _, _ in self.sampler_variables
        } | self.current_sampler_parameters

    def structured_buffer_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_structured_buffer_type(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_requires_counter(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "uint"
        element_type_name = type_name.split("<", 1)[1][:-1].strip()
        return self.metal_native_narrow_bitcast_storage_type(
            element_type_name
        ) or self.map_type(element_type_name)

    def structured_buffer_load_source(self, expr):
        if not isinstance(expr, FunctionCallNode):
            return None
        function = getattr(expr, "function", None) or getattr(expr, "name", None)
        name = getattr(function, "name", function)
        args = getattr(expr, "args", [])
        if (
            name == "buffer_load"
            and name not in self.user_function_names
            and len(args) == 2
            and self.is_structured_buffer_type(self.expression_result_type(args[0]))
        ):
            return args[0]
        return None

    def metal_thread_pointer_return_surrogate_type(self, func):
        """Recover a pointer ABI from one canonical CrossGL buffer surrogate.

        Older Metal imports represented a source ``thread T*`` return as a
        ``RWStructuredBuffer<T>`` (or ``StructuredBuffer<T>`` for a const
        pointee), retaining the exact pointer type on the sole returned
        ``PointerReinterpretNode``. Preserve compatibility with that narrow
        shape without mistaking arbitrary resource returns for local views.
        New imports carry the pointer return type directly.
        """

        declared_type = getattr(func, "return_type", None)
        buffer_kind = self.structured_buffer_type_name(declared_type)
        if buffer_kind not in {"StructuredBuffer", "RWStructuredBuffer"}:
            return None

        body = getattr(func, "body", None)
        statements = list(getattr(body, "statements", ()) or ())
        if len(statements) != 1 or not isinstance(
            statements[0], (ReturnNode, BackendReturnNode)
        ):
            return None
        reinterpret = getattr(statements[0], "value", None)
        if not isinstance(reinterpret, PointerReinterpretNode):
            return None
        pointer_type = getattr(reinterpret, "target_type", None)
        if not isinstance(pointer_type, PointerType):
            return None
        if (
            self.normalized_address_space(getattr(pointer_type, "address_space", None))
            != "thread"
        ):
            return None
        if getattr(pointer_type, "access_mode", None) not in {
            None,
            "read",
            "write",
            "read_write",
        }:
            return None
        if getattr(pointer_type, "resource_qualifiers", None):
            return None

        is_mutable = bool(getattr(pointer_type, "is_mutable", True))
        if is_mutable != (buffer_kind == "RWStructuredBuffer"):
            return None
        if getattr(pointer_type, "access_mode", None) == "read" and is_mutable:
            return None
        if (
            getattr(pointer_type, "access_mode", None)
            in {
                "write",
                "read_write",
            }
            and not is_mutable
        ):
            return None

        pointee_type = self.metal_native_narrow_bitcast_storage_type(
            pointer_type.pointee_type
        ) or self.map_type(pointer_type.pointee_type)
        if pointee_type != self.structured_buffer_element_type(declared_type):
            return None
        return pointer_type

    def metal_effective_function_return_type(self, func):
        if isinstance(getattr(func, "return_type", None), PointerType):
            return func.return_type
        pointer_type = self.metal_thread_pointer_return_surrogate_type(func)
        if pointer_type is not None:
            return pointer_type
        return self.type_name_string(getattr(func, "return_type", "void"))

    def metal_value_array_type(self, type_name):
        """Spell a sized C-style array type as nested Metal ``array<T, N>``.

        Returns None for unsized or malformed array suffixes.
        """
        base_type, suffix = split_array_type_suffix(str(type_name or "").strip())
        base_type = base_type.strip()
        sizes = []
        while suffix:
            match = re.match(r"\[([^\[\]]+)\]\s*", suffix)
            if match is None or not match.group(1).strip():
                return None
            sizes.append(match.group(1).strip())
            suffix = suffix[match.end() :]
        if not sizes or not base_type:
            return None
        spelled = self.map_type(base_type)
        for size in reversed(sizes):
            spelled = f"array<{spelled}, {size}>"
        return spelled

    def metal_value_array_returned_locals(self, func):
        """Name the locals a function returns as its array-valued result."""
        return_type = self.metal_effective_function_return_type(func)
        if not isinstance(return_type, str) or "[" not in return_type:
            return frozenset()
        names = set()
        for node in self.metal_render_walks.body_nodes_of_type(
            func, (ReturnNode, BackendReturnNode)
        ):
            value = getattr(node, "value", None)
            if isinstance(value, (IdentifierNode, VariableNode)) and isinstance(
                getattr(value, "name", None), str
            ):
                names.add(value.name)
        return frozenset(names)

    def metal_expression_returns_value_array(self, expression):
        if not isinstance(expression, FunctionCallNode):
            return False
        name = self.function_call_name(expression)
        return_type = self.function_return_types.get(name)
        if not (isinstance(return_type, str) and "[" in return_type):
            # A generic call returns the result type of the specialization it calls, ``float[2]`` for
            # ``T pick<T>(T x)`` called with a ``float[2]``.
            specialized = generic_function_call_name(
                self,
                name,
                getattr(expression, "arguments", getattr(expression, "args", [])),
            )
            if specialized is None:
                return False
            return_type = self.function_return_types.get(specialized)
            if return_type is None:
                clone = generic_function_specialization_named(self, specialized)
                if clone is None:
                    return False
                return_type = self.type_name_string(
                    self.metal_effective_function_return_type(clone)
                )
        return isinstance(return_type, str) and "[" in return_type

    def metal_expression_is_value_array(self, expression):
        """Whether ``expression`` is a Metal ``array<T, N>`` value: a user-function result, such a local or a row of one."""
        if self.metal_expression_returns_value_array(expression):
            return True
        if isinstance(expression, ArrayAccessNode):
            return "[" in self.type_name_string(
                self.expression_result_type(expression)
            ) and self.metal_expression_is_value_array(expression.array_expr)
        return isinstance(expression, (IdentifierNode, VariableNode)) and (
            getattr(expression, "name", None)
            in (getattr(self, "current_metal_value_array_names", None) or ())
        )

    def metal_c_array_return(self, value):
        """Brace list returning C-array ``value`` as the function's ``array<T, N>`` result, or ``None``.

        Metal returns a CrossGL array as ``metal::array<T, N>``, an aggregate with no conversion from a C array or
        from the pointer a C-array parameter is adjusted to ([dcl.fct]/5), so ``return x;`` of a parameter, of a
        structure member or of a program-scope constant does not compile. The brace list ``{x[0], x[1]}``
        initializes the result element by element, which keeps the value semantics of the source return.
        ``None`` leaves an ``array<T, N>`` value, or a function that returns no array, to plain ``return``.
        """
        return_type = getattr(self, "current_function_return_type", None)
        if return_type is None or "[" not in self.type_name_string(return_type):
            return None
        if self.metal_expression_is_value_array(value):
            return None
        return generate_metal_array_initializer(
            self, value, return_type, value_array=True
        )

    def metal_value_array_initializer(self, value, declared_type):
        """Brace list initializing an ``array<T, N>`` local from C-array ``value``, or ``None`` to render it as is.

        ``array<T, N>`` has no conversion from a C array either, so a returned local initialized with a parameter
        or another C array takes its elements.
        """
        if self.metal_expression_is_value_array(value):
            return None
        if isinstance(value, ArrayLiteralNode):
            return generate_metal_value_array_literal(self, value, declared_type)
        return generate_metal_array_initializer(
            self, value, declared_type, value_array=True
        )

    def metal_local_value_array_type(self, node, declared_type):
        """Return ``array<T, N>`` for a local that holds an array result."""
        if "[" not in str(declared_type):
            return None
        if self.local_variable_address_space(node) == "threadgroup":
            return None
        if not (
            getattr(node, "name", None) in self.current_metal_value_array_locals
            or self.metal_expression_returns_value_array(
                getattr(node, "initial_value", None)
            )
        ):
            return None
        return self.metal_value_array_type(declared_type)

    def format_metal_pointer_return_type(self, pointer_type):
        address_space = self.normalized_address_space(pointer_type.address_space)
        pointee_type = self.metal_native_narrow_bitcast_storage_type(
            pointer_type.pointee_type
        ) or self.map_type(pointer_type.pointee_type)
        const_prefix = "" if pointer_type.is_mutable else "const "
        return f"{const_prefix}{address_space} {pointee_type}*"

    def structured_buffer_address_space(self, vtype, node=None):
        qualifiers = self.parameter_qualifier_names(node)
        explicit_spaces = qualifiers & {"constant", "device"}
        if len(explicit_spaces) > 1:
            name = getattr(node, "name", "<anonymous>")
            raise MetalRefusal(
                f"Metal structured buffer '{name}' has conflicting address-space "
                f"metadata: {', '.join(sorted(explicit_spaces))}"
            )
        if "constant" in explicit_spaces:
            if self.structured_buffer_type_name(vtype) != "StructuredBuffer":
                name = getattr(node, "name", "<anonymous>")
                raise MetalRefusal(
                    f"Writable Metal structured buffer '{name}' cannot use the "
                    "constant address space"
                )
            return "constant"
        if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
            return "const device"
        return "device"

    def format_structured_buffer_parameter(
        self, vtype, name, array_size=None, node=None
    ):
        element_type = self.structured_buffer_element_type(vtype)
        layout_storage = self.metal_hlsl_structured_buffers.get(name)
        if layout_storage is not None:
            element_type = layout_storage[0]
        address_space = self.structured_buffer_address_space(vtype, node)
        memory_qualifiers = self.metal_parameter_memory_qualifier_prefix(
            node, vtype, address_space
        )
        pointer_type = f"{memory_qualifiers}{address_space} {element_type}*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type}{metal_restrict_suffix(node)} {name}"

    def structured_buffer_length_resource_name(self, name):
        return self.metal_generated_companion_name(f"{name}Length")

    def structured_buffer_length_parameter_name(self, name):
        return self.metal_generated_companion_name(f"{name}Length")

    def format_structured_buffer_length_parameter(self, name, array_size=None):
        length_name = self.structured_buffer_length_parameter_name(name)
        pointer_type = "constant uint*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {length_name}"
        return f"{pointer_type} {length_name}"

    def structured_buffer_counter_resource_name(self, name):
        return f"{name}Counter"

    def structured_buffer_counter_parameter_name(self, name):
        return f"{name}Counter"

    def format_structured_buffer_counter_parameter(self, name, array_size=None):
        counter_name = self.structured_buffer_counter_parameter_name(name)
        pointer_type = "device atomic_uint*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {counter_name}"
        return f"{pointer_type} {counter_name}"

    def format_glsl_buffer_block_parameter(self, block, name, array_size=None):
        address_space = "const device" if block.get("readonly") else "device"
        pointer_type = f"{block.get('memory', '')}{address_space} uchar*"
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{pointer_type}, {array_size}> {name}"
        return f"{pointer_type} {name}"

    def visible_function_table_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_visible_function_table_type(self, vtype):
        return self.visible_function_table_type_name(vtype) in {
            "visible_function_table",
            "visibleFunctionTable",
        }

    def visible_function_table_signature(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "void()"

        payload_type = type_name.split("<", 1)[1][:-1].strip()
        if not payload_type or payload_type == "void":
            return "void()"
        return f"void(thread {self.map_type(payload_type)}&)"

    def map_visible_function_table_type(self, vtype):
        return f"visible_function_table<{self.visible_function_table_signature(vtype)}>"

    def format_visible_function_table_parameter(self, vtype, name, array_size=None):
        table_type = self.visible_function_table_parameter_type(vtype)
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{table_type}, {array_size}> {name}"
        return f"{table_type} {name}"

    def visible_function_table_parameter_type(self, vtype):
        type_name = str(vtype)
        if type_name.startswith("visible_function_table<"):
            signature = type_name.split("<", 1)[1][:-1].strip()
            if signature.startswith("void"):
                return type_name
        return self.map_visible_function_table_type(vtype)

    def intersection_function_table_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_intersection_function_table_type(self, vtype):
        return self.intersection_function_table_type_name(vtype) in {
            "intersection_function_table",
            "intersectionFunctionTable",
        }

    def map_intersection_function_table_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" in type_name and type_name.endswith(">"):
            tags = type_name.split("<", 1)[1][:-1].strip()
            return f"intersection_function_table<{tags}>"
        return "intersection_function_table<>"

    def format_intersection_function_table_parameter(
        self, vtype, name, array_size=None
    ):
        table_type = (
            vtype
            if str(vtype).startswith("intersection_function_table<")
            else self.map_intersection_function_table_type(vtype)
        )
        if array_size is not None:
            array_size = array_size or "1"
            return f"array<{table_type}, {array_size}> {name}"
        return f"{table_type} {name}"

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}

    def is_acceleration_structure_type(self, vtype):
        return self.resource_base_type(vtype) in {
            "accelerationStructureEXT",
            "RaytracingAccelerationStructure",
            "AccelerationStructure",
            "acceleration_structure",
            "instance_acceleration_structure",
            "primitive_acceleration_structure",
        }

    def is_resource_parameter_type(self, vtype):
        return (
            self.is_structured_buffer_type(vtype)
            or self.is_acceleration_structure_type(vtype)
            or self.is_visible_function_table_type(vtype)
            or self.is_intersection_function_table_type(vtype)
            or self.resource_base_type(vtype)
            in {
                "comparison_sampler",
                "sampler",
                "sampler1D",
                "sampler1DArray",
                "sampler2D",
                "sampler3D",
                "samplerCube",
                "sampler2DArray",
                "samplerCubeArray",
                "sampler2DMS",
                "sampler2DMSArray",
                "sampler2DShadow",
                "sampler2DArrayShadow",
                "samplerCubeShadow",
                "samplerCubeArrayShadow",
                "isampler1D",
                "isampler1DArray",
                "isampler2D",
                "isampler3D",
                "isamplerCube",
                "isampler2DArray",
                "isamplerCubeArray",
                "isampler2DMS",
                "isampler2DMSArray",
                "usampler1D",
                "usampler1DArray",
                "usampler2D",
                "usampler3D",
                "usamplerCube",
                "usampler2DArray",
                "usamplerCubeArray",
                "usampler2DMS",
                "usampler2DMSArray",
                "iimage1D",
                "iimage1DArray",
                "iimage2D",
                "iimage3D",
                "iimage2DArray",
                "iimage2DMS",
                "iimage2DMSArray",
                "uimage1D",
                "uimage1DArray",
                "uimage2D",
                "uimage3D",
                "uimage2DArray",
                "uimage2DMS",
                "uimage2DMSArray",
                "image1D",
                "image1DArray",
                "image2D",
                "image3D",
                "imageCube",
                "image2DArray",
                "image2DMS",
                "image2DMSArray",
                *TEXEL_BUFFER_TYPES,
            }
        )

    def is_bound_scalar_value_parameter(
        self, raw_param_type, node=None, shader_type=None
    ):
        if node is None or shader_type != "compute":
            return False
        if self.explicit_buffer_binding_index(node) is None:
            return False
        qualifiers = self.parameter_qualifier_names(node)
        if not qualifiers & {"uniform", "constant"}:
            return False
        if self.is_resource_parameter_type(raw_param_type):
            return False
        if self.is_raw_buffer_parameter_type(raw_param_type, node):
            return False
        type_name = self.type_name_string(raw_param_type)
        return is_metal_buffer_scalar_type_name(type_name, self.map_type)

    def is_texture_or_image_resource_type(self, vtype):
        return (
            self.is_resource_parameter_type(vtype)
            and not self.is_sampler_type(vtype)
            and not self.is_structured_buffer_type(vtype)
            and not self.is_acceleration_structure_type(vtype)
        )

    def is_integer_coordinate_type(self, vtype):
        type_name = self.type_name_string(vtype)
        base_type = self.resource_base_type(type_name)
        mapped_type = self.map_type(base_type)
        return is_integer_coordinate_type_name(base_type, mapped_type)

    def texture_dimension_descriptor(self, texture_type):
        texture_type = self.resource_base_type(texture_type)
        sampling = self.texture_sampling_capabilities(texture_type)
        is_multisample = "_ms" in texture_type
        is_storage_image = "access::" in texture_type

        coordinate_dimension = None
        if texture_type and "cube" not in texture_type:
            if texture_type.startswith("texture1d_array<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture1d<"):
                coordinate_dimension = 1
            elif texture_type.startswith("texture2d_ms_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d_ms<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("texture2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("depth2d_array<"):
                coordinate_dimension = 3
            elif texture_type.startswith("depth2d<"):
                coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                coordinate_dimension = 3

        texel_fetch_offset_dimension = None
        if texture_type and "cube" not in texture_type and not is_multisample:
            if texture_type.startswith("texture1d_array<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture1d<"):
                texel_fetch_offset_dimension = 1
            elif texture_type.startswith("texture2d_array<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture2d<"):
                texel_fetch_offset_dimension = 2
            elif texture_type.startswith("texture3d<"):
                texel_fetch_offset_dimension = 3

        sample_offset_dimension = None
        if texture_type.startswith("texture2d_array<"):
            sample_offset_dimension = 2
        elif texture_type.startswith("texture2d<"):
            sample_offset_dimension = 2
        elif texture_type.startswith("texture3d<"):
            sample_offset_dimension = 3
        if not sampling["sample_offset"]:
            sample_offset_dimension = None

        gradient_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                gradient_dimension = 2
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                gradient_dimension = 2
            elif texture_type.startswith("texture3d<"):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                gradient_dimension = 3
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                gradient_dimension = 3

        query_lod_coordinate_dimension = None
        if texture_type and not is_storage_image and not is_multisample:
            if texture_type.startswith(("texture2d_array<", "depth2d_array<")):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texture2d<", "depth2d<")):
                query_lod_coordinate_dimension = 2
            elif texture_type.startswith("texture3d<"):
                query_lod_coordinate_dimension = 3
            elif texture_type.startswith(("texturecube_array<", "depthcube_array<")):
                query_lod_coordinate_dimension = 4
            elif texture_type.startswith(("texturecube<", "depthcube<")):
                query_lod_coordinate_dimension = 3

        return texture_resource_dimension_descriptor(
            texture_type,
            sampling,
            coordinate_dimension=coordinate_dimension,
            offset_dimension=sample_offset_dimension,
            sample_offset_dimension=sample_offset_dimension,
            texel_fetch_offset_dimension=texel_fetch_offset_dimension,
            gradient_dimension=gradient_dimension,
            query_lod_coordinate_dimension=query_lod_coordinate_dimension,
            is_multisample=is_multisample,
        )

    def resource_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)["coordinate_dimension"]

    def resource_offset_dimension(self, func_name, texture_type):
        descriptor = self.texture_dimension_descriptor(texture_type)
        key = texture_resource_offset_dimension_key(
            func_name, collapse_compare_offsets=True
        )
        return descriptor[key]

    def resource_gradient_dimension(self, func_name, texture_type):
        return self.texture_dimension_descriptor(texture_type)["gradient_dimension"]

    def resource_query_lod_coordinate_dimension(self, texture_type):
        return self.texture_dimension_descriptor(texture_type)[
            "query_lod_coordinate_dimension"
        ]

    def parameter_attribute(self, raw_param_type, semantic, shader_type, node=None):
        ray_semantic_role = self.metal_ray_semantic_role(node, shader_type)
        if shader_type == "ray_generation" and ray_semantic_role == "payload":
            return ""
        if self.is_metal_ray_stage(shader_type) and ray_semantic_role == "payload":
            return " [[payload]]"
        if (
            self.is_metal_ray_stage(shader_type)
            and ray_semantic_role == "hit_attribute"
        ):
            return ""
        if (
            shader_type in {"ray_callable", "callable"}
            and ray_semantic_role == "callable_data"
        ):
            return ""
        if self.is_metal_tessellation_helper_semantic(semantic) and shader_type in {
            None,
            "tessellation_control",
            "tessellation_evaluation",
        }:
            return ""
        if semantic:
            if shader_type == "fragment" and node is not None:
                # A no-perspective barycentric input takes [[center_no_perspective]], as a member does.
                implied = self.metal_semantic_implied_interpolation_attribute(node)
                if implied is not None:
                    return f"{self.map_semantic(semantic)} [[{implied}]]"
            return self.map_semantic(semantic)
        if shader_type in {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "ray_generation",
            "ray_intersection",
            "intersection",
            "object",
            "task",
            "amplification",
            "mesh",
        }:
            threadgroup_attr = metal_threadgroup_parameter_attribute(self, node)
            if threadgroup_attr:
                return threadgroup_attr
            resource_attr = self.resource_parameter_attribute(raw_param_type, node)
            if resource_attr:
                return resource_attr
        if self.is_resource_parameter_type(raw_param_type):
            return ""
        if is_metal_colour_attachment_input_parameter(self, node, shader_type):
            return ""
        if shader_type in {"vertex", "fragment"}:
            return " [[stage_in]]"
        return ""

    def metal_conditional_resource_attribute(self, node):
        attributes = [
            attr
            for attr in getattr(node, "attributes", []) or []
            if self.normalized_metal_abi_attribute_name(attr) == "function_constant"
        ]
        if not attributes:
            return ""
        name = getattr(node, "name", "<anonymous>")
        if len(attributes) != 1:
            raise MetalRefusal(
                f"Metal parameter '{name}' has multiple function constant conditions"
            )
        arguments = getattr(attributes[0], "arguments", []) or []
        if len(arguments) != 1:
            raise MetalRefusal(
                f"Metal parameter '{name}' requires one function constant condition"
            )
        condition = self.attribute_value_to_string(arguments[0])
        if not condition:
            raise MetalRefusal(
                f"Metal parameter '{name}' requires a function constant condition"
            )
        return f" [[function_constant({condition})]]"

    def parameter_resource_binding_metadata(self, raw_param_type, node=None):
        if node is None:
            return None
        if self.is_sampler_type(raw_param_type):
            namespace = "sampler"
            attribute_names = {"binding", "sampler"}
            prefixes = ("s",)
        elif self.is_visible_function_table_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_intersection_function_table_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_acceleration_structure_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_structured_buffer_type(raw_param_type):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_texture_or_image_resource_type(raw_param_type):
            namespace = "texture"
            attribute_names = {"binding", "texture"}
            prefixes = ("t", "u")
        elif self.is_bound_scalar_value_parameter(raw_param_type, node, "compute"):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        elif self.is_raw_buffer_parameter_type(raw_param_type, node):
            namespace = "buffer"
            attribute_names = {"binding", "buffer"}
            prefixes = ("b", "u", "t")
        else:
            return None

        binding = self.explicit_resource_binding_index(node, attribute_names, prefixes)
        if binding is None:
            return None
        return (
            namespace,
            binding,
            self.parameter_resource_binding_count(raw_param_type, node),
            getattr(node, "name", "<anonymous>"),
        )

    def parameter_resource_binding_count(self, raw_param_type, node=None):
        array_size = None
        if self.is_structured_buffer_type(raw_param_type):
            array_size = self.structured_buffer_parameter_array_size(
                raw_param_type, node
            )
        else:
            array_type = self.resource_array_parameter(raw_param_type, node)
            if array_type is not None:
                _, array_size = array_type
        return self.resource_array_count(array_size)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return (
                self.type_name_string(parameter.param_type)
                if getattr(parameter.param_type, "generic_args", None)
                else parameter.param_type
            )
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "float"

    def validate_stage_parameter_resource_bindings(self, parameters, func_name=None):
        used_bindings = {}
        self.reserve_global_parameter_resource_bindings(used_bindings, func_name)
        for parameter in parameters or []:
            metadata = self.parameter_resource_binding_metadata(
                self.parameter_raw_type(parameter), parameter
            )
            if metadata is None:
                continue
            namespace, binding, resource_count, name = metadata
            self.reserve_resource_binding_range(
                used_bindings,
                "Metal",
                namespace,
                binding,
                resource_count,
                name,
            )
