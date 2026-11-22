"""Local declarations of the Metal generator: global resource and constant buffer parameters, local variable types and resource aliases, local storage and pointer types, address spaces, program-scope groupshared and value globals, readonly local aliases and atomic value types."""

from ...backend.common_ast import VariableNode as BackendVariableNode
from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    IdentifierNode,
    NamedType,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import MetalRefusal
from ..validation import collect_duplicate_cbuffer_member_names
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
    parse_array_type,
    split_array_type_suffix,
)
from .builtin_result_types import (
    C_FAMILY,
    inferred_value_type,
)
from .image_access_contracts import (
    explicit_image_access,
    explicit_image_format,
)
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    stage_matches,
)


class MetalLocalDeclarationMixin:
    """Local declaration members of ``MetalCodeGen``."""

    def append_required_global_resource_parameters(self, params_str, func_name):
        resource_params = []
        for (
            texture_variable,
            texture_type,
            array_size,
        ) in self.required_function_textures(func_name):
            texture_name = getattr(texture_variable, "name", None)
            if texture_name:
                resource_params.append(
                    self.format_resource_parameter(
                        self.metal_coherent_texture_type(
                            texture_type, None, texture_name
                        ),
                        texture_name,
                        array_size,
                    )
                )
        for (
            acceleration_structure_variable,
            acceleration_structure_type,
            array_size,
        ) in self.required_function_acceleration_structures(func_name):
            acceleration_structure_name = getattr(
                acceleration_structure_variable, "name", None
            )
            if acceleration_structure_name:
                resource_params.append(
                    self.format_resource_parameter(
                        acceleration_structure_type,
                        acceleration_structure_name,
                        array_size,
                    )
                )
        for (
            visible_function_table_variable,
            visible_function_table_type,
            array_size,
        ) in self.required_function_visible_function_tables(func_name):
            visible_function_table_name = getattr(
                visible_function_table_variable, "name", None
            )
            if visible_function_table_name:
                resource_params.append(
                    self.format_visible_function_table_parameter(
                        visible_function_table_type,
                        visible_function_table_name,
                        array_size,
                    )
                )
        for (
            intersection_function_table_variable,
            intersection_function_table_type,
            array_size,
        ) in self.required_function_intersection_function_tables(func_name):
            intersection_function_table_name = getattr(
                intersection_function_table_variable, "name", None
            )
            if intersection_function_table_name:
                resource_params.append(
                    self.format_intersection_function_table_parameter(
                        intersection_function_table_type,
                        intersection_function_table_name,
                        array_size,
                    )
                )
        for (
            buffer_variable,
            buffer_type,
            array_size,
            address_space,
        ) in self.required_function_metal_buffer_resources(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_metal_buffer_resource_parameter(
                        buffer_type,
                        buffer_name,
                        array_size,
                        address_space,
                        buffer_variable,
                    )
                )
        for (
            buffer_variable,
            buffer_type,
            array_size,
        ) in self.required_function_structured_buffers(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_structured_buffer_parameter(
                        buffer_type, buffer_name, array_size, buffer_variable
                    )
                )
                if self.structured_buffer_requires_length(buffer_name):
                    resource_params.append(
                        self.format_structured_buffer_length_parameter(
                            buffer_name, array_size
                        )
                    )
                if self.structured_buffer_requires_counter(buffer_type):
                    resource_params.append(
                        self.format_structured_buffer_counter_parameter(
                            buffer_name, array_size
                        )
                    )
        for (
            buffer_variable,
            block,
            array_size,
        ) in self.required_function_glsl_buffer_blocks(func_name):
            buffer_name = getattr(buffer_variable, "name", None)
            if buffer_name:
                resource_params.append(
                    self.format_glsl_buffer_block_parameter(
                        block, buffer_name, array_size
                    )
                )
        for sampler_variable, array_size in self.required_function_samplers(func_name):
            sampler_name = getattr(sampler_variable, "name", None)
            if sampler_name:
                resource_params.append(
                    self.format_resource_parameter("sampler", sampler_name, array_size)
                )
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def cbuffer_parameter_name(self, cbuffer):
        parameter_name = self.cbuffer_parameter_names.get(id(cbuffer))
        if parameter_name:
            return parameter_name
        return self.default_cbuffer_parameter_name(cbuffer)

    def default_cbuffer_parameter_name(self, cbuffer):
        name = getattr(cbuffer, "name", "constants")
        if not name:
            return "constants"
        name = name[:1].lower() + name[1:]
        # ``Kernel`` would become the keyword ``kernel``, which MSL rejects as a parameter name.
        if name in self.METAL_RESERVED_LOCAL_IDENTIFIERS:
            return f"{name}_"
        return name

    def collect_cbuffer_parameter_names(self, cbuffers, reserved_names=None):
        parameter_names = {}
        used_names = set(reserved_names or [])
        for cbuffer in cbuffers:
            base_name = self.default_cbuffer_parameter_name(cbuffer)
            parameter_name = base_name
            suffix = 1
            while parameter_name in used_names:
                parameter_name = f"{base_name}{suffix}"
                suffix += 1
            used_names.add(parameter_name)
            parameter_names[id(cbuffer)] = parameter_name
        return parameter_names

    def collect_cbuffer_member_references(self, cbuffers):
        references = {}
        self.metal_cbuffer_member_types = {}
        ambiguous_members = collect_duplicate_cbuffer_member_names(cbuffers)
        layout_plans = getattr(self, "metal_hlsl_cbuffer_plans", {}) or {}
        for cbuffer in cbuffers:
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            plan = layout_plans.get(id(cbuffer))
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", None)
                if not member_name or member_name in ambiguous_members:
                    continue
                reference = f"{parameter_name}.{member_name}"
                layout_member = plan.member(member_name) if plan is not None else None
                if (
                    layout_member is not None
                    and layout_member.read_conversion
                    and layout_member.array_count is None
                ):
                    reference = f"{layout_member.read_conversion}({reference})"
                references[member_name] = reference
                member_type = (
                    getattr(member, "member_type", None)
                    or getattr(member, "vtype", None)
                    or getattr(member, "var_type", None)
                )
                if member_type is not None:
                    self.metal_cbuffer_member_types[member_name] = (
                        self.type_name_string(member_type)
                    )
        self.ambiguous_cbuffer_members = ambiguous_members
        return references

    def generate_tail_expression_statement(self, stmt, indent=0):
        if not getattr(stmt, "is_tail_expression", False):
            return None
        return_type = self.type_name_string(self.current_function_return_type)
        if not return_type or return_type == "void":
            return None

        indent_str = "    " * indent
        return_wrapper = self.current_function_return_wrapper
        if return_wrapper is not None:
            value = self.generate_expression_with_expected(
                stmt.expression,
                return_wrapper["source_type"],
            )
            return f"{indent_str}return {return_wrapper['struct_name']}{{{value}}};\n"

        value = self.generate_expression_with_expected(stmt.expression, return_type)
        return f"{indent_str}return {value};\n"

    def generate_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        if not lines:
            return ""

        result = ""
        for line in lines:
            stripped = line.strip()
            terminator = (
                ""
                if stripped.endswith((";", "{", ":"))
                or stripped == "}"
                or stripped.startswith(("case ", "default:"))
                else ";"
            )
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def local_variable_declared_type(self, stmt, refuse_untyped=False):
        """The type a local declaration is emitted with.

        A local with no written type takes its initializer's; when that cannot be inferred either, analyses get
        ``None`` and the declaration itself (``refuse_untyped``) is refused rather than given a guessed type.
        """
        var_type = getattr(stmt, "var_type", None)
        if var_type is None:
            var_type = getattr(stmt, "vtype", None)
        if var_type is None:
            var_type = inferred_value_type(
                self, getattr(stmt, "initial_value", None), C_FAMILY
            )
        declared_type = self.type_name_string(var_type)
        if declared_type or not refuse_untyped:
            return declared_type or None
        return self.refuse_lowering(
            f"local declaration: the type of '{getattr(stmt, 'name', '?')}' is not "
            "written and cannot be inferred from its initializer",
            "float",
        )

    def local_variable_texture_alias_resource_type(self, node, declared_type):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None

        initial_type = self.texture_argument_resource_type(initial_value)
        if initial_type is None:
            return None

        declared_type_name = self.type_name_string(declared_type)
        if not self.is_texture_or_image_resource_type(declared_type_name):
            if (
                self.local_variable_type_node(node) is None
                and explicit_image_access(node, self.attribute_value_to_string) is None
                and explicit_image_format(node, self.attribute_value_to_string) is None
            ):
                return initial_type
            return None

        declared_resource_type = self.map_resource_type_with_format(
            declared_type_name, node
        )
        self.validate_local_resource_alias_compatibility(
            node, declared_type_name, declared_resource_type, initial_type
        )
        if (
            explicit_image_access(node, self.attribute_value_to_string) is None
            and explicit_image_format(node, self.attribute_value_to_string) is None
        ):
            return initial_type
        if explicit_image_access(node, self.attribute_value_to_string) is None:
            return initial_type
        return declared_resource_type

    def validate_local_resource_alias_compatibility(
        self, node, declared_type_name, declared_resource_type, initial_type
    ):
        alias_name = getattr(node, "name", "<anonymous>")
        declared_compat_type = self.local_resource_alias_compatibility_type(
            declared_resource_type
        )
        initial_compat_type = self.local_resource_alias_compatibility_type(initial_type)
        if declared_compat_type != initial_compat_type:
            raise MetalRefusal(
                f"Metal local resource alias '{alias_name}' declares "
                f"{declared_type_name} but initializer has {initial_type}"
            )

        declared_access = explicit_image_access(node, self.attribute_value_to_string)
        if declared_access is not None:
            initial_access = self.storage_image_access_mode(initial_type)
            if initial_access is not None and declared_access != initial_access:
                raise MetalRefusal(
                    f"Metal local resource alias '{alias_name}' declares "
                    f"access::{declared_access} but initializer has "
                    f"access::{initial_access}"
                )

        declared_format = explicit_image_format(node, self.attribute_value_to_string)
        if declared_format is not None:
            initial_format = self.image_resource_format(
                getattr(node, "initial_value", None)
            )
            if initial_format is not None and declared_format != initial_format:
                raise MetalRefusal(
                    f"Metal local resource alias '{alias_name}' declares "
                    f"format {declared_format} but initializer has {initial_format}"
                )

    def local_resource_alias_compatibility_type(self, resource_type):
        resource_type = self.resource_base_type(resource_type)
        if self.is_storage_image_resource(resource_type):
            return self.storage_image_access_agnostic_type(resource_type)
        return resource_type

    def local_variable_texture_alias_declaration_type(self, node, resource_type):
        array_size = self.texture_argument_resource_array_size(
            getattr(node, "initial_value", None)
        )
        if array_size is None:
            return resource_type
        return self.format_resource_array_type(resource_type, array_size)

    def texture_argument_resource_array_size(self, texture_arg):
        if isinstance(texture_arg, ArrayAccessNode):
            return None
        member_array_size = self.struct_member_resource_array_size(texture_arg)
        if member_array_size is not None:
            return member_array_size
        texture_name = self.expression_name(texture_arg)
        if not texture_name:
            return None

        local_type = self.local_variable_types.get(texture_name)
        _element_type, array_size = self.metal_array_type_parts(local_type)
        if array_size is not None:
            return array_size

        if texture_name in self.current_texture_parameter_array_sizes:
            return self.current_texture_parameter_array_sizes[texture_name]

        for texture_variable, _, _texture_type, array_size in self.texture_variables:
            if getattr(texture_variable, "name", None) == texture_name:
                return array_size
        return None

    def format_resource_array_type(self, resource_type, array_size):
        return f"array<{resource_type}, {array_size or '1'}>"

    def metal_array_type_parts(self, type_name):
        if type_name is None:
            return None, None
        type_text = str(type_name).strip()
        if not (type_text.startswith("array<") and type_text.endswith(">")):
            return None, None

        inner = type_text[len("array<") : -1]
        depth = 0
        for index, char in enumerate(inner):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                return inner[:index].strip(), inner[index + 1 :].strip()
        return inner.strip(), None

    def metal_array_element_type(self, type_name):
        element_type, _array_size = self.metal_array_type_parts(type_name)
        return element_type

    def local_variable_sampler_alias_resource_type(self, node, declared_type):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None

        initial_name = self.expression_name(initial_value)
        initial_type = self.expression_result_type(initial_value)
        initial_array_element_type = self.metal_array_element_type(initial_type)
        if not (
            self.is_sampler_type(initial_type)
            or self.is_sampler_type(initial_array_element_type)
            or initial_name in self.sampler_variable_names()
            or self.struct_member_is_sampler_resource(
                self.struct_resource_member_node(initial_value)
            )
        ):
            return None

        declared_type_name = self.type_name_string(declared_type)
        if self.local_variable_type_node(node) is None:
            array_size = self.sampler_argument_resource_array_size(initial_value)
            if array_size is not None:
                return self.format_resource_array_type("sampler", array_size)
            return "sampler"
        if self.is_sampler_type(declared_type_name):
            return "sampler"
        return None

    def local_variable_acceleration_structure_alias_resource_type(
        self, node, declared_type
    ):
        initial_type = self.acceleration_structure_argument_resource_type(
            getattr(node, "initial_value", None)
        )
        if initial_type is None:
            return None

        declared_type_name = self.type_name_string(declared_type)
        if self.local_variable_type_node(node) is None:
            return initial_type
        if self.is_acceleration_structure_type(declared_type_name):
            declared_resource_type = self.map_resource_type_with_format(
                declared_type_name, node
            )
            if declared_resource_type != initial_type:
                alias_name = getattr(node, "name", "<anonymous>")
                raise MetalRefusal(
                    f"Metal local acceleration structure alias '{alias_name}' "
                    f"declares {declared_type_name} but initializer has "
                    f"{initial_type}"
                )
            return declared_resource_type
        return None

    def acceleration_structure_argument_resource_type(self, argument):
        if argument is None:
            return None

        argument_type = self.expression_result_type(argument)
        if argument_type and self.is_acceleration_structure_type(argument_type):
            if self.metal_type_is_pointer_like(argument_type):
                return None
            return self.map_resource_type_with_format(argument_type)

        argument_name = self.expression_name(argument)
        if not argument_name:
            return None

        local_type = self.local_variable_types.get(argument_name)
        if local_type and self.is_acceleration_structure_type(local_type):
            if self.metal_type_is_pointer_like(local_type):
                return None
            return self.map_resource_type_with_format(local_type)

        for (
            acceleration_structure_variable,
            _,
            mapped_type,
            array_size,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None) != argument_name:
                continue
            if array_size is not None and not isinstance(argument, ArrayAccessNode):
                return None
            return mapped_type
        return None

    def sampler_argument_resource_array_size(self, sampler_arg):
        if isinstance(sampler_arg, ArrayAccessNode):
            return None

        member_array_size = self.struct_member_resource_array_size(sampler_arg)
        if member_array_size is not None:
            return member_array_size

        sampler_name = self.expression_name(sampler_arg)
        if not sampler_name:
            return None

        local_type = self.local_variable_types.get(sampler_name)
        _element_type, array_size = self.metal_array_type_parts(local_type)
        if array_size is not None:
            return array_size

        if sampler_name in self.current_sampler_parameter_array_sizes:
            return self.current_sampler_parameter_array_sizes[sampler_name]

        for sampler_variable, _, array_size in self.sampler_variables:
            if getattr(sampler_variable, "name", None) == sampler_name:
                return array_size
        return None

    def sampler_argument_resource_type(self, sampler_arg):
        sampler_name = self.expression_name(sampler_arg)
        if sampler_name in self.sampler_variable_names():
            return "sampler"

        arg_type = self.expression_result_type(sampler_arg)
        array_element_type = self.metal_array_element_type(arg_type)
        if self.is_sampler_type(arg_type) or self.is_sampler_type(array_element_type):
            return "sampler"

        if self.struct_member_is_sampler_resource(
            self.struct_resource_member_node(sampler_arg)
        ):
            return "sampler"
        return None

    def record_local_texture_alias_sampler_source(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            self.current_texture_alias_sources[node.name] = initial_value

    def record_local_image_alias_metadata(self, node):
        image_format = explicit_image_format(node, self.attribute_value_to_string)
        if image_format is None:
            image_format = self.image_resource_format(
                getattr(node, "initial_value", None)
            )
        if image_format is not None:
            self.current_image_format_parameters[node.name] = image_format

    def local_variable_type_node(self, stmt):
        return getattr(stmt, "var_type", None) or getattr(stmt, "vtype", None)

    def metal_native_width_local_array_declaration_ids(self, func):
        declarations = {}
        for current in self.metal_render_walks.body_nodes_of_type(
            func, (VariableNode, BackendVariableNode)
        ):
            name = getattr(current, "name", None)
            declared_type = self.local_variable_declared_type(current)
            base_type, array_suffix = split_array_type_suffix(
                self.type_name_string(declared_type)
            )
            native_type = self.metal_native_narrow_bitcast_storage_type(base_type)
            if name and array_suffix and native_type is not None:
                declarations.setdefault(name, []).append((current, native_type))
        if not declarations:
            return set()

        required = set()
        for current in self.metal_render_walks.body_nodes_of_type(
            func, FunctionCallNode
        ):
            function_name = self.function_call_name(current)
            arguments = list(
                getattr(current, "arguments", getattr(current, "args", [])) or []
            )
            parameters = self.metal_user_function_parameter_nodes_for_call(
                function_name,
                arguments,
            )
            if parameters is None:
                continue
            for index, argument in enumerate(arguments):
                if index >= len(parameters):
                    continue
                argument_name = self.assignment_target_root_name(argument)
                candidates = declarations.get(argument_name, ())
                if not candidates:
                    continue
                parameter = parameters[index]
                parameter_type = getattr(
                    parameter,
                    "param_type",
                    getattr(parameter, "vtype", None),
                )
                if not isinstance(parameter_type, PointerType):
                    continue
                pointee_type = parameter_type.pointee_type
                native_pointee = self.metal_native_narrow_bitcast_storage_type(
                    pointee_type
                )
                if native_pointee is None:
                    continue
                required.update(
                    id(declaration)
                    for declaration, native_type in candidates
                    if native_type == native_pointee
                )
        return required

    def metal_local_storage_declaration_type(self, node, declared_type):
        """Map local storage while preserving proven native array element widths.

        Arithmetic values intentionally use CrossGL's widened fixed-width integer
        mapping.  Metal threadgroup storage retains its native physical ABI, and a
        fixed local array passed to a ``thread T*`` helper must retain the helper's
        ``char``/``uchar``/``short``/``ushort`` pointee width.
        """
        pointee = self.local_variable_crossgl_pointer_pointee(node)
        if pointee is not None:
            # A CrossGL `ptr<T>` local is a pointer into the address space
            # the qualifier prefix supplies, not a `ptr<T>` value: Metal has
            # no such type, so it is declared as a C-style pointer.
            return self.metal_crossgl_pointer_local_type(node, pointee)
        mapped_type = self.map_type(declared_type)
        aggregate_type, array_suffix = split_array_type_suffix(mapped_type)
        if aggregate_type in self.structs_by_name and (
            aggregate_type == self.current_function_name
            or aggregate_type in self.function_return_types
            or aggregate_type in self.local_variable_types
        ):
            # Expanding source `auto` must not look up a same-named value.
            tag = "union" if aggregate_type in self.metal_union_layouts else "struct"
            mapped_type = f"{tag} {aggregate_type}{array_suffix}"
        if (
            self.local_variable_address_space(node) != "threadgroup"
            and id(node)
            not in self.current_metal_native_width_local_array_declaration_ids
        ):
            return mapped_type

        source_type = self.type_name_string(declared_type)
        base_type, array_suffix = split_array_type_suffix(source_type)
        native_type = self.metal_native_narrow_bitcast_storage_type(base_type)
        if native_type is None:
            return mapped_type
        return f"{native_type}{array_suffix}"

    def array_indirect_element_type_node(self, raw_type):
        if not self.is_array_type_node(raw_type):
            return None
        element_type = raw_type
        while self.is_array_type_node(element_type):
            element_type = getattr(element_type, "element_type", None)
        return (
            element_type
            if isinstance(element_type, (PointerType, ReferenceType))
            else None
        )

    def local_variable_crossgl_pointer_pointee(self, node):
        """The pointee type node of a CrossGL ``ptr<T>`` local, else ``None``.

        The CrossGL parser represents ``ptr<T>`` as a ``NamedType``, not a
        ``PointerType``, so without this the address-space alias machinery
        below misses it and Metal output copies ``ptr<T>`` unchanged, which
        the Metal compiler rejects as an undeclared identifier.
        """
        raw_type = self.local_variable_type_node(node)
        if (
            isinstance(raw_type, NamedType)
            and raw_type.name == "ptr"
            and len(raw_type.generic_args) == 1
        ):
            return raw_type.generic_args[0]
        return None

    def is_crossgl_pointer_type(self, vtype):
        """Whether ``vtype`` is a CrossGL ``ptr<T>``, as a ``NamedType`` or spelled."""
        if isinstance(vtype, NamedType):
            return vtype.name == "ptr" and len(vtype.generic_args) == 1
        type_name = str(self.type_name_string(vtype) or "").replace(" ", "")
        return (
            type_name.startswith("ptr<")
            and type_name.endswith(">")
            and "," not in type_name
        )

    def metal_crossgl_pointer_local_type(self, node, pointee):
        """The C-style Metal pointer type of a CrossGL ``ptr<T>`` local.

        The CrossGL spelling carries no address space, so the local takes the
        one its initializer points into. Without an initializer, as for
        ``var p: ptr<f32>;``, or with one whose address space is unknown, the
        local is refused instead of being declared a ``thread`` pointer. A
        pointer into device, constant or threadgroup memory keeps the native
        width of a ``char`` or ``short`` pointee, as the buffers and
        threadgroup arrays it points into do.
        """
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return self.refuse_lowering(
                f"pointer local: '{node.name}' has no initializer, so the address "
                "space it points into is unknown",
                "float*",
            )
        if (
            self.local_variable_initializer_address_space(node) is None
            # Branches in different address spaces are refused on their own.
            and self.argument_address_space_conflict(initial_value) is None
        ):
            return self.refuse_lowering(
                f"pointer local: the address space the initializer of '{node.name}' "
                "points into is unknown",
                "float*",
            )
        pointee_type = self.map_type(self.convert_type_node_to_string(pointee))
        if self.local_variable_address_space(node) != "thread":
            pointee_type = (
                self.metal_native_narrow_bitcast_storage_type(pointee) or pointee_type
            )
        return f"{pointee_type}*"

    def local_variable_inherits_readonly_initializer(self, node):
        """Whether a pointer local aliases a read-only parameter.

        A CrossGL pointer local carries no qualifiers of its own, so a local
        such as ``var src: ptr<f32> = in_;`` must inherit the constness of
        the read-only storage parameter it aliases; otherwise Metal output
        would drop ``const`` from the pointee and reject the initialization.
        """
        if self.local_variable_crossgl_pointer_pointee(node) is None:
            return False
        return self.points_into_readonly_metal_storage(
            getattr(node, "initial_value", None)
        )

    def local_variable_is_array_alias(self, node):
        raw_type = self.local_variable_type_node(node)
        if self.is_array_type_node(raw_type):
            return True
        type_name = self.type_name_string(raw_type)
        if type_name is None:
            type_name = self.local_variable_declared_type(node)
        return "[" in str(type_name) and str(type_name).rstrip().endswith("]")

    def local_pointer_initializer_needs_address(self, node, initial_value):
        if not isinstance(self.local_variable_type_node(node), PointerType):
            return False
        if initial_value is None:
            return False
        if (
            isinstance(initial_value, UnaryOpNode)
            and getattr(initial_value, "operator", getattr(initial_value, "op", None))
            == "&"
        ):
            return False
        if self.metal_type_is_pointer_like(self.expression_result_type(initial_value)):
            return False
        return self.assignment_target_root_name(initial_value) is not None

    def expression_is_raw_pointer_identifier(self, value):
        if not isinstance(value, (IdentifierNode, VariableNode, BackendVariableNode)):
            return False
        name = getattr(value, "name", getattr(value, "identifier", None))
        if not name:
            return False
        for parameter in self.function_parameter_nodes.get(
            self.current_function_name,
            [],
        ):
            if getattr(parameter, "name", None) != name:
                continue
            raw_type = getattr(
                parameter,
                "param_type",
                getattr(parameter, "vtype", None),
            )
            if isinstance(raw_type, PointerType):
                return True
        return any(
            getattr(node, "name", None) == name
            for node, _binding, _type, _size, _space in (
                self.metal_buffer_resource_variables
            )
        )

    def pointer_assignment_needs_address(self, target, value):
        target_type = self.expression_result_type(target)
        if self.pointer_pointee_type_name(target_type) is None:
            return False
        if value is None:
            return False
        if self.expression_is_raw_pointer_identifier(value):
            return False
        if (
            isinstance(value, UnaryOpNode)
            and getattr(value, "operator", getattr(value, "op", None)) == "&"
        ):
            return False
        if self.metal_type_is_pointer_like(self.expression_result_type(value)):
            return False
        return self.assignment_target_root_name(value) is not None

    def metal_pointer_offset_is_integral_scalar_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if not type_name or "[" in type_name:
            return False
        # Reference expressions supply the referenced value as the offset.
        type_name = type_name.strip()
        if type_name.endswith("&"):
            type_name = type_name[:-1].rstrip()
        integral_names = {
            "char",
            "uchar",
            "short",
            "ushort",
            "int",
            "uint",
            "long",
            "ulong",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int32_t",
            "uint32_t",
            "int64_t",
            "uint64_t",
            "size_t",
        }
        mapped_type = self.map_type(type_name)
        return type_name in integral_names or mapped_type in integral_names

    def address_space_assignment_diagnostic(self, target, value, operator="="):
        target_type = self.expression_result_type(target)
        # A CrossGL `ptr<T>` local is a pointer too; `*p` and `p[i]` are not
        # reassignments of it, so only the bare name is checked.
        if self.pointer_pointee_type_name(target_type) is None and not (
            isinstance(target, IdentifierNode)
            and self.is_crossgl_pointer_type(target_type)
        ):
            return None
        if operator in {"+=", "-="}:
            value_type = self.expression_result_type(value)
            if self.metal_pointer_offset_is_integral_scalar_type(value_type):
                return None
            target_name = self.assignment_target_display_name(target) or "<target>"
            value_name = self.assignment_target_display_name(value) or "<expr>"
            return self.refuse_lowering(
                f"pointer offset assignment: value '{value_name}' is not an integer "
                f"scalar for target '{target_name}'",
                "(void)0",
            )

        expected_address_space = self.argument_address_space(target)
        if expected_address_space is None:
            return None

        target_name = self.assignment_target_display_name(target) or "<target>"
        conflict = self.argument_address_space_conflict(value)
        if conflict is not None:
            return self.refuse_lowering(
                "address-space assignment: value "
                f"{self.address_space_conflict_description(conflict)} use "
                f"different address spaces; assignment to '{target_name}' "
                f"requires {expected_address_space}",
                "(void)0",
            )

        actual_address_space = self.argument_address_space(value)
        if (
            actual_address_space is None
            or actual_address_space == expected_address_space
        ):
            return None

        value_name = self.assignment_target_display_name(value) or "<expr>"
        return self.refuse_lowering(
            f"address-space assignment: value '{value_name}' uses "
            f"{actual_address_space} address space but target '{target_name}' uses "
            f"{expected_address_space}",
            "(void)0",
        )

    def local_reference_assignment_target(self, target):
        if not isinstance(target, BinaryOpNode):
            return None
        if getattr(target, "operator", getattr(target, "op", None)) != "&":
            return None
        type_name = self.expression_name(getattr(target, "left", None))
        name = self.expression_name(getattr(target, "right", None))
        if not type_name or not name:
            return None
        if type_name in self.local_variable_types:
            return None
        if (
            type_name not in self.type_mapping
            and type_name not in self.struct_member_types
        ):
            return None
        return type_name, name

    def generate_local_reference_assignment_declaration(self, target, value, rhs):
        reference_target = self.local_reference_assignment_target(target)
        if reference_target is None:
            return None

        type_name, name = reference_target
        address_space = self.argument_address_space(value) or "thread"
        self.local_variable_types[name] = f"{type_name}&"
        self.current_address_space_variables[name] = address_space

        readonly_key = self.readonly_metal_mesh_payload_key(value)
        root_name = self.assignment_target_root_name(value)
        qualifier = address_space
        if readonly_key is not None:
            reason = self.current_readonly_metal_mesh_payload_reasons[readonly_key]
            self.current_readonly_metal_mesh_payload_parameters.add(name)
            self.current_readonly_metal_mesh_payload_reasons[name] = reason
            qualifier = "const object_data"
        elif root_name in self.current_readonly_metal_parameters:
            reason = self.current_readonly_metal_parameter_reasons[root_name]
            self.current_readonly_metal_parameters.add(name)
            self.current_readonly_metal_parameter_reasons[name] = reason
            if address_space != "constant":
                qualifier = f"const {address_space}"

        mapped_type = self.map_type(type_name)
        local_name = self.metal_local_identifier_name(name)
        return f"{qualifier} {mapped_type}& {local_name} = {rhs}"

    def metal_program_scope_groupshared_stage_entries(self, ast, target_stage):
        entries = []
        seen = set()

        def add(stage_name, func, stage_local_variables=None):
            if func is None or id(func) in seen:
                return
            normalized_stage_name = normalize_stage_name(stage_name)
            if not normalized_stage_name:
                return
            if not stage_matches(target_stage, normalized_stage_name):
                return
            seen.add(id(func))
            entries.append(
                (normalized_stage_name, func, list(stage_local_variables or []))
            )

        for func in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(func)
            if stage_name in self.stage_entry_types():
                add(stage_name, func)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            add(
                normalize_stage_name(stage_type),
                getattr(stage, "entry_point", None),
                getattr(stage, "local_variables", []),
            )

        return entries

    def metal_stage_local_variables_with_lowered_program_groupshared(
        self, func, stage_local_variables=None
    ):
        lowered_globals = (
            self.metal_lowered_program_scope_groupshared_globals_by_function.get(
                id(func), []
            )
        )
        if not lowered_globals:
            return stage_local_variables
        return list(lowered_globals) + list(stage_local_variables or [])

    def metal_entry_local_names(self, func, stage_local_variables=None):
        names = {
            getattr(variable, "name", None)
            for variable in stage_local_variables or []
            if getattr(variable, "name", None)
        }
        for node in self.metal_render_walks.body_nodes_of_type(
            func, (VariableNode, BackendVariableNode)
        ):
            name = getattr(node, "name", None)
            if name:
                names.add(name)
        return names

    def metal_variable_name(self, node):
        return getattr(node, "name", getattr(node, "variable_name", None))

    def program_scope_groupshared_initializer(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            return initial_value
        return getattr(node, "value", None)

    def global_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        for address_space in (
            "constant",
            "device",
            "threadgroup_imageblock",
            "threadgroup",
            "thread",
        ):
            if address_space in qualifiers:
                return f"{address_space} "
        if qualifiers & {"const", "readonly"}:
            return "constant "
        return "constant "

    def is_program_scope_groupshared_global(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        }
        return bool(
            (qualifiers | attributes)
            & {"groupshared", "shared", "threadgroup", "workgroup"}
        )

    def record_metal_program_scope_value_global(
        self, name, vtype, qualifier, node=None
    ):
        if not name or not self.global_value_variable_requires_initializer(qualifier):
            return
        self.metal_program_scope_value_globals.add(name)
        if self.is_program_scope_groupshared_global(node):
            self.metal_program_scope_groupshared_globals.add(name)
        self.metal_program_scope_value_global_types[name] = self.type_name_string(vtype)

    def global_value_variable_requires_initializer(self, qualifier):
        return str(qualifier or "").strip().startswith("constant")

    def metal_program_scope_global_default_initializer(
        self, mapped_type, array_suffix=""
    ):
        _base_type, mapped_array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix or mapped_array_suffix:
            return "{}"
        return self.metal_default_value_expression(mapped_type)

    def refuse_program_scope_groupshared_global(self, name, node=None):
        """Refuse groupshared storage Metal cannot declare at program scope.

        Other uninitialised value globals are zero-initialised in the constant address space: that is HLSL's
        ``static`` semantics and a valid choice for GLSL, which leaves them undefined, and every store to them is
        refused separately.
        """
        if self.is_program_scope_groupshared_global(node):
            self.refuse_lowering(
                f"program-scope groupshared global: '{name}' cannot be emitted as "
                "Metal threadgroup storage at program scope"
            )

    def local_variable_qualifier(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        attributes = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        }
        if self.local_variable_is_address_space_alias(node):
            raw_type = self.local_variable_type_node(node)
            memory_qualifiers = self.resource_memory_qualifier_prefix(node, raw_type)
            address_space = self.local_variable_address_space(node)
            if address_space is not None:
                const_prefix = ""
                if address_space != "constant" and (
                    "const" in qualifiers
                    or "readonly" in qualifiers
                    or self.local_variable_inherits_readonly_mesh_payload(node)
                    or self.local_variable_inherits_readonly_initializer(node)
                    or self.local_variable_address_space_mismatch_diagnostic(node)
                    is not None
                ):
                    const_prefix = "const "
                return f"{memory_qualifiers}{const_prefix}{address_space} "
        if "threadgroup_imageblock" in qualifiers | attributes:
            return "threadgroup_imageblock "
        if (qualifiers | attributes) & {
            "shared",
            "groupshared",
            "threadgroup",
            "workgroup",
        }:
            return "threadgroup "
        if self.is_metal_atomic_value_type(self.local_variable_declared_type(node)):
            return "threadgroup "
        if self.local_variable_uses_constexpr_qualifier(node, qualifiers):
            return "constexpr "
        if qualifiers & {"const", "constant"}:
            return "const "
        return ""

    def local_variable_uses_constexpr_qualifier(self, node, qualifiers):
        if "constant" not in qualifiers:
            return False
        if self.local_variable_is_address_space_alias(node):
            return False
        if self.local_variable_is_array_alias(node):
            return False

        type_name = self.local_variable_declared_type(node)
        mapped_type = self.map_type(type_name)
        if mapped_type not in {"int", "uint"}:
            return False
        return (
            self.literal_int_value(
                getattr(node, "initial_value", None), self.literal_int_constants
            )
            is not None
        )

    def local_variable_address_space(self, node):
        explicit_address_space = self.normalized_address_space(
            self.parameter_address_space(node)
        )
        initializer_address_space = self.local_variable_initializer_address_space(node)
        if (
            self.local_variable_is_address_space_alias(node)
            and explicit_address_space is not None
            and initializer_address_space is not None
            and explicit_address_space != initializer_address_space
        ):
            return initializer_address_space
        if explicit_address_space is not None:
            return explicit_address_space
        if (
            self.local_variable_is_address_space_alias(node)
            and initializer_address_space is not None
        ):
            return initializer_address_space
        if self.is_metal_atomic_value_type(self.local_variable_declared_type(node)):
            return "threadgroup"
        return "thread"

    def local_variable_initializer_address_space(self, node):
        initial_value = getattr(node, "initial_value", None)
        if initial_value is None:
            return None
        return self.argument_address_space(initial_value)

    def local_variable_inherits_readonly_mesh_payload(self, node):
        if not self.local_variable_is_address_space_alias(node):
            return False
        initial_value = getattr(node, "initial_value", None)
        return self.readonly_metal_mesh_payload_key(initial_value) is not None

    def local_variable_address_space_mismatch_diagnostic(self, node):
        if not self.local_variable_is_address_space_alias(node):
            return None
        explicit_address_space = self.normalized_address_space(
            self.parameter_address_space(node)
        )
        initializer_address_space = self.local_variable_initializer_address_space(node)
        if (
            explicit_address_space is None
            or initializer_address_space is None
            or explicit_address_space == initializer_address_space
        ):
            return None
        initial_value = getattr(node, "initial_value", None)
        initializer_name = self.assignment_target_root_name(initial_value) or "<expr>"
        return self.refuse_lowering(
            f"address-space local alias: initializer '{initializer_name}' uses "
            f"{initializer_address_space} address space but local '{node.name}' was "
            f"declared {explicit_address_space}"
        )

    def local_variable_address_space_conflict_diagnostic(self, node):
        raw_type = self.local_variable_type_node(node)
        if (
            not isinstance(raw_type, (PointerType, ReferenceType))
            and self.local_variable_crossgl_pointer_pointee(node) is None
        ):
            return None
        conflict = self.argument_address_space_conflict(
            getattr(node, "initial_value", None)
        )
        if conflict is None:
            return None
        return self.refuse_lowering(
            "address-space local alias: initializer "
            f"{self.address_space_conflict_description(conflict)} use different "
            "address spaces"
        )

    def local_variable_address_space_conflict_declaration(self, node, declaration):
        raw_type = self.local_variable_type_node(node)
        if not isinstance(raw_type, ReferenceType):
            return declaration
        address_space = self.local_variable_address_space(node)
        referent_type = self.map_resource_type_with_format(
            raw_type.referenced_type, node
        )
        value_declaration = format_c_style_array_declaration(
            referent_type, self.metal_local_identifier_name(node.name)
        )
        return self.maybe_format_unused_local_declaration(
            f"{address_space} {value_declaration}",
            node.name,
        )

    def record_readonly_metal_mesh_payload_local_alias(self, node):
        if self.local_variable_address_space(node) != "object_data":
            return
        if not self.local_variable_is_address_space_alias(node):
            return

        qualifiers = self.parameter_qualifier_names(node)
        reason = None
        if self.local_variable_address_space_mismatch_diagnostic(node) is not None:
            reason = "address-space mismatch alias"
        elif qualifiers & {"const", "constant"}:
            reason = "const-qualified payload parameter"
        else:
            initial_value = getattr(node, "initial_value", None)
            readonly_key = self.readonly_metal_mesh_payload_key(initial_value)
            reason = self.current_readonly_metal_mesh_payload_reasons.get(readonly_key)

        if reason is None:
            return
        self.current_readonly_metal_mesh_payload_parameters.add(node.name)
        self.current_readonly_metal_mesh_payload_reasons[node.name] = reason

    def readonly_metal_local_alias_reason(self, node):
        is_address_space_alias = self.local_variable_is_address_space_alias(node)
        is_array_alias = self.local_variable_is_array_alias(node)
        qualifiers = self.parameter_qualifier_names(node)
        if (
            self.array_indirect_element_type_node(self.local_variable_type_node(node))
            is not None
        ):
            # In `const device T* values[N]`, prefix const/constant qualifies
            # the pointee. The thread-local array slots and their pointer values
            # remain assignable (MLX initializes and advances those slots).
            qualifiers -= {"const", "constant"}
        if not (is_address_space_alias or is_array_alias):
            if "const" in qualifiers:
                return "const-qualified local"
            if "constant" in qualifiers:
                return "constant-qualified local"
            return None
        if (
            is_address_space_alias
            and self.local_variable_address_space(node) == "object_data"
        ):
            return None
        if "const" in qualifiers and is_array_alias:
            return "const-qualified local array"
        if "const" in qualifiers:
            return "const-qualified local alias"
        if "constant" in qualifiers and is_array_alias:
            return "constant-qualified local array"
        if "constant" in qualifiers:
            return "constant address-space local alias"
        if "readonly" in qualifiers:
            return "readonly local alias"
        if self.local_variable_inherits_readonly_initializer(node):
            return "const-qualified local alias"
        return None

    def record_readonly_metal_local_alias(self, node):
        reason = self.readonly_metal_local_alias_reason(node)
        if reason is None:
            # A writable declaration hides a read-only one of the same name, such as a
            # ``const`` local of an earlier loop body.
            self.current_readonly_metal_parameters.discard(node.name)
            self.current_readonly_metal_parameter_reasons.pop(node.name, None)
            return
        self.current_readonly_metal_parameters.add(node.name)
        self.current_readonly_metal_parameter_reasons[node.name] = reason

    def is_metal_atomic_value_type(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name is None:
            return False
        type_name = type_name.split("[", 1)[0].strip()
        return type_name in {
            "atomic_bool",
            "atomic_int",
            "atomic_uint",
        }

    def metal_atomic_array_type_parts(self, vtype):
        type_name = self.type_name_string(vtype)
        if type_name is None or "[" not in type_name:
            return None

        atomic_type = type_name.split("[", 1)[0].strip()
        if not self.is_metal_atomic_value_type(atomic_type):
            return None

        size = None
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            size = evaluate_literal_int_expression(
                getattr(vtype, "size", None),
                self.literal_int_constants,
            )
        if size is None:
            _base_type, size = parse_array_type(type_name)
        if size is None:
            size = self.literal_int_array_size_from_type_name(type_name)
        return atomic_type, size
