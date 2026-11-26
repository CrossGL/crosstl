"""Structured buffers in the GLSL generator: fixed resource array sizes, buffer blocks, bindings, declarations, layouts and call names."""

from ..ast import (
    ArrayNode,
    ArrayType,
    ConstructorNode,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from ..resource_identity import resource_identity_marker
from ..semantics.shader_records import (
    is_shader_record_buffer,
    shader_record_memory_layout,
)
from .array_utils import (
    evaluate_literal_int_expression,
    parse_array_type,
    split_array_type_suffix,
)
from .ast_walk import walk_ast_values
from .glsl_buffer_layout import glsl_buffer_block_node_type
from .glsl_zero_values import zero_array_value
from .image_access_contracts import (
    image_access_diagnostic_name,
    is_glsl_storage_image_type,
)
from .push_constant_blocks import (
    glsl_push_constant_struct_block,
    glsl_push_constant_struct_pads,
    is_push_constant,
)
from .stage_utils import collect_stage_local_cbuffers


class OpenGLStructuredBufferMixin:
    """Structured buffer members of ``GLSLCodeGen``."""

    def fixed_resource_array_size(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is None:
                return None
            base_type = self.convert_type_node_to_string(vtype.element_type)
            if not self.is_resource_array_hint_type(base_type):
                return None
            size = self.literal_int_value(vtype.size, self.literal_int_constants)
            return size if size is not None and size > 0 else None
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None
        base_type, size = parse_array_type(type_string)
        if size is None or not self.is_resource_array_hint_type(base_type):
            return None
        return max(size, 1)

    def is_unsized_sampled_texture_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_inferable_resource_array_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_inferable_resource_array_type(base_type)

    def is_unsized_resource_array_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            if vtype.size is not None:
                return False
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return self.is_resource_array_hint_type(base_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return False
        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return False
        base_type, size = parse_array_type(type_string)
        return size is None and self.is_resource_array_hint_type(base_type)

    def collect_functions(self, root):
        functions = []
        for node in self.walk_ast(root):
            if hasattr(node, "body") and hasattr(node, "parameters"):
                functions.append(node)
        return functions

    def collect_generic_type_parameter_names(self, root):
        names = set()
        for node in self.walk_ast(root):
            for param in getattr(node, "generic_params", []) or []:
                name = getattr(param, "name", None)
                if name:
                    names.add(str(name))
        return names

    def walk_ast(self, root):
        return walk_ast_values(root)

    def is_sampler_type(self, vtype):
        return self.resource_base_type(vtype) in {"sampler", "comparison_sampler"}

    def structured_buffer_type_name(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0]

    def is_structured_buffer_type(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "StructuredBuffer",
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def is_constant_buffer_type(self, vtype):
        return str(self.resource_base_type(vtype)).split("<", 1)[0] == "ConstantBuffer"

    def constant_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return None
        return type_name.split("<", 1)[1][:-1].strip()

    def structured_buffer_requires_counter(self, vtype):
        return self.structured_buffer_type_name(vtype) in {
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def structured_buffer_element_type(self, vtype):
        type_name = str(self.resource_base_type(vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return "uint"
        element_type = type_name.split("<", 1)[1][:-1].strip()
        return self.map_type(element_type)

    def structured_buffer_source_element_type(self, vtype):
        base_type, _declared_size = self.structured_buffer_array_base_type(vtype)
        type_name = str(self.resource_base_type(base_type or vtype))
        if "<" not in type_name or not type_name.endswith(">"):
            return None
        return type_name.split("<", 1)[1][:-1].strip()

    def structured_buffer_array_parameter_info(
        self, vtype, param_name=None, function_name=None
    ):
        base_type, declared_size = self.structured_buffer_array_base_type(vtype)
        if base_type is None or not self.is_structured_buffer_type(base_type):
            return None

        count = declared_size
        if count is None and function_name and param_name:
            hint = self.function_resource_array_size_hints.get(function_name, {}).get(
                param_name
            )
            count = self.positive_int_value(hint)
        if count is None or count <= 0:
            return None
        return {
            "base_type": base_type,
            "element_type": self.structured_buffer_element_type(base_type),
            "count": count,
        }

    def structured_buffer_array_base_type(self, vtype):
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            base_type = self.convert_type_node_to_string(vtype.element_type)
            return base_type, self.positive_int_value(vtype.size)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            return None, None

        type_string = str(vtype)
        if "[" not in type_string or "]" not in type_string:
            return None, None
        base_type, size = parse_array_type(type_string)
        return base_type, self.positive_int_value(size)

    def collect_unsupported_structured_buffer_array_functions(self, functions):
        unsupported_functions = {}
        for func in functions or []:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            unsupported_params = []
            for index, param in enumerate(
                getattr(func, "parameters", getattr(func, "params", []))
            ):
                raw_type = getattr(param, "param_type", getattr(param, "vtype", None))
                base_type, _ = self.structured_buffer_array_base_type(raw_type)
                if base_type is None or not self.is_structured_buffer_type(base_type):
                    continue
                if (
                    self.structured_buffer_array_parameter_info(
                        raw_type, getattr(param, "name", None), func_name
                    )
                    is not None
                ):
                    continue
                unsupported_params.append(
                    {"index": index, "name": getattr(param, "name", None)}
                )
            if unsupported_params:
                unsupported_functions[func_name] = {
                    "indices": {param["index"] for param in unsupported_params},
                    "params": unsupported_params,
                }
        return unsupported_functions

    def skipped_function_parameter_indices(self, func_name):
        skipped = set(self.function_sampler_parameter_indices.get(func_name, set()))
        preserved = getattr(self, "preserved_sampler_parameters", {}).get(
            func_name, set()
        )
        if preserved:
            parameter_names = self.function_parameter_names.get(func_name, [])
            skipped = {
                index
                for index in skipped
                if index >= len(parameter_names)
                or parameter_names[index] not in preserved
            }
        unsupported = self.unsupported_structured_buffer_array_functions.get(
            func_name, {}
        )
        skipped.update(unsupported.get("indices", set()))
        return skipped

    def unsupported_structured_buffer_array_function_body(
        self, func, unsupported_info, indent
    ):
        indent_str = "    " * indent
        code = ""
        func_name = getattr(func, "name", "<anonymous>")
        for param in unsupported_info.get("params", []):
            param_name = param.get("name") or "<anonymous>"
            self.refuse_lowering(
                f"structured buffer array parameter {func_name}.{param_name}: "
                "fixed array size is required for helper lowering"
            )

        return_type = (
            self.current_function_return_type
            or self.type_name_string(getattr(func, "return_type", None))
            or "void"
        )
        mapped_return_type = self.map_type(return_type)
        if mapped_return_type == "void" or self.current_stage_output is not None:
            return f"{code}{indent_str}return;\n"
        return f"{code}{indent_str}return {self.zero_value_expression(return_type)};\n"

    def zero_value_expression(self, vtype):
        if isinstance(vtype, ArrayType):
            return zero_array_value(self, vtype, self.map_type(vtype))
        mapped_type = self.map_type(vtype)
        if mapped_type == "void":
            return ""
        if mapped_type == "bool":
            return "false"
        if mapped_type == "uint":
            return "0u"
        if mapped_type == "int":
            return "0"
        if mapped_type in {"float", "double"}:
            return "0.0"
        if mapped_type.startswith("bvec"):
            return f"{mapped_type}(false)"
        if mapped_type.startswith("uvec"):
            return f"{mapped_type}(0u)"
        if mapped_type.startswith("ivec"):
            return f"{mapped_type}(0)"
        if mapped_type.startswith(("vec", "dvec", "mat", "dmat")):
            return f"{mapped_type}(0.0)"
        array = zero_array_value(self, None, mapped_type)
        if array is not None:
            return array
        struct = self.glsl_struct_node(mapped_type)
        if struct is not None:
            member_values = []
            for member in getattr(struct, "members", []) or []:
                member_type = getattr(
                    member, "member_type", getattr(member, "vtype", None)
                )
                member_values.append(self.zero_value_expression(member_type))
            return f"{mapped_type}({', '.join(member_values)})"
        return f"{mapped_type}(0)"

    def positive_int_value(self, value):
        literal = self.literal_int_value(value, self.literal_int_constants)
        if literal is None and isinstance(value, int) and not isinstance(value, bool):
            literal = value
        if literal is None and isinstance(value, str) and value.isdigit():
            literal = int(value)
        return literal if literal is not None and literal > 0 else None

    def structured_buffer_access_metadata(self, node):
        choices = self.resource_access_metadata_choices(node)
        self.validate_resource_access_metadata_consistency(node, choices)
        if not choices:
            return None
        return choices[0][1]

    def structured_buffer_effective_access(self, vtype, node=None):
        type_name = self.structured_buffer_type_name(vtype)
        explicit_access = self.structured_buffer_access_metadata(node)

        if type_name == "StructuredBuffer":
            if explicit_access in {"write", "read_write"}:
                access_name = image_access_diagnostic_name(explicit_access)
                node_name = self.resource_node_name(node, "<unnamed>")
                raise OpenGLRefusal(
                    "OpenGL StructuredBuffer resource "
                    f"'{node_name}' cannot use {access_name} access metadata"
                )
            return "read"

        return explicit_access

    def structured_buffer_memory_qualifiers(self, vtype, node=None):
        effective_access = self.structured_buffer_effective_access(vtype, node)
        qualifiers = set()
        for qualifier in self.resource_memory_qualifiers(node).split():
            qualifiers.add(qualifier)
        if self.structured_buffer_type_name(vtype) == "StructuredBuffer":
            qualifiers.add("readonly")
        elif effective_access == "read":
            qualifiers.add("readonly")
        elif effective_access == "write":
            qualifiers.add("writeonly")

        if "globallycoherent" in qualifiers:
            qualifiers.add("coherent")

        order = ("coherent", "volatile", "restrict", "readonly", "writeonly")
        return " ".join(qualifier for qualifier in order if qualifier in qualifiers)

    def record_structured_buffer_access_metadata(
        self, name, vtype, node=None, parameter=False
    ):
        if not name or not self.is_structured_buffer_type(vtype):
            return
        access = self.structured_buffer_effective_access(vtype, node)
        if access is None or access == "read_write":
            return
        target = (
            self.current_structured_buffer_access_parameters
            if parameter
            else self.structured_buffer_variable_accesses
        )
        target[name] = access

    def structured_buffer_block_declaration(
        self, vtype, name, binding, array_size=None, node=None
    ):
        element_type = self.structured_buffer_element_type(vtype)
        self.union_storage.validate_buffer(element_type)
        storage_declarations = ""
        if array_size is None:
            storage = self.glsl_hlsl_structured_storage_type(node, name, element_type)
            if storage is not None:
                element_type, storage_declarations = storage
        element_type = self.float_atomic_storage.declaration(element_type, name)
        definitions = self.float_atomic_storage.take_definitions()
        memory_qualifiers = self.structured_buffer_memory_qualifiers(vtype, node)
        qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
        layout = self.glsl_resource_layout_prefix("std430", binding=binding, node=node)
        block_name = self.glsl_generated_module_identifier(
            ("structured-buffer-block", name),
            f"{name}Buffer",
        )
        definitions += resource_identity_marker(
            node, name if array_size is not None else block_name
        )
        if array_size is not None:
            instance_member = "data"
            self.structured_buffer_instance_members[name] = instance_member
            array_suffix = f"[{array_size}]" if array_size else "[]"
            return (
                definitions + f"{layout} {qualifier_prefix}buffer "
                f"{block_name} {{ {element_type} {instance_member}[]; }} "
                f"{name}{array_suffix};\n"
            )
        return storage_declarations + (
            definitions + f"{layout} {qualifier_prefix}buffer "
            f"{block_name} {{ {element_type} {name}[]; }};\n"
        )

    def structured_buffer_counter_resource_name(self, name):
        return self.glsl_generated_module_identifier(
            ("structured-buffer-counter", name),
            f"{name}Counter",
        )

    def structured_buffer_counter_parameter_name(self, name):
        return self.glsl_synthetic_local_identifier(f"{name}Counter")

    def structured_buffer_counter_block_declaration(
        self, name, binding, array_size=None
    ):
        counter_member = self.structured_buffer_counter_resource_name(name)
        block_name = self.glsl_generated_module_identifier(
            ("structured-buffer-counter-block", name),
            f"{name}CounterBuffer",
        )
        if array_size is not None:
            counter_member = "counter"
            counter_instance = self.glsl_generated_module_identifier(
                ("structured-buffer-counter-instance", name),
                f"{name}Counters",
            )
            self.structured_buffer_counter_members[name] = counter_member
            self.structured_buffer_counter_instances[name] = counter_instance
            array_suffix = f"[{array_size}]" if array_size else "[]"
            layout = self.glsl_resource_layout_prefix("std430", binding=binding)
            return (
                f"{layout} buffer {block_name} "
                f"{{ uint {counter_member}[]; }} {counter_instance}{array_suffix};\n"
            )

        self.structured_buffer_counter_members[name] = counter_member
        layout = self.glsl_resource_layout_prefix("std430", binding=binding)
        return f"{layout} buffer {block_name} {{ uint {counter_member}[]; }};\n"

    def glsl_buffer_block_attribute(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return attr
        return None

    def glsl_buffer_block_layout(self, node):
        attr = self.glsl_buffer_block_attribute(node)
        arguments = getattr(attr, "arguments", []) if attr is not None else []
        layout_parts = [
            self.attribute_value_to_string(argument) for argument in arguments or []
        ]
        layout_parts = [part for part in layout_parts if part]
        if is_shader_record_buffer(node) and not any(
            self.normalized_glsl_buffer_block_layout_part(part) == "shaderrecordext"
            for part in layout_parts
        ):
            # ``layout(shaderRecordEXT, std430) buffer B {...} b;`` parses with shaderRecordEXT as its own attribute.
            layout_parts.insert(0, "shaderRecordEXT")
            if len(layout_parts) == 1:
                layout_parts.append(shader_record_memory_layout(node))
        if layout_parts:
            self.validate_glsl_buffer_block_layout_parts(node, layout_parts)
            return ", ".join(layout_parts)
        return "std430"

    def validate_glsl_buffer_block_layout_parts(self, node, layout_parts):
        node_name = self.resource_node_name(node, "<unnamed>")
        seen = {}
        memory_layouts = []
        for layout_part in layout_parts:
            normalized = self.normalized_glsl_buffer_block_layout_part(layout_part)
            if normalized in seen:
                raise OpenGLRefusal(
                    "Duplicate OpenGL buffer block layout metadata for "
                    f"'{node_name}': {layout_part}"
                )
            seen[normalized] = layout_part
            if normalized in self.GLSL_BUFFER_BLOCK_MEMORY_LAYOUT_NAMES:
                memory_layouts.append(layout_part)

        if len(memory_layouts) > 1:
            first_layout = memory_layouts[0]
            conflicting_layout = memory_layouts[1]
            raise OpenGLRefusal(
                "Conflicting OpenGL buffer block memory layout metadata for "
                f"'{node_name}': {first_layout} differs from {conflicting_layout}"
            )

    def normalized_glsl_buffer_block_layout_part(self, layout_part):
        return str(layout_part).strip().lower().replace("_", "")

    def is_shader_record_buffer_block(self, node):
        layout_parts = {
            part.strip().lower().replace("_", "")
            for part in self.glsl_buffer_block_layout(node).split(",")
        }
        return "shaderrecordext" in layout_parts

    def is_glsl_buffer_block_variable(self, node, vtype=None):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if (
            self.glsl_buffer_block_attribute(node) is None
            and "buffer" not in qualifiers
        ):
            return False
        type_name = self.resource_base_type(vtype or glsl_buffer_block_node_type(node))
        return str(type_name) in self.structs_by_name

    def collect_glsl_buffer_block_struct_names(self, global_vars):
        names = set()
        for node in global_vars:
            node_type = glsl_buffer_block_node_type(node)
            if not self.is_glsl_buffer_block_variable(node, node_type):
                continue
            type_name = self.resource_base_type(node_type)
            names.add(str(type_name))
        return names

    def collect_layout_bound_struct_uniform_names(self, global_vars):
        names = set()
        for node in global_vars:
            node_type = self.resource_node_type(node)
            if self.is_constant_buffer_type(node_type):
                element_type = self.constant_buffer_element_type(node_type)
                if element_type:
                    names.add(str(element_type))
            if not self.is_layout_bound_struct_uniform(node, node_type):
                continue
            names.add(str(self.resource_base_type(node_type)))
        return names

    def collect_glsl_interface_value_struct_names(
        self, ast, structs, global_vars, functions
    ):
        candidates = set(self.glsl_buffer_block_struct_names)
        candidates.update(self.layout_bound_struct_uniform_names)
        if not candidates:
            return set()

        value_structs = set()

        def referenced_types(type_node):
            return {
                name
                for name in candidates
                if self.type_node_references_name(type_node, name)
            }

        def record_type(type_node):
            value_structs.update(referenced_types(type_node))

        for function in functions or []:
            record_type(getattr(function, "return_type", None))
            for parameter in (
                getattr(function, "parameters", getattr(function, "params", [])) or []
            ):
                parameter_type = self.resource_node_type(parameter)
                referenced = referenced_types(parameter_type)
                if not referenced:
                    continue
                if self.is_stage_entry_resource_parameter(parameter):
                    continue
                value_structs.update(referenced)

            for node in self.walk_ast(getattr(function, "body", None)):
                if isinstance(node, (VariableNode, ArrayNode)):
                    record_type(self.resource_node_type(node))
                    record_type(getattr(node, "element_type", None))
                if isinstance(node, ConstructorNode):
                    record_type(getattr(node, "constructor_type", None))
                record_type(getattr(node, "target_type", None))

        for struct in structs or []:
            if getattr(struct, "name", None) in candidates:
                continue
            for member in getattr(struct, "members", []) or []:
                record_type(getattr(member, "member_type", None))

        for node in global_vars or []:
            node_type = self.resource_node_type(node)
            if self.is_glsl_buffer_block_variable(node, node_type):
                continue
            if self.is_layout_bound_struct_uniform(node, node_type):
                continue
            if self.is_constant_buffer_type(node_type):
                # A ConstantBuffer<T> declares a uniform block of T's members, not a T value.
                continue
            record_type(node_type)

        for node in getattr(ast, "constants", []) or []:
            record_type(getattr(node, "const_type", getattr(node, "vtype", None)))
        return value_structs

    def prepare_glsl_interface_block_names(self, ast, target_stage, structs):
        ordinary_struct_names = {
            getattr(node, "name", None)
            for node in structs or []
            if getattr(node, "name", None)
            and not self.is_glsl_interface_block_struct(node)
            and (
                node.name in self.glsl_interface_value_struct_names
                or (
                    node.name not in self.layout_bound_struct_uniform_names
                    and node.name not in self.glsl_buffer_block_struct_names
                )
            )
        }
        for qualifier in ("uniform", "buffer"):
            for name in ordinary_struct_names:
                self.reserve_glsl_interface_block_name(qualifier, name)

        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        cbuffers += collect_stage_local_cbuffers(ast, target_stage)
        for cbuffer in cbuffers:
            self.reserve_glsl_interface_block_name(
                "uniform", getattr(cbuffer, "name", None)
            )

        for node in structs or []:
            if not self.is_glsl_interface_block_struct(node):
                continue
            block_name = self.glsl_interface_block_name(node)
            for qualifier in self.glsl_interface_block_qualifiers(node):
                qualifier = str(qualifier).lower()
                if qualifier in {"uniform", "buffer", "in", "out"}:
                    self.reserve_glsl_interface_block_name(qualifier, block_name)

    def reserve_glsl_interface_block_name(self, qualifier, name):
        if not name:
            return
        emitted_name = (
            str(name)
            if str(name) in self.GLSL_BUILTIN_INTERFACE_BLOCK_NAMES
            else self.glsl_module_identifier_name(name)
        )
        self.glsl_interface_block_names_by_qualifier.setdefault(
            str(qualifier).lower(), set()
        ).add(emitted_name)

    def unique_glsl_interface_block_name(self, qualifier, source_name, instance_name):
        qualifier = str(qualifier).lower()
        used_names = self.glsl_interface_block_names_by_qualifier.setdefault(
            qualifier, set()
        )
        if str(source_name) in self.GLSL_BUILTIN_INTERFACE_BLOCK_NAMES:
            return str(source_name)
        source_name = self.glsl_sanitized_identifier_base(
            source_name,
            fallback="InterfaceBlock",
        )
        if source_name not in used_names and not self.glsl_identifier_requires_escape(
            source_name
        ):
            used_names.add(source_name)
            return source_name

        instance_name = self.glsl_sanitized_identifier_base(
            instance_name,
            fallback="instance",
        )
        base_name = f"{source_name}_{instance_name}"
        candidate = self.glsl_unique_identifier(base_name, used_names)
        used_names.add(candidate)
        return candidate

    def glsl_buffer_block_declaration(
        self, node, vtype, name, binding, array_suffix=""
    ):
        source_block_name = str(self.resource_base_type(vtype))
        struct = self.structs_by_name.get(source_block_name)
        if struct is None:
            return None

        block_name = self.unique_glsl_interface_block_name(
            "buffer", source_block_name, name
        )

        layout = self.glsl_buffer_block_layout(node)
        memory_qualifiers = self.resource_memory_qualifiers(node)
        qualifier_prefix = f"{memory_qualifiers} " if memory_qualifiers else ""
        layout_prefix = self.glsl_resource_layout_prefix(
            layout, binding=binding, node=node
        )
        code = f"{layout_prefix} {qualifier_prefix}buffer {block_name} {{\n"
        for member in getattr(struct, "members", []) or []:
            # A member of structure type keeps its row_major, which reaches the structure's matrices (GLSL 4.60
            # section 4.4.5); row_major_members has already transposed the matrix members and dropped theirs.
            orientation = self.glsl_uniform_member_orientation(member)
            code += self.generate_struct_member_declaration(
                member,
                indent=f"    {orientation}",
                struct_name=source_block_name,
                preserve_unsized_arrays=True,
            )
        code += f"}} {name}{array_suffix};\n"
        return code

    def is_sampled_texture_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return (
            mapped_type != "sampler"
            and self.is_opaque_resource_type(mapped_type)
            and not mapped_type.startswith(("image", "iimage", "uimage"))
        )

    def is_storage_image_type(self, vtype):
        mapped_type = self.map_type(self.resource_base_type(vtype))
        return self.is_opaque_resource_type(mapped_type) and is_glsl_storage_image_type(
            mapped_type
        )

    def is_inferable_resource_array_type(self, vtype):
        return self.is_sampled_texture_type(vtype) or self.is_storage_image_type(vtype)

    def is_resource_array_hint_type(self, vtype):
        return self.is_inferable_resource_array_type(
            vtype
        ) or self.is_structured_buffer_type(vtype)

    def is_inferable_resource_type(self, vtype):
        return (
            self.is_sampler_type(vtype)
            or self.is_structured_buffer_type(vtype)
            or self.is_inferable_resource_array_type(vtype)
        )

    def canonical_sampled_resource_type(self, type_name):
        """Return GLSL sampler spelling for HLSL-style sampled resources."""
        if not isinstance(type_name, str):
            return None
        base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
        return self.sampled_resource_type_aliases.get(base_type)

    def canonical_storage_resource_type(self, type_name):
        """Return GLSL image spelling for HLSL-style writable resources."""
        if not isinstance(type_name, str):
            return None

        base_type = type_name.split("[", 1)[0].strip()
        base_name = base_type.split("<", 1)[0].strip()
        image_type = self.storage_resource_type_aliases.get(base_name)
        if image_type is None:
            return None

        if "<" not in base_type or ">" not in base_type:
            return image_type

        value_type = base_type.split("<", 1)[1].rsplit(">", 1)[0].strip()
        value_type = value_type.split(",", 1)[0].strip().lower()
        if value_type in {"int", "i32"}:
            return f"i{image_type}"
        if value_type in {"uint", "u32"}:
            return f"u{image_type}"
        return image_type

    def canonical_resource_type(self, type_name):
        """Return canonical GLSL sampler/image/resource type for known aliases."""
        if isinstance(type_name, str):
            base_type = type_name.split("[", 1)[0].split("<", 1)[0].strip()
            sampler_type = self.sampler_state_type_aliases.get(base_type)
            if sampler_type is not None:
                return sampler_type
        return self.canonical_sampled_resource_type(
            type_name
        ) or self.canonical_storage_resource_type(type_name)

    def resource_base_type(self, vtype):
        if vtype is None:
            return ""
        if hasattr(vtype, "element_type") and str(type(vtype)).find("ArrayType") != -1:
            return self.resource_base_type(vtype.element_type)
        if hasattr(vtype, "name") or hasattr(vtype, "element_type"):
            vtype = self.convert_type_node_to_string(vtype)
        vtype = str(vtype)
        if "[" in vtype and "]" in vtype:
            base_type, _ = parse_array_type(vtype)
            return self.canonical_resource_type(base_type) or base_type
        return self.canonical_resource_type(vtype) or vtype

    def resource_array_count(self, size):
        if size is None:
            return 1
        resolved_size = self.literal_int_value(size, self.literal_int_constants)
        if resolved_size is not None:
            return max(resolved_size, 1)
        size_str = str(size)
        return max(int(size_str), 1) if size_str.isdigit() else 1

    def resource_node_type(self, node):
        return getattr(
            node,
            "var_type",
            getattr(node, "param_type", getattr(node, "vtype", "float")),
        )

    def resource_node_name(self, node, default=None):
        return getattr(node, "name", getattr(node, "variable_name", default))

    def is_layout_bound_struct_uniform(self, node, vtype):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "uniform" not in qualifiers:
            return False
        return str(self.resource_base_type(vtype)) in self.structs_by_name and (
            self.explicit_resource_binding_index(node) is not None
            or is_push_constant(node)
        )

    def glsl_constant_buffer_block_declaration(
        self, vtype, var_name, binding, array_suffix="", node=None
    ):
        struct_name = self.constant_buffer_element_type(vtype)
        if struct_name not in self.structs_by_name:
            return ""
        return self.glsl_struct_uniform_block_declaration(
            struct_name,
            var_name,
            binding,
            array_suffix,
            node=node,
        )

    def glsl_struct_uniform_block_declaration(
        self, vtype, var_name, binding, array_suffix="", node=None
    ):
        struct_name = str(self.resource_base_type(vtype))
        self.union_storage.validate_buffer(struct_name)
        struct = self.structs_by_name[struct_name]
        block_name = self.unique_glsl_interface_block_name(
            "uniform", struct_name, var_name
        )
        layout = self.glsl_resource_layout_prefix("std140", binding=binding, node=node)
        code = resource_identity_marker(node, var_name)
        code += f"{layout} uniform {block_name} {{\n"
        pads = glsl_push_constant_struct_pads(
            self, node, struct, var_name, array_suffix
        )
        for member in getattr(struct, "members", []) or []:
            code += pads.get(getattr(member, "name", None), "")
            code += self.generate_struct_member_declaration(
                member,
                struct_name=struct_name,
                preserve_unsized_arrays=True,
            )
        code += f"}} {var_name}{array_suffix};\n"
        return glsl_push_constant_struct_block(
            self,
            node,
            struct,
            block_name,
            var_name,
            array_suffix,
            code,
            resource_identity_marker(node, var_name),
        )

    def pointer_buffer_structured_type(self, node, node_type):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "buffer" not in qualifiers or not hasattr(node_type, "pointee_type"):
            return None
        element_type = self.convert_type_node_to_string(node_type.pointee_type)
        return f"RWStructuredBuffer<{element_type}>"

    def resource_declaration_shape(self, node):
        node_type = self.resource_node_type(node)
        node_name = self.resource_node_name(node, "")
        resource_count = 1
        array_size = None
        array_suffix = ""

        pointer_buffer_type = self.pointer_buffer_structured_type(node, node_type)
        if pointer_buffer_type is not None:
            return pointer_buffer_type, None, "", resource_count

        if (
            hasattr(node_type, "element_type")
            and str(type(node_type)).find("ArrayType") != -1
        ):
            vtype = self.convert_type_node_to_string(node_type.element_type)
            array_size = (
                self.generate_expression(node_type.size)
                if node_type.size
                else (
                    self.resource_array_size_hints.get(node_name, "")
                    if self.is_resource_array_hint_type(vtype)
                    else ""
                )
            )
            if not array_size and self.is_inferable_resource_array_type(vtype):
                array_size = "1"
            # An array of arrays: the outer dimension is the first suffix, the element's own ones follow it.
            vtype, element_suffix = split_array_type_suffix(vtype)
            array_suffix = (f"[{array_size}]" if array_size else "[]") + element_suffix
            resource_count = self.resource_array_count(
                node_type.size if node_type.size else array_size
            )
            return vtype, array_size, array_suffix, resource_count

        if hasattr(node_type, "name") or hasattr(node_type, "element_type"):
            return self.convert_type_node_to_string(node_type), None, "", 1

        vtype = str(node_type) if node_type is not None else "float"
        if "[" in vtype and "]" in vtype:
            base_type, parsed_size = parse_array_type(vtype)
            array_size = parsed_size
            if array_size is None and self.is_inferable_resource_array_type(base_type):
                array_size = "1"
            array_suffix = f"[{parsed_size}]" if parsed_size else "[]"
            if array_size and not parsed_size:
                array_suffix = f"[{array_size}]"
            resource_count = self.resource_array_count(parsed_size)
            if array_size and not parsed_size:
                resource_count = self.resource_array_count(array_size)
            return base_type, array_size, array_suffix, resource_count

        return vtype, None, "", resource_count

    def resource_declaration_identity(self, node):
        self.validate_resource_access_metadata_operands(node)
        vtype, _, array_suffix, resource_count = self.resource_declaration_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            return (
                "buffer binding",
                "glsl_buffer_block",
                str(self.resource_base_type(vtype)),
                array_suffix,
                resource_count,
                self.glsl_buffer_block_layout(node),
                self.resource_memory_qualifiers(node),
            )
        if self.is_structured_buffer_type(vtype):
            return (
                "buffer binding",
                self.structured_buffer_type_name(vtype),
                self.structured_buffer_element_type(vtype),
                array_suffix,
                resource_count,
            )
        if self.is_constant_buffer_type(vtype):
            return (
                "uniform buffer binding",
                "ConstantBuffer",
                self.constant_buffer_element_type(vtype),
                array_suffix,
                resource_count,
            )

        mapped_type = self.map_resource_type_with_format(vtype, node)
        if mapped_type == "sampler" or not self.is_opaque_resource_type(mapped_type):
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if qualifiers & {"uniform", "buffer"}:
                return (
                    "qualified resource",
                    self.map_type(vtype),
                    array_suffix,
                    resource_count,
                    self.glsl_variable_layout_prefix(node),
                    self.glsl_variable_qualifier_prefix(node),
                )
            return None
        namespace = (
            "image binding" if self.is_storage_image_type(vtype) else "texture binding"
        )
        return (
            namespace,
            mapped_type,
            array_suffix,
            resource_count,
            self.image_format_qualifier(mapped_type, node),
            self.resource_memory_qualifiers(node),
            self.resource_precision_qualifiers(node),
        )

    def global_resource_shape(self, node):
        vtype, _, _, resource_count = self.resource_declaration_shape(node)
        return vtype, resource_count

    def global_resource_binding_metadata(self, node):
        classified = self.global_resource_binding_namespace(node)
        if classified is None:
            return None
        namespace, resource_count, var_name = classified
        binding = self.explicit_resource_binding_index(node)
        if binding is None:
            return None
        return namespace, binding, resource_count, var_name

    def global_resource_binding_namespace(self, node):
        """``(namespace, count, name)`` of a resource declaration that takes an OpenGL binding point."""
        source_var_name = self.resource_node_name(node)
        if not source_var_name:
            return None
        var_name = self.glsl_global_identifier_name(source_var_name)

        vtype, resource_count = self.global_resource_shape(node)
        if self.is_glsl_buffer_block_variable(node, vtype):
            namespace = "buffer binding"
        elif self.is_structured_buffer_type(vtype):
            namespace = "buffer binding"
        elif self.is_constant_buffer_type(vtype):
            namespace = "uniform buffer binding"
        elif self.is_layout_bound_struct_uniform(node, vtype):
            namespace = "uniform buffer binding"
        else:
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if mapped_type == "sampler":
                if not self.should_preserve_sampler_resource(source_var_name):
                    return None
                namespace = "sampler binding"
            elif mapped_type == "atomic_uint":
                # Atomic counters have their own binding table (:mod:`.glsl_atomic_counters`).
                return None
            elif not self.is_opaque_resource_type(mapped_type):
                return None
            else:
                namespace = (
                    "image binding"
                    if self.is_storage_image_type(vtype)
                    else "texture binding"
                )
        return namespace, resource_count, var_name

    def reserve_explicit_global_resource_bindings(self, global_vars, used_bindings):
        self.glsl_register_target_bindings = {}
        self.glsl_hlsl_register_by_id = {}
        used_source_bindings = {}
        deferred_registers = []
        for order, node in enumerate(global_vars):
            metadata = self.global_resource_binding_metadata(node)
            if metadata is None:
                continue
            namespace, binding, resource_count, var_name = metadata
            register = self.hlsl_register_binding(node)
            if register is not None:
                deferred_registers.append(
                    (node, namespace, register, max(resource_count or 1, 1), var_name)
                    + (order,)
                )
                continue
            source_binding = self.source_descriptor_binding_index(node)
            if source_binding is not None:
                # Vulkan descriptors are unique per (set, binding), whatever OpenGL binding point they get.
                self.reserve_resource_binding_range(
                    used_source_bindings,
                    "source descriptor",
                    f"descriptor set {self.resource_descriptor_set_index(node)} binding",
                    self.raw_descriptor_binding_index(node),
                    1,
                    var_name,
                )
            if self.should_remap_descriptor_array_target_binding(node, resource_count):
                continue
            if self.resource_binding_range_conflicts(
                used_bindings,
                namespace,
                binding,
                resource_count,
                var_name,
            ):
                if self.should_remap_stage_entry_resource_binding_conflict(
                    used_bindings,
                    namespace,
                    binding,
                    resource_count,
                    var_name,
                ):
                    continue
            self.reserve_resource_binding_range(
                used_bindings,
                "OpenGL",
                namespace,
                binding,
                resource_count,
                var_name,
            )
        self.reserve_deferred_hlsl_register_bindings(deferred_registers, used_bindings)

    def next_resource_binding(self, binding_cursors, namespace):
        return binding_cursors.get(namespace, 0)

    def next_available_resource_binding(
        self, used_bindings, binding_cursors, namespace, count
    ):
        count = max(count or 1, 1)
        binding = self.next_resource_binding(binding_cursors, namespace)
        return self.next_available_resource_binding_from(
            used_bindings, namespace, binding, count
        )

    def next_available_resource_binding_from(
        self, used_bindings, namespace, binding, count
    ):
        # The first window of ``count`` slots at or after ``binding`` that no used
        # range overlaps. One sweep in start order finds it: a range that
        # overlaps the current window forces the window past its end, and once a
        # range starts after the window every later one does too.
        count = max(count or 1, 1)
        for used_start, used_end, _ in sorted(
            used_bindings.get(namespace, []), key=lambda used: used[0]
        ):
            if used_end < binding:
                continue
            if used_start > binding + count - 1:
                break
            binding = used_end + 1
        return binding

    def resource_binding_range_conflict_names(
        self, used_bindings, namespace, start, count, name=None
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        conflicts = []
        for used_start, used_end, used_name in used_bindings.get(namespace, []):
            if start <= used_end and used_start <= end:
                if (
                    name is not None
                    and used_start == start
                    and used_end == end
                    and used_name == name
                ):
                    continue
                conflicts.append(used_name)
        return conflicts

    def resource_binding_range_conflicts(
        self, used_bindings, namespace, start, count, name=None
    ):
        return bool(
            self.resource_binding_range_conflict_names(
                used_bindings, namespace, start, count, name
            )
        )

    def should_remap_stage_entry_resource_binding_conflict(
        self, used_bindings, namespace, start, count, name
    ):
        if name not in self.stage_entry_resource_declaration_names:
            return False
        conflicts = self.resource_binding_range_conflict_names(
            used_bindings,
            namespace,
            start,
            count,
            name,
        )
        return bool(conflicts) and all(
            used_name in self.stage_entry_resource_declaration_names
            for used_name in conflicts
        )

    def should_remap_stage_entry_cbuffer_binding_conflict(
        self,
        used_bindings,
        binding,
        name,
    ):
        if name not in self.stage_entry_constant_value_cbuffer_names:
            return False
        conflicts = self.resource_binding_range_conflict_names(
            used_bindings,
            "uniform buffer binding",
            binding,
            1,
            name,
        )
        return bool(conflicts) and all(
            used_name in self.stage_entry_constant_value_cbuffer_names
            for used_name in conflicts
        )

    def opengl_resource_binding_for_declaration(
        self, node, used_bindings, binding_cursors, namespace, count
    ):
        explicit_binding = self.explicit_resource_binding_index(node)
        if explicit_binding is None:
            return self.next_available_resource_binding(
                used_bindings, binding_cursors, namespace, count
            )
        node_name = self.resource_node_name(node)
        should_remap_overlap = self.should_remap_stage_entry_resource_binding_conflict(
            used_bindings,
            namespace,
            explicit_binding,
            count,
            node_name,
        )
        if (
            self.should_remap_descriptor_array_target_binding(node, count)
            or should_remap_overlap
        ):
            return self.next_available_resource_binding_from(
                used_bindings, namespace, explicit_binding, count
            )
        return explicit_binding

    def advance_resource_binding(self, binding_cursors, namespace, start, count):
        count = max(count or 1, 1)
        binding_cursors[namespace] = max(
            binding_cursors.get(namespace, 0), start + count
        )

    def reserve_resource_binding_range(
        self, used_bindings, target, namespace, start, count, name
    ):
        count = max(count or 1, 1)
        end = start + count - 1
        ranges = used_bindings.setdefault(namespace, [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise OpenGLRefusal(
                    f"Conflicting {target} resource binding for '{name}': "
                    f"{self.resource_binding_range_label(namespace, start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.resource_binding_range_label(namespace, used_start, used_end)}"
                )
        ranges.append((start, end, name))

    def resource_binding_range_label(self, namespace, start, end):
        if start == end:
            return f"{namespace} {start}"
        return f"{namespace} {start}-{end}"

    def literal_int_value(self, expr, constants=None):
        return evaluate_literal_int_expression(expr, constants)

    def initial_literal_int_constants(self, func):
        visible_constants = dict(self.literal_int_constants)
        for param in getattr(func, "parameters", []) or []:
            visible_constants.pop(getattr(param, "name", None), None)
        return visible_constants

    def visible_literal_int_constants(self, func):
        visible_constants = self.initial_literal_int_constants(func)

        for node in self.walk_ast(getattr(func, "body", [])):
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if not name:
                    continue

                visible_constants.pop(name, None)
                if "const" not in getattr(node, "qualifiers", []):
                    continue

                value = self.literal_int_value(
                    getattr(node, "initial_value", None), visible_constants
                )
                if value is not None:
                    visible_constants[name] = value

        return visible_constants

    def function_call_name(self, call):
        func_expr = getattr(call, "function", None)
        if func_expr is None:
            func_expr = getattr(call, "name", None)
        if isinstance(func_expr, str):
            return func_expr
        if hasattr(func_expr, "name") and isinstance(func_expr.name, str):
            return func_expr.name
        return None
