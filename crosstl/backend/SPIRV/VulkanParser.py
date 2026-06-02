"""Parser for Vulkan SPIR-V source AST construction."""

import shlex

from .VulkanAst import *
from .VulkanLexer import *


class VulkanParser:
    """Parse Vulkan/SPIR-V style tokens into the Vulkan backend AST."""

    PARAMETER_QUALIFIER_TOKENS = {"IN", "OUT", "INOUT"}
    PRECISION_QUALIFIER_TOKENS = {"HIGHP", "MEDIUMP", "LOWP"}
    LAYOUT_DECLARATION_QUALIFIERS = {
        "centroid",
        "coherent",
        "flat",
        "highp",
        "invariant",
        "lowp",
        "mediump",
        "noperspective",
        "patch",
        "pervertexEXT",
        "precise",
        "readonly",
        "restrict",
        "sample",
        "smooth",
        "volatile",
        "writeonly",
    }
    ASSIGNMENT_TOKENS = (
        "EQUALS",
        "PLUS_EQUALS",
        "MINUS_EQUALS",
        "MULTIPLY_EQUALS",
        "DIVIDE_EQUALS",
        "ASSIGN_AND",
        "ASSIGN_OR",
        "ASSIGN_XOR",
        "ASSIGN_MOD",
        "ASSIGN_SHIFT_LEFT",
        "ASSIGN_SHIFT_RIGHT",
    )
    SPIRV_INTERFACE_STORAGE_CLASSES = {"Input": "IN", "Output": "OUT"}
    SPIRV_INTERFACE_DECORATIONS = {
        "BuiltIn": "builtin",
        "Location": "location",
        "Component": "component",
        "Index": "index",
    }
    SPIRV_DECLARATION_DECORATION_QUALIFIERS = {
        "Centroid": "centroid",
        "Flat": "flat",
        "Invariant": "invariant",
        "NoPerspective": "noperspective",
        "Patch": "patch",
        "Sample": "sample",
    }
    SPIRV_VECTOR_TYPES = {
        ("float", "2"): "vec2",
        ("float", "3"): "vec3",
        ("float", "4"): "vec4",
        ("int", "2"): "ivec2",
        ("int", "3"): "ivec3",
        ("int", "4"): "ivec4",
        ("uint", "2"): "uvec2",
        ("uint", "3"): "uvec3",
        ("uint", "4"): "uvec4",
        ("bool", "2"): "bvec2",
        ("bool", "3"): "bvec3",
        ("bool", "4"): "bvec4",
    }
    SPIRV_BUILTIN_VARIABLE_NAMES = {
        "BaseInstance": "gl_BaseInstance",
        "BaseVertex": "gl_BaseVertex",
        "ClipDistance": "gl_ClipDistance",
        "CullDistance": "gl_CullDistance",
        "FragCoord": "gl_FragCoord",
        "FragDepth": "gl_FragDepth",
        "FrontFacing": "gl_FrontFacing",
        "GlobalInvocationId": "gl_GlobalInvocationID",
        "InstanceIndex": "gl_InstanceID",
        "LocalInvocationId": "gl_LocalInvocationID",
        "LocalInvocationIndex": "gl_LocalInvocationIndex",
        "NumWorkgroups": "gl_NumWorkGroups",
        "PointCoord": "gl_PointCoord",
        "PointSize": "gl_PointSize",
        "Position": "gl_Position",
        "PrimitiveId": "gl_PrimitiveID",
        "SubgroupLocalInvocationId": "gl_SubgroupInvocationID",
        "SubgroupSize": "gl_SubgroupSize",
        "VertexIndex": "gl_VertexID",
        "WorkgroupId": "gl_WorkGroupID",
    }

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.loop_depth = 0
        self.breakable_depth = 0
        self.skip_comments()

    def skip_comments(self):
        while self.current_token[0] in ["COMMENT_SINGLE", "COMMENT_MULTI"]:
            self.eat(self.current_token[0])

    def peek(self, offset):
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][0]
        return None

    def peek_value(self, offset):
        peek_index = self.pos + offset
        if peek_index < len(self.tokens):
            return self.tokens[peek_index][1]
        return None

    def skip_until(self, token_type):
        while self.current_token[0] != token_type and self.current_token[0] != "EOF":
            self.pos += 1
            if self.pos < len(self.tokens):
                self.current_token = self.tokens[self.pos]
            else:
                self.current_token = ("EOF", None)
        return

    def eat(self, token_type):
        if self.current_token[0] == token_type:
            self.pos += 1
            self.current_token = (
                self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)
            )
            self.skip_comments()
        else:
            raise SyntaxError(f"Expected {token_type}, got {self.current_token[0]}")

    def parse(self):
        module = self.parse_module()
        self.eat("EOF")
        return module

    def parse_module(self):
        if self.current_token[0] == "SPIRV_ASSEMBLY":
            code = self.current_token[1]
            self.eat("SPIRV_ASSEMBLY")
            return self.parse_spirv_assembly_module(code)

        functions = []
        structs = []
        global_variables = []
        while self.current_token[0] != "EOF":
            if self.current_token[0] == "PRECISION":
                self.parse_precision_declaration()
            elif self.current_token[0] == "LAYOUT":
                global_variables.append(self.parse_layout())
            elif self.current_token[0] == "STRUCT":
                structs.append(self.parse_struct())
            elif self.current_token[0] == "UNIFORM":
                global_variables.append(self.parse_uniform())
            elif (
                (
                    self.current_token[0]
                    in [
                        "VOID",
                        "FLOAT",
                        "INT",
                        "UINT",
                        "BOOL",
                        "VEC2",
                        "VEC3",
                        "VEC4",
                        "MAT2",
                        "MAT3",
                        "MAT4",
                    ]
                    or self.current_token[1] in VALID_DATA_TYPES
                )
                and self.peek(1) == "IDENTIFIER"
                and self.peek(2) == "LPAREN"
            ):
                functions.append(self.parse_function())
            elif (
                (self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER")
                or (
                    self.current_token[1] in VALID_DATA_TYPES
                    and self.peek(1) == "IDENTIFIER"
                )
                or (
                    self.current_token[0] == "CONST"
                    and (
                        self.peek_value(1) in VALID_DATA_TYPES
                        or self.peek(1) == "IDENTIFIER"
                    )
                )
            ):
                global_variables.append(self.parse_assignment_or_function_call())
            else:
                self.eat(self.current_token[0])
        return ShaderNode(
            functions=functions,
            structs=structs,
            global_variables=global_variables,
        )

    def parse_spirv_assembly_module(self, code):
        instructions = self.parse_spirv_assembly_instructions(code)
        names = {}
        decorations = {}
        member_decorations = {}
        member_names = {}
        types = {}
        constants = {}
        constant_types = {}
        spec_constant_ids = []
        variables = []
        entry_points = []

        for result_id, opcode, operands, _line_number in instructions:
            if opcode == "OpName" and len(operands) >= 2:
                names[operands[0]] = operands[1]
            elif opcode == "OpMemberName" and len(operands) >= 3:
                target, member, name = operands[0], operands[1], operands[2]
                member_names.setdefault(target, {})[member] = name
            elif opcode == "OpDecorate" and len(operands) >= 2:
                target, decoration = operands[0], operands[1]
                decorations.setdefault(target, []).append((decoration, operands[2:]))
            elif opcode == "OpMemberDecorate" and len(operands) >= 3:
                target, member, decoration = operands[0], operands[1], operands[2]
                member_decorations.setdefault(target, []).append(
                    (member, decoration, operands[3:])
                )
            elif opcode == "OpEntryPoint" and len(operands) >= 3:
                entry_points.append(
                    {
                        "execution_model": operands[0],
                        "id": operands[1],
                        "name": operands[2],
                        "interface_ids": operands[3:],
                    }
                )
            elif result_id and opcode == "OpTypeVoid":
                types[result_id] = {"kind": "scalar", "name": "void"}
            elif result_id and opcode == "OpTypeBool":
                types[result_id] = {"kind": "scalar", "name": "bool"}
            elif result_id and opcode == "OpTypeFloat" and operands:
                types[result_id] = {
                    "kind": "scalar",
                    "name": self.spirv_float_type_name(operands[0]),
                }
            elif result_id and opcode == "OpTypeInt" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "scalar",
                    "name": self.spirv_int_type_name(operands[0], operands[1]),
                }
            elif result_id and opcode == "OpTypeVector" and len(operands) >= 2:
                component_type = self.spirv_type_name(operands[0], types)
                types[result_id] = {
                    "kind": "vector",
                    "name": self.SPIRV_VECTOR_TYPES.get(
                        (component_type, operands[1]),
                        f"{component_type}{operands[1]}" if component_type else None,
                    ),
                    "component_type": operands[0],
                    "component_count": operands[1],
                }
            elif result_id and opcode == "OpTypeMatrix" and len(operands) >= 2:
                column_type = types.get(operands[0], {})
                component_type = self.spirv_type_name(
                    column_type.get("component_type"), types
                )
                types[result_id] = {
                    "kind": "matrix",
                    "name": self.spirv_matrix_type_name(
                        component_type,
                        column_type.get("component_count"),
                        operands[1],
                    ),
                    "column_type": operands[0],
                    "column_count": operands[1],
                }
            elif result_id and opcode == "OpTypeArray" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "array",
                    "element_type": operands[0],
                    "length_id": operands[1],
                }
            elif result_id and opcode == "OpTypeRuntimeArray" and len(operands) >= 1:
                types[result_id] = {
                    "kind": "runtime_array",
                    "element_type": operands[0],
                }
            elif result_id and opcode == "OpTypeStruct":
                types[result_id] = {"kind": "struct", "member_types": operands}
            elif result_id and opcode == "OpTypeImage" and len(operands) >= 7:
                sampled_type = self.spirv_type_name(operands[0], types)
                types[result_id] = {
                    "kind": "image",
                    "name": self.spirv_image_type_name(
                        sampled_type,
                        operands[1],
                        operands[2],
                        operands[3],
                        operands[4],
                        operands[5],
                    ),
                    "sampled_type": operands[0],
                    "dim": operands[1],
                    "depth": operands[2],
                    "arrayed": operands[3],
                    "multisampled": operands[4],
                    "sampled": operands[5],
                    "format": operands[6],
                    "access_qualifier": operands[7] if len(operands) >= 8 else None,
                }
            elif result_id and opcode == "OpTypeSampledImage" and operands:
                image_type = types.get(operands[0], {})
                types[result_id] = {
                    "kind": "sampled_image",
                    "name": self.spirv_sampled_image_type_name(image_type, types),
                    "image_type": operands[0],
                }
            elif result_id and opcode == "OpTypeSampler":
                types[result_id] = {"kind": "sampler", "name": "sampler"}
            elif result_id and opcode == "OpTypePointer" and len(operands) >= 2:
                types[result_id] = {
                    "kind": "pointer",
                    "storage_class": operands[0],
                    "type_id": operands[1],
                }
            elif result_id and opcode in {"OpConstant", "OpSpecConstant"}:
                if len(operands) >= 2:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = operands[1]
                    if opcode == "OpSpecConstant":
                        spec_constant_ids.append(result_id)
            elif result_id and opcode in {
                "OpConstantFalse",
                "OpConstantTrue",
                "OpSpecConstantFalse",
                "OpSpecConstantTrue",
            }:
                if operands:
                    constant_types[result_id] = operands[0]
                    constants[result_id] = (
                        "true" if opcode.endswith("True") else "false"
                    )
                    if opcode.startswith("OpSpecConstant"):
                        spec_constant_ids.append(result_id)
            elif result_id and opcode == "OpVariable" and len(operands) >= 2:
                variables.append(
                    {
                        "id": result_id,
                        "pointer_type_id": operands[0],
                        "storage_class": operands[1],
                    }
                )

        entry_interface_ids = {
            interface_id
            for entry_point in entry_points
            for interface_id in entry_point["interface_ids"]
        }
        global_variables = self.spirv_assembly_interface_variables(
            variables,
            entry_interface_ids,
            names,
            decorations,
            member_decorations,
            member_names,
            types,
            constants,
        )
        global_variables = (
            self.spirv_assembly_specialization_constants(
                spec_constant_ids, names, decorations, types, constants, constant_types
            )
            + global_variables
        )
        if not global_variables:
            raise SyntaxError(SPIRV_ASSEMBLY_ERROR)

        return ShaderNode(
            functions=[],
            structs=[],
            global_variables=global_variables,
            spirv_assembly=True,
            spirv_entry_points=entry_points,
            spirv_names=names,
            spirv_decorations=decorations,
            spirv_member_decorations=member_decorations,
            spirv_member_names=member_names,
            spirv_types=types,
        )

    def parse_spirv_assembly_instructions(self, code):
        instructions = []
        for line_number, line in enumerate(code.splitlines(), start=1):
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ";"
            try:
                parts = list(lexer)
            except ValueError as exc:
                raise SyntaxError(
                    f"Invalid SPIR-V assembly syntax on line {line_number}: {exc}"
                ) from exc

            if not parts:
                continue

            result_id = None
            if len(parts) >= 3 and parts[1] == "=":
                result_id = parts[0]
                opcode = parts[2]
                operands = parts[3:]
            else:
                opcode = parts[0]
                operands = parts[1:]

            if not opcode.startswith("Op"):
                raise SyntaxError(
                    f"Expected SPIR-V opcode on line {line_number}, got {opcode}"
                )
            instructions.append((result_id, opcode, operands, line_number))

        return instructions

    def spirv_assembly_interface_variables(
        self,
        variables,
        entry_interface_ids,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        layouts = []
        for variable in variables:
            pointer_type = types.get(variable["pointer_type_id"], {})
            storage_class = variable["storage_class"] or pointer_type.get(
                "storage_class"
            )
            if pointer_type.get("kind") != "pointer":
                continue

            if storage_class == "UniformConstant":
                resource_layout = self.spirv_assembly_uniform_constant_layout(
                    variable, pointer_type, names, decorations, types, constants
                )
                if resource_layout is not None:
                    layouts.append(resource_layout)
                continue

            if storage_class in {"PushConstant", "StorageBuffer", "Uniform"}:
                resource_block_layout = self.spirv_assembly_resource_block_layout(
                    variable,
                    pointer_type,
                    storage_class,
                    names,
                    decorations,
                    member_decorations,
                    member_names,
                    types,
                    constants,
                )
                if resource_block_layout is not None:
                    layouts.append(resource_block_layout)
                continue

            layout_type = self.SPIRV_INTERFACE_STORAGE_CLASSES.get(storage_class)
            if layout_type is None:
                continue
            if entry_interface_ids and variable["id"] not in entry_interface_ids:
                continue

            struct_layouts = self.spirv_assembly_struct_interface_layouts(
                variable,
                pointer_type,
                storage_class,
                names,
                member_decorations,
                member_names,
                types,
                constants,
            )
            if struct_layouts:
                layouts.extend(struct_layouts)
                continue

            variable_decorations = decorations.get(variable["id"], [])
            qualifiers = self.spirv_layout_qualifiers(variable_decorations)
            if not self.spirv_has_interface_qualifier(qualifiers):
                continue
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                variable_decorations
            )

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                pointer_type.get("type_id"), types, constants
            )
            if data_type is None:
                continue

            variable_name = names.get(variable["id"]) or variable["id"].lstrip("%")
            variable_name = self.spirv_builtin_variable_name_from_qualifiers(
                qualifiers, variable_name
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=layout_type,
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=variable["id"],
                    spirv_decorations=variable_decorations,
                    spirv_storage_class=storage_class,
                )
            )

        return layouts

    def spirv_assembly_uniform_constant_layout(
        self, variable, pointer_type, names, decorations, types, constants
    ):
        data_type, array_suffix = self.spirv_type_name_and_suffix(
            pointer_type.get("type_id"), types, constants
        )
        if data_type is None:
            return None

        variable_decorations = decorations.get(variable["id"], [])
        qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
        qualifier_names = {name for name, _value in qualifiers}
        if not {"set", "binding"}.issubset(qualifier_names):
            return None

        variable_name = names.get(variable["id"]) or variable["id"].lstrip("%")
        variable_name += array_suffix
        return LayoutNode(
            qualifiers,
            layout_type="UNIFORM",
            data_type=data_type,
            variable_name=variable_name,
            spirv_id=variable["id"],
            spirv_decorations=variable_decorations,
            spirv_storage_class="UniformConstant",
        )

    def spirv_assembly_specialization_constants(
        self, spec_constant_ids, names, decorations, types, constants, constant_types
    ):
        layouts = []
        for result_id in spec_constant_ids:
            constant_id = self.spirv_spec_constant_id(decorations.get(result_id, []))
            if constant_id is None:
                continue

            data_type = self.spirv_type_name(constant_types.get(result_id), types)
            if data_type is None:
                continue

            variable_name = names.get(result_id) or result_id.lstrip("%")
            declaration = AssignmentNode(
                VariableNode(f"const {data_type}", variable_name),
                constants.get(result_id),
            )
            layouts.append(
                LayoutNode(
                    [("constant_id", constant_id)],
                    declaration=declaration,
                    layout_type="CONST",
                    spirv_id=result_id,
                    spirv_decorations=decorations.get(result_id, []),
                )
            )

        return layouts

    def spirv_spec_constant_id(self, decorations):
        for decoration, operands in decorations:
            if decoration == "SpecId" and operands:
                return operands[0]
        return None

    def spirv_assembly_resource_block_layout(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        decorations,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return None

        struct_decorations = decorations.get(struct_type_id, [])
        variable_id = variable["id"]
        variable_decorations = decorations.get(variable_id, [])
        has_block = self.spirv_has_decoration(struct_decorations, "Block")
        has_buffer_block = self.spirv_has_decoration(struct_decorations, "BufferBlock")
        if storage_class == "PushConstant":
            if not has_block:
                return None
            layout_type = "UNIFORM"
        elif storage_class == "Uniform":
            if has_buffer_block:
                layout_type = "BUFFER"
            elif has_block:
                layout_type = "UNIFORM"
            else:
                return None
        elif storage_class == "StorageBuffer":
            if not has_block:
                return None
            layout_type = "BUFFER"
        else:
            return None

        struct_fields = []
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue

            member_key = str(member_index)
            field_name = member_names.get(struct_type_id, {}).get(
                member_key, f"member{member_key}"
            )
            struct_fields.append((data_type, f"{field_name}{array_suffix}"))

        if not struct_fields:
            return None

        variable_name = names.get(variable_id) or variable_id.lstrip("%")
        block_name = (
            names.get(struct_type_id) or variable_name or struct_type_id.lstrip("%")
        )
        qualifiers = []
        if storage_class in {"StorageBuffer", "Uniform"}:
            qualifiers = self.spirv_descriptor_qualifiers(variable_decorations)
            qualifier_names = {name for name, _value in qualifiers}
            if not {"set", "binding"}.issubset(qualifier_names):
                return None

        return LayoutNode(
            qualifiers,
            layout_type=layout_type,
            push_constant=storage_class == "PushConstant",
            block_name=block_name,
            variable_name=variable_name,
            struct_fields=struct_fields,
            spirv_id=variable_id,
            spirv_decorations=struct_decorations + variable_decorations,
            spirv_storage_class=storage_class,
        )

    def spirv_has_decoration(self, decorations, target_decoration):
        return any(decoration == target_decoration for decoration, _ in decorations)

    def spirv_descriptor_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            if not operands:
                continue
            if decoration == "DescriptorSet":
                qualifiers.append(("set", operands[0]))
            elif decoration == "Binding":
                qualifiers.append(("binding", operands[0]))
        return qualifiers

    def spirv_assembly_struct_interface_layouts(
        self,
        variable,
        pointer_type,
        storage_class,
        names,
        member_decorations,
        member_names,
        types,
        constants,
    ):
        struct_type_id = pointer_type.get("type_id")
        struct_type = types.get(struct_type_id, {})
        if struct_type.get("kind") != "struct":
            return []

        layouts = []
        block_name = names.get(variable["id"]) or variable["id"].lstrip("%")
        for member_index, member_type_id in enumerate(
            struct_type.get("member_types", [])
        ):
            member_key = str(member_index)
            member_layout_decorations = [
                (decoration, operands)
                for member, decoration, operands in member_decorations.get(
                    struct_type_id, []
                )
                if member == member_key
            ]
            qualifiers = self.spirv_layout_qualifiers(member_layout_decorations)
            if not self.spirv_has_interface_qualifier(qualifiers):
                continue
            declaration_qualifiers = self.spirv_declaration_qualifiers(
                member_layout_decorations
            )

            data_type, array_suffix = self.spirv_type_name_and_suffix(
                member_type_id, types, constants
            )
            if data_type is None:
                continue

            variable_name = self.spirv_struct_member_variable_name(
                struct_type_id,
                member_key,
                block_name,
                qualifiers,
                member_names,
            )
            variable_name += array_suffix
            layouts.append(
                LayoutNode(
                    qualifiers,
                    layout_type=self.SPIRV_INTERFACE_STORAGE_CLASSES[storage_class],
                    data_type=data_type,
                    variable_name=variable_name,
                    declaration_qualifiers=declaration_qualifiers,
                    spirv_id=f"{variable['id']}.{member_key}",
                    spirv_decorations=member_layout_decorations,
                    spirv_storage_class=storage_class,
                )
            )

        return layouts

    def spirv_layout_qualifiers(self, decorations):
        qualifiers = []
        for decoration, operands in decorations:
            qualifier_name = self.SPIRV_INTERFACE_DECORATIONS.get(decoration)
            if qualifier_name and operands:
                qualifiers.append((qualifier_name, operands[0]))
        return qualifiers

    def spirv_declaration_qualifiers(self, decorations):
        qualifiers = []
        for decoration, _operands in decorations:
            qualifier = self.SPIRV_DECLARATION_DECORATION_QUALIFIERS.get(decoration)
            if qualifier:
                qualifiers.append(qualifier)
        return qualifiers

    def spirv_has_interface_qualifier(self, qualifiers):
        return any(name in {"builtin", "location"} for name, _value in qualifiers)

    def spirv_struct_member_variable_name(
        self,
        struct_type_id,
        member_key,
        block_name,
        qualifiers,
        member_names,
    ):
        member_name = member_names.get(struct_type_id, {}).get(member_key)
        if member_name:
            return member_name

        builtin_name = self.spirv_builtin_variable_name_from_qualifiers(qualifiers)
        if builtin_name:
            return builtin_name

        if block_name:
            return f"{block_name}_{member_key}"
        return f"member{member_key}"

    def spirv_builtin_variable_name_from_qualifiers(
        self, qualifiers, fallback_name=None
    ):
        for name, value in qualifiers:
            if name == "builtin":
                return self.SPIRV_BUILTIN_VARIABLE_NAMES.get(value, value)
        return fallback_name

    def spirv_type_name_and_suffix(self, type_id, types, constants):
        type_info = types.get(type_id)
        if type_info is None:
            return None, ""

        if type_info.get("kind") == "array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants
            )
            if base_type is None:
                return None, ""
            length_id = type_info.get("length_id")
            length = constants.get(length_id, str(length_id).lstrip("%"))
            return base_type, f"[{length}]{suffix}"

        if type_info.get("kind") == "runtime_array":
            base_type, suffix = self.spirv_type_name_and_suffix(
                type_info.get("element_type"), types, constants
            )
            if base_type is None:
                return None, ""
            return base_type, f"[]{suffix}"

        return type_info.get("name"), ""

    def spirv_type_name(self, type_id, types):
        type_info = types.get(type_id)
        if type_info is None:
            return None
        return type_info.get("name")

    def spirv_sampled_image_type_name(self, image_type, types):
        sampled_type = self.spirv_type_name(image_type.get("sampled_type"), types)
        return self.spirv_image_type_name(
            sampled_type,
            image_type.get("dim"),
            image_type.get("depth"),
            image_type.get("arrayed"),
            image_type.get("multisampled"),
            "1",
        )

    def spirv_image_type_name(
        self, sampled_type, dim, depth, arrayed, multisampled, sampled
    ):
        if dim == "SubpassData":
            return "subpassInputMS" if multisampled == "1" else "subpassInput"

        base_name = self.spirv_image_base_type_name(dim, arrayed, multisampled)
        if base_name is None:
            return None

        if sampled == "2":
            prefix = {"int": "i", "uint": "u"}.get(sampled_type, "")
            return f"{prefix}image{base_name}"

        prefix = {"int": "i", "uint": "u"}.get(sampled_type, "")
        suffix = "Shadow" if depth == "1" and not prefix else ""
        return f"{prefix}sampler{base_name}{suffix}"

    def spirv_image_base_type_name(self, dim, arrayed, multisampled):
        if dim == "Buffer":
            return "Buffer"
        if dim == "Cube":
            return "CubeArray" if arrayed == "1" else "Cube"
        if dim == "1D":
            return "1DArray" if arrayed == "1" else "1D"
        if dim == "2D":
            if multisampled == "1":
                return "2DMSArray" if arrayed == "1" else "2DMS"
            return "2DArray" if arrayed == "1" else "2D"
        if dim == "3D":
            return "3D"
        return None

    def spirv_matrix_type_name(self, component_type, row_count, column_count):
        if not component_type or not row_count or not column_count:
            return None

        prefix = {"double": "dmat", "float": "mat"}.get(component_type)
        if prefix is None:
            return None

        if row_count == column_count:
            return f"{prefix}{column_count}"
        return f"{prefix}{column_count}x{row_count}"

    def spirv_float_type_name(self, width):
        if width == "64":
            return "double"
        return "float"

    def spirv_int_type_name(self, width, signedness):
        if width == "1":
            return "bool"
        if signedness == "0":
            return "uint"
        return "int"

    def parse_precision_declaration(self):
        self.eat("PRECISION")
        if self.current_token[0] in self.PRECISION_QUALIFIER_TOKENS:
            self.eat(self.current_token[0])

        if self.current_token[1] not in VALID_DATA_TYPES:
            raise SyntaxError(f"Unexpected precision type: {self.current_token[1]}")
        self.eat(self.current_token[0])
        self.eat("SEMICOLON")

    def parse_layout(self):
        self.eat("LAYOUT")
        self.eat("LPAREN")
        bindings = []
        push_constant = False
        if self.current_token[0] == "PUSH_CONSTANT":
            push_constant = True
            self.eat("PUSH_CONSTANT")
        if self.current_token[0] == "COMMA":
            self.eat("COMMA")

        while self.current_token[0] != "RPAREN":
            binding_name = self.current_token[1]
            self.eat("IDENTIFIER")

            if self.current_token[0] == "EQUALS":
                self.eat("EQUALS")
                binding_value = self.parse_layout_qualifier_value()
                bindings.append((binding_name, binding_value))
            else:
                bindings.append((binding_name, None))

            if self.current_token[0] == "COMMA":
                self.eat("COMMA")

        self.eat("RPAREN")

        declaration_qualifiers = self.parse_layout_declaration_qualifiers()
        if (
            self.has_specialization_constant_qualifier(bindings)
            and self.current_token[0] == "CONST"
        ):
            declaration = self.parse_assignment_or_function_call()
            return LayoutNode(
                bindings,
                declaration=declaration,
                layout_type="CONST",
                declaration_qualifiers=declaration_qualifiers,
            )

        layout_type = None
        block_name = None
        if self.current_token[0] in ["IN", "OUT", "UNIFORM", "BUFFER"]:
            layout_type = self.current_token[0]
            self.eat(layout_type)
            declaration_qualifiers.extend(self.parse_layout_declaration_qualifiers())
            if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "LBRACE":
                block_name = self.current_token[1]
                self.eat(self.current_token[0])

        data_type = None
        struct_fields = None
        if layout_type in ["UNIFORM", "BUFFER"]:
            if self.current_token[0] == "LBRACE":
                self.eat("LBRACE")
                struct_fields = []

                while self.current_token[0] != "RBRACE":
                    field_type = self.parse_data_type(
                        allow_identifier=True,
                        error_message="Expected some data type before an identifier",
                    )
                    field_name = self.current_token[1]
                    self.eat("IDENTIFIER")
                    field_name += self.parse_array_suffixes_as_text()
                    self.eat("SEMICOLON")
                    struct_fields.append((field_type, field_name))

                self.eat("RBRACE")
                data_type = "struct"
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SyntaxError(
                    "Expected structured data block after 'uniform' or 'buffer'"
                )
        else:
            if layout_type in ["IN", "OUT"] and self.current_token[0] == "SEMICOLON":
                pass
            elif self.is_data_type_token(allow_identifier=True):
                data_type = self.parse_data_type(allow_identifier=True)
            else:
                raise SyntaxError(f"Unexpected type: {self.current_token[1]}")

        variable_name = None
        if self.current_token[0] == "IDENTIFIER":
            variable_name = self.current_token[1]
            self.eat("IDENTIFIER")
            variable_name += self.parse_array_suffixes_as_text()

        self.eat("SEMICOLON")
        return LayoutNode(
            bindings,
            push_constant=push_constant,
            layout_type=layout_type,
            data_type=data_type,
            variable_name=variable_name,
            struct_fields=struct_fields,
            block_name=block_name,
            declaration_qualifiers=declaration_qualifiers,
        )

    def parse_layout_qualifier_value(self):
        if self.current_token[0] in {"NUMBER", "IDENTIFIER"}:
            value = self.current_token[1]
            self.eat(self.current_token[0])
            return value
        raise SyntaxError(
            f"Expected layout qualifier value, got {self.current_token[0]}"
        )

    def has_specialization_constant_qualifier(self, qualifiers):
        return any(str(name).lower() == "constant_id" for name, _ in qualifiers)

    def is_data_type_token(self, allow_identifier=False):
        return self.current_token[1] in VALID_DATA_TYPES or (
            allow_identifier and self.current_token[0] == "IDENTIFIER"
        )

    def parse_data_type(self, allow_identifier=False, error_message=None):
        if not self.is_data_type_token(allow_identifier=allow_identifier):
            raise SyntaxError(
                error_message or f"Unexpected type: {self.current_token[1]}"
            )

        type_name = self.current_token[1]
        self.eat(self.current_token[0])
        return type_name

    def parse_layout_declaration_qualifiers(self):
        qualifiers = []
        while self.current_token[1] in self.LAYOUT_DECLARATION_QUALIFIERS:
            qualifiers.append(self.current_token[1])
            self.eat(self.current_token[0])
        return qualifiers

    def parse_push_constant(self):
        self.eat("PUSH_CONSTANT")
        self.eat("LBRACE")
        members = []
        while self.current_token[0] != "RBRACE":
            members.append(self.parse_variable())
        self.eat("RBRACE")
        return PushConstantNode(members)

    def parse_descriptor_set(self):
        self.eat("DESCRIPTOR_SET")
        set_number = self.current_token[1]
        self.eat("NUMBER")
        self.eat("LBRACE")
        bindings = []
        while self.current_token[0] != "RBRACE":
            bindings.append(self.parse_variable())
        self.eat("RBRACE")
        return DescriptorSetNode(set_number, bindings)

    def parse_struct(self):
        self.eat("STRUCT")
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LBRACE")
        members = []

        while self.current_token[0] != "RBRACE":
            if self.current_token[0] in [
                "VEC2",
                "VEC3",
                "VEC4",
                "IVEC2",
                "IVEC3",
                "IVEC4",
                "UVEC2",
                "UVEC3",
                "UVEC4",
                "FLOAT",
                "INT",
                "UINT",
                "BOOL",
                "MAT2",
                "MAT3",
                "MAT4",
            ]:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[1] in VALID_DATA_TYPES:
                type_name = self.current_token[1]
                self.eat(self.current_token[0])
            elif self.current_token[0] == "IDENTIFIER":
                type_name = self.current_token[1]
                self.eat("IDENTIFIER")
            else:
                raise SyntaxError(
                    f"Unexpected token in struct member: {self.current_token}"
                )

            member_name = self.current_token[1]
            self.eat("IDENTIFIER")
            member_name += self.parse_array_suffixes_as_text()

            self.eat("SEMICOLON")

            members.append(VariableNode(type_name, member_name))

        self.eat("RBRACE")
        self.eat("SEMICOLON")

        return StructNode(name, members)

    def parse_function(self):
        return_type = self.current_token[1]
        if self.current_token[1] in VALID_DATA_TYPES:
            self.eat(self.current_token[0])
        else:
            raise SyntaxError(f"Unexpected type: {self.current_token[1]}")
        func_name = self.current_token[1]
        self.eat("IDENTIFIER")
        self.eat("LPAREN")
        params = self.parse_parameters()
        self.eat("RPAREN")
        body = self.parse_block()
        return FunctionNode(return_type, func_name, params, body)

    def parse_parameters(self):
        params = []
        if self.current_token[0] == "VOID" and self.peek(1) == "RPAREN":
            self.eat("VOID")
            return params

        while self.current_token[0] != "RPAREN":
            qualifiers = []
            while self.current_token[0] in self.PARAMETER_QUALIFIER_TOKENS:
                qualifiers.append(self.current_token[1])
                self.eat(self.current_token[0])

            vtype = self.current_token[1]
            self.eat(self.current_token[0])
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            name += self.parse_array_suffixes_as_text()
            params.append(VariableNode(vtype, name, qualifiers=qualifiers))
            if self.current_token[0] == "COMMA":
                self.eat("COMMA")
        return params

    def parse_block(self):
        self.eat("LBRACE")
        statements = []
        while self.current_token[0] != "RBRACE":
            statements.append(self.parse_body())
        self.eat("RBRACE")
        return statements

    def parse_body(self):
        token_type = self.current_token[0]

        if token_type == "CONST":
            return self.parse_assignment_or_function_call()
        if token_type == "IDENTIFIER" and (
            self.peek(1) in ["LPAREN", "LBRACKET"]
            or self.looks_like_member_call_statement()
        ):
            return self.parse_expression_statement()
        if token_type == "IDENTIFIER" or self.current_token[1] in VALID_DATA_TYPES:
            return self.parse_assignment_or_function_call()
        elif token_type == "IF":
            return self.parse_if_statement()
        elif token_type == "FOR":
            return self.parse_for_statement()
        elif token_type == "WHILE":
            return self.parse_while_statement()
        elif token_type == "DO":
            return self.parse_do_while_statement()
        elif token_type == "SWITCH":
            return self.parse_switch_statement()
        elif token_type == "BREAK":
            if self.breakable_depth == 0:
                raise SyntaxError("break used outside loop or switch")
            self.eat("BREAK")
            self.eat("SEMICOLON")
            return BreakNode()
        elif token_type == "CONTINUE":
            if self.loop_depth == 0:
                raise SyntaxError("continue used outside loop")
            self.eat("CONTINUE")
            self.eat("SEMICOLON")
            return ContinueNode()
        elif token_type == "RETURN":
            return self.parse_return_statement()
        elif token_type == "DISCARD":
            self.eat("DISCARD")
            self.eat("SEMICOLON")
            return DiscardNode()
        else:
            return self.parse_expression_statement()

    def parse_return_statement(self):
        self.eat("RETURN")
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return ReturnNode()

        value = self.parse_expression()
        self.eat("SEMICOLON")
        return ReturnNode(value)

    def parse_update(self):
        if self.current_token[0] == "IDENTIFIER":
            target = self.parse_update_target()
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                return UnaryOpNode("POST_INCREMENT", target)
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                return UnaryOpNode("POST_DECREMENT", target)
            elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
                op_name = self.current_token[1]
                self.eat(self.current_token[0])
                value = self.parse_expression()
                return AssignmentNode(target, value, op_name)
            else:
                raise SyntaxError(
                    f"Unexpected token in update: {self.current_token[0]}"
                )
        elif self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_update_target())
        elif self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_update_target())
        else:
            raise SyntaxError(f"Unexpected token in update: {self.current_token[0]}")

    def parse_update_target(self):
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(f"Expected update target, got {self.current_token[0]}")

        target = VariableNode("", self.current_token[1])
        self.eat("IDENTIFIER")
        target = self.parse_postfix_suffixes(target)
        if not isinstance(target, (VariableNode, MemberAccessNode, ArrayAccessNode)):
            raise SyntaxError(f"Invalid update target: {type(target).__name__}")
        return target

    def parse_if_statement(self):
        self.eat("IF")
        self.eat("LPAREN")
        if_condition = self.parse_expression()
        self.eat("RPAREN")
        if_body = self.parse_block()
        else_body = None
        else_if_chain = []
        while self.current_token[0] == "ELSE" and self.peek(1) == "IF":
            self.eat("ELSE")
            self.eat("IF")
            self.eat("LPAREN")
            else_if_condition = self.parse_expression()
            self.eat("RPAREN")
            else_if_chain.append((else_if_condition, self.parse_block()))
        if self.current_token[0] == "ELSE":
            self.eat("ELSE")
            else_body = self.parse_block()
        return IfNode(
            if_condition,
            if_body,
            else_body,
            else_if_chain=else_if_chain,
        )

    def parse_for_statement(self):
        self.eat("FOR")
        self.eat("LPAREN")
        initialization = self.parse_for_initializer()
        condition = self.parse_for_condition()
        increment = self.parse_for_update()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return ForNode(initialization, condition, increment, body)

    def parse_for_initializer(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        items = [
            self.parse_assignment_or_function_call(
                terminators={"COMMA", "SEMICOLON"},
                consume_terminator=False,
            )
        ]
        declaration_type = self.for_initializer_declaration_type(items[0])
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            if declaration_type and self.current_token[0] == "IDENTIFIER":
                items.append(
                    self.parse_variable(
                        declaration_type,
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
            else:
                items.append(
                    self.parse_assignment_or_function_call(
                        terminators={"COMMA", "SEMICOLON"},
                        consume_terminator=False,
                    )
                )
        self.eat("SEMICOLON")
        return items if len(items) > 1 else items[0]

    def for_initializer_declaration_type(self, item):
        target = item.left if isinstance(item, AssignmentNode) else item
        if isinstance(target, VariableNode) and target.vtype:
            return target.vtype
        return ""

    def parse_for_condition(self):
        if self.current_token[0] == "SEMICOLON":
            self.eat("SEMICOLON")
            return None

        condition = self.parse_expression()
        self.eat("SEMICOLON")
        return condition

    def parse_for_update(self):
        if self.current_token[0] == "RPAREN":
            return None
        updates = [self.parse_update()]
        while self.current_token[0] == "COMMA":
            self.eat("COMMA")
            updates.append(self.parse_update())
        return updates if len(updates) > 1 else updates[0]

    def parse_variable(
        self,
        type_name="",
        terminators=None,
        consume_terminator=True,
    ):
        terminators = terminators or {"SEMICOLON"}
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        if type_name:
            name += self.parse_array_suffixes_as_text()
        target = VariableNode(type_name, name)
        if not type_name:
            target = self.parse_postfix_suffixes(target)

        if self.current_token[0] in terminators:
            if consume_terminator:
                self.eat(self.current_token[0])
            return target

        elif self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            value = self.parse_expression()
            self.consume_terminator(terminators, consume_terminator)
            return AssignmentNode(target, value, op_name)

        elif self.current_token[0] in ("BINARY_AND", "BINARY_OR", "BINARY_XOR"):
            op = self.current_token[0]
            op_symbol = (
                "&" if op == "BINARY_AND" else ("|" if op == "BINARY_OR" else "^")
            )
            self.eat(op)
            right = self.parse_expression()
            self.consume_terminator(terminators, consume_terminator)
            return BinaryOpNode(target, op_symbol, right)

        elif self.current_token[0] in (
            "EQUAL",
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
            "BITWISE_SHIFT_RIGHT",
            "BITWISE_SHIFT_LEFT",
            "BITWISE_XOR",
        ):
            op = self.current_token[0]
            op_name = self.current_token[1]
            self.eat(op)
            value = self.parse_expression()
            self.consume_terminator(terminators, consume_terminator)
            return BinaryOpNode(target, op_name, value)
        else:
            raise SyntaxError(
                f"Unexpected token after identifier {name}: {self.current_token[0]}"
            )

    def consume_terminator(self, terminators, consume_terminator):
        if self.current_token[0] not in terminators:
            expected = " or ".join(sorted(terminators))
            raise SyntaxError(f"Expected {expected}, got {self.current_token[0]}")
        if consume_terminator:
            self.eat(self.current_token[0])

    def parse_member_access(self, object):
        self.eat("DOT")
        if self.current_token[0] != "IDENTIFIER":
            raise SyntaxError(
                f"Expected identifier after dot, got {self.current_token[0]}"
            )
        member = self.current_token[1]
        self.eat("IDENTIFIER")

        if self.current_token[0] == "DOT":
            return self.parse_member_access(MemberAccessNode(object, member))

        return MemberAccessNode(object, member)

    def parse_function_call(self, name):
        args = self.parse_call_arguments()
        return FunctionCallNode(name, args)

    def parse_call_arguments(self):
        self.eat("LPAREN")
        args = []
        if self.current_token[0] != "RPAREN":
            args.append(self.parse_expression())
            while self.current_token[0] == "COMMA":
                self.eat("COMMA")
                args.append(self.parse_expression())
        self.eat("RPAREN")
        return args

    def parse_function_call_or_identifier(self):
        func_name = self.current_token[1]
        self.eat(self.current_token[0])

        if self.current_token[0] == "LPAREN":
            node = self.parse_function_call(func_name)
        else:
            node = VariableNode("", func_name)
        return self.parse_postfix_suffixes(node)

    def parse_postfix_suffixes(self, node):
        while True:
            if self.current_token[0] == "DOT":
                self.eat("DOT")
                member = self.current_token[1]
                self.eat("IDENTIFIER")
                if self.current_token[0] == "LPAREN":
                    node = MethodCallNode(node, member, self.parse_call_arguments())
                else:
                    node = MemberAccessNode(node, member)
                continue

            if self.current_token[0] == "LBRACKET":
                self.eat("LBRACKET")
                index = self.parse_expression()
                self.eat("RBRACKET")
                node = ArrayAccessNode(node, index)
                continue

            return node

    def looks_like_member_call_statement(self):
        index = self.pos
        if self.tokens[index][0] != "IDENTIFIER":
            return False

        while index + 2 < len(self.tokens):
            if self.tokens[index + 1][0] != "DOT":
                return False
            if self.tokens[index + 2][0] != "IDENTIFIER":
                return False
            index += 2
            if index + 1 < len(self.tokens) and self.tokens[index + 1][0] == "LPAREN":
                return True

        return False

    def parse_array_suffixes_as_text(self):
        suffix = ""
        while self.current_token[0] == "LBRACKET":
            suffix += "["
            self.eat("LBRACKET")
            while self.current_token[0] != "RBRACKET":
                if self.current_token[0] == "EOF":
                    raise SyntaxError("Unterminated array suffix")
                suffix += str(self.current_token[1])
                self.eat(self.current_token[0])
            self.eat("RBRACKET")
            suffix += "]"
        return suffix

    def parse_primary(self):
        if self.current_token[0] == "MINUS":
            self.eat("MINUS")
            value = self.parse_primary()
            return UnaryOpNode("-", value)

        if (
            self.current_token[0] == "BITWISE_NOT"
            or self.current_token[0] == "BINARY_NOT"
        ):
            self.eat(self.current_token[0])
            value = self.parse_primary()
            return UnaryOpNode("~", value)

        if (
            self.current_token[0] == "IDENTIFIER"
            or self.current_token[1] in VALID_DATA_TYPES
        ):
            return self.parse_function_call_or_identifier()
        elif self.current_token[0] == "NUMBER":
            value = self.current_token[1]
            self.eat("NUMBER")
            if value[-1:] in {"u", "U", "f", "F"}:
                value = value[:-1]
            return value
        elif self.current_token[0] == "LPAREN":
            self.eat("LPAREN")
            expr = self.parse_expression()
            self.eat("RPAREN")
            return expr
        else:
            raise SyntaxError(
                f"Unexpected token in expression: {self.current_token[0]}"
            )

    def parse_multiplicative(self):
        left = self.parse_unary()
        while self.current_token[0] in ["MULTIPLY", "DIVIDE", "MOD"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_additive(self):
        left = self.parse_multiplicative()
        while self.current_token[0] in ["PLUS", "MINUS"]:
            token_type = self.current_token[0]
            op = self.current_token[1]
            self.eat(token_type)
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
        return left

    def parse_assignment_or_function_call(
        self,
        terminators=None,
        consume_terminator=True,
    ):
        terminators = terminators or {"SEMICOLON"}
        type_name = ""
        qualifiers = []
        while self.current_token[0] == "CONST":
            qualifiers.append(self.current_token[1])
            self.eat("CONST")

        if qualifiers:
            if (
                self.current_token[0] == "IDENTIFIER"
                or self.current_token[1] in VALID_DATA_TYPES
            ):
                type_name = " ".join([*qualifiers, self.current_token[1]])
                self.eat(self.current_token[0])
            else:
                raise SyntaxError(
                    f"Unexpected token after const: {self.current_token[0]}"
                )
        elif self.current_token[0] == "IDENTIFIER" and self.peek(1) in [
            "POST_INCREMENT",
            "POST_DECREMENT",
        ]:
            name = self.current_token[1]
            self.eat("IDENTIFIER")
            if self.current_token[0] == "POST_INCREMENT":
                self.eat("POST_INCREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_INCREMENT", VariableNode("", name))
            elif self.current_token[0] == "POST_DECREMENT":
                self.eat("POST_DECREMENT")
                self.consume_terminator(terminators, consume_terminator)
                return UnaryOpNode("POST_DECREMENT", VariableNode("", name))
            else:
                raise SyntaxError(
                    f"Unexpected token after identifier: {self.current_token[0]}"
                )
        if self.current_token[0] == "IDENTIFIER" and self.peek(1) == "IDENTIFIER":
            type_name = self.current_token[1]
            self.eat("IDENTIFIER")
        elif self.current_token[1] in VALID_DATA_TYPES:
            type_name = self.current_token[1]
            self.eat(self.current_token[0])
        if self.current_token[0] == "IDENTIFIER":
            return self.parse_variable(
                type_name,
                terminators=terminators,
                consume_terminator=consume_terminator,
            )

    def parse_expression(self):
        return self.parse_assignment_expression()

    def parse_assignment_expression(self):
        left = self.parse_ternary_expression()
        if self.current_token[0] in self.ASSIGNMENT_TOKENS:
            op_name = self.current_token[1]
            self.eat(self.current_token[0])
            right = self.parse_assignment_expression()
            return AssignmentNode(left, right, op_name)
        return left

    def parse_ternary_expression(self):
        left = self.parse_logical_or_expression()

        if self.current_token[0] == "QUESTION":
            self.eat("QUESTION")
            true_expr = self.parse_expression()
            self.eat("COLON")
            false_expr = self.parse_expression()
            left = TernaryOpNode(left, true_expr, false_expr)

        return left

    def parse_logical_or_expression(self):
        left = self.parse_logical_and_expression()
        while self.current_token[0] == "OR":
            op_symbol = self.current_token[1]
            self.eat("OR")
            right = self.parse_logical_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_logical_and_expression(self):
        left = self.parse_bitwise_or_expression()
        while self.current_token[0] == "AND":
            op_symbol = self.current_token[1]
            self.eat("AND")
            right = self.parse_bitwise_or_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_or_expression(self):
        left = self.parse_bitwise_xor_expression()
        while self.current_token[0] == "BINARY_OR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_OR")
            right = self.parse_bitwise_xor_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_xor_expression(self):
        left = self.parse_bitwise_and_expression()
        while self.current_token[0] == "BINARY_XOR":
            op_symbol = self.current_token[1]
            self.eat("BINARY_XOR")
            right = self.parse_bitwise_and_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_bitwise_and_expression(self):
        left = self.parse_equality_expression()
        while self.current_token[0] == "BINARY_AND":
            op_symbol = self.current_token[1]
            self.eat("BINARY_AND")
            right = self.parse_equality_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_equality_expression(self):
        left = self.parse_relational_expression()
        while self.current_token[0] in ["EQUAL", "NOT_EQUAL"]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_relational_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_relational_expression(self):
        left = self.parse_shift_expression()
        while self.current_token[0] in [
            "LESS_THAN",
            "GREATER_THAN",
            "LESS_EQUAL",
            "GREATER_EQUAL",
        ]:
            op = self.current_token[0]
            op_symbol = self.current_token[1]
            self.eat(op)
            right = self.parse_shift_expression()
            left = BinaryOpNode(left, op_symbol, right)
        return left

    def parse_shift_expression(self):
        left = self.parse_additive()
        while self.current_token[0] in [
            "BITWISE_SHIFT_LEFT",
            "BITWISE_SHIFT_RIGHT",
        ]:
            op = self.current_token[0]
            self.eat(op)
            right = self.parse_additive()
            op_symbol = "<<" if op == "BITWISE_SHIFT_LEFT" else ">>"
            left = BinaryOpNode(left, op_symbol, right)

        return left

    def parse_expression_statement(self):
        expr = self.parse_expression()
        self.eat("SEMICOLON")
        return expr

    def parse_while_statement(self):
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        return WhileNode(condition, body)

    def parse_do_while_statement(self):
        self.eat("DO")
        self.loop_depth += 1
        self.breakable_depth += 1
        try:
            body = self.parse_block()
        finally:
            self.breakable_depth -= 1
            self.loop_depth -= 1
        self.eat("WHILE")
        self.eat("LPAREN")
        condition = self.parse_expression()
        self.eat("RPAREN")
        self.eat("SEMICOLON")
        return DoWhileNode(body, condition)

    def parse_switch_statement(self):
        self.eat("SWITCH")
        self.eat("LPAREN")
        expr = self.parse_expression()
        self.eat("RPAREN")
        self.eat("LBRACE")
        cases = []
        seen_default = False
        self.breakable_depth += 1
        try:
            while self.current_token[0] not in ["RBRACE", "EOF"]:
                if self.current_token[0] == "DEFAULT":
                    if seen_default:
                        raise SyntaxError("duplicate default label in switch")
                    seen_default = True
                cases.append(self.parse_case_statement())
            if self.current_token[0] == "EOF":
                raise SyntaxError("Unterminated switch statement")
        finally:
            self.breakable_depth -= 1
        self.eat("RBRACE")
        return SwitchNode(expr, cases)

    def parse_case_statement(self):
        if self.current_token[0] == "CASE":
            self.eat("CASE")
            value = self.parse_expression()
            self.eat("COLON")
        elif self.current_token[0] == "DEFAULT":
            self.eat("DEFAULT")
            value = None
            self.eat("COLON")
        else:
            raise SyntaxError(
                f"Expected CASE or DEFAULT in switch, got {self.current_token[0]}"
            )
        statements = []
        while self.current_token[0] not in ["CASE", "DEFAULT", "RBRACE", "EOF"]:
            statements.append(self.parse_body())
        if self.current_token[0] == "EOF":
            raise SyntaxError("Unterminated switch case")
        return CaseNode(value, statements)

    def parse_default_statement(self):
        self.eat("DEFAULT")
        self.eat("COLON")
        statements = []
        while self.current_token[0] not in ["CASE", "RBRACE"]:
            statements.append(self.parse_body())
        return DefaultNode(statements)

    def parse_uniform(self):
        self.eat("UNIFORM")
        var_type = self.parse_data_type(allow_identifier=True)
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        name += self.parse_array_suffixes_as_text()
        self.eat("SEMICOLON")
        return UniformNode(var_type, name)

    def parse_unary(self):
        if self.current_token[0] in ["PLUS", "MINUS", "BITWISE_NOT", "NOT"]:
            op = self.current_token[1]
            self.eat(self.current_token[0])
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
        if self.current_token[0] == "PRE_INCREMENT":
            self.eat("PRE_INCREMENT")
            return UnaryOpNode("PRE_INCREMENT", self.parse_unary())
        if self.current_token[0] == "PRE_DECREMENT":
            self.eat("PRE_DECREMENT")
            return UnaryOpNode("PRE_DECREMENT", self.parse_unary())
        return self.parse_primary()
