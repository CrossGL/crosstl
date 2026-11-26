"""Program declarations in the HLSL to CrossGL converter: stage entry attributes, flattened stage structs, patch constant outputs, legacy samplers, integer constants and global variable declarations."""

import re

from crosstl.translator.codegen.hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_STRUCTURED_LAYOUT,
)

from ...storage_image_texels import IMAGE_FORMAT_ATTRIBUTES
from ..DirectxAst import (
    BinaryOpNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
)
from ..shader_record_import import (
    is_hlsl_shader_record,
    shader_record_layout,
)
from ..subpass_inputs import SUBPASS_INPUT_TYPES
from ..vk_constant_ids import (
    VK_CONSTANT_ID_ATTRIBUTE,
    has_vk_constant_id,
    vk_constant_id_suffix,
)


class ProgramDeclarationMixin:
    """Program declaration members of ``HLSLToCrossGLConverter``."""

    def source_function_accepts_argument_count(self, name, count):
        """Whether a function the shader defines can be called with ``count`` arguments."""
        for function in self.source_functions.get(name, []):
            params = getattr(function, "params", []) or []
            required = sum(
                1 for param in params if getattr(param, "value", None) is None
            )
            if required <= count <= len(params):
                return True
        return False

    def should_emit_stage_entry_attribute(self, func, stage_name, entry_name=None):
        """Whether the stage block's function keeps its HLSL name through ``@ stage_entry``.

        Every function the importer writes into a stage block is that stage's entry. CrossGL treats a stage
        function named ``main`` as the entry and needs ``@ stage_entry`` on any other, or it synthesizes an empty
        entry and the shader computes nothing.
        """
        if entry_name is not None or getattr(func, "name", None) == "main":
            return False
        return bool(stage_name)

    def collect_flattened_stage_struct_names(self, ast):
        flattened = set()
        used_elsewhere = set()
        for func in getattr(ast, "functions", []) or []:
            stage_name = str(getattr(func, "qualifier", "") or "").lower()
            params = getattr(func, "params", []) or []
            flattens_entry_structs = stage_name in {
                "vertex",
                "fragment",
            } and self.has_stage_struct_output_parameter(params)
            for param in params:
                type_name = self.raw_type_base(getattr(param, "vtype", None))
                if type_name not in self.struct_type_names:
                    continue
                if flattens_entry_structs:
                    flattened.add(type_name)
                else:
                    used_elsewhere.add(type_name)
            return_type = self.raw_type_base(getattr(func, "return_type", None))
            if return_type in self.struct_type_names:
                used_elsewhere.add(return_type)
            for node in self.iter_ast_children(getattr(func, "body", []) or []):
                if isinstance(node, VariableNode):
                    type_name = self.raw_type_base(getattr(node, "vtype", None))
                    if type_name in self.struct_type_names:
                        used_elsewhere.add(type_name)
        for node in getattr(ast, "global_variables", []) or []:
            type_name = self.raw_type_base(getattr(node, "vtype", None))
            if type_name in self.struct_type_names:
                used_elsewhere.add(type_name)
        return flattened - used_elsewhere

    def generate_patch_constant_interface_outputs(
        self, func, functions_by_name, structs_by_name, indent=2
    ):
        patch_function_name = self.patch_constant_function_name(func)
        if not patch_function_name:
            return ""

        patch_function = functions_by_name.get(patch_function_name)
        if patch_function is None:
            return ""

        return_struct_name = self.raw_type_base(
            getattr(patch_function, "return_type", "")
        )
        patch_struct = structs_by_name.get(return_struct_name)
        if patch_struct is None:
            return ""

        existing_semantics = self.function_return_cross_stage_semantics(
            func, structs_by_name
        )
        lines = ""
        for member in getattr(patch_struct, "members", []) or []:
            if not isinstance(member, VariableNode):
                continue
            semantic_names = self.cross_stage_semantic_names(
                getattr(member, "semantic", None)
            )
            if not semantic_names or semantic_names & existing_semantics:
                continue

            attributes = self.format_semantic_and_interpolation_attributes(
                member, member.semantic
            )
            if not attributes:
                continue
            lines += self.format_matrix_layout_attributes(member, indent)
            lines += (
                "    " * indent
                + f"out {self.map_variable_type(member)} "
                + f"{self.render_identifier(member.name)}"
                + f"{self.format_array_suffixes(member)}{attributes};\n"
            )
        return lines

    def patch_constant_function_name(self, func):
        for attr in getattr(func, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "patchconstantfunc":
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", [])) or []
            if not args:
                return None
            value = args[0]
            if isinstance(value, str):
                return value.strip().strip("\"'")
            return str(value)
        return None

    def function_return_cross_stage_semantics(self, func, structs_by_name):
        return_type = self.raw_type_base(getattr(func, "return_type", ""))
        return_struct = structs_by_name.get(return_type)
        if return_struct is None:
            return self.cross_stage_semantic_names(getattr(func, "semantic", None))

        names = set()
        for member in getattr(return_struct, "members", []) or []:
            names.update(
                self.cross_stage_semantic_names(getattr(member, "semantic", None))
            )
        return names

    def cross_stage_semantic_names(self, semantic):
        mapped = self.map_semantic(semantic)
        names = {
            name.lower() for name in re.findall(r"@\s*([A-Za-z_][A-Za-z0-9_]*)", mapped)
        }
        return {
            name
            for name in names
            if not (
                name.startswith("gl_")
                or name.startswith("sv_")
                or name.startswith("hlsl_")
                or name in {"builtin", "position"}
            )
        }

    def collect_struct_variable_declarations(self, structs):
        declarations = []
        for struct in structs or []:
            declarations.extend(getattr(struct, "variable_declarations", []) or [])
        return declarations

    def is_legacy_sampler_declaration(self, node):
        return self.raw_type_base(getattr(node, "vtype", None)) in {
            "sampler",
            "sampler_state",
        }

    def sampler_state_texture_value(self, node):
        if not self.is_legacy_sampler_declaration(node):
            return None
        for state_name, state_value in getattr(node, "sampler_state", []) or []:
            if str(state_name).lower() == "texture":
                return state_value
        return None

    def collect_legacy_sampler_bindings(self, ast):
        bindings = {}
        declaration_ids = set()
        for node in getattr(ast, "global_variables", []) or []:
            sampler_name = getattr(node, "name", None)
            texture_value = self.sampler_state_texture_value(node)
            if sampler_name and texture_value is not None:
                bindings[sampler_name] = texture_value
                declaration_ids.add(id(node))
        return bindings, declaration_ids

    def collect_legacy_bound_texture_types(self, root):
        texture_types = {}

        def visit(node):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, FunctionCallNode) and isinstance(node.name, str):
                func_name = self.normalize_hlsl_intrinsic_name(node.name)
                sampler_type = self.legacy_texture_function_bound_sampler_type(
                    func_name
                )
                if sampler_type and getattr(node, "args", None):
                    texture_expr = self.legacy_sampler_bound_texture(node.args[0])
                    texture_name = self.expression_base_name(texture_expr)
                    if texture_name:
                        texture_types.setdefault(texture_name, sampler_type)
            for child in self.iter_ast_children(node):
                visit(child)

        visit(root)
        return texture_types

    def collect_integer_constant_values(self, ast):
        constants = {}
        for node in getattr(ast, "global_variables", []) or []:
            if not self.is_scalar_integer_constant_declaration(node):
                continue
            value = self.evaluate_integer_constant_node(
                getattr(node, "value", None), constants
            )
            if value is not None:
                constants[getattr(node, "name", "")] = value
        return constants

    def is_scalar_integer_constant_declaration(self, node):
        qualifiers = {
            str(qualifier).lower() for qualifier in getattr(node, "qualifiers", [])
        }
        if "const" not in qualifiers:
            return False
        type_name = self.canonical_composite_type(
            str(getattr(node, "vtype", "")).strip()
        )
        return type_name in {"int", "uint", "dword"}

    def evaluate_integer_constant_node(self, node, constants):
        if isinstance(node, bool):
            return None
        if isinstance(node, int):
            return node
        if isinstance(node, str):
            return constants.get(node)
        if isinstance(node, UnaryOpNode):
            value = self.evaluate_integer_constant_node(node.operand, constants)
            if value is None:
                return None
            if node.op == "+":
                return value
            if node.op == "-":
                return -value
            if node.op == "~":
                return ~value
            return None
        if isinstance(node, BinaryOpNode):
            left = self.evaluate_integer_constant_node(node.left, constants)
            right = self.evaluate_integer_constant_node(node.right, constants)
            if left is None or right is None:
                return None
            return self.apply_integer_constant_operator(left, node.op, right)
        return None

    def apply_integer_constant_operator(self, left, operator, right):
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/" and right != 0:
            return left // right
        if operator == "%" and right != 0:
            return left % right
        if operator == "<<":
            return left << right
        if operator == ">>":
            return left >> right
        if operator == "&":
            return left & right
        if operator == "|":
            return left | right
        if operator == "^":
            return left ^ right
        return None

    def generate_global_variable_declaration(self, node):
        self.record_variable_type(
            node, self.global_variable_types, self.global_resource_array_dims
        )
        if id(node) in self.legacy_sampler_declaration_ids:
            return ""
        record = is_hlsl_shader_record(node)
        if record is not None:
            struct_name, layout = shader_record_layout(node, self.structs_by_name)
            return (
                f"    {self.render_identifier(struct_name)} {self.render_identifier(node.name)}"
                f" @glsl_buffer_block({record}, {layout});\n"
            )
        if self.raw_type_base(getattr(node, "vtype", None)) in SUBPASS_INPUT_TYPES:
            code = self.format_subpass_input_attributes(
                getattr(node, "attributes", []), 1
            )
        else:
            # A typed storage texture's format attribute becomes ``@ format`` below.
            format_attributes = (
                IMAGE_FORMAT_ATTRIBUTES
                if self.raw_type_base(getattr(node, "vtype", None))
                in self.RUNTIME_FORMAT_STORAGE_TEXTURES
                else ()
            )
            # A specialization id follows the name, as the GLSL importer writes it.
            code = self.format_attributes(
                getattr(node, "attributes", []),
                1,
                skip_names=(*format_attributes, VK_CONSTANT_ID_ATTRIBUTE),
            )
        if id(node) in self.hlsl_program_constant_global_ids:
            code += "    @ hlsl_program_constant\n"
        code += self.format_resource_qualifier_attributes(node, 1)
        code += self.format_binding_attributes(node, 1)
        code += self.format_matrix_layout_attributes(node, 1)
        if self.is_structured_buffer_resource_type(getattr(node, "vtype", None)):
            code += f"    @ {BUFFER_LAYOUT_ATTRIBUTE}({HLSL_STRUCTURED_LAYOUT})\n"
        code += self.format_storage_image_format_attributes(node, 1)
        storage_prefix = self.format_global_storage_qualifier_prefix(node)
        precise_prefix = self.format_precise_qualifier_prefix(node)
        array_suffix = self.format_array_suffixes(node)
        variable_type = self.map_variable_type(node)
        if self.is_vk_push_constant_global(node) and str(
            getattr(node, "vtype", "")
        ) in getattr(self, "struct_type_names", set()):
            # DXC reads '[[vk::push_constant]] T name;' as one push-constant buffer holding a T, which CrossGL
            # spells 'ConstantBuffer<T> name @push_constant'.
            variable_type = f"ConstantBuffer<{variable_type}>"
        variable_name = self.render_identifier(node.name) + vk_constant_id_suffix(
            self, node
        )
        if storage_prefix == "const " and array_suffix:
            variable_type = f"{variable_type}{array_suffix}"
            array_suffix = ""
        initializer = ""
        if (
            getattr(node, "value", None) is not None
            and id(node) not in self.hlsl_program_constant_global_ids
        ):
            # DXC folds global initializers with exact math at compile time.
            self.constant_initializer_depth += 1
            try:
                initializer = f" = {self.generate_expression(node.value, expected_type=node.vtype)}"
            finally:
                self.constant_initializer_depth -= 1
        if storage_prefix == "const " and precise_prefix:
            declaration_prefix = f"{precise_prefix}{storage_prefix}"
        else:
            declaration_prefix = f"{storage_prefix}{precise_prefix}"
        if (
            getattr(node, "attributes", None)
            and not declaration_prefix
            and str(getattr(node, "vtype", ""))
            in getattr(self, "struct_type_names", set())
            and not variable_type.startswith("ConstantBuffer<")
        ):
            declaration_prefix = "var "
        return (
            code + f"    {declaration_prefix}{variable_type} "
            f"{variable_name}{array_suffix}{initializer};\n"
        )

    def is_vk_push_constant_global(self, node):
        """Whether a global carries DXC's ``[[vk::push_constant]]``, which binds it outside ``$Globals``."""
        return any(
            str(getattr(attr, "name", "")) == "vk::push_constant"
            for attr in getattr(node, "attributes", None) or []
        )

    def is_hlsl_program_constant_global(self, node):
        # A module variable without ``static`` is a $Globals member, ``const`` or not, even with an initializer, which
        # DXC ignores ("Initializer of external global will be ignored").
        if self.is_vk_push_constant_global(node):
            return False
        if has_vk_constant_id(node):
            # DXC's SPIR-V back end makes it a specialization constant (vk_constant_ids).
            return False
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if qualifiers & {"groupshared", "static", "shared"}:
            return False

        type_name = getattr(node, "vtype", None)
        base_type = self.raw_type_base(type_name)
        if not base_type:
            return False
        mapped_base_type = self.raw_type_base(self.map_type(type_name))
        if (
            self.is_uav_resource_type(type_name)
            or self.is_buffer_resource_type(type_name)
            or self.is_crossgl_resource_type(mapped_base_type)
            or base_type.startswith(("Texture", "FeedbackTexture"))
            or base_type in {
                # A ConstantBuffer<T> binds a constant buffer of its own, outside $Globals.
                "ConstantBuffer",
                "Sampler",
                "SamplerState",
                "SamplerComparisonState",
                "sampler",
                "sampler_state",
                "RaytracingAccelerationStructure",
                "RayQuery",
                "InputPatch",
                "OutputPatch",
                "PointStream",
                "LineStream",
                "TriangleStream",
                *SUBPASS_INPUT_TYPES,
            }
        ):
            return False
        return True

    def is_crossgl_resource_type(self, type_name):
        base_type = self.raw_type_base(type_name)
        if not base_type:
            return False
        return base_type in {
            "sampler",
            "comparison_sampler",
            "accelerationStructureEXT",
        } or base_type.startswith(
            (
                "sampler",
                "isampler",
                "texture",
                "itexture",
                "utexture",
                "usampler",
                "image",
                "iimage",
                "uimage",
                "feedbackTexture",
            )
        )

    def collect_hlsl_program_constant_global_ids(self, ast):
        candidates = {
            getattr(node, "name", None): node
            for node in getattr(ast, "global_variables", []) or []
            if self.is_hlsl_program_constant_global(node)
            and getattr(node, "name", None)
        }
        if not candidates:
            return set()

        referenced_names = set()

        def visit(node):
            if node is None or isinstance(node, (int, float, bool)):
                return
            if isinstance(node, str):
                if node in candidates:
                    referenced_names.add(node)
                return
            if isinstance(node, VariableNode):
                name = getattr(node, "name", None)
                if name in candidates:
                    referenced_names.add(name)
            for child in self.iter_ast_children(node):
                visit(child)

        for func in getattr(ast, "functions", []) or []:
            visit(getattr(func, "body", []))

        return {id(candidates[name]) for name in referenced_names}
