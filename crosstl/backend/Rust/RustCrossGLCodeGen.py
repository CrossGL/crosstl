"""Reverse code generator that emits CrossGL from Rust AST nodes."""

import re

from ..import_refusals import ImportRefusalMixin, RustImportRefused
from .calls import RustCallMixin
from .closures import RustClosureMixin
from .control_flow import RustControlFlowMixin
from .declarations import RustDeclarationMixin
from .default_values import RustDefaultValueMixin
from .enums import RustEnumMixin
from .execution_modes import RustExecutionModeMixin
from .expression_results import RustExpressionResultMixin
from .expressions import RustExpressionMixin
from .import_tables import (
    BLOCK_EXPRESSION_NODE_TYPES,
    CROSSGL_RESERVED_IDENTIFIERS,
    RUST_ATTRIBUTES,
    RUST_FUNCTION_MAP,
    RUST_INTERPOLATION_SEMANTICS,
    RUST_NUMERIC_LITERAL_RE,
    RUST_RAW_STRING_RE,
    RUST_SEMANTICS,
    RUST_SPIRV_STAGES,
    RUST_TYPE_MAP,
    TRANSPARENT_BLOCK_NODE_TYPES,
)
from .loop_control import RustLoopControlMixin
from .match_lowering import RustMatchMixin
from .operator_spelling import RustOperatorSpellingMixin
from .RustAst import *
from .RustLexer import *
from .RustParser import *
from .try_operator import RustTryOperatorMixin
from .type_mapping import RustTypeMappingMixin
from .value_types import RustValueTypeMixin


class RustToCrossGLConverter(
    RustValueTypeMixin,
    RustDefaultValueMixin,
    RustOperatorSpellingMixin,
    RustEnumMixin,
    RustExecutionModeMixin,
    RustDeclarationMixin,
    RustExpressionResultMixin,
    RustTryOperatorMixin,
    RustCallMixin,
    RustControlFlowMixin,
    RustMatchMixin,
    RustLoopControlMixin,
    RustExpressionMixin,
    RustClosureMixin,
    RustTypeMappingMixin,
    ImportRefusalMixin,
):
    """Serialize Rust backend AST nodes back into CrossGL source."""

    IMPORT_REFUSAL_TYPE = RustImportRefused

    GLOBAL_INITIALIZER_MACROS = {
        "concat",
        "env",
        "include_bytes",
        "include_str",
        "option_env",
    }
    SOURCE_VALUE_MACROS = GLOBAL_INITIALIZER_MACROS | {"include_wgsl"}
    VALUE_PASSTHROUGH_MACROS = {
        "comptime",
    }
    OPAQUE_VALUE_MACROS = {
        "cfg_if",
        "intrinsic",
        "quote",
    }
    REPARSEABLE_STATEMENT_MACROS = {
        "assert",
        "assert_eq",
        "assert_ne",
        "debug",
        "debug_assert",
        "debug_assert_eq",
        "debug_assert_ne",
        "error",
        "format",
        "format_args",
        "log::debug",
        "log::error",
        "log::info",
        "log::trace",
        "log::warn",
        "info",
        "panic",
        "println",
        "trace",
        "todo",
        "unexpanded",
        "unimplemented",
        "unreachable",
        "warn",
        "write",
        "writeln",
    }
    SCALAR_ZERO_ARG_METHOD_MAP = {
        "abs": "abs",
        "floor": "floor",
        "ceil": "ceil",
        "round": "round",
        "trunc": "trunc",
        "round_ties_even": "roundEven",
        "fract": "fract",
        "sqrt": "sqrt",
        "exp": "exp",
        "exp2": "exp2",
        "ln": "log",
        "log2": "log2",
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "asin": "asin",
        "acos": "acos",
        "atan": "atan",
        "sinh": "sinh",
        "cosh": "cosh",
        "tanh": "tanh",
        "to_degrees": "degrees",
        "to_radians": "radians",
        "signum": "sign",
        "is_nan": "isnan",
        "is_infinite": "isinf",
        "is_finite": "isfinite",
        "count_ones": "bitCount",
        "reverse_bits": "bitfieldReverse",
    }
    SCALAR_ONE_ARG_METHOD_MAP = {
        "min": "min",
        "max": "max",
        "powf": "pow",
        "powi": "pow",
        "atan2": "atan2",
        "rem_euclid": "mod",
        "step": "step",
    }
    SCALAR_TWO_ARG_METHOD_MAP = {
        "clamp": "clamp",
        "lerp": "mix",
        "mul_add": "fma",
    }
    RUST_GPU_DERIVATIVE_METHOD_MAP = {
        "dfdx": "dFdx",
        "dfdy": "dFdy",
        "dfdx_fine": "dFdxFine",
        "dfdy_fine": "dFdyFine",
        "dfdx_coarse": "dFdxCoarse",
        "dfdy_coarse": "dFdyCoarse",
        "fwidth": "fwidth",
        "fwidth_fine": "fwidthFine",
        "fwidth_coarse": "fwidthCoarse",
    }
    RUST_GPU_ASSOCIATED_INTRINSIC_TYPES = {"Derivative"}
    RESOURCE_METHOD_PREFIXES = ("sample", "texture_", "image_", "buffer_")
    RESOURCE_METHOD_NAMES = {
        "fetch",
        "fetch_with",
        "fetch_with_lod",
        "gather",
        "gather_with",
        "query_levels",
        "query_lod",
        "query_samples",
        "query_size",
        "query_size_lod",
        "read",
        "write",
    }
    RESOURCE_TYPE_PREFIXES = (
        "sampler",
        "image",
        "uimage",
        "iimage",
        "StructuredBuffer",
        "RWStructuredBuffer",
        "AppendStructuredBuffer",
        "ConsumeStructuredBuffer",
        "ByteAddressBuffer",
        "RWByteAddressBuffer",
    )
    RUST_GPU_SAMPLE_METHOD_MAP = {
        "gather": "textureGather",
        "sample_by_lod": "textureLod",
        "sample_by_gradient": "textureGrad",
        "sample_with_project_coordinate": "textureProj",
        "sample_with_project_coordinate_by_lod": "textureProjLod",
        "sample_with_project_coordinate_by_gradient": "textureProjGrad",
        "sample_depth_reference": "textureCompare",
        "sample_depth_reference_by_lod": "textureCompareLod",
        "sample_depth_reference_by_gradient": "textureCompareGrad",
        "sample_depth_reference_with_project_coordinate": "textureCompareProj",
        "sample_depth_reference_with_project_coordinate_by_lod": (
            "textureCompareProjLod"
        ),
        "sample_depth_reference_with_project_coordinate_by_gradient": (
            "textureCompareProjGrad"
        ),
    }
    RUST_GPU_SAMPLE_WITH_METHOD_MAP = {
        "sample_with": "texture",
        "sample_with_project_coordinate_with": "textureProj",
        "sample_depth_reference_with": "textureCompare",
        "sample_depth_reference_with_project_coordinate_with": "textureCompareProj",
    }
    VECTOR_EXTEND_CONSTRUCTOR_MAP = {
        "vec2": "vec3",
        "vec3": "vec4",
        "ivec2": "ivec3",
        "ivec3": "ivec4",
        "uvec2": "uvec3",
        "uvec3": "uvec4",
        "bvec2": "bvec3",
        "bvec3": "bvec4",
    }
    VECTOR_TRUNCATE_CONSTRUCTOR_MAP = {
        "vec3": "vec2",
        "vec4": "vec3",
        "ivec3": "ivec2",
        "ivec4": "ivec3",
        "uvec3": "uvec2",
        "uvec4": "uvec3",
        "bvec3": "bvec2",
        "bvec4": "bvec3",
    }
    VECTOR_CONSTRUCTOR_RETURN_TYPES = {
        "vec2": "vec2",
        "vec3": "vec3",
        "vec4": "vec4",
        "ivec2": "ivec2",
        "ivec3": "ivec3",
        "ivec4": "ivec4",
        "uvec2": "uvec2",
        "uvec3": "uvec3",
        "uvec4": "uvec4",
        "bvec2": "bvec2",
        "bvec3": "bvec3",
        "bvec4": "bvec4",
    }
    VECTOR_SPLAT_CONSTRUCTORS = set(VECTOR_CONSTRUCTOR_RETURN_TYPES.values())
    VECTOR_COMPONENT_COUNTS = {
        "vec2": 2,
        "vec3": 3,
        "vec4": 4,
        "ivec2": 2,
        "ivec3": 3,
        "ivec4": 4,
        "uvec2": 2,
        "uvec3": 3,
        "uvec4": 4,
        "bvec2": 2,
        "bvec3": 3,
        "bvec4": 4,
    }
    SCALAR_ASSOCIATED_CONSTANTS = {
        "u8::MAX": "255",
        "u8::MIN": "0",
        "u16::MAX": "65535",
        "u16::MIN": "0",
        "u32::MAX": "4294967295u",
        "u32::MIN": "0u",
        "i8::MAX": "127",
        "i8::MIN": "(-128)",
        "i16::MAX": "32767",
        "i16::MIN": "(-32768)",
        "i32::MAX": "2147483647",
        "i32::MIN": "(-2147483648)",
        "f32::EPSILON": "1.1920929e-7",
        "f32::INFINITY": "1.0 / 0.0",
        "f32::NEG_INFINITY": "-1.0 / 0.0",
    }
    VECTOR_AXIS_CONSTANT_INDICES = {"X": 0, "Y": 1, "Z": 2, "W": 3}

    def __init__(self):
        self.type_map = dict(RUST_TYPE_MAP)

        self.semantic_map = dict(RUST_SEMANTICS)
        self.interpolation_semantic_map = dict(RUST_INTERPOLATION_SEMANTICS)
        self.spirv_stage_map = dict(RUST_SPIRV_STAGES)

        self.function_map = dict(RUST_FUNCTION_MAP)

        self.attribute_map = dict(RUST_ATTRIBUTES)

        self.indentation = 0
        self.code = []
        self.type_aliases = {}
        self.imported_type_aliases = {}
        self.imported_module_aliases = {}
        self.associated_consts = {}
        self.struct_member_types = {}
        self.struct_generics = {}
        self.return_result_target = object()
        self.labeled_control_counter = 0
        self.switch_break_counter = 0
        self.for_loop_index_counter = 0
        self.for_loop_step_counter = 0
        self.for_loop_bound_counter = 0
        self.for_loop_iterable_counter = 0
        self.match_chain_counter = 0
        self.match_subject_counter = 0
        self.match_payload_counter = 0
        self.match_or_counter = 0
        self.match_tuple_counter = 0
        self.match_array_counter = 0
        self.matches_result_counter = 0
        self.closure_param_counter = 0
        self.try_subject_counter = 0
        self.try_value_counter = 0
        self.try_block_value_counter = 0
        self.transparent_block_value_counter = 0
        self.inline_expression_value_counter = 0
        self.current_function_return_type = None
        self.user_function_names = set()
        self.function_return_types = {}
        self.function_definitions = {}
        self.function_type_signatures = {}
        self.impl_method_signatures = {}
        self.impl_receiver_method_names = set()
        self.impl_associated_function_names = set()
        self.option_unwrap_or_helpers = {}
        self.current_option_unwrap_fallback_name = None
        self.value_type_scopes = [{}]
        self.closure_helper_counter = 0
        self.closure_helper_names = set()
        self.local_function_item_counter = 0
        self.local_function_item_names = set()
        self.local_function_item_helper_names = {}
        self.current_closure_helpers = None
        self.closure_helper_generation_depth = 0
        self.reparseable_macro_expression_depth = 0
        self.name_alias_scopes = []
        self.local_binding_name_scopes = []
        self.local_callable_scopes = []

    def get_indent(self):
        return "    " * self.indentation

    def visit(self, node):
        """Dispatch a Rust AST node to a visitor method when available."""
        if isinstance(node, StructNode):
            return self.visit_StructNode(node)
        elif isinstance(node, FunctionNode):
            return self.visit_FunctionNode(node)
        elif isinstance(node, BinaryOpNode):
            return self.visit_BinaryOpNode(node)
        elif isinstance(node, UnaryOpNode):
            return self.visit_UnaryOpNode(node)
        elif isinstance(node, ImplNode):
            return self.visit_ImplNode(node)
        elif isinstance(node, UseNode):
            return self.visit_UseNode(node)
        elif isinstance(node, ConstNode):
            return self.visit_ConstNode(node)
        elif isinstance(node, StaticNode):
            return self.visit_StaticNode(node)

        # For other node types, use existing methods
        if hasattr(self, f"generate_{type(node).__name__}"):
            method = getattr(self, f"generate_{type(node).__name__}")
            return method(node)
        return self.generate_expression(node)

    def generate(self, ast):
        """Render ``ast`` as CrossGL; raise :class:`RustImportRefused` if a construct has no CrossGL form."""
        return self.refusals.run(self.render_program, ast)

    def render_program(self, ast):
        self.type_aliases = {}
        self.imported_type_aliases = {}
        self.imported_module_aliases = {}
        self.associated_consts = {}
        self.struct_member_types = self.collect_struct_member_types(ast)
        self.struct_generics = self.collect_struct_generics(ast)
        self.labeled_control_counter = 0
        self.switch_break_counter = 0
        self.for_loop_index_counter = 0
        self.for_loop_step_counter = 0
        self.for_loop_bound_counter = 0
        self.match_chain_counter = 0
        self.match_subject_counter = 0
        self.match_payload_counter = 0
        self.match_or_counter = 0
        self.match_tuple_counter = 0
        self.match_array_counter = 0
        self.matches_result_counter = 0
        self.closure_param_counter = 0
        self.try_subject_counter = 0
        self.try_value_counter = 0
        self.try_block_value_counter = 0
        self.transparent_block_value_counter = 0
        self.inline_expression_value_counter = 0
        self.current_function_return_type = None
        self.user_function_names = self.collect_user_function_names(ast)
        self.function_return_types = self.collect_function_return_types(ast)
        self.function_definitions = self.collect_function_definitions(ast)
        self.function_type_signatures = self.collect_function_type_signatures(ast)
        self.impl_method_signatures = self.collect_impl_method_signatures(ast)
        (
            self.impl_receiver_method_names,
            self.impl_associated_function_names,
        ) = self.collect_impl_method_name_sets()
        self.value_type_scopes = [{}]
        self.closure_helper_counter = 0
        self.closure_helper_names = set()
        self.local_function_item_counter = 0
        self.local_function_item_names = set()
        self.local_function_item_helper_names = {}
        self.option_unwrap_or_helpers = {}
        self.current_option_unwrap_fallback_name = None
        self.current_closure_helpers = None
        self.closure_helper_generation_depth = 0
        self.reparseable_macro_expression_depth = 0
        self.name_alias_scopes = []
        self.local_binding_name_scopes = []
        self.local_callable_scopes = []
        code = "shader main {\n"

        for use_stmt in ast.use_statements:
            self.register_use_alias(use_stmt)
            code += f"    // use {use_stmt.path}\n"

        for alias in getattr(ast, "type_aliases", []):
            self.register_type_alias(alias)

        self.register_associated_consts(ast)

        code += self.generate_unit_enums(ast)

        for alias in getattr(ast, "type_aliases", []):
            alias_code = self.generate_type_alias(alias)
            if alias_code:
                code += alias_code

        for global_var in ast.global_variables:
            if isinstance(global_var, ConstNode):
                self.add_value_type(global_var.name, global_var.vtype)
                declarator = self.format_global_declarator(
                    global_var.vtype, global_var.name
                )
                value = self.generate_global_initializer(global_var.value)
                code += f"    const {declarator} = {value};\n"
            elif isinstance(global_var, StaticNode):
                self.add_value_type(global_var.name, global_var.vtype)
                mutability = "mut " if global_var.is_mutable else ""
                declarator = self.format_global_declarator(
                    global_var.vtype, global_var.name
                )
                value = self.generate_global_initializer(global_var.value)
                code += f"    static {mutability}{declarator} = {value};\n"

        for struct in ast.structs:
            if isinstance(struct, StructNode):
                code += f"    struct {struct.name} {{\n"
                for member in struct.members:
                    semantic = self.get_semantic_from_attributes(member.attributes)
                    declarator = self.format_typed_declarator(member.vtype, member.name)
                    code += f"        {declarator}{semantic};\n"
                code += "    }\n\n"

        for impl_block in ast.impl_blocks:
            if getattr(impl_block, "is_negative", False):
                continue

            consts_code = ""
            for const in getattr(impl_block, "associated_consts", []):
                consts_code += self.generate_impl_associated_const(
                    impl_block.struct_name,
                    const,
                    indent=1,
                )
            if consts_code:
                code += f"    // Associated constants for {impl_block.struct_name}\n"
                code += consts_code

        for func in ast.functions:
            shader_type = self.get_shader_type_from_attributes(func.attributes)
            if shader_type:
                code += f"    // {shader_type.title()} Shader\n"
                code += self.generate_shader_stage_function(func, shader_type)
            else:
                code += self.generate_function(func, indent=1)

        for impl_block in ast.impl_blocks:
            if getattr(impl_block, "is_negative", False):
                code += (
                    f"    // Negative implementation for {impl_block.trait_name} "
                    f"on {impl_block.struct_name}\n"
                )
                continue
            code += f"    // Implementation for {impl_block.struct_name}\n"
            for func in impl_block.functions:
                code += self.generate_function(
                    func, indent=1, struct_name=impl_block.struct_name
                )

        code += self.generate_option_unwrap_or_helpers(indent=1)
        code += "}\n"
        return code

    def collect_user_function_names(self, ast):
        names = {getattr(func, "name", None) for func in getattr(ast, "functions", [])}
        for impl_block in getattr(ast, "impl_blocks", []):
            for func in getattr(impl_block, "functions", []):
                names.add(getattr(func, "name", None))
        names.discard(None)
        return names

    def push_name_alias_scope(self, aliases=None):
        self.name_alias_scopes.append(dict(aliases or {}))

    def current_name_alias_scope(self):
        if not self.name_alias_scopes:
            self.push_name_alias_scope()
        return self.name_alias_scopes[-1]

    def pop_name_alias_scope(self):
        if self.name_alias_scopes:
            self.name_alias_scopes.pop()

    def push_local_callable_scope(self, entries=None):
        scope = {}
        if entries:
            for name, alias in entries:
                if name:
                    scope[name] = alias or name
        self.local_callable_scopes.append(scope)

    def pop_local_callable_scope(self):
        if self.local_callable_scopes:
            self.local_callable_scopes.pop()

    def add_local_callable(self, name, alias=None):
        if not name:
            return
        if not self.local_callable_scopes:
            self.push_local_callable_scope()
        self.local_callable_scopes[-1][name] = alias or name

    def lookup_local_callable_name(self, name):
        if not isinstance(name, str):
            return None
        for scope in reversed(self.local_callable_scopes):
            callable_name = scope.get(name)
            if callable_name is not None:
                return callable_name
        return None

    def get_local_function_item_helper_name(self, func):
        key = id(func)
        helper_name = self.local_function_item_helper_names.get(key)
        if helper_name is None:
            helper_name = self.next_local_function_item_name(func.name)
            self.local_function_item_helper_names[key] = helper_name
        return helper_name

    def predeclare_local_function_items(self, body):
        for stmt in body or []:
            if isinstance(stmt, FunctionNode):
                self.add_local_callable(
                    stmt.name,
                    self.get_local_function_item_helper_name(stmt),
                )

    def generate_scoped_function_body(
        self,
        body,
        indent=1,
        loop_contexts=None,
        allow_implicit_final_return=False,
    ):
        self.push_local_callable_scope()
        try:
            return self.generate_function_body(
                body,
                indent,
                loop_contexts,
                allow_implicit_final_return=allow_implicit_final_return,
            )
        finally:
            self.pop_local_callable_scope()

    def resolve_name_alias(self, name):
        if not isinstance(name, str):
            return name

        for scope in reversed(self.name_alias_scopes):
            alias = scope.get(name)
            if alias is not None:
                return alias
        return name

    def crossgl_identifier(self, name, forbidden=None):
        if not isinstance(name, str) or not name:
            return name

        forbidden = set(forbidden or ())
        if self.is_valid_crossgl_identifier(name) and name not in forbidden:
            return name

        base = re.sub(r"\W+", "_", name).strip("_") or "value"
        if base[0].isdigit():
            base = f"_{base}"

        candidate = f"{base}_"
        suffix = 1
        while not self.is_valid_crossgl_identifier(candidate) or candidate in forbidden:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def is_valid_crossgl_identifier(self, name):
        return (
            self.is_plain_identifier(name) and name not in CROSSGL_RESERVED_IDENTIFIERS
        )

    def prepare_function_parameters(
        self, params, struct_name=None, extra_forbidden=None, shader_type=None
    ):
        original_names = {
            param.name for param in params if getattr(param, "name", None)
        }
        extra_forbidden = set(extra_forbidden or ())
        used_names = set()
        aliases = {}
        declarations = []
        param_types = []

        for param in params:
            forbidden = (original_names - {param.name}) | used_names | extra_forbidden
            name = self.crossgl_identifier(param.name, forbidden)
            aliases[param.name] = name
            used_names.add(name)
            semantic = self.get_semantic_from_attributes(
                getattr(param, "attributes", [])
            )
            qualifier = self.get_parameter_qualifier_from_attributes(
                getattr(param, "attributes", [])
            )
            param_type = self.normalize_parameter_type(
                param.vtype,
                struct_name,
                attributes=getattr(param, "attributes", []),
            )
            if self.is_graphics_mutable_output_parameter(
                param, shader_type, qualifier, semantic
            ):
                qualifier = "out"
                if not semantic and str(shader_type).lower() == "fragment":
                    semantic = " @ gl_FragColor"
            elif (
                shader_type is None
                and not qualifier
                and self.is_mutable_reference_value_parameter(param, param_type)
            ):
                # A `&mut T` parameter writes the caller's place. Safe Rust lets nothing else reach that place
                # during the call, so CrossGL's copy-in/copy-out `inout` observes the same values.
                qualifier = "inout"
            declaration = self.format_typed_declarator(param_type, name)
            if qualifier:
                declaration = f"{qualifier} {declaration}"
            declarations.append(f"{declaration}{semantic}")
            param_types.append((name, param_type))

        return ", ".join(declarations), param_types, aliases

    _OPAQUE_TYPE_WORDS = (
        "sampler",
        "image",
        "texture",
        "buffer",
        "accelerationstructure",
        "subpass",
        "rayquery",
    )

    def is_mutable_reference_value_parameter(self, param, param_type):
        """Whether ``param`` is ``&mut T`` of a sized value ``T``, not a resource handle, buffer or slice."""
        raw_type = str(getattr(param, "vtype", "")).strip()
        if not re.match(r"^&\s*(?:'\w+\s+)?mut\b", raw_type):
            return False
        crossgl_type = str(param_type).strip()
        if "[]" in crossgl_type or "&" in crossgl_type:
            return False
        base = re.split(r"[\s<\[]", crossgl_type, maxsplit=1)[0]
        if base in self.struct_member_types:
            return True
        return not any(word in base.lower() for word in self._OPAQUE_TYPE_WORDS)

    def get_parameter_qualifier_from_attributes(self, attributes):
        if not attributes:
            return ""

        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue
            if "uniform" in attr.args:
                return "uniform"

        return ""

    def is_graphics_mutable_output_parameter(
        self, param, shader_type, qualifier="", semantic=""
    ):
        shader_type = str(shader_type).lower() if shader_type is not None else None
        if shader_type not in {"vertex", "fragment"}:
            return False
        if qualifier in {"uniform"}:
            return False

        raw_type = str(getattr(param, "vtype", "")).strip()
        if not raw_type.startswith("&mut "):
            return False

        semantic_text = str(semantic or "")
        if shader_type == "fragment" and "gl_FragColor" in semantic_text:
            return True
        if shader_type == "fragment" and not semantic_text:
            return True
        if shader_type == "vertex" and "gl_Position" in semantic_text:
            return True
        return False

    def local_aliasing_enabled(self):
        return bool(self.local_binding_name_scopes)

    def collect_simple_local_binding_names(self, statements):
        names = set()

        def visit_statement(stmt):
            if isinstance(stmt, LetNode):
                if isinstance(stmt.name, str):
                    names.add(stmt.name)
                for child in (stmt.value, stmt.else_body):
                    visit(child)
                return
            if isinstance(stmt, IfNode):
                visit(stmt.if_body)
                for _, body in getattr(stmt, "if_chain", []) or []:
                    visit(body)
                for _, body in getattr(stmt, "else_if_chain", []) or []:
                    visit(body)
                visit(stmt.else_body)
                return
            if isinstance(stmt, (ForNode, WhileNode, LoopNode)):
                pattern = getattr(stmt, "pattern", None)
                if isinstance(pattern, str):
                    names.add(pattern)
                visit(getattr(stmt, "body", None))
                return
            if isinstance(stmt, MatchNode):
                for arm in getattr(stmt, "arms", []) or []:
                    visit(getattr(arm, "body", None))

        def visit(node):
            if node is None:
                return
            if isinstance(node, list):
                for item in node:
                    visit_statement(item)
                return
            if isinstance(node, BlockNode):
                visit(node.statements)
                return
            visit_statement(node)

        visit(statements)
        return names

    def declare_local_alias(self, name):
        if not self.local_aliasing_enabled() or not isinstance(name, str):
            return name

        forbidden = set(self.local_binding_name_scopes[-1])
        forbidden.discard(name)
        for scope in self.name_alias_scopes:
            forbidden.update(scope.values())

        alias = self.crossgl_identifier(name, forbidden)
        self.current_name_alias_scope()[name] = alias
        return alias

    def collect_function_return_types(self, ast):
        return_types = {}
        for func in getattr(ast, "functions", []):
            name = getattr(func, "name", None)
            return_type = getattr(func, "return_type", None)
            if name and return_type:
                return_types[name] = return_type
        return return_types

    def collect_function_definitions(self, ast):
        definitions = {}
        for func in getattr(ast, "functions", []):
            name = getattr(func, "name", None)
            if name:
                definitions[name] = func
        return definitions

    def is_block_expression_node(self, node):
        return isinstance(node, BLOCK_EXPRESSION_NODE_TYPES)

    def is_transparent_block_expression_node(self, node):
        return isinstance(node, TRANSPARENT_BLOCK_NODE_TYPES)

    def get_block_expression_node(self, node):
        if isinstance(node, BlockNode):
            return node
        if self.is_transparent_block_expression_node(node):
            return node.block
        return None

    def effective_attributes(self, attributes):
        for attr in attributes or []:
            yield attr
            if isinstance(attr, AttributeNode) and attr.name == "cfg_attr":
                yield from self.extract_cfg_attr_codegen_attributes(attr.args)

    def extract_cfg_attr_codegen_attributes(self, args):
        codegen_attribute_names = {
            "spirv",
            "vertex_shader",
            "fragment_shader",
            "compute_shader",
            "location",
            "binding",
        }
        extracted = []
        index = 0
        args = list(args or [])

        while index < len(args):
            name = args[index]
            if name not in codegen_attribute_names:
                index += 1
                continue

            if index + 1 < len(args) and args[index + 1] == "(":
                nested_args, index = self.collect_nested_cfg_attr_args(
                    args,
                    index + 1,
                )
                extracted.append(AttributeNode(name, nested_args))
                continue

            extracted.append(AttributeNode(name, []))
            index += 1

        return extracted

    def collect_nested_cfg_attr_args(self, args, open_index):
        nested_args = []
        depth = 0
        index = open_index + 1

        while index < len(args):
            token = args[index]
            if token == "(":
                depth += 1
                nested_args.append(token)
            elif token == ")":
                if depth == 0:
                    return nested_args, index + 1
                depth -= 1
                nested_args.append(token)
            else:
                nested_args.append(token)
            index += 1

        return nested_args, index

    def get_shader_type_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if isinstance(attr, AttributeNode):
                mapped = self.attribute_map.get(attr.name)
                if mapped in ["vertex", "fragment", "compute"]:
                    return mapped
                if attr.name == "spirv":
                    for arg in attr.args:
                        mapped = self.spirv_stage_map.get(arg)
                        if mapped:
                            return mapped
        return None

    def get_entry_point_name_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue

            entry_point_name = self.attribute_arg_value(attr.args, "entry_point_name")
            if entry_point_name is not None:
                return self.unquote_attribute_string(entry_point_name)

        return None

    def get_numthreads_from_attributes(self, attributes):
        if not attributes:
            return None

        for attr in self.effective_attributes(attributes):
            if not isinstance(attr, AttributeNode) or attr.name != "spirv":
                continue
            args = list(attr.args or [])
            if "threads" not in args:
                continue
            threads_index = args.index("threads")
            if threads_index + 1 >= len(args) or args[threads_index + 1] != "(":
                continue

            values = []
            depth = 0
            for token in args[threads_index + 2 :]:
                if token == "(":
                    depth += 1
                    continue
                if token == ")":
                    if depth == 0:
                        break
                    depth -= 1
                    continue
                if depth == 0:
                    values.append(self.normalize_attribute_metadata_value(str(token)))

            if values:
                while len(values) < 3:
                    values.append("1")
                return values[:3]

        return None

    def unquote_attribute_string(self, value):
        if not isinstance(value, str):
            return value

        raw_string = RUST_RAW_STRING_RE.match(value)
        if raw_string:
            return raw_string.group(2)

        if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]

        return value

    def get_semantic_from_attributes(self, attributes):
        if not attributes:
            return ""

        for attr in self.effective_attributes(attributes):
            if isinstance(attr, AttributeNode):
                if attr.name in self.semantic_map:
                    return f" @ {self.semantic_map[attr.name]}"
                if attr.name == "spirv":
                    semantics = []
                    for arg in attr.args:
                        if arg in self.semantic_map:
                            semantics.append(self.semantic_map[arg])
                        elif arg in self.interpolation_semantic_map:
                            semantics.append(self.interpolation_semantic_map[arg])

                    location = self.attribute_arg_value(attr.args, "location")
                    binding_index = self.attribute_arg_value(attr.args, "binding")
                    descriptor_set = self.attribute_arg_value(
                        attr.args, "descriptor_set"
                    )
                    spec_constant_id = None
                    if "spec_constant" in attr.args:
                        spec_constant_id = self.attribute_arg_value(attr.args, "id")
                    input_attachment_index = self.attribute_arg_value(
                        attr.args, "input_attachment_index"
                    )

                    if location is not None:
                        semantics.append(
                            f"location({self.normalize_attribute_metadata_value(location)})"
                        )
                    if descriptor_set is not None:
                        semantics.append(
                            f"set({self.normalize_attribute_metadata_value(descriptor_set)})"
                        )
                    if binding_index is not None:
                        semantics.append(
                            f"binding({self.normalize_attribute_metadata_value(binding_index)})"
                        )
                    if "push_constant" in attr.args:
                        semantics.append("push_constant")
                    if spec_constant_id is not None:
                        semantics.append(
                            "constant_id("
                            f"{self.normalize_attribute_metadata_value(spec_constant_id)}"
                            ")"
                        )
                    if input_attachment_index is not None:
                        semantics.append(
                            "input_attachment_index("
                            f"{self.normalize_attribute_metadata_value(input_attachment_index)}"
                            ")"
                        )
                    if "workgroup" in attr.args:
                        semantics.append("groupshared")
                    if semantics:
                        return "".join(f" @ {semantic}" for semantic in semantics)
                elif attr.name == "location" and attr.args:
                    return (
                        " @ location("
                        f"{self.normalize_attribute_metadata_value(attr.args[0])}"
                        ")"
                    )
                elif attr.name == "binding" and attr.args:
                    return (
                        " @ binding("
                        f"{self.normalize_attribute_metadata_value(attr.args[0])}"
                        ")"
                    )
        return ""

    def attribute_arg_value(self, args, key):
        for index, arg in enumerate(args):
            if arg != key:
                continue
            if index + 2 < len(args) and args[index + 1] == "=":
                return args[index + 2]
            if index + 1 < len(args):
                return args[index + 1]
        return None

    def normalize_attribute_metadata_value(self, value):
        if not isinstance(value, str):
            return value

        match = RUST_NUMERIC_LITERAL_RE.match(value)
        if not match:
            return value

        return match.group("body").replace("_", "")

    def visit_StructNode(self, node):
        code = f"struct {node.name} {{\n"
        self.indentation += 1

        for member in node.members:
            semantic = self.get_semantic_from_attributes(member.attributes)
            type_str = self.map_type(member.vtype)
            code += f"{self.get_indent()}{type_str} {member.name}{semantic};\n"

        self.indentation -= 1
        code += f"{self.get_indent()}}}\n"
        return code

    def visit_FunctionNode(self, node):
        shader_type = self.get_shader_type_from_attributes(node.attributes)
        if shader_type:
            body = self.generate_function_body(
                node.body,
                1,
                allow_implicit_final_return=True,
            )
            return f"{shader_type} {{\n{body}}}\n"
        else:
            return self.generate_function(node, 0)

    def visit_BinaryOpNode(self, node):
        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = self.generate_expression(node.operand)
        return f"({node.op}{operand})"

    def visit_ImplNode(self, node):
        if getattr(node, "is_negative", False):
            return f"// Negative implementation for {node.trait_name} on {node.struct_name}\n"
        code = f"// Implementation for {node.struct_name}\n"
        for func in node.functions:
            code += self.generate_function(func, 0, node.struct_name)
        return code

    def visit_UseNode(self, node):
        return f"// use {node.path}\n"

    def visit_ConstNode(self, node):
        declarator = self.format_global_declarator(node.vtype, node.name)
        value = self.generate_global_initializer(node.value)
        return f"const {declarator} = {value};\n"

    def visit_StaticNode(self, node):
        mutability = "mut " if node.is_mutable else ""
        declarator = self.format_global_declarator(node.vtype, node.name)
        value = self.generate_global_initializer(node.value)
        return f"static {mutability}{declarator} = {value};\n"
