"""Rust traits, structs, enums, generic parameters and semantic checks emitted for CrossGL type declarations."""

from ..ast import (
    ArrayNode,
    EnumNode,
    FunctionNode,
)
from ..matrix_shape import matrix_type_spelling
from ..refusals import RustRefusal
from .array_utils import prepend_array_dimension
from .union_layout import UnionLayoutCollector


class RustTypeDefinitionMixin:
    """Type-definition members of ``RustCodeGen``."""

    def generate_trait(self, node):
        """Render a simple CrossGL trait declaration as a Rust trait."""
        code = f"pub trait {node.name}{self.format_generic_params(node)}: Sized {{\n"
        for method in getattr(node, "members", []) or []:
            if isinstance(method, FunctionNode):
                code += self.generate_trait_method(method)
        code += "}\n\n"
        if self.is_auto_numeric_trait(node):
            code += self.generate_numeric_trait_primitive_impls(node.name)
        return code

    def is_auto_numeric_trait(self, node):
        """Detect the canonical value-style Numeric trait used by generic shaders."""
        if getattr(node, "name", None) != "Numeric":
            return False

        methods = {
            method.name: method
            for method in getattr(node, "members", []) or []
            if isinstance(method, FunctionNode)
        }
        if set(methods) != {"add", "mul", "zero", "one"}:
            return False

        return (
            self.is_binary_self_method(methods["add"])
            and self.is_binary_self_method(methods["mul"])
            and self.is_nullary_self_method(methods["zero"])
            and self.is_nullary_self_method(methods["one"])
        )

    def is_binary_self_method(self, method):
        params = getattr(method, "parameters", getattr(method, "params", []))
        return (
            len(params) == 2
            and params[0].name == "self"
            and self.function_parameter_type(params[0]) == "Self"
            and self.function_parameter_type(params[1]) == "Self"
            and self.convert_type_node_to_string(method.return_type) == "Self"
        )

    def is_nullary_self_method(self, method):
        params = getattr(method, "parameters", getattr(method, "params", []))
        return (
            len(params) == 0
            and self.convert_type_node_to_string(method.return_type) == "Self"
        )

    def generate_numeric_trait_primitive_impls(self, trait_name):
        """Provide primitive Rust implementations for the canonical Numeric trait."""
        primitive_types = ["f32", "f64", "i32", "u32", "i16", "u16", "i64", "u64"]
        code = ""
        for rust_type in primitive_types:
            one = "1.0" if rust_type in {"f32", "f64"} else "1"
            zero = "0.0" if rust_type in {"f32", "f64"} else "0"
            code += f"impl {trait_name} for {rust_type} {{\n"
            code += "    fn add(self, other: Self) -> Self {\n"
            code += "        self + other\n"
            code += "    }\n"
            code += "    fn mul(self, other: Self) -> Self {\n"
            code += "        self * other\n"
            code += "    }\n"
            code += "    fn zero() -> Self {\n"
            code += f"        {zero}\n"
            code += "    }\n"
            code += "    fn one() -> Self {\n"
            code += f"        {one}\n"
            code += "    }\n"
            code += "}\n\n"
        return code

    def generate_required_generic_math_traits(self):
        code = ""
        if "CglSqrt" in self.required_generic_math_traits:
            code += self.generate_cgl_sqrt_trait()
        return code

    def generate_cgl_sqrt_trait(self):
        code = "pub trait CglSqrt {\n"
        code += "    fn cgl_sqrt(self) -> Self;\n"
        code += "}\n\n"
        for rust_type in ["f32", "f64"]:
            code += f"impl CglSqrt for {rust_type} {{\n"
            code += "    fn cgl_sqrt(self) -> Self {\n"
            code += "        self.sqrt()\n"
            code += "    }\n"
            code += "}\n\n"
        for rust_type in ["i32", "u32", "i16", "u16", "i64", "u64"]:
            code += f"impl CglSqrt for {rust_type} {{\n"
            code += "    fn cgl_sqrt(self) -> Self {\n"
            code += f"        (self as f64).sqrt() as {rust_type}\n"
            code += "    }\n"
            code += "}\n\n"
        return code

    def generate_trait_method(self, method):
        param_list = getattr(method, "parameters", getattr(method, "params", []))
        params = []
        for param in param_list:
            param_type = self.function_parameter_type(param)
            if param.name == "self" and param_type == "Self":
                params.append("self")
            else:
                params.append(f"{param.name}: {self.map_type(param_type)}")

        return_type = getattr(method, "return_type", None)
        if return_type is not None:
            return_type = self.convert_type_node_to_string(return_type)
        else:
            return_type = "void"

        generic_params = self.format_generic_params(method)
        params_str = ", ".join(params)
        return (
            f"    fn {method.name}{generic_params}({params_str}) "
            f"-> {self.map_type(return_type)};\n"
        )

    def generate_struct(self, node):
        if UnionLayoutCollector.attribute(node, "union_layout") is not None:
            # A Rust struct gives every field its own storage, so writing one member of a union and reading
            # another would read a field that was never written.
            raise RustRefusal(
                f"Rust struct '{node.name}' has a @union_layout: the Rust generator has no union lowering, "
                "and a struct would give its overlapping members separate storage"
            )
        wrapper_enum = self.struct_enum_wrapper(node)
        if wrapper_enum is not None:
            self.struct_member_types[node.name] = {}
            self.struct_member_semantics[node.name] = {}
            self.struct_generic_params[node.name] = self.generic_param_names(node)
            return self.generate_enum(
                wrapper_enum,
                enum_name=node.name,
                generic_owner=node,
                derive_traits=self.derive_traits_for_enum(
                    wrapper_enum,
                    generic_owner=node,
                    include_default=self.enum_has_unit_variant(wrapper_enum),
                ),
                default_variant_name=self.enum_default_variant_name(wrapper_enum),
            )

        members = getattr(node, "members", [])
        derive_traits = self.derive_traits_for_members(members, include_default=True)
        code = f"#[repr(C)]\n#[derive({derive_traits})]\n"
        code += f"pub struct {node.name}{self.format_generic_params(node)} {{\n"
        member_types = {}
        member_semantics = {}

        for member in members:
            if isinstance(member, ArrayNode):
                member_name = self.rust_identifier(member.name)
                element_type = getattr(
                    member, "element_type", getattr(member, "vtype", "float")
                )
                member_types[member.name] = (
                    f"{self.convert_type_node_to_string(element_type)}[{member.size}]"
                    if member.size
                    else f"{self.convert_type_node_to_string(element_type)}[]"
                )
                if member.size:
                    code += (
                        f"    pub {member_name}: "
                        f"[{self.map_type_to_rust(element_type)}; {member.size}],\n"
                    )
                else:
                    code += (
                        f"    pub {member_name}: "
                        f"Vec<{self.map_type_to_rust(element_type)}>,\n"
                    )
            else:
                if not hasattr(member, "member_type") and not hasattr(member, "vtype"):
                    continue

                if hasattr(member, "member_type"):
                    member_type = self.convert_type_node_to_string(member.member_type)
                elif hasattr(member, "vtype"):
                    member_type = member.vtype
                else:
                    member_type = "float"
                member_types[member.name] = member_type

                semantic = self.node_semantic(member)
                if semantic:
                    member_semantics[member.name] = semantic
                    self.validate_rust_builtin_semantic_type(
                        semantic, member_type, "struct member semantic"
                    )
                semantic_comment = (
                    f"  // {self.map_semantic(semantic)}" if semantic else ""
                )
                member_name = self.rust_identifier(member.name)
                code += (
                    f"    pub {member_name}: "
                    f"{self.map_type(member_type)},{semantic_comment}\n"
                )

        self.struct_member_types[node.name] = member_types
        self.struct_member_semantics[node.name] = member_semantics
        self.struct_generic_params[node.name] = self.generic_param_names(node)
        code += "}\n\n"
        code += self.generate_struct_constructor_impl(node, member_types)
        if not self.members_default_derive_supported(members):
            code += self.generate_struct_default_impl(node, member_types)
        return code

    def generate_struct_constructor_impl(self, node, member_types):
        """Emit a positional constructor for generated Rust structs."""
        if not member_types:
            return ""

        generic_decl = self.format_generic_params(node)
        generic_use = self.format_generic_argument_params(node)
        params = []
        fields = []
        for name, member_type in member_types.items():
            param_name = self.struct_constructor_param_name(name)
            field_ident = self.rust_identifier(name)
            param_ident = self.rust_identifier(param_name)
            params.append(f"{param_ident}: {self.map_type(member_type)}")
            if param_name == name:
                fields.append(field_ident)
            else:
                fields.append(f"{field_ident}: {param_ident}")

        code = f"impl{generic_decl} {node.name}{generic_use} {{\n"
        code += f"    pub fn new({', '.join(params)}) -> Self {{\n"
        code += f"        Self {{ {', '.join(fields)} }}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def generate_struct_default_impl(self, node, member_types):
        generic_decl = self.format_generic_params(node)
        generic_use = self.format_generic_argument_params(node)
        field_initializers = []
        for name, member_type in member_types.items():
            field_initializers.append(
                f"{self.rust_identifier(name)}: "
                f"{self.rust_runtime_default_initializer(member_type)}"
            )

        code = f"impl{generic_decl} Default for {node.name}{generic_use} {{\n"
        code += "    fn default() -> Self {\n"
        code += f"        Self {{ {', '.join(field_initializers)} }}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def struct_constructor_param_name(self, field_name):
        if field_name in self.static_variable_names:
            return f"{field_name}_value"
        return field_name

    def struct_enum_wrapper(self, node):
        """Return a nested enum when a legacy wrapper struct should lower as enum."""
        members = getattr(node, "members", []) or []
        enum_members = [member for member in members if isinstance(member, EnumNode)]
        data_members = [
            member for member in members if not isinstance(member, EnumNode)
        ]

        if len(enum_members) != 1 or len(data_members) != 1:
            return None

        variant_member = data_members[0]
        if getattr(variant_member, "name", None) != "variant":
            return None

        variant_type = self.get_member_type(variant_member)
        if variant_type != enum_members[0].name:
            return None

        return enum_members[0]

    def enum_has_unit_variant(self, node):
        return self.enum_default_variant_name(node) is not None

    def enum_default_variant_name(self, node):
        for variant in getattr(node, "variants", []) or []:
            if self.enum_variant_data(variant):
                continue
            if getattr(variant, "value", None) is not None:
                continue
            return variant.name
        return None

    def generate_enum(
        self,
        node,
        enum_name=None,
        generic_owner=None,
        derive_traits=None,
        default_variant_name=None,
    ):
        """Render a CrossGL enum declaration as a Rust enum."""
        if derive_traits is None:
            default_variant_name = self.enum_default_variant_name(node)
            derive_traits = self.derive_traits_for_enum(
                node,
                generic_owner=generic_owner,
                include_default=default_variant_name is not None,
            )

        derived_traits = {
            trait.strip() for trait in derive_traits.split(",") if trait.strip()
        }

        if derive_traits:
            code = f"#[derive({derive_traits})]\n"
        else:
            code = ""
        name = enum_name or node.name
        generic_node = generic_owner or node
        self.enum_generic_params[name] = self.generic_param_names(generic_node)
        code += f"pub enum {name}{self.format_generic_params(generic_node)} {{\n"
        for variant in getattr(node, "variants", []) or []:
            if variant.name == default_variant_name:
                code += "    #[default]\n"
            code += f"    {self.generate_enum_variant(variant)},\n"
        code += "}\n\n"

        if "Default" not in derived_traits:
            code += self.generate_enum_default_impl(node, name, generic_node)

        return code

    def generate_enum_default_impl(self, node, name, generic_node):
        variant = self.enum_default_impl_variant(node)
        if variant is None:
            return ""

        default_expression = self.enum_variant_default_expression(variant)
        generic_decl = self.format_default_impl_generic_params(generic_node)
        generic_use = self.format_generic_argument_params(generic_node)
        code = f"impl{generic_decl} Default for {name}{generic_use} {{\n"
        code += "    fn default() -> Self {\n"
        code += f"        {default_expression}\n"
        code += "    }\n"
        code += "}\n\n"
        return code

    def enum_default_impl_variant(self, node):
        variants = getattr(node, "variants", []) or []
        for variant in variants:
            if not self.enum_variant_data(variant):
                return variant
        return variants[0] if variants else None

    def enum_variant_default_expression(self, variant):
        data = self.enum_variant_data(variant)
        if not data:
            return f"Self::{variant.name}"

        if all(isinstance(item, tuple) and len(item) == 2 for item in data):
            fields = ", ".join(
                f"{self.rust_identifier(name)}: Default::default()"
                for name, _field_type in data
            )
            return f"Self::{variant.name} {{ {fields} }}"

        args = ", ".join("Default::default()" for _field_type in data)
        return f"Self::{variant.name}({args})"

    def enum_variant_data(self, variant):
        data = getattr(variant, "data", None)
        if data:
            return data
        return getattr(variant, "fields", None) or []

    def format_default_impl_generic_params(self, node):
        generic_params = getattr(node, "generic_params", []) or []
        if not generic_params:
            return ""
        params = []
        for param in generic_params:
            name = self.generic_param_name(param)
            if not name:
                continue
            constraints = self.generic_param_constraints(param)
            if "Default" not in constraints:
                constraints.append("Default")
            params.append(f"{name}: {' + '.join(constraints)}")
        return f"<{', '.join(params)}>" if params else ""

    def generate_enum_variant(self, variant):
        data = self.enum_variant_data(variant)
        if data:
            if all(isinstance(item, tuple) and len(item) == 2 for item in data):
                fields = ", ".join(
                    f"{self.rust_identifier(name)}: "
                    f"{self.map_type(self.convert_type_node_to_string(field_type))}"
                    for name, field_type in data
                )
                return f"{variant.name} {{ {fields} }}"

            fields = ", ".join(
                self.map_type(self.convert_type_node_to_string(field_type))
                for field_type in data
            )
            return f"{variant.name}({fields})"

        value = getattr(variant, "value", None)
        if value is not None:
            return f"{variant.name} = {self.generate_expression(value)}"

        return variant.name

    def format_generic_params(self, node, extra_constraints=None):
        """Render AST generic parameters as a Rust generic parameter list."""
        extra_constraints = extra_constraints or {}
        params = []
        for param in getattr(node, "generic_params", []) or []:
            name = self.generic_param_name(param)
            if not name:
                continue
            constraints = self.generic_param_constraints(param)
            for constraint in extra_constraints.get(name, []):
                if constraint not in constraints:
                    constraints.append(constraint)
            if constraints:
                params.append(f"{name}: {' + '.join(constraints)}")
            else:
                params.append(name)
        return f"<{', '.join(params)}>" if params else ""

    def format_generic_argument_params(self, node):
        """Render only generic parameter names for a Rust type reference."""
        params = self.generic_param_names(node)
        return f"<{', '.join(params)}>" if params else ""

    def generic_param_names(self, node):
        """Return unique generic parameter names from an AST declaration."""
        params = []
        for param in getattr(node, "generic_params", []) or []:
            name = self.generic_param_name(param)

            if name and name not in params:
                params.append(name)

        return params

    def generic_param_name(self, param):
        if hasattr(param, "name"):
            return param.name
        if isinstance(param, (tuple, list)) and param:
            return param[0]
        return str(param)

    def generic_param_constraints(self, param):
        constraints = []
        for constraint in getattr(param, "constraints", []) or []:
            constraint_name = self.generic_constraint_name(constraint)
            if constraint_name and constraint_name not in constraints:
                constraints.append(constraint_name)
        return constraints

    def generic_constraint_name(self, constraint):
        if hasattr(constraint, "name") or hasattr(constraint, "element_type"):
            constraint = self.convert_type_node_to_string(constraint)
        return self.map_type(str(constraint))

    def convert_type_node_to_string(self, type_node) -> str:
        """Convert new AST TypeNode to string representation."""
        if hasattr(type_node, "value") and getattr(type_node, "name", None) is None:
            value = type_node.value
            return str(value).lower() if isinstance(value, bool) else str(value)

        type_class = type_node.__class__.__name__
        if type_class == "ReferenceType":
            referenced_type = self.convert_type_node_to_string(
                type_node.referenced_type
            )
            mutability = "mut " if type_node.is_mutable else ""
            return f"&{mutability}{referenced_type}"
        if type_class == "PointerType":
            pointee_type = self.convert_type_node_to_string(type_node.pointee_type)
            mutability = "mut" if type_node.is_mutable else "const"
            return f"*{mutability} {pointee_type}"
        if type_class == "FunctionType":
            params = ", ".join(
                self.convert_type_node_to_string(param_type)
                for param_type in type_node.param_types
            )
            return_type = self.convert_type_node_to_string(type_node.return_type)
            return f"fn({params}) -> {return_type}"
        if type_class == "ArrayType":
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = self.format_array_size(type_node.size)
            return prepend_array_dimension(element_type, size)
        if getattr(type_node, "name", None) is not None:
            generic_args = getattr(type_node, "generic_args", [])
            if generic_args:
                args = ", ".join(
                    self.convert_type_node_to_string(arg) for arg in generic_args
                )
                return f"{type_node.name}<{args}>"
            return type_node.name
        elif hasattr(type_node, "element_type") and hasattr(type_node, "size"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            size = type_node.size
            if element_type == "float":
                return f"vec{size}"
            elif element_type == "int":
                return f"ivec{size}"
            elif element_type == "uint":
                return f"uvec{size}"
            elif element_type == "double":
                return f"dvec{size}"
            elif element_type == "bool":
                return f"bvec{size}"
            elif element_type in {"i8", "u8", "i16", "u16", "f16"}:
                return f"vec{size}<{element_type}>"
            else:
                return f"{element_type}{size}"
        elif hasattr(type_node, "element_type") and hasattr(type_node, "rows"):
            element_type = self.convert_type_node_to_string(type_node.element_type)
            return matrix_type_spelling(type_node, element_type)
        else:
            return str(type_node)

    def format_array_size(self, size):
        if size is None:
            return None
        if hasattr(size, "value"):
            return size.value
        return size

    def extract_semantic_from_attributes(self, attributes):
        """Extract semantic information from new AST attributes."""
        for attr in attributes:
            if hasattr(attr, "name") and attr.name in self.semantic_map:
                return attr.name
        return None

    def node_semantic(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        return self.extract_semantic_from_attributes(getattr(node, "attributes", []))

    def rust_semantic_output_kind(self, semantic):
        if semantic is None:
            return None

        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        input_only_sources = {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_instanceid",
            "gl_pointcoord",
            "gl_vertexid",
        }
        input_only_sv_sources = {
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }
        if lower_name in input_only_sources or upper_name in input_only_sv_sources:
            return "input_only"

        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def is_rust_vec4_float_type(self, type_name):
        return self.map_type(type_name) == "Vec4<f32>"

    def is_rust_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "f32"

    def validate_rust_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.rust_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return

        if kind in {"position", "color"}:
            if self.is_rust_vec4_float_type(type_name):
                return
            raise RustRefusal(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "expected vec4-compatible type"
            )

        if kind == "depth" and not self.is_rust_float_scalar_type(type_name):
            raise RustRefusal(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "expected float type"
            )

    def validate_rust_output_semantic_stage(self, shader_type, semantic, context):
        kind = self.rust_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise RustRefusal(
                f"Unsupported {semantic} {context} for Rust codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if shader_type is None:
            return

        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if shader_type not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise RustRefusal(
                f"Unsupported {semantic} {context} for Rust {shader_type} stage; "
                f"valid stage is {allowed}"
            )

    def validate_rust_return_semantic(self, shader_type, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "()":
            raise RustRefusal(
                f"Unsupported {semantic} return semantic for Rust codegen; "
                "void return type"
            )
        self.validate_rust_output_semantic_stage(
            shader_type, semantic, "return semantic"
        )
        self.validate_rust_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_rust_struct_return_semantics(self, shader_type, return_type):
        if shader_type is None:
            return
        base_type = str(return_type).split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        if not member_semantics:
            return
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_rust_output_semantic_stage(shader_type, semantic, context)
            self.validate_rust_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def rust_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def rust_stage_parameter_semantic_rules(self):
        ray_stages = self.rust_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "u32", {"vertex"}),
            "gl_instanceid": ("instance_id", "u32", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "i32", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "u32", {"vertex"}),
            "gl_drawid": ("draw_id", "u32", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "u32", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "u32", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "i32", {"vertex"}),
            "SV_STARTINSTANCELOCATION": ("start_instance_location", "u32", {"vertex"}),
            "SV_DRAWID": ("draw_id", "u32", {"vertex"}),
            "gl_position": ("position", "Vec4<f32>", {"fragment"}),
            "gl_fragcoord": ("frag_coord", "Vec4<f32>", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "Vec2<f32>", {"fragment"}),
            "gl_primitiveid": ("primitive_id", "u32", {"fragment"} | ray_hit_stages),
            "gl_sampleid": ("sample_index", "u32", {"fragment"}),
            "SV_POSITION": ("position", "Vec4<f32>", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": ("primitive_id", "u32", {"fragment"} | ray_hit_stages),
            "SV_SAMPLEINDEX": ("sample_index", "u32", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "Vec3<u32>", {"compute"}),
            "gl_localinvocationid": (
                "local_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "gl_globalinvocationid": (
                "global_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "gl_localinvocationindex": (
                "local_invocation_index",
                "u32",
                {"compute"},
            ),
            "SV_GROUPID": ("workgroup_id", "Vec3<u32>", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "Vec3<u32>", {"compute"}),
            "SV_DISPATCHTHREADID": (
                "global_invocation_id",
                "Vec3<u32>",
                {"compute"},
            ),
            "SV_GROUPINDEX": ("local_invocation_index", "u32", {"compute"}),
            "gl_launchidext": ("launch_id", "Vec3<u32>", ray_stages),
            "gl_launchsizeext": ("launch_size", "Vec3<u32>", ray_stages),
            "gl_hittext": ("hit_t", "f32", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "u32", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "Vec3<u32>", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": (
                "launch_size",
                "Vec3<u32>",
                ray_stages,
            ),
        }

    def validate_rust_stage_parameter_semantic_type(
        self, parameter, semantic, expected_type
    ):
        actual_type = self.map_type(self.function_parameter_type(parameter))
        if actual_type == expected_type:
            return
        semantic_key = self.rust_stage_parameter_semantic_key(semantic)
        if (
            expected_type == "Vec3<u32>"
            and semantic_key in {"gl_globalinvocationid", "SV_DISPATCHTHREADID"}
            and actual_type in {"u32", "Vec2<u32>"}
        ):
            return
        raise RustRefusal(
            f"Unsupported {semantic} stage parameter semantic for Rust codegen; "
            f"expected {expected_type} type"
        )

    def validate_rust_stage_parameter_semantics(self, shader_type, parameters):
        if shader_type is None:
            return

        rules = self.rust_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for parameter in parameters or []:
            semantic = self.node_semantic(parameter)
            if semantic is None:
                continue

            semantic_key = self.rust_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if shader_type not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise RustRefusal(
                        f"Unsupported {semantic} stage parameter semantic for Rust "
                        f"{shader_type} stage; valid stage is {allowed}"
                    )
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise RustRefusal(
                        f"Duplicate Rust stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{parameter.name}'"
                    )
                seen_system_semantics[mapped_semantic] = parameter.name
                self.validate_rust_stage_parameter_semantic_type(
                    parameter, semantic, expected_type
                )
                continue

            kind = self.rust_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise RustRefusal(
                    f"Unsupported {semantic} stage parameter semantic for Rust "
                    f"{shader_type} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )

    def get_member_type(self, member):
        if hasattr(member, "member_type"):
            member_type = self.convert_type_node_to_string(member.member_type)
            return self.resource_type_with_attributes(member_type, member)
        if hasattr(member, "vtype"):
            return self.resource_type_with_attributes(member.vtype, member)
        return "float"

    def get_cbuffer_nodes(self, ast):
        nodes = []
        seen = set()
        for attr in ("cbuffers", "constants"):
            for node in getattr(ast, attr, None) or []:
                node_id = id(node)
                if node_id not in seen:
                    nodes.append(node)
                    seen.add(node_id)
        return nodes

    def map_type_to_rust(self, type_str):
        """The Rust spelling of a constant-buffer member type: the generator's one type mapping."""
        return self.map_type(type_str)
