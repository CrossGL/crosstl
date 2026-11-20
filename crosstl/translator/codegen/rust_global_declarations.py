"""Rust constant buffers, statics, constants and default initializers emitted for CrossGL globals."""

from ..ast import (
    ArrayLiteralNode,
    ArrayNode,
    ConstantNode,
    StructNode,
)
from .rust_resource_bindings import refuse_unbound_rust_resource


class RustGlobalDeclarationMixin:
    """Global-declaration members of ``RustCodeGen``."""

    def generate_cbuffers(self, ast):
        code = ""
        cbuffers = self.get_cbuffer_nodes(ast)
        for node in cbuffers:
            if isinstance(node, StructNode):
                code += self.rust_resource_metadata_comment(
                    node, getattr(node, "name", None), kind="cbuffer"
                )
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    member_name = self.rust_identifier(member.name)
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += (
                                f"    pub {member_name}: "
                                f"[{self.map_type(member.element_type)}; "
                                f"{member.size}],\n"
                            )
                        else:
                            code += (
                                f"    pub {member_name}: "
                                f"Vec<{self.map_type(member.element_type)}>,\n"
                            )
                    else:
                        code += (
                            f"    pub {member_name}: "
                            f"{self.map_type(self.get_member_type(member))},\n"
                        )
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
            elif hasattr(node, "name") and hasattr(node, "members"):  # CbufferNode
                code += self.rust_resource_metadata_comment(
                    node, getattr(node, "name", None), kind="cbuffer"
                )
                members = getattr(node, "members", [])
                derive_traits = self.derive_traits_for_members(members)
                code += f"#[repr(C)]\n#[derive({derive_traits})]\n"
                code += f"pub struct {node.name} {{\n"
                for member in members:
                    member_name = self.rust_identifier(member.name)
                    if isinstance(member, ArrayNode):
                        if member.size:
                            code += (
                                f"    pub {member_name}: "
                                f"[{self.map_type(member.element_type)}; "
                                f"{member.size}],\n"
                            )
                        else:
                            code += (
                                f"    pub {member_name}: "
                                f"Vec<{self.map_type(member.element_type)}>,\n"
                            )
                    else:
                        code += (
                            f"    pub {member_name}: "
                            f"{self.map_type(self.get_member_type(member))},\n"
                        )
                code += "}\n\n"
                code += self.generate_cbuffer_member_statics(members)
        return code

    def generate_cbuffer_member_statics(self, members):
        code = ""
        for member in members:
            if isinstance(member, ArrayNode):
                if member.size:
                    member_type = (
                        f"[{self.map_type(member.element_type)}; {member.size}]"
                    )
                else:
                    member_type = f"Vec<{self.map_type(member.element_type)}>"
            else:
                member_type = self.map_type(self.get_member_type(member))
            initializer = self.rust_static_default_initializer(member_type)
            code += self.generate_static_declaration(
                member.name,
                member_type,
                initializer,
            )
        return code

    def generate_variable_static_declaration(self, node):
        """Render a module-level static for a global or stage-local variable."""
        var_type = self.get_variable_type(node)
        if var_type is None:
            var_type = "float"
        elif hasattr(var_type, "name") or hasattr(var_type, "element_type"):
            var_type = self.convert_type_node_to_string(var_type)

        self.register_glsl_buffer_block_metadata(node)
        self.register_variable_type(node.name, var_type, scope="static")
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))
        rust_type = self.map_type(var_type)

        if initial_value is not None:
            init_expr = self.generate_expression_with_type(
                initial_value,
                var_type,
                static_context=True,
            )
            lazy_lock = self.static_array_literal_requires_lazy_lock(
                var_type,
                initial_value,
            )
        else:
            init_expr = self.rust_static_default_initializer(rust_type)
            lazy_lock = False

        return (
            self.rust_resource_metadata_comment(node, var_type)
            + refuse_unbound_rust_resource(self, node, var_type)
            + self.generate_static_declaration(
                node.name,
                rust_type,
                init_expr,
                lazy_lock=lazy_lock,
            )
        )

    def generate_global_array_declaration(self, node):
        element_type_name = self.convert_type_node_to_string(
            getattr(node, "element_type", "float")
        )
        element_type_name = self.resource_type_with_attributes(element_type_name, node)
        element_type = self.map_type(element_type_name)
        size = self.format_array_size(getattr(node, "size", None))
        initial_value = getattr(node, "initial_value", getattr(node, "value", None))

        if size is None:
            rust_type = f"Vec<{element_type}>"
            source_type = f"{element_type_name}[]"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = source_type
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False
        else:
            rust_type = f"[{element_type}; {size}]"
            source_type = f"{element_type_name}[{size}]"
            if isinstance(initial_value, ArrayLiteralNode):
                target_type = source_type
                initializer = self.generate_expression_with_type(
                    initial_value, target_type, static_context=True
                )
                lazy_lock = self.static_array_literal_requires_lazy_lock(
                    target_type, initial_value
                )
            else:
                initializer = self.rust_static_default_initializer(rust_type)
                lazy_lock = False

        self.register_variable_type(node.name, source_type, scope="static")
        return (
            self.rust_resource_metadata_comment(node, source_type)
            + refuse_unbound_rust_resource(self, node, source_type)
            + self.generate_static_declaration(
                node.name, rust_type, initializer, lazy_lock=lazy_lock
            )
        )

    def register_constant_types(self, ast):
        for node in getattr(ast, "constants", []) or []:
            if not isinstance(node, ConstantNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
            self.register_variable_type(name, const_type, scope="static")

    def generate_constant_declarations(self, ast):
        code = ""
        for node in getattr(ast, "constants", []) or []:
            if isinstance(node, ConstantNode):
                code += self.generate_constant_declaration(node)
        return f"{code}\n" if code else ""

    def generate_constant_declaration(self, node):
        name = getattr(node, "name", None)
        if not name:
            return ""

        const_type = getattr(node, "const_type", getattr(node, "vtype", "float"))
        if hasattr(const_type, "name") or hasattr(const_type, "element_type"):
            const_type = self.convert_type_node_to_string(const_type)
        else:
            const_type = str(const_type)

        self.register_variable_type(name, const_type, scope="static")
        rust_type = self.map_type(const_type)
        value = getattr(node, "value", None)
        if value is None:
            initializer = self.rust_static_default_initializer(rust_type)
        else:
            initializer = self.generate_expression_with_type(
                value,
                const_type,
                static_context=True,
            )
            initializer = self.normalize_assignment_rhs(
                const_type,
                value,
                initializer,
                "=",
            )

        if self.rust_constant_initializer_is_const(rust_type):
            symbol = self.static_symbol_name(name)
            return f"const {symbol}: {rust_type} = {initializer};\n"

        return self.generate_static_declaration(
            name,
            rust_type,
            initializer,
            lazy_lock=True,
        )

    def rust_constant_initializer_is_const(self, rust_type):
        return self.rust_scalar_default_literal(rust_type) is not None

    def generate_static_declaration(
        self, name, rust_type, initializer, lazy_lock=False
    ):
        symbol = self.static_symbol_name(name)
        if self.pointer_type_parts(rust_type) is not None:
            self.lazy_static_names.discard(name)
            initializer = self.rust_static_pointer_storage_initializer(initializer)
            return f"static {symbol}: usize = {initializer};\n"
        lazy_lock = lazy_lock or self.static_initializer_requires_lazy_lock(
            rust_type,
            initializer,
        )
        if lazy_lock:
            self.lazy_static_names.add(name)
            return (
                f"static {symbol}: std::sync::LazyLock<{rust_type}> = "
                f"std::sync::LazyLock::new(|| {initializer});\n"
            )
        self.lazy_static_names.discard(name)
        return f"static {symbol}: {rust_type} = {initializer};\n"

    def static_initializer_requires_lazy_lock(self, rust_type, initializer):
        return str(initializer) == "Default::default()"

    def rust_static_pointer_storage_initializer(self, initializer):
        initializer = str(initializer).strip()
        if initializer in {
            "",
            "0",
            "Default::default()",
            "std::ptr::null()",
            "std::ptr::null_mut()",
        }:
            return "0"
        return f"({initializer} as usize)"

    def static_array_literal_requires_lazy_lock(self, target_type, initial_value):
        if not isinstance(initial_value, ArrayLiteralNode):
            return False
        if not self.is_array_type_name(target_type):
            return False

        base_type, sizes = self.c_array_type_parts(target_type)
        size = sizes[0] if sizes else None
        if size is None:
            return True

        return self.is_rust_shader_pod_value_type(self.map_type(base_type))

    def rust_static_default_initializer(self, rust_type):
        rust_type = str(rust_type)

        if rust_type.startswith("Vec<"):
            return "Vec::new()"

        array_type = self.parse_rust_array_type(rust_type)
        if array_type is not None:
            element_type, size = array_type
            element_initializer = self.rust_scalar_default_literal(element_type)
            if element_initializer is not None:
                return f"[{element_initializer}; {size}]"
            if self.rust_type_is_zeroable(element_type):
                return "unsafe { std::mem::zeroed() }"
            if self.is_rust_shader_pod_value_type(element_type):
                return "unsafe { std::mem::zeroed() }"
            return "Default::default()"

        scalar_initializer = self.rust_scalar_default_literal(rust_type)
        if scalar_initializer is not None:
            return scalar_initializer

        return "Default::default()"

    def rust_runtime_default_initializer(self, type_name):
        rust_type = self.map_type(type_name)
        if self.parse_rust_array_type(
            rust_type
        ) is not None and self.rust_type_is_zeroable(rust_type):
            return "unsafe { std::mem::zeroed() }"
        return self.rust_static_default_initializer(rust_type)

    def local_array_default_initializer(self, rust_type):
        rust_type = str(rust_type)
        if rust_type.startswith("Vec<"):
            return "Vec::new()"
        if self.parse_rust_array_type(rust_type) is not None:
            if not self.rust_type_default_derive_supported(
                rust_type
            ) and self.rust_type_is_zeroable(rust_type):
                return "unsafe { std::mem::zeroed() }"
            return "std::array::from_fn(|_| Default::default())"
        return None

    def parse_rust_array_type(self, rust_type):
        if not (rust_type.startswith("[") and rust_type.endswith("]")):
            return None
        body = rust_type[1:-1]
        if ";" not in body:
            return None
        element_type, size = body.rsplit(";", 1)
        return element_type.strip(), size.strip()

    def rust_scalar_default_literal(self, rust_type):
        rust_type = str(rust_type)
        if rust_type in {"f16", "f32", "f64"}:
            return "0.0"
        if rust_type in {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
        }:
            return "0"
        if rust_type == "bool":
            return "false"
        if rust_type == "&'static str":
            return '""'
        return None

    def rust_type_is_zeroable(self, rust_type):
        rust_type = str(rust_type).strip()
        if self.pointer_type_parts(rust_type) is not None:
            return True

        array_type = self.parse_rust_array_type(rust_type)
        if array_type is not None:
            element_type, _size = array_type
            return self.rust_type_is_zeroable(element_type)

        if self.rust_scalar_default_literal(rust_type) is not None:
            return rust_type != "&'static str"

        return self.is_rust_shader_pod_value_type(rust_type)
