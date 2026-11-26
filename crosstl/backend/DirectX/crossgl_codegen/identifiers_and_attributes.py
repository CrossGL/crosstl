"""Identifiers and attributes in the HLSL to CrossGL converter: identifier renames, intrinsic names, array suffixes, type aliases, enums, constant buffer names, binding, matrix layout, storage, interpolation and parameter attributes, parameters, geometry patch shapes and the variable type registry."""

import re

from crosstl.translator.codegen.hlsl_buffer_layout import (
    BUFFER_LAYOUT_ATTRIBUTE,
    HLSL_STRUCTURED_LAYOUT,
)

from ...common_ast import ArrayAccessNode
from ..DirectxAst import (
    EnumNode,
    FunctionNode,
    MemberAccessNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)
from ..vk_constant_ids import has_vk_constant_id
from ..vk_locations import vk_location_suffix
from ..work_graph_import import crossgl_work_graph_parameter_attributes


class IdentifierAttributeMixin:
    """Identifier attribute members of ``HLSLToCrossGLConverter``."""

    def get_indent(self):
        return "    " * self.indentation

    def is_opaque_template_value_path(self, name):
        """Return whether an HLSL template value path needs CrossGL-safe spelling."""
        return isinstance(name, str) and "<" in name and ">" in name and "::" in name

    def function_parameter_renames(self, params):
        reserved_names = {param.name for param in params if param.name}
        renames = {}
        for param in params:
            name = param.name
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or name not in self.crossgl_reserved_identifiers
            ):
                continue
            candidate = f"{name}_"
            while candidate in reserved_names or candidate in renames.values():
                candidate = f"{candidate}_"
            renames[name] = candidate
        return renames

    def collect_function_identifier_renames(self, functions):
        declared_names = {
            func.name
            for func in functions or []
            if isinstance(getattr(func, "name", None), str)
        }
        used_names = set(declared_names)
        renames = {}
        for name in sorted(declared_names):
            if "::" in name:
                candidate = self.sanitize_scoped_identifier_name(name)
                if candidate in self.crossgl_reserved_identifiers:
                    candidate = f"{candidate}_"
                while candidate in used_names or candidate in renames.values():
                    candidate = f"{candidate}_"
                renames[name] = candidate
                used_names.add(candidate)
                continue
            if not name.isidentifier() or name not in self.crossgl_reserved_identifiers:
                continue
            candidate = f"{name}_"
            while candidate in used_names or candidate in renames.values():
                candidate = f"{candidate}_"
            renames[name] = candidate
            used_names.add(candidate)
        return renames

    def collect_type_identifier_renames(self, ast):
        declared_names = {
            name
            for nodes in (
                getattr(ast, "structs", []) or [],
                getattr(ast, "typedefs", []) or [],
                getattr(ast, "enums", []) or [],
            )
            for node in nodes
            for name in [getattr(node, "name", None)]
            if isinstance(name, str) and name
        }
        used_names = set(declared_names) | set(
            self.function_identifier_renames.values()
        )
        renames = {}
        for name in sorted(declared_names):
            if (
                name.isidentifier()
                and name not in self.crossgl_reserved_identifiers
                and "::" not in name
            ):
                continue
            if name.isidentifier() and name in self.crossgl_reserved_identifiers:
                candidate = f"{name}_"
            else:
                candidate = self.sanitize_scoped_identifier_name(name)
                if candidate in self.crossgl_reserved_identifiers:
                    candidate = f"{candidate}_"
            while candidate in used_names or candidate in renames.values():
                candidate = f"{candidate}_"
            renames[name] = candidate
            used_names.add(candidate)
        return renames

    def crossgl_intrinsic_name(self, name):
        """The CrossGL name of callee ``name``: the CrossGL spelling of an HLSL intrinsic ``function_map`` renames.

        Any other callee is carried as written: a function or type the program declares, a constructor, or an HLSL
        built-in CrossGL spells the same way. The IR accepts HLSL built-ins as plain calls for the generators to
        map or refuse, and ``declared_functions`` refuses any other name (``builtin_function_names``).
        """
        if name in self.function_map:
            return self.function_map[name]
        return name

    def normalize_hlsl_intrinsic_name(self, name):
        """Drop HLSL's intrinsic namespace so builtin lowering still applies."""
        if not isinstance(name, str):
            return name
        if name.startswith("::"):
            name = name[2:]
        if name.startswith("hlsl::"):
            return name.split("::", 1)[1]
        return name

    def format_array_suffixes(self, node, is_main=False):
        sizes = getattr(node, "array_sizes", None)
        if not sizes:
            return ""
        parts = []
        for size in sizes:
            if size is None:
                parts.append("[]")
            else:
                parts.append(f"[{self.generate_expression(size, is_main)}]")
        return "".join(parts)

    def generate_type_alias_declaration(self, alias):
        """Write an HLSL ``typedef`` or ``using`` alias as a CrossGL ``typedef``.

        CrossGL spells an alias as a C declarator, so the array dimensions of
        ``typedef int Pair[2];`` and ``using Pair = int[2];`` follow the name.
        """
        alias_type = getattr(alias, "alias_type", None) or getattr(
            alias, "original_type", None
        )
        if alias_type is None:
            return ""
        return (
            f"typedef {self.map_type(alias_type)} "
            f"{self.render_type_identifier(alias.name)}"
            f"{self.format_array_suffixes(alias)};"
        )

    def generate_enum(self, enum, indent=1):
        code = "    " * indent + f"enum {self.render_type_identifier(enum.name)} {{\n"
        for member_name, member_value in enum.members:
            if member_value is None:
                code += "    " * (indent + 1) + f"{member_name},\n"
            else:
                code += (
                    "    " * (indent + 1)
                    + f"{member_name} = {self.generate_expression(member_value)},\n"
                )
        code += "    " * indent + "}\n"
        return code

    def collect_cbuffer_reserved_names(self, ast):
        names = set()
        for collection in (
            getattr(ast, "structs", []) or [],
            getattr(ast, "global_variables", []) or [],
            getattr(ast, "functions", []) or [],
            getattr(ast, "typedefs", []) or [],
            getattr(ast, "enums", []) or [],
        ):
            for node in collection:
                name = getattr(node, "name", None)
                if isinstance(name, str) and name:
                    if isinstance(node, FunctionNode):
                        name = self.render_function_identifier(name)
                    elif isinstance(node, (StructNode, TypeAliasNode, EnumNode)):
                        name = self.render_type_identifier(name)
                    else:
                        name = self.render_identifier(name)
                    names.add(name)
        return names

    def unique_cbuffer_name(self, name, used_names):
        base = name if isinstance(name, str) and name else "CBuffer"
        candidate = base
        if candidate not in used_names:
            return candidate

        index = 1
        while f"{base}_{index}" in used_names:
            index += 1
        return f"{base}_{index}"

    def format_attributes(self, attributes, indent, skip_names=None):
        if not attributes:
            return ""
        skip_names = {str(name).lower() for name in skip_names or ()}
        lines = ""
        for attr in attributes:
            attr_name = str(getattr(attr, "name", ""))
            if attr_name.lower() in skip_names:
                continue
            attr_name = self.crossgl_attribute_name(attr_name)
            prefix = "@" if attr_name in self.crossgl_reserved_identifiers else "@ "
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                rendered_args = ", ".join(self.generate_expression(arg) for arg in args)
                lines += "    " * indent + f"{prefix}{attr_name}({rendered_args})\n"
            else:
                lines += "    " * indent + f"{prefix}{attr_name}\n"
        return lines

    def crossgl_attribute_name(self, attr_name):
        """Return a CrossGL-lexer-safe identifier for HLSL/DXC attributes."""
        if str(attr_name) == "vk::push_constant":
            # CrossGL spells a push-constant block '@push_constant' whatever its source language.
            return "push_constant"
        safe_name = re.sub(r"\W+", "_", str(attr_name)).strip("_")
        if not safe_name:
            return "attribute"
        if safe_name[0].isdigit():
            return f"attr_{safe_name}"
        return safe_name

    def format_binding_attributes(self, node, indent):
        lines = ""
        register = getattr(node, "register", None)
        packoffset = getattr(node, "packoffset", None)
        if register:
            parts = [part.strip() for part in str(register).split(",") if part.strip()]
            # A register class without a number (``register(s)``) denotes
            # register 0 of that class, as DXC compiles it.
            parts = [
                f"{part}0" if part.lower() in {"b", "s", "t", "u"} else part
                for part in parts
            ]
            rendered = ", ".join(parts)
            lines += "    " * indent + f"@ register({rendered})\n"
        if packoffset:
            lines += "    " * indent + f"@ packoffset({packoffset})\n"
        return lines

    def matrix_layout_qualifiers(self, node):
        """The CrossGL orientation of a declaration's ``row_major`` or ``column_major`` qualifier.

        With row-major packing CrossGL holds the transpose of every HLSL matrix
        (:meth:`hlsl_matrix_rows_are_columns`), so the stored HLSL rows of a
        ``row_major`` matrix are CrossGL columns, and the reverse.
        """
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        found = [
            qualifier
            for qualifier in ("row_major", "column_major")
            if qualifier in qualifiers
        ]
        if self.hlsl_matrix_rows_are_columns():
            swapped = {"row_major": "column_major", "column_major": "row_major"}
            found = [swapped[qualifier] for qualifier in found]
        return found

    def format_matrix_layout_attributes(self, node, indent):
        return "".join(
            "    " * indent + f"@ {qualifier}\n"
            for qualifier in self.matrix_layout_qualifiers(node)
        )

    def format_inline_matrix_layout_attributes(self, node):
        return " ".join(
            f"@ {qualifier}" for qualifier in self.matrix_layout_qualifiers(node)
        )

    def is_uav_resource_type(self, hlsl_type):
        if not hlsl_type:
            return False
        type_name = str(hlsl_type)
        if "<" in type_name:
            type_name = type_name.split("<", 1)[0]
        return type_name.startswith(
            ("RWTexture", "RWBuffer", "RasterizerOrdered")
        ) or type_name in {
            "RWStructuredBuffer",
            "AppendStructuredBuffer",
            "ConsumeStructuredBuffer",
        }

    def format_resource_qualifier_attributes(self, node, indent):
        if not self.is_uav_resource_type(getattr(node, "vtype", None)):
            return ""

        lines = ""
        if self.is_rasterizer_ordered_resource_type(getattr(node, "vtype", None)):
            lines += "    " * indent + "@ rasterizer_ordered\n"

        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "globallycoherent" in qualifiers:
            lines += "    " * indent + "@ globallycoherent\n"
        if "reordercoherent" in qualifiers:
            lines += "    " * indent + "@ reordercoherent\n"
        return lines

    def format_storage_qualifier_prefix(self, node, allowed_qualifiers):
        qualifiers = []
        allowed = {str(qualifier).lower() for qualifier in allowed_qualifiers}
        for qualifier in getattr(node, "qualifiers", []) or []:
            qualifier_name = str(qualifier).lower()
            if qualifier_name in allowed:
                qualifiers.append(qualifier_name)
        return f"{' '.join(qualifiers)} " if qualifiers else ""

    def should_emit_const_qualifier(self, node):
        # A $Globals member is read-only already and has no initializer for ``const`` to protect.
        return getattr(node, "value", None) is not None and id(node) not in getattr(
            self, "hlsl_program_constant_global_ids", ()
        )

    def format_global_storage_qualifier_prefix(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        ordered = []
        if "shared" in qualifiers:
            ordered.append("shared")
        if "groupshared" in qualifiers:
            ordered.append("groupshared")
        if "static" in qualifiers:
            ordered.append("static")
        # A specialization constant is ``const`` in CrossGL; DXC also accepts one declared without it.
        if (
            "const" in qualifiers or has_vk_constant_id(node)
        ) and self.should_emit_const_qualifier(node):
            ordered.append("const")
        return f"{' '.join(ordered)} " if ordered else ""

    def format_local_storage_qualifier_prefix(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        ordered = []
        if "static" in qualifiers:
            ordered.append("static")
        if "const" in qualifiers and self.should_emit_const_qualifier(node):
            ordered.append("const")
        return f"{' '.join(ordered)} " if ordered else ""

    def format_precise_qualifier_prefix(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        return "precise " if "precise" in qualifiers else ""

    def function_precise_return_needs_attribute(self, func):
        qualifiers = {str(q).lower() for q in getattr(func, "qualifiers", []) or []}
        return "precise" in qualifiers and bool(getattr(func, "attributes", []))

    def format_interpolation_attributes(self, node):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        attributes = []

        if "nointerpolation" in qualifiers:
            attributes.append("flat")
        elif "noperspective" in qualifiers:
            attributes.append("noperspective")
        elif "linear" in qualifiers:
            attributes.append("smooth")

        if "centroid" in qualifiers:
            attributes.append("centroid")
        elif "sample" in qualifiers:
            attributes.append("sample")

        return " ".join(f"@ {attribute}" for attribute in attributes)

    def format_semantic_and_interpolation_attributes(self, node, semantic):
        attributes = []
        mapped_semantic = self.map_semantic(semantic, parameter=node)
        if mapped_semantic:
            attributes.append(mapped_semantic)
        location = vk_location_suffix(
            self, node, semantic, f"'{getattr(node, 'name', '<member>')}'"
        ).strip()
        if location:
            attributes.append(location)
        interpolation_attributes = self.format_interpolation_attributes(node)
        if interpolation_attributes:
            attributes.append(interpolation_attributes)
        return f" {' '.join(attributes)}" if attributes else ""

    def format_inline_parameter_attributes(self, parameter):
        mesh_role_attributes = {"mesh_payload", "vertices", "indices", "primitives"}
        rendered = []
        for attr in getattr(parameter, "attributes", []) or []:
            attr_name = str(getattr(attr, "name", ""))
            if attr_name.lower() not in mesh_role_attributes:
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                rendered_args = ", ".join(self.generate_expression(arg) for arg in args)
                rendered.append(f"@ {attr_name}({rendered_args})")
            else:
                rendered.append(f"@ {attr_name}")
        return " ".join(rendered)

    def format_inline_resource_qualifier_attributes(self, parameter):
        if not self.is_uav_resource_type(getattr(parameter, "vtype", None)):
            return ""

        qualifiers = {
            str(q).lower() for q in getattr(parameter, "qualifiers", []) or []
        }
        rendered = []
        if "globallycoherent" in qualifiers:
            rendered.append("@ globallycoherent")
        if "reordercoherent" in qualifiers:
            rendered.append("@ reordercoherent")
        return " ".join(rendered)

    def format_geometry_primitive_parameter_qualifier(self, parameter):
        for attr in getattr(parameter, "attributes", []) or []:
            if str(getattr(attr, "name", "")).lower() != "primitive":
                continue
            args = getattr(attr, "args", getattr(attr, "arguments", []))
            if args:
                return str(args[0]).lower()
        return ""

    def infer_ray_parameter_semantic(self, function_qualifier, parameter, index):
        if getattr(parameter, "semantic", None):
            return parameter.semantic

        qualifier = str(function_qualifier or "").lower()
        if qualifier == "ray_miss" and index == 0:
            return "payload"
        if qualifier in {"ray_closest_hit", "ray_any_hit"}:
            if (
                str(getattr(parameter, "vtype", ""))
                == "BuiltInTriangleIntersectionAttributes"
            ):
                return "hit_attribute"
            if index == 0:
                return "payload"
            if index == 1:
                return "hit_attribute"
        if qualifier == "ray_callable" and index == 0:
            return "callable_data"
        return None

    def format_parameter(
        self, parameter, function_qualifier=None, parameter_index=None
    ):
        prefixes = []
        attributes = self.format_inline_parameter_attributes(parameter)
        if attributes:
            prefixes.append(attributes)
        work_graph_attributes = crossgl_work_graph_parameter_attributes(self, parameter)
        if work_graph_attributes:
            prefixes.append(work_graph_attributes)
        resource_attributes = self.format_inline_resource_qualifier_attributes(
            parameter
        )
        if resource_attributes:
            prefixes.append(resource_attributes)
        matrix_layout_attributes = self.format_inline_matrix_layout_attributes(
            parameter
        )
        if matrix_layout_attributes:
            prefixes.append(matrix_layout_attributes)
        # A buffer parameter reads the words of the buffer passed to it, so it carries the same layout.
        if self.is_structured_buffer_resource_type(getattr(parameter, "vtype", None)):
            prefixes.append(f"@ {BUFFER_LAYOUT_ATTRIBUTE}({HLSL_STRUCTURED_LAYOUT})")
        qualifier_prefix = self.format_storage_qualifier_prefix(
            parameter, {"in", "out", "inout", "const", "precise"}
        ).strip()
        if function_qualifier is None and self.is_implicit_value_array_parameter(
            parameter
        ):
            qualifier_prefix = f"in {qualifier_prefix}".strip()
        if qualifier_prefix:
            prefixes.append(qualifier_prefix)
        primitive_qualifier = self.format_geometry_primitive_parameter_qualifier(
            parameter
        )
        if primitive_qualifier:
            prefixes.append(primitive_qualifier)
        parameter_name = parameter.name or f"_param{parameter_index}"
        parameter_name = self.render_identifier(parameter_name)
        parameter_type, parameter_array_suffix = self.geometry_patch_parameter_shape(
            parameter, function_qualifier
        )
        if parameter_type is None:
            parameter_type = self.map_variable_type(parameter)
            parameter_array_suffix = self.format_array_suffixes(parameter)
        parameter_text = f"{parameter_type} {parameter_name}{parameter_array_suffix}"
        semantic = self.map_semantic(
            self.infer_ray_parameter_semantic(
                function_qualifier, parameter, parameter_index
            ),
            function_qualifier=function_qualifier,
            parameter=parameter,
        )
        location = vk_location_suffix(
            self,
            parameter,
            getattr(parameter, "semantic", None),
            f"parameter '{parameter_name}'",
        ).strip()
        interpolation_attributes = self.format_interpolation_attributes(parameter)
        if semantic or location or interpolation_attributes:
            parameter_text += " " + " ".join(
                attribute
                for attribute in (semantic, location, interpolation_attributes)
                if attribute
            )
        prefixes.append(parameter_text)
        return " ".join(prefixes)

    def geometry_patch_parameter_shape(self, parameter, function_qualifier):
        """Lower HLSL geometry InputPatch<T, N> parameters to CrossGL arrays."""
        if function_qualifier != "geometry":
            return None, ""

        type_name = str(getattr(parameter, "vtype", "")).strip()
        if self.raw_type_base(type_name) != "InputPatch":
            return None, ""
        if "<" not in type_name or not type_name.endswith(">"):
            return None, ""

        generic_type = type_name.split("<", 1)[1][:-1].strip()
        args = self.split_generic_arguments(generic_type)
        if len(args) != 2:
            return None, ""

        element_type = self.map_type(args[0])
        patch_count = self.parse_template_dimension(args[1])
        if patch_count is None:
            return None, ""
        return element_type, f"[{patch_count}]"

    def record_variable_type(self, node, type_map=None, array_dim_map=None):
        name = getattr(node, "name", None)
        if not name:
            return
        if type_map is None:
            type_map = self.current_variable_types
        if array_dim_map is None:
            array_dim_map = self.current_resource_array_dims
        type_map[name] = getattr(node, "vtype", None)
        array_dim_map[name] = len(getattr(node, "array_sizes", None) or [])

    def expression_base_name(self, expr):
        if isinstance(expr, str):
            return expr
        if isinstance(expr, VariableNode):
            return expr.name
        if isinstance(expr, ArrayAccessNode):
            return self.expression_base_name(expr.array)
        if isinstance(expr, MemberAccessNode):
            return self.expression_base_name(expr.object)
        return None

    def collect_struct_member_types(self, structs):
        member_types = {}
        for struct in structs or []:
            if not isinstance(struct, StructNode):
                continue
            if getattr(struct, "is_forward_declaration", False):
                continue
            member_types[struct.name] = {
                member.name: getattr(member, "vtype", None)
                for member in getattr(struct, "members", []) or []
            }
        return member_types

    def collect_struct_member_array_dims(self, structs):
        member_array_dims = {}
        for struct in structs or []:
            if not isinstance(struct, StructNode):
                continue
            if getattr(struct, "is_forward_declaration", False):
                continue
            member_array_dims[struct.name] = {
                member.name: len(getattr(member, "array_sizes", None) or [])
                for member in getattr(struct, "members", []) or []
            }
        return member_array_dims
