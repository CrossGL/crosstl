"""Program-wide analysis for the Rust generator: user functions, statics, entry resources, enums, derivable traits and captures."""

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayNode,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    DoWhileNode,
    EnumNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    MatchNode,
    MemberAccessNode,
    ParameterNode,
    ReturnNode,
    StructNode,
    TernaryOpNode,
    TextureNode,
    TextureOpNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from .array_utils import prepend_array_dimension
from .rust_gpu_constant_blocks import CONSTANT_BLOCK_FLAG
from .rust_gpu_images import rust_gpu_image_parameter_type
from .rust_gpu_interface import (
    STORAGE_SLICE_KIND,
    UNIFORM_VALUE_KIND,
    WORKGROUP_KIND,
    builtin_input_infos,
    declared_builtin_input_info,
    is_uniform_value,
    is_workgroup_array,
    slice_type_name,
    storage_slice_parts,
    workgroup_type_name,
)
from .rust_gpu_layouts import STD140, STD430, layout_mismatch
from .rust_gpu_storage import PADDED_ELEMENTS
from .rust_resource_bindings import (
    PROMOTABLE_RESOURCE_KINDS,
    resource_parameters,
)
from .specialization_constants import declares_specialization_id
from .stage_utils import normalize_stage_name


class RustProgramAnalysisMixin:
    """Program-analysis members of ``RustCodeGen``."""

    def collect_user_function_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                names.add(current.name)
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        names.discard(None)
        return names

    def collect_user_function_nodes(self, node):
        functions = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode) and current.name:
                functions[current.name] = current
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return functions

    def collect_static_variable_names(self, node):
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            for variable in getattr(current, "global_variables", []):
                name = getattr(variable, "name", None)
                if name:
                    names.add(name)
            for variable in getattr(current, "local_variables", []):
                if self.is_rust_shared_variable(variable):
                    continue
                name = getattr(variable, "name", None)
                if name:
                    names.add(name)
            for constant in getattr(current, "constants", []):
                name = getattr(constant, "name", None)
                if name:
                    names.add(name)
            for cbuffer in self.get_cbuffer_nodes(current):
                for member in getattr(cbuffer, "members", []) or []:
                    name = getattr(member, "name", None)
                    if name:
                        names.add(name)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return names

    def build_static_symbol_names(self, names):
        """Map source static names to Rust-style uppercase static symbols."""
        symbol_names = {}
        used_symbols = set()
        for name in sorted(names):
            symbol = self.uppercase_static_symbol_name(name)
            base_symbol = symbol
            suffix = 2
            while symbol in used_symbols:
                symbol = f"{base_symbol}_{suffix}"
                suffix += 1
            symbol_names[name] = symbol
            used_symbols.add(symbol)
        return symbol_names

    def rust_variable_source_type(self, node):
        """Return the source-level type string used for Rust resource metadata."""
        if isinstance(node, ArrayNode):
            element_type_name = self.convert_type_node_to_string(
                getattr(node, "element_type", "float")
            )
            element_type_name = self.resource_type_with_attributes(
                element_type_name,
                node,
            )
            size = self.format_array_size(getattr(node, "size", None))
            return prepend_array_dimension(element_type_name, size)
        return self.get_variable_type(node) or "float"

    def rust_function_stage_name(self, func):
        """Return the normalized shader stage name for a function, if any."""
        qualifiers = list(getattr(func, "qualifiers", []) or [])
        qualifier = getattr(func, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)

        stage_names = {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            *self.rust_mesh_stage_names(),
            *self.rust_ray_stage_names(),
        }
        for qualifier in qualifiers:
            stage_name = normalize_stage_name(qualifier)
            if stage_name in stage_names:
                return stage_name
        return None

    def rust_global_entry_resource_infos(self, ast):
        """Collect global and stage-local resources that can be promoted to entry parameters.

        Stage-local variables share one module-level namespace with the globals (a name is declared once), so a
        stage-local resource whose name a global or an earlier stage already declares is the same resource.
        """
        resources = {}
        declared = set()
        candidates = list(getattr(ast, "global_variables", []) or [])
        stages = getattr(ast, "stages", None)
        if isinstance(stages, dict):
            for stage in stages.values():
                candidates.extend(getattr(stage, "local_variables", []) or [])
        for node in candidates:
            name = getattr(node, "name", None)
            if not name or name in declared:
                continue
            declared.add(name)
            type_name = self.rust_variable_source_type(node)
            builtin = declared_builtin_input_info(node)
            if builtin is not None and self.map_type(type_name) == self.map_type(
                builtin["type_name"]
            ):
                resources[name] = builtin
                continue
            interface = self.rust_gpu_interface_info(node, type_name)
            if interface is not None:
                resources[name] = interface
                continue
            if not self.can_promote_rust_entry_resource(node, type_name):
                continue
            resources[name] = {"node": node, "type_name": type_name}
        for name, info in builtin_input_infos(ast).items():
            if name not in declared:
                resources[name] = info
        for name, info in self.rust_spec_constant_infos.items():
            resources[name] = info
        return resources

    def rust_gpu_interface_info(self, node, type_name):
        """The entry-parameter record of a storage buffer pointer, uniform value or workgroup variable, else None."""
        slice_parts = storage_slice_parts(node)
        if slice_parts is not None:
            pointee, writable = slice_parts
            element = self.convert_type_node_to_string(pointee)
            if element not in PADDED_ELEMENTS:
                self.refuse_rust_gpu_layout(
                    node, element, STD430, element_of_slice=True
                )
            return {
                "node": node,
                "type_name": slice_type_name(element, writable),
                "kind": STORAGE_SLICE_KIND,
            }
        if is_workgroup_array(node):
            return {
                "node": node,
                "type_name": workgroup_type_name(self.type_name_string(type_name)),
                "kind": WORKGROUP_KIND,
            }
        if is_uniform_value(node, self.type_name_string(type_name)):
            # A constant block's struct already places every member at its block offset.
            if not getattr(node, CONSTANT_BLOCK_FLAG, None):
                self.refuse_rust_gpu_layout(
                    node,
                    self.type_name_string(type_name),
                    STD140,
                    element_of_slice=False,
                )
            return {"node": node, "type_name": type_name, "kind": UNIFORM_VALUE_KIND}
        return None

    def refuse_rust_gpu_layout(self, node, type_name, rule, element_of_slice):
        """Refuse a buffer binding whose rust-gpu layout is not the CrossGL ``rule`` layout (:mod:`.rust_gpu_layouts`)."""
        reason = layout_mismatch(self, type_name, rule, element_of_slice)
        if reason is not None and getattr(node, "name", None) in getattr(
            self, "rust_referenced_names", ()
        ):
            kind = "storage buffer" if element_of_slice else "uniform"
            self.refuse_lowering(
                f"resource: {kind} '{getattr(node, 'name', '?')}': {reason}"
            )

    def can_promote_rust_entry_resource(self, node, type_name):
        """Return whether a resource can be an entry parameter, with its explicit or assigned binding.

        A fixed-size resource array is one parameter of array type bound to ``count`` consecutive bindings.
        """
        if getattr(node, "initial_value", getattr(node, "value", None)) is not None:
            return False

        resource_kind = self.rust_resource_kind(type_name, node=node)
        if resource_kind not in PROMOTABLE_RESOURCE_KINDS:
            return False

        _base_type, count = self.rust_resource_base_type_and_count(type_name)
        if self.is_array_type_name(self.type_name_string(type_name)):
            _base, sizes = self.c_array_type_parts(self.type_name_string(type_name))
        return isinstance(count, int) and count >= 1

    def collect_rust_entry_resource_params(self, ast):
        """Map stage entry functions to their Rust resource parameters, and helpers to the resources they take.

        The helper parameters go to :attr:`rust_resource_capture_params`, keyed by helper name.
        """
        self.rust_resource_capture_params = {}
        resources = self.rust_global_entry_resource_infos(ast)
        self.rust_entry_resource_infos = resources
        if not resources:
            return {}

        stage_entry_ids = set()
        for stage in (
            getattr(ast, "stages", {}).values() if hasattr(ast, "stages") else []
        ):
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is not None:
                stage_entry_ids.add(id(entry_point))

        entry_functions = {}
        helpers = []
        seen_functions = set()

        def visit(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    visit(item)
                return
            if isinstance(current, FunctionNode):
                function_id = id(current)
                if function_id in seen_functions:
                    return
                seen_functions.add(function_id)
                if (
                    function_id in stage_entry_ids
                    or self.rust_function_stage_name(current) is not None
                ):
                    entry_functions[function_id] = current
                elif not self.is_function_prototype(current):
                    helpers.append(current)
                return

            for function in getattr(current, "functions", []) or []:
                visit(function)
            for function in getattr(current, "local_functions", []) or []:
                visit(function)
            visit(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    visit(stage)

        visit(ast)
        params_by_function, self.rust_resource_capture_params = resource_parameters(
            entry_functions, helpers, resources
        )
        return params_by_function

    def is_rust_gpu_interface_declaration(self, node):
        """Whether ``node`` is a buffer pointer, uniform value, workgroup variable or specialization constant, which
        only entry-point parameters declare: a module-level declaration of one would be a mutable or defaulted static.
        """
        if getattr(node, "name", None) is None:
            return False
        if node.name in self.rust_spec_constant_infos and declares_specialization_id(
            node
        ):
            return True
        type_name = self.rust_variable_source_type(node)
        if self.rust_gpu_interface_info(node, type_name) is None and (
            rust_gpu_image_parameter_type(self, type_name) is None
        ):
            return False
        if not self.is_promoted_rust_entry_resource(node) and (
            node.name in self.rust_referenced_names
        ):
            self.refuse_lowering(
                f"resource: '{node.name}' is referenced where no rust-gpu entry-point parameter reaches"
            )
        return True

    def is_promoted_rust_entry_resource(self, node):
        """Return whether a global resource is emitted as an entry parameter."""
        return id(node) in self.rust_promoted_resource_node_ids

    def uppercase_static_symbol_name(self, name):
        """Convert a CrossGL identifier to a Rust static identifier."""
        name = str(name)
        words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        words = re.sub(r"[^0-9A-Za-z_]+", "_", words)
        words = re.sub(r"_+", "_", words).strip("_")
        symbol = words.upper() or "STATIC"
        if symbol[0].isdigit():
            symbol = f"STATIC_{symbol}"
        return symbol

    def collect_runtime_type_collisions(self, node):
        """Find user declarations that shadow Rust math prelude type names."""
        names = set()

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, (StructNode, EnumNode)):
                name = getattr(current, "name", None)
                if name in self.runtime_math_type_names():
                    names.add(name)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return names

    def runtime_math_type_names(self):
        return {
            "Vec2",
            "Vec3",
            "Vec4",
            "Mat2",
            "Mat3",
            "Mat4",
            "Mat2x3",
            "Mat2x4",
            "Mat3x2",
            "Mat3x4",
            "Mat4x2",
            "Mat4x3",
        }

    def collect_trait_methods(self, node):
        traits = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, StructNode) and getattr(current, "is_trait", False):
                traits[current.name] = {
                    method.name
                    for method in getattr(current, "members", []) or []
                    if isinstance(method, FunctionNode)
                }
            for struct in getattr(current, "structs", []):
                collect(struct)
            for function in getattr(current, "functions", []):
                collect(function)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return traits

    def collect_enum_variant_names(self, node):
        """Collect enum variant names keyed by the Rust enum type name."""
        variants = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variants[name] = [
                variant.name
                for variant in getattr(enum_node, "variants", []) or []
                if getattr(variant, "name", None)
            ]

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return variants

    def collect_enum_nodes_by_name(self, node):
        """Map each enumeration name to its AST node for empty brace initializers."""
        enums = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            enums[name] = enum_node

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return enums

    def collect_enum_variant_field_types(self, node):
        """Collect named payload fields for struct-like enum variants."""
        field_types = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variant_fields = {}
            for variant in getattr(enum_node, "variants", []) or []:
                data = self.enum_variant_data(variant)
                if not data or not all(
                    isinstance(item, tuple) and len(item) == 2 for item in data
                ):
                    continue
                variant_fields[variant.name] = {
                    field_name: self.convert_type_node_to_string(field_type)
                    for field_name, field_type in data
                }
            if variant_fields:
                field_types[name] = variant_fields

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return field_types

    def collect_enum_variant_payload_type_map(self, node):
        """Collect positional payload types for enum variant constructor calls."""
        payload_types = {}

        def add_enum(name, enum_node):
            if not name or enum_node is None:
                return
            variant_payloads = {}
            for variant in getattr(enum_node, "variants", []) or []:
                variant_payloads[variant.name] = [
                    self.convert_type_node_to_string(field_type)
                    for field_type in self.enum_variant_payload_types(variant)
                ]
            payload_types[name] = variant_payloads

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, wrapper_enum)
                    add_enum(wrapper_enum.name, wrapper_enum)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return payload_types

    def collect_enum_generic_params(self, node):
        """Collect generic parameter names for enums and enum-wrapper structs."""
        generic_params = {}

        def add_enum(name, generic_owner):
            params = self.generic_param_names(generic_owner)
            if name and params:
                generic_params[name] = params

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, EnumNode):
                add_enum(current.name, current)
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    add_enum(current.name, current)
                    add_enum(wrapper_enum.name, current)
                for member in getattr(current, "members", []) or []:
                    if isinstance(member, EnumNode):
                        collect(member)
            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return generic_params

    def is_user_defined_function(self, func_name):
        return isinstance(func_name, str) and func_name in self.user_function_names

    def collect_user_function_return_types(self, node):
        return_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode):
                return_type = getattr(current, "return_type", None)
                if current.name and return_type is not None:
                    return_types[current.name] = self.convert_type_node_to_string(
                        return_type
                    )
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return return_types

    def collect_user_function_param_types(self, node):
        param_types = {}

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return
            if isinstance(current, FunctionNode) and current.name:
                params = getattr(current, "parameters", getattr(current, "params", []))
                param_types[current.name] = [
                    self.function_parameter_type(param) for param in params
                ]
            for function in getattr(current, "functions", []):
                collect(function)
            for function in getattr(current, "local_functions", []):
                collect(function)
            collect(getattr(current, "entry_point", None))
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        return param_types

    def collect_stage_local_function_captures(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        entry_params = getattr(
            entry_point, "parameters", getattr(entry_point, "params", [])
        )
        entry_param_types = {
            param.name: self.function_parameter_type(param) for param in entry_params
        }
        if not entry_param_types:
            return {}

        captures = {}
        for function in getattr(stage, "local_functions", []) or []:
            if self.is_function_prototype(function):
                continue
            used_names = self.collect_free_identifier_names(function)
            captured_params = [
                ParameterNode(name, entry_param_types[name])
                for name in entry_param_types
                if name in used_names
            ]
            if captured_params:
                captures[function.name] = captured_params
        return captures

    def collect_free_identifier_names(self, function):
        bound_names = {
            param.name
            for param in getattr(
                function, "parameters", getattr(function, "params", [])
            )
        }
        used_names = set()

        def visit(node, bound):
            if node is None:
                return
            if isinstance(node, list):
                for item in node:
                    visit(item, bound)
                return
            if isinstance(node, IdentifierNode):
                if node.name not in bound:
                    used_names.add(node.name)
                return
            if isinstance(node, VariableNode):
                visit(getattr(node, "initial_value", None), bound)
                bound.add(node.name)
                return
            if isinstance(node, ArrayNode):
                visit(
                    getattr(node, "initial_value", getattr(node, "value", None)), bound
                )
                bound.add(node.name)
                return
            if isinstance(node, FunctionCallNode):
                function_expr = getattr(node, "function", getattr(node, "name", None))
                if isinstance(function_expr, MemberAccessNode):
                    visit(function_expr, bound)
                for arg in getattr(node, "arguments", getattr(node, "args", [])) or []:
                    visit(arg, bound)
                return
            if isinstance(node, TextureNode):
                visit(getattr(node, "texture_expr", None), bound)
                visit(getattr(node, "sampler_expr", None), bound)
                visit(getattr(node, "coordinates", None), bound)
                visit(getattr(node, "level", None), bound)
                visit(getattr(node, "offset", None), bound)
                return
            if isinstance(node, TextureOpNode):
                visit(getattr(node, "texture_expr", None), bound)
                visit(getattr(node, "sampler_expr", None), bound)
                visit(getattr(node, "arguments", []), bound)
                return
            if isinstance(node, MemberAccessNode):
                visit(
                    getattr(node, "object_expr", getattr(node, "object", None)), bound
                )
                return
            if isinstance(node, AssignmentNode):
                visit(getattr(node, "target", getattr(node, "left", None)), bound)
                visit(getattr(node, "value", getattr(node, "right", None)), bound)
                return
            if isinstance(
                node, (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode, IfNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)
                return
            if isinstance(node, ReturnNode):
                visit(getattr(node, "value", None), bound)
                return
            if isinstance(node, ExpressionStatementNode):
                visit(getattr(node, "expression", None), bound)
                return
            if isinstance(
                node, (BinaryOpNode, TernaryOpNode, UnaryOpNode, ArrayAccessNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)
                return
            if isinstance(
                node, (ConstructorNode, ArrayLiteralNode, BlockNode, MatchNode)
            ):
                for value in getattr(node, "__dict__", {}).values():
                    visit(value, bound)

        visit(getattr(function, "body", None), set(bound_names))
        return used_names

    def derive_traits_for_members(self, members, include_default=False):
        traits = ["Debug", "Clone"]
        if self.members_are_copy_derivable(members):
            traits.append("Copy")
        if include_default and self.members_default_derive_supported(members):
            traits.append("Default")
        return ", ".join(traits)

    def members_default_derive_supported(self, members):
        return all(
            self.member_default_derive_supported(member)
            for member in members
            if not isinstance(member, EnumNode)
        )

    def member_default_derive_supported(self, member):
        member_type = self.member_type_for_copy_check(member)
        if member_type is None:
            return True
        return self.rust_type_default_derive_supported(self.map_type(member_type))

    def rust_type_default_derive_supported(self, rust_type):
        array_type = self.parse_rust_array_type(str(rust_type))
        if array_type is None:
            return True
        element_type, size = array_type
        if size is not None:
            try:
                if int(size) > 32:
                    return False
            except ValueError:
                return False
        return self.rust_type_default_derive_supported(element_type)

    def derive_traits_for_enum(self, node, generic_owner=None, include_default=False):
        traits = ["Debug", "Clone"]
        generic_names = set(self.generic_param_names(generic_owner or node))
        if self.enum_payloads_are_copy_derivable(node, generic_names):
            traits.append("Copy")
        if include_default:
            traits.append("Default")
        if not any(
            self.enum_variant_payload_types(variant)
            for variant in getattr(node, "variants", []) or []
        ):
            # CrossGL compares enum values with == and switches on them.
            traits.extend(["PartialEq", "Eq"])
        return ", ".join(traits)

    def collect_non_copy_type_names(self, node):
        declarations = []

        def collect(current):
            if current is None:
                return
            if isinstance(current, list):
                for item in current:
                    collect(item)
                return

            if isinstance(current, EnumNode):
                declarations.append((current.name, "enum", current, current))
            elif isinstance(current, StructNode):
                wrapper_enum = self.struct_enum_wrapper(current)
                if wrapper_enum is not None:
                    declarations.append((current.name, "enum", wrapper_enum, current))
                    declarations.append(
                        (wrapper_enum.name, "enum", wrapper_enum, current)
                    )
                else:
                    declarations.append((current.name, "struct", current, current))
                    for member in getattr(current, "members", []) or []:
                        if isinstance(member, EnumNode):
                            collect(member)

            for struct in getattr(current, "structs", []):
                collect(struct)
            for struct in getattr(current, "local_structs", []):
                collect(struct)
            stages = getattr(current, "stages", {})
            if isinstance(stages, dict):
                for stage in stages.values():
                    collect(stage)

        collect(node)
        for cbuffer in self.get_cbuffer_nodes(node):
            collect(cbuffer)

        non_copy_type_names = set()
        changed = True
        while changed:
            changed = False
            for name, declaration_kind, declaration, generic_owner in declarations:
                if not name or name in non_copy_type_names:
                    continue
                generic_names = set(self.generic_param_names(generic_owner))
                if declaration_kind == "enum":
                    copy_derivable = self.enum_payloads_are_copy_derivable(
                        declaration,
                        generic_names,
                        non_copy_type_names,
                    )
                else:
                    copy_derivable = self.members_are_copy_derivable(
                        getattr(declaration, "members", []),
                        generic_names,
                        non_copy_type_names,
                    )
                if not copy_derivable:
                    non_copy_type_names.add(name)
                    changed = True

        return non_copy_type_names

    def members_are_copy_derivable(
        self,
        members,
        generic_names=None,
        non_copy_type_names=None,
    ):
        generic_names = generic_names or set()
        for member in members:
            if isinstance(member, EnumNode):
                continue
            member_type = self.member_type_for_copy_check(member)
            if member_type is None:
                continue
            if not self.type_is_copy_derivable(
                member_type,
                generic_names,
                non_copy_type_names,
            ):
                return False
        return True

    def member_type_for_copy_check(self, member):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", None)
            )
            element_type = self.convert_type_node_to_string(element_type)
            size = self.format_array_size(getattr(member, "size", None))
            return prepend_array_dimension(element_type, size)

        if not hasattr(member, "member_type") and not hasattr(member, "vtype"):
            return None
        return self.get_member_type(member)

    def enum_payloads_are_copy_derivable(
        self,
        node,
        generic_names,
        non_copy_type_names=None,
    ):
        for variant in getattr(node, "variants", []) or []:
            for field_type in self.enum_variant_payload_types(variant):
                if not self.type_is_copy_derivable(
                    field_type,
                    generic_names,
                    non_copy_type_names,
                ):
                    return False
        return True

    def enum_variant_payload_types(self, variant):
        field_types = []
        for item in self.enum_variant_data(variant):
            if isinstance(item, tuple) and len(item) == 2:
                field_types.append(item[1])
            else:
                field_types.append(item)
        return field_types

    def type_is_copy_derivable(
        self,
        type_name,
        generic_names=None,
        non_copy_type_names=None,
    ):
        generic_names = generic_names or set()
        non_copy_type_names = non_copy_type_names or self.non_copy_type_names
        type_name = self.convert_type_node_to_string(type_name)

        if type_name in generic_names:
            return True

        if self.is_array_type_name(type_name):
            base_type, sizes = self.c_array_type_parts(type_name)
            return all(
                size is not None for size in sizes
            ) and self.type_is_copy_derivable(
                base_type,
                generic_names,
                non_copy_type_names,
            )

        rust_type = self.unqualify_runtime_type_name(self.map_type(type_name))
        if rust_type.startswith("Vec<"):
            return False

        base_type, generic_args = self.generic_type_parts(type_name)
        if base_type in non_copy_type_names:
            return False
        if generic_args:
            return all(
                self.type_is_copy_derivable(
                    generic_arg,
                    generic_names,
                    non_copy_type_names,
                )
                for generic_arg in generic_args
            )

        return True

    def members_have_unsized_arrays(self, members):
        return any(self.member_has_unsized_array(member) for member in members)

    def member_has_unsized_array(self, member):
        if isinstance(member, ArrayNode):
            return getattr(member, "size", None) is None

        member_type = getattr(member, "member_type", None)
        if member_type is not None and member_type.__class__.__name__ == "ArrayType":
            return getattr(member_type, "size", None) is None

        vtype = getattr(member, "vtype", None)
        return isinstance(vtype, str) and vtype.endswith("[]")
