"""Metal emission of HLSL resource semantics.

The members here give ``MetalCodeGen`` the HLSL resource model that Metal needs
spelled out: binding records for each HLSL register, HLSL buffer layouts
(packed constant buffers and structured buffers), register-class binding
assignment, fences for image stores the same thread reads back, explicit 1D
texture levels of detail and strong atomic compare-exchange on pointers.
"""

import re

from ..ast import (
    ArrayAccessNode,
    BinaryOpNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerAccessNode,
    PointerType,
    PrimitiveType,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from .buffer_member_layout import member_explicit_offset, member_is_row_major
from .glsl_block_layout import (
    GLSL_BLOCK_LAYOUTS,
    declared_buffer_layout,
    member_explicit_alignment,
)
from .hlsl_buffer_layout import (
    HLSL_CBUFFER_LAYOUT,
    HLSL_STRUCTURED_LAYOUT,
    HlslLayoutError,
    buffer_layout_from_attributes,
)
from .metal_hlsl_buffer_layout import LAYOUT_LABELS, MetalHlslBufferLayouts


def metal_layout_refusal(message, layout):
    """The typed diagnostic for a buffer layout Metal declarations cannot reproduce."""
    from .metal_codegen import UnsupportedMetalFeatureError

    return UnsupportedMetalFeatureError(
        "buffer layout",
        message,
        missing_capabilities=(f"metal.buffer_layout.{layout}",),
    )


class MetalHlslResourceMixin:
    """HLSL resource, binding and buffer-layout members of ``MetalCodeGen``."""

    def resource_binding_records(self):
        """Describe the argument-table slot assigned to each resource.

        Valid after ``generate``. Each record names the source resource, its Metal
        namespace (``buffer``, ``texture`` or ``sampler``) and slot, the slot count
        for arrays, and the HLSL register that bound it when the source used one.
        Hidden buffers synthesized for a resource (structured-buffer lengths and
        counters) carry ``of`` with the owning resource name.
        """

        def register_of(node):
            return self.metal_hlsl_register_bindings_by_id.get(id(node))

        def count_of(node):
            try:
                return max(int(self.global_resource_shape(node)[1] or 1), 1)
            except Exception:  # noqa: BLE001 - shape inference is best effort here
                return 1

        records = []
        for cbuffer in self.cbuffer_variables:
            records.append(
                {
                    "name": getattr(cbuffer, "name", None),
                    "parameter": self.cbuffer_parameter_name(cbuffer),
                    "kind": "constant-buffer",
                    "namespace": "buffer",
                    "binding": self.cbuffer_binding_indices.get(id(cbuffer), 0),
                    "count": 1,
                    "register": register_of(cbuffer),
                }
            )
        groups = (
            ("buffer", "buffer", self.metal_buffer_resource_variables),
            ("texture", "texture", self.texture_variables),
            ("acceleration-structure", "buffer", self.acceleration_structure_variables),
            ("visible-function-table", "buffer", self.visible_function_table_variables),
            (
                "intersection-function-table",
                "buffer",
                self.intersection_function_table_variables,
            ),
            ("structured-buffer", "buffer", self.structured_buffer_variables),
            ("buffer-block", "buffer", self.glsl_buffer_block_variables),
        )
        for kind, namespace, entries in groups:
            for entry in entries:
                node, binding = entry[0], entry[1]
                record = {
                    "name": getattr(node, "name", None),
                    "kind": kind,
                    "namespace": namespace,
                    "binding": binding,
                    "count": count_of(node),
                    "register": register_of(node),
                }
                if (
                    kind in {"texture", "structured-buffer", "buffer"}
                    and len(entry) > 2
                ):
                    record["type"] = str(entry[2])
                records.append(record)
        for kind, entries in (
            ("structured-buffer-length", self.structured_buffer_length_variables),
            ("structured-buffer-counter", self.structured_buffer_counter_variables),
        ):
            for node, binding, _type, _array_size in entries:
                records.append(
                    {
                        "name": None,
                        "of": getattr(node, "name", None),
                        "kind": kind,
                        "namespace": "buffer",
                        "binding": binding,
                        "count": count_of(node),
                        "register": None,
                    }
                )
        for node, binding, _array_size in self.sampler_variables:
            records.append(
                {
                    "name": getattr(node, "name", None),
                    "kind": "sampler",
                    "namespace": "sampler",
                    "binding": binding,
                    "count": count_of(node),
                    "register": register_of(node),
                }
            )
        return records

    def metal_hlsl_layout_type_resolver(self, raw_type):
        """Resolve a CrossGL member type for the HLSL buffer layout planner."""
        name = str(raw_type).strip()
        struct = self.structs_by_name.get(name)
        if struct is not None:
            static_members = self.metal_static_struct_members.get(name, set())
            members = [
                (
                    member.name,
                    self.struct_member_raw_type(member),
                    member_is_row_major(member),
                )
                for member in getattr(struct, "members", []) or []
                if getattr(member, "name", None)
                and not isinstance(member, FunctionNode)
                and member.name not in static_members
            ]
            return ("struct", name, members)
        return self.map_type(name)

    def collect_metal_hlsl_buffer_layouts(self, ast):
        """Plan Metal storage that reproduces the HLSL buffer layouts of a program.

        Constant buffers marked ``@ buffer_layout(hlsl_cbuffer)`` or, for GLSL
        uniform and push-constant blocks, ``@ buffer_layout(std140)``,
        ``std430`` or ``scalar``, and the element types of
        structured buffers marked ``@ buffer_layout(hlsl_structured)`` get Metal
        declarations with the exact member offsets of those layouts.
        """
        self.metal_hlsl_cbuffer_plans = {}
        self.metal_hlsl_struct_plans = {}
        self.metal_hlsl_structured_buffers = {}
        self.metal_hlsl_pointer_parameters = {}
        self.metal_hlsl_cbuffer_array_members = {}
        planner = MetalHlslBufferLayouts(
            self.metal_hlsl_layout_type_resolver,
            int_constants=getattr(self, "literal_int_constants", None),
        )
        self.metal_hlsl_layout_planner = planner
        for cbuffer in self.cbuffer_variables or []:
            layout = declared_buffer_layout(getattr(cbuffer, "attributes", None))
            if layout != HLSL_CBUFFER_LAYOUT and layout not in GLSL_BLOCK_LAYOUTS:
                continue
            try:
                members = [
                    (
                        member.name,
                        self.struct_member_raw_type(member),
                        member_is_row_major(member),
                        member_explicit_offset(member),
                        member_explicit_alignment(member),
                    )
                    for member in getattr(cbuffer, "members", []) or []
                    if getattr(member, "name", None)
                ]
                plan = planner.plan_cbuffer(cbuffer.name, members, layout)
            except HlslLayoutError as exc:
                raise metal_layout_refusal(
                    f"Cannot reproduce the {LAYOUT_LABELS[layout]} layout of cbuffer "
                    f"'{cbuffer.name}' in Metal: {exc}",
                    layout,
                ) from exc
            self.metal_hlsl_cbuffer_plans[id(cbuffer)] = plan
            for member in plan.members:
                if member.element_load or (
                    member.array_count is not None
                    and (member.element_suffix or member.read_conversion)
                ):
                    self.metal_hlsl_cbuffer_array_members[member.name] = member
        for node in self.metal_hlsl_structured_declarations(ast):
            layout = buffer_layout_from_attributes(getattr(node, "attributes", None))
            if layout != HLSL_STRUCTURED_LAYOUT:
                continue
            vtype = getattr(
                node,
                "var_type",
                getattr(node, "vtype", getattr(node, "param_type", None)),
            )
            if isinstance(vtype, PointerType):
                # A helper's buffer pointer: its pointee gets the storage of the buffers passed to it.
                self.metal_hlsl_pointer_parameters[id(node)] = (
                    self.metal_hlsl_pointer_storage(planner, node, vtype)
                )
                continue
            type_name = str(self.resource_base_type(vtype))
            if "<" not in type_name or not type_name.endswith(">"):
                continue
            element_type = type_name.split("<", 1)[1][:-1].strip()
            try:
                storage = planner.plan_structured_element(element_type)
            except HlslLayoutError as exc:
                raise metal_layout_refusal(
                    f"Cannot reproduce the HLSL layout of structured buffer "
                    f"'{node.name}' in Metal: {exc}",
                    HLSL_STRUCTURED_LAYOUT,
                ) from exc
            previous = self.metal_hlsl_structured_buffers.get(node.name)
            if previous is not None and previous != storage:
                raise ValueError(
                    f"Cannot reproduce the HLSL layouts of the structured buffers named "
                    f"'{node.name}' in Metal: they hold different element types"
                )
            self.metal_hlsl_structured_buffers[node.name] = storage
        self.metal_hlsl_struct_plans = dict(planner.struct_plans)

    def metal_hlsl_pointer_storage(self, planner, node, pointer_type):
        """The Metal storage of the pointee of a pointer parameter marked ``@buffer_layout(hlsl_structured)``."""
        element_type = self.convert_type_node_to_string(pointer_type.pointee_type)
        try:
            return planner.plan_structured_element(str(element_type))
        except HlslLayoutError as exc:
            raise metal_layout_refusal(
                f"Cannot reproduce the HLSL layout of pointer parameter "
                f"'{node.name}' in Metal: {exc}",
                HLSL_STRUCTURED_LAYOUT,
            ) from exc

    @staticmethod
    def metal_hlsl_structured_declarations(ast):
        """Global structured buffers, then the parameters of every function, in program order."""
        yield from getattr(ast, "global_variables", []) or []
        functions = list(getattr(ast, "functions", []) or [])
        stages = getattr(ast, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            functions.extend(getattr(stage, "local_functions", None) or [])
            if getattr(stage, "entry_point", None) is not None:
                functions.append(stage.entry_point)
        for function in functions:
            yield from getattr(function, "parameters", None) or []

    def metal_hlsl_layout_struct_code(self, plan):
        member_names = {member.name for member in plan.members}

        def padding_name(index):
            candidate = f"_pad{index}"
            while candidate in member_names:
                candidate = f"_{candidate}"
            return candidate

        lines = plan.declaration_lines(padding_name)
        body = "".join(f"    {line}\n" for line in lines)
        storage = self.metal_hlsl_layout_planner.storage_definitions(plan)
        return f"{storage}struct {plan.name} {{\n{body}}};\n"

    def metal_hlsl_layout_member(self, expr):
        """The planned storage of a struct member access, if its struct has one."""
        plans = getattr(self, "metal_hlsl_struct_plans", None)
        if not plans or not isinstance(expr, MemberAccessNode):
            return None
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        object_type = self.expression_result_type(object_expr)
        if object_type is None:
            return None
        struct_name = self.type_name_string(self.member_lookup_type_name(object_type))
        plan = plans.get(struct_name)
        if plan is None:
            return None
        return plan.member(str(getattr(expr, "member", "")))

    def metal_hlsl_structured_buffer_storage(self, array_expr):
        storages = getattr(self, "metal_hlsl_structured_buffers", None)
        if not storages:
            return None
        name = self.expression_name(array_expr)
        if not isinstance(name, str) or name not in storages:
            return None
        local_type = self.local_variable_types.get(name)
        if local_type is not None and not self.is_structured_buffer_type(
            self.type_name_string(local_type)
        ):
            return None
        return storages[name]

    def metal_hlsl_layout_element_access(self, expr):
        """``(access, read_conversion)`` for an element read with HLSL storage."""
        array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        index_expr = getattr(expr, "index", getattr(expr, "index_expr", None))
        name = self.expression_name(array_expr)
        cbuffer_arrays = getattr(self, "metal_hlsl_cbuffer_array_members", {}) or {}
        if (
            isinstance(array_expr, IdentifierNode)
            and isinstance(name, str)
            and name in cbuffer_arrays
            and name not in self.local_variable_types
            and name in self.cbuffer_member_references
        ):
            member = cbuffer_arrays[name]
            base = self.cbuffer_member_references[name]
            index = self.generate_expression(index_expr)
            if member.element_load:
                return f"{member.element_load}({base}, {index})", None
            return f"{base}[{index}]{member.element_suffix}", member.read_conversion
        storage = self.metal_hlsl_structured_buffer_storage(array_expr)
        if storage is not None and storage[1]:
            array = self.generate_expression(array_expr)
            index = self.generate_expression(index_expr)
            return f"{array}[{index}]", storage[1]
        if isinstance(array_expr, MemberAccessNode):
            member = self.metal_hlsl_layout_member(array_expr)
            if member is not None and member.element_load:
                array = self.generate_expression(array_expr)
                index = self.generate_expression(index_expr)
                return f"{member.element_load}({array}, {index})", None
            if (
                member is not None
                and member.array_count is not None
                and (member.element_suffix or member.read_conversion)
            ):
                array = self.generate_expression(array_expr)
                index = self.generate_expression(index_expr)
                return (
                    f"{array}[{index}]{member.element_suffix}",
                    member.read_conversion,
                )
        return None

    def metal_hlsl_layout_member_object_code(self, expr):
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        obj = self.generate_expression_with_expected(object_expr, None)
        if isinstance(
            object_expr, (BinaryOpNode, TernaryOpNode, UnaryOpNode)
        ) and not self.metal_expression_is_parenthesized(obj):
            obj = f"({obj})"
        return obj

    def metal_hlsl_layout_store_target(self, target):
        """``(lhs, write_conversion)`` for a store into HLSL layout storage."""
        planner = getattr(self, "metal_hlsl_layout_planner", None)
        node = target
        while planner is not None and isinstance(
            node, (MemberAccessNode, ArrayAccessNode)
        ):
            node = node.object if isinstance(node, MemberAccessNode) else node.array
            member = self.metal_hlsl_layout_member(node)
            if member is not None and member.storage_type in planner.matrix_storages:
                kind = "array" if member.element_load else "matrix"
                raise ValueError(
                    f"Cannot store to part of HLSL constant-buffer {kind} "
                    f"'{node.member}' in Metal: it is held as register words"
                )
        if isinstance(target, MemberAccessNode):
            member = self.metal_hlsl_layout_member(target)
            if (
                member is not None
                and member.array_count is None
                and (member.read_conversion or member.write_conversion)
            ):
                obj = self.metal_hlsl_layout_member_object_code(target)
                return f"{obj}.{target.member}", member.write_conversion
            object_expr = getattr(
                target, "object", getattr(target, "object_expr", None)
            )
            inner = self.metal_hlsl_layout_member(object_expr)
            if (
                inner is not None
                and inner.array_count is None
                and inner.read_conversion
            ):
                obj = self.metal_hlsl_layout_member_object_code(object_expr)
                return f"{obj}.{object_expr.member}.{target.member}", None
            if isinstance(object_expr, (MemberAccessNode, ArrayAccessNode)):
                # A component of a vector whose reads convert (``uint3(v[i])``) is
                # stored through the storage vector: ``v[i].y``.
                outer = self.metal_hlsl_layout_store_target(object_expr)
                if outer is not None:
                    lhs, conversion = outer
                    if conversion:
                        width = len(str(target.member))
                        conversion = re.sub(
                            r"[234]$", str(width) if width > 1 else "", conversion
                        )
                    return f"{lhs}.{target.member}", conversion
            return None
        if isinstance(target, ArrayAccessNode):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            index_expr = getattr(target, "index", getattr(target, "index_expr", None))
            storage = self.metal_hlsl_structured_buffer_storage(array_expr)
            if storage is not None and storage[1]:
                array = self.generate_expression(array_expr)
                index = self.generate_expression(index_expr)
                return f"{array}[{index}]", storage[2]
            if isinstance(array_expr, MemberAccessNode):
                member = self.metal_hlsl_layout_member(array_expr)
                if (
                    member is not None
                    and member.array_count is not None
                    and member.read_conversion
                ):
                    array = self.generate_expression(array_expr)
                    index = self.generate_expression(index_expr)
                    return (
                        f"{array}[{index}]{member.element_suffix}",
                        member.write_conversion,
                    )
        return None

    def metal_global_resource_value_type(self, name):
        """Declared type of a program-scope constant-buffer member or buffer.

        Constant-buffer members are referenced by bare name in CrossGL and
        structured buffers are program-scope resources; neither is recorded
        as a local or program-scope value, so expression typing would
        otherwise treat them as unknown.
        """
        if not isinstance(name, str) or not name:
            return None
        if name not in getattr(self, "ambiguous_cbuffer_members", set()):
            for cbuffer in getattr(self, "cbuffer_variables", []) or []:
                for member in getattr(cbuffer, "members", []) or []:
                    if getattr(member, "name", None) == name:
                        return (
                            getattr(member, "member_type", None)
                            or getattr(member, "vtype", None)
                            or getattr(member, "var_type", None)
                        )
        for node, _binding, vtype, _size in (
            getattr(self, "structured_buffer_variables", []) or []
        ):
            if getattr(node, "name", None) == name:
                return vtype
        return None

    METAL_STORAGE_READ_NODES = (
        IdentifierNode,
        ArrayAccessNode,
        MemberAccessNode,
        PointerAccessNode,
    )

    def is_metal_bfloat_storage_read(self, expr):
        """Whether ``expr`` reads a variable, element or member of type bfloat."""
        if (
            isinstance(expr, FunctionCallNode)
            and self.function_call_name(expr) == "buffer_load"
            and len(getattr(expr, "args", None) or []) == 2
        ):
            # buffer_load(buffer, index) is emitted as buffer[index].
            expr = ArrayAccessNode(expr.args[0], expr.args[1])
        elif (
            isinstance(expr, UnaryOpNode)
            and expr.operator == "*"
            and not getattr(expr, "is_postfix", False)
        ):
            # A dereference reads the first element: *pointer is pointer[0].
            expr = ArrayAccessNode(expr.operand, LiteralNode(0, PrimitiveType("int")))
        if not isinstance(expr, self.METAL_STORAGE_READ_NODES):
            return False
        expr_type = self.expression_result_type(expr)
        return expr_type is not None and self.map_type(expr_type) == "bfloat"

    HLSL_REGISTER_CLASS_ORDER = {"b": 0, "t": 1, "u": 2, "s": 3}

    @staticmethod
    def metal_namespace_register_classes(namespace):
        if namespace == "buffer":
            return ("b", "t", "u")
        if namespace == "texture":
            return ("t", "u")
        if namespace == "sampler":
            return ("s",)
        return ()

    def hlsl_register_binding(self, node, register_classes):
        """Return ``(class, index)`` when ``node`` is bound only by an HLSL register.

        HLSL register classes (``b``, ``t``, ``u`` and ``s``) are separate binding
        namespaces, while Metal shares one argument table between constant,
        shader-resource and unordered-access buffers and another between read-only
        and read-write textures. Register numbers are therefore source bindings,
        not target slots; explicit ``@binding``/``@buffer``/``@texture`` attributes
        keep their exact target meaning.
        """
        selected = None
        for attr in getattr(node, "attributes", None) or []:
            attr_name = str(getattr(attr, "name", "") or "").lower()
            arguments = getattr(attr, "arguments", []) or []
            if attr_name in {"binding", "buffer", "texture", "sampler"} and arguments:
                return None
            if attr_name != "register" or not arguments or len(arguments) > 1:
                continue
            raw_value = self.attribute_value_to_string(arguments[0])
            if raw_value is None:
                continue
            raw_value = str(raw_value).strip().lower()
            register_class = raw_value[:1]
            digits = raw_value[1:]
            if register_class not in register_classes or not digits.isdigit():
                continue
            selected = (register_class, int(digits))
        return selected

    def defer_hlsl_register_binding(
        self, node, namespace, resource_count, name, register_classes
    ):
        register = self.hlsl_register_binding(node, register_classes)
        if register is None:
            return False
        if any(entry[0] is node for entry in self.metal_deferred_register_bindings):
            return True
        self.metal_deferred_register_bindings.append(
            (
                node,
                namespace,
                register,
                max(resource_count or 1, 1),
                name,
                len(self.metal_deferred_register_bindings),
            )
        )
        return True

    def reserve_deferred_hlsl_register_bindings(self, used_bindings, source_bindings):
        """Assign Metal slots to register-bound resources after exact bindings.

        Each resource keeps its register number when that slot is free in the Metal
        namespace and otherwise takes the next free slot. Resources are visited in
        register-class order (b, t, u, s), then register number, so the mapping does
        not depend on declaration order. Overlapping registers of the same class
        remain errors.
        """
        deferred = sorted(
            self.metal_deferred_register_bindings,
            key=lambda entry: (
                self.HLSL_REGISTER_CLASS_ORDER.get(entry[2][0], 99),
                entry[2][1],
                entry[5],
            ),
        )
        for node, namespace, register, count, name, _order in deferred:
            if id(node) in self.metal_resource_binding_indices_by_id:
                continue
            register_class, register_index = register
            self.reserve_source_resource_binding_range(
                source_bindings,
                f"register {register_class}",
                0,
                register_index,
                count,
                name,
                stage_scope=self.metal_source_binding_stage_scope(node),
            )
            target_binding = self.next_available_resource_binding(
                used_bindings, namespace, register_index, count
            )
            self.reserve_resource_binding_range(
                used_bindings, "Metal", namespace, target_binding, count, name
            )
            self.metal_resource_binding_indices_by_id[id(node)] = target_binding
            self.metal_hlsl_register_bindings_by_id[id(node)] = (
                f"{register_class}{register_index}"
            )
        self.metal_deferred_register_bindings = []

    def metal_texture1d_explicit_lod_matches_kernel_sampling(self, lod_arg):
        """Return whether an explicit 1D LOD equals Metal kernel sampling.

        Metal 1D and 1D-array textures always have exactly one mip level and
        their sample() overloads take no LOD argument. Kernel functions have no
        derivatives, so sample() selects level 0 with the magnification filter,
        which is exactly what an explicit LOD of zero selects. Graphics stages
        derive an implicit LOD that can select the minification filter instead,
        so only programs whose emitted entry points are all kernels qualify.
        """
        if not getattr(self, "metal_compute_only_program", False):
            return False
        if self.literal_int_value(lod_arg, self.literal_int_constants) == 0:
            return True
        value = getattr(lod_arg, "value", lod_arg)
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return float(value) == 0.0
        if isinstance(value, str):
            text = value.strip()
            while text and text[-1] in "fFhHlLuU":
                text = text[:-1]
            try:
                return float(text) == 0.0
            except ValueError:
                return False
        return False

    def pointer_atomic_compare_helper_name(self, address_space, component_type):
        return f"__crossgl_{address_space}_atomic_compare_exchange_{component_type}"

    def generate_pointer_atomic_compare_helpers(self):
        """Strong compare-exchange returning the original value.

        Metal only provides the weak form, which may fail spuriously; retry
        while the observed value still equals the comparand.
        """
        helpers = []
        for address_space, component_type in sorted(
            getattr(self, "required_pointer_atomic_compare_helpers", set())
        ):
            atomic_type = f"atomic_{component_type}"
            helper_name = self.pointer_atomic_compare_helper_name(
                address_space, component_type
            )
            helpers.append(
                f"{component_type} {helper_name}({address_space} {component_type}* target, "
                f"{component_type} compareValue, {component_type} value) {{\n"
                f"    {address_space} {atomic_type}* atomicTarget = "
                f"reinterpret_cast<{address_space} {atomic_type}*>(target);\n"
                f"    {component_type} original;\n"
                "    do {\n"
                "        original = compareValue;\n"
                "    } while (!atomic_compare_exchange_weak_explicit(atomicTarget, "
                "&original, value, memory_order_relaxed, memory_order_relaxed) "
                "&& original == compareValue);\n"
                "    return original;\n"
                "}\n\n"
            )
        return "".join(helpers)

    def collect_metal_read_after_write_images(self, ast):
        """Names of storage images the program both loads and stores.

        A thread only observes its own earlier ``write`` to a read_write
        texture after a texture fence. ``True`` means every stored image needs
        one because a load goes through a name other than a global image.
        """
        loaded = set()
        stored = False
        for node in self.iter_unique_ast_nodes(ast):
            if not isinstance(node, FunctionCallNode):
                continue
            name = self.function_call_name(node)
            args = list(getattr(node, "args", []) or [])
            if name == "imageStore":
                stored = True
            if name != "imageLoad" or not args:
                continue
            image = args[0]
            image_name = getattr(image, "name", None)
            if not isinstance(image, (IdentifierNode, VariableNode)) or not image_name:
                return True
            loaded.add(image_name)
        if not stored:
            return set()
        global_images = {
            getattr(node, "name", None)
            for node in getattr(ast, "global_variables", []) or []
        }
        if loaded - global_images:
            return True
        return loaded

    def iter_unique_ast_nodes(self, root):
        """Yield every AST node reachable from ``root`` once, iteratively."""
        seen = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if node is None or isinstance(node, (str, int, float, bool)):
                continue
            if isinstance(node, (list, tuple, set)):
                stack.extend(node)
                continue
            if isinstance(node, dict):
                stack.extend(node.values())
                continue
            if not hasattr(node, "__dict__") or id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            for key, value in vars(node).items():
                if key not in {"parent", "annotations"}:
                    stack.append(value)

    def metal_image_store_fence_statement(self, stmt, indent_str):
        """Fence after an image store whose image the program also reads.

        Metal only makes a thread's ``write`` to a read_write texture visible to
        its own later reads after ``fence()``.
        """
        if not isinstance(stmt, FunctionCallNode):
            return ""
        if self.function_call_name(stmt) != "imageStore":
            return ""
        args = list(getattr(stmt, "args", []) or [])
        if len(args) < 3:
            return ""
        fenced = getattr(self, "metal_fenced_image_names", set())
        image_name = getattr(args[0], "name", None)
        if fenced is not True and image_name not in fenced:
            return ""
        image_type = self.texture_resource_type(args[0])
        if self.is_multisample_storage_image_resource(image_type):
            return ""
        return f"{indent_str}{self.generate_expression(args[0])}.fence();\n"
