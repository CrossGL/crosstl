"""Typed buffer atomics and aggregate conditionals in the HLSL generator: image atomics, helpers, statements and returns, render targets, assignments and lvalues."""

from hashlib import sha1

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from .array_utils import format_c_style_array_declaration
from .directx_values import (
    DirectXAggregateConditionalError,
    DirectXAggregateInitializerError,
)
from .generic_function_utils import numeric_trait_method_result_type
from .hlsl_array_list_assignment import generate_hlsl_array_list_assignment
from .hlsl_groupshared_atomics import is_hlsl_groupshared_atomic_target
from .hlsl_word_views import (
    hlsl_word_float_atomic_body,
    hlsl_word_view_atomic_lvalue,
    hlsl_word_view_atomic_storage,
)
from .image_access_contracts import (
    image_atomic_helper_descriptor_fields,
    image_atomic_helper_resource_metadata,
    image_atomic_result_kind_mismatch,
)


class DirectXTypedBufferAtomicMixin:
    """Typed buffer atomic members of ``HLSLCodeGen``."""

    def image_atomic_helper_descriptor(self, operation, texture_type):
        intrinsic = self.image_atomic_intrinsic(operation)
        if not intrinsic:
            return None

        texture_type = self.resource_base_type(texture_type)
        if "<" not in texture_type or ">" not in texture_type:
            return None

        texture_family = texture_type.split("<", 1)[0]
        component_type = texture_type.split("<", 1)[1].split(">", 1)[0].strip()
        metadata = image_atomic_helper_resource_metadata(
            texture_family,
            {
                "RWTexture1D": "image1D",
                "RWTexture1DArray": "image1DArray",
                "RWTexture2D": "image2D",
                "RWTexture3D": "image3D",
                "RWTexture2DArray": "image2DArray",
                "RWTexture2DMS": "image2DMS",
                "RWTexture2DMSArray": "image2DMSArray",
                "RasterizerOrderedTexture1D": "image1D",
                "RasterizerOrderedTexture1DArray": "image1DArray",
                "RasterizerOrderedTexture2D": "image2D",
                "RasterizerOrderedTexture3D": "image3D",
                "RasterizerOrderedTexture2DArray": "image2DArray",
            },
            {
                "RWTexture1D": "int",
                "RWTexture1DArray": "int2",
                "RWTexture2D": "int2",
                "RWTexture3D": "int3",
                "RWTexture2DArray": "int3",
                "RWTexture2DMS": "int2",
                "RWTexture2DMSArray": "int3",
                "RasterizerOrderedTexture1D": "int",
                "RasterizerOrderedTexture1DArray": "int2",
                "RasterizerOrderedTexture2D": "int2",
                "RasterizerOrderedTexture3D": "int3",
                "RasterizerOrderedTexture2DArray": "int3",
            },
            sample_families={"RWTexture2DMS", "RWTexture2DMSArray"},
        )
        if metadata is None:
            return None

        descriptor = image_atomic_helper_descriptor_fields(
            operation,
            component_type,
            metadata["suffix_family"],
            metadata["coord_type"],
        )
        if descriptor is None:
            return None

        descriptor["intrinsic"] = intrinsic
        if metadata.get("has_sample"):
            descriptor["has_sample"] = True
        return descriptor

    def image_atomic_expression(self, operation, args):
        if not self.image_atomic_intrinsic(operation):
            return None
        if operation == "imageAtomicCompSwap":
            if len(args) < 4:
                return None
            image_name = self.generate_expression(args[0])
            coord = self.generate_expression(args[1])
            image_type = self.texture_resource_type(args[0])
            if self.is_multisample_storage_image_resource_type(image_type):
                return self.unsupported_multisample_image_atomic_call(
                    operation, image_type
                )
            descriptor = self.image_atomic_helper_descriptor(operation, image_type)
            if descriptor is None:
                return self.unsupported_image_atomic_call(operation, image_type)
            helper_name = descriptor["helper_name"]
            if descriptor.get("has_sample"):
                if len(args) < 5:
                    return None
                sample = self.generate_expression(args[2])
                compare = self.generate_expression(args[3])
                value = self.generate_expression(args[4])
                self.required_image_atomic_helpers.add((operation, image_type))
                return (
                    f"{helper_name}({image_name}, {coord}, {sample}, "
                    f"{compare}, {value})"
                )
            compare = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            self.required_image_atomic_helpers.add((operation, image_type))
            return f"{helper_name}({image_name}, {coord}, {compare}, {value})"
        if len(args) < 3:
            return None
        image_name = self.generate_expression(args[0])
        coord = self.generate_expression(args[1])
        image_type = self.texture_resource_type(args[0])
        if self.is_multisample_storage_image_resource_type(image_type):
            return self.unsupported_multisample_image_atomic_call(operation, image_type)
        descriptor = self.image_atomic_helper_descriptor(operation, image_type)
        if descriptor is None:
            return self.unsupported_image_atomic_call(operation, image_type)
        helper_name = descriptor["helper_name"]
        if descriptor.get("has_sample"):
            if len(args) < 4:
                return None
            sample = self.generate_expression(args[2])
            value = self.generate_expression(args[3])
            self.required_image_atomic_helpers.add((operation, image_type))
            return f"{helper_name}({image_name}, {coord}, {sample}, {value})"
        value = self.generate_expression(args[2])
        self.required_image_atomic_helpers.add((operation, image_type))
        return f"{helper_name}({image_name}, {coord}, {value})"

    def generate_image_atomic_helpers(self):
        if not self.required_image_atomic_helpers:
            return ""

        helpers = []
        for operation, texture_type in sorted(self.required_image_atomic_helpers):
            descriptor = self.image_atomic_helper_descriptor(operation, texture_type)
            if descriptor is None:
                continue
            helper_name = descriptor["helper_name"]
            return_type = descriptor["return_type"]
            coord_type = descriptor["coord_type"]
            intrinsic = descriptor["intrinsic"]
            has_sample = descriptor.get("has_sample")
            sample_param = ", int sample" if has_sample else ""
            target = "image[coord, sample]" if has_sample else "image[coord]"
            if operation == "imageAtomicCompSwap":
                helpers.append(
                    f"{return_type} {helper_name}({texture_type} image, {coord_type} coord{sample_param}, {return_type} compareValue, {return_type} value) {{\n"
                    f"    {return_type} original;\n"
                    f"    InterlockedCompareExchange({target}, compareValue, value, original);\n"
                    "    return original;\n"
                    "}\n\n"
                )
                continue
            helpers.append(
                f"{return_type} {helper_name}({texture_type} image, {coord_type} coord{sample_param}, {return_type} value) {{\n"
                f"    {return_type} original;\n"
                f"    {intrinsic}({target}, value, original);\n"
                "    return original;\n"
                "}\n\n"
            )

        return "".join(helpers)

    def hlsl_typed_buffer_element_type(self, vtype, resource_types=None):
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        if resource_types is None:
            resource_types = {"RWBuffer", "RWStructuredBuffer"}
        base_type = self.resource_base_type(type_name)
        if "<" not in base_type or not base_type.endswith(">"):
            return None
        resource_type, generic_args = base_type.split("<", 1)
        if resource_type not in resource_types:
            return None
        generic_args = generic_args[:-1].strip()
        if not generic_args or "," in generic_args:
            return None
        return generic_args

    def hlsl_struct_buffer_expression_resource_type(self, expr):
        name = self.expression_name(expr)
        if name in self.hlsl_struct_buffer_resource_types:
            return self.hlsl_struct_buffer_resource_types[name]
        if isinstance(expr, MemberAccessNode) or (
            hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__)
        ):
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.hlsl_struct_buffer_expression_resource_type(object_expr)
        if isinstance(expr, ArrayAccessNode) or (
            hasattr(expr, "__class__") and "ArrayAccess" in str(expr.__class__)
        ):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            return self.hlsl_struct_buffer_expression_resource_type(array_expr)
        return None

    def hlsl_struct_buffer_member_access_expression(self, expr):
        if not (
            isinstance(expr, MemberAccessNode)
            or (hasattr(expr, "__class__") and "MemberAccess" in str(expr.__class__))
        ):
            return None
        object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
        if not (
            isinstance(object_expr, (IdentifierNode, VariableNode, str))
            or (
                hasattr(object_expr, "__class__")
                and "Identifier" in str(object_expr.__class__)
            )
        ):
            return None
        object_name = self.expression_name(object_expr)
        if object_name not in self.hlsl_struct_buffer_resource_types:
            return None
        member = getattr(expr, "member", "")
        return f"{object_name}[0].{member}"

    def hlsl_typed_buffer_atomic_operations(self):
        return {
            "atomicLoad": ("InterlockedOr", 0),
            "atomicStore": ("InterlockedExchange", 1),
            "atomicAdd": ("InterlockedAdd", 1),
            "atomicMin": ("InterlockedMin", 1),
            "atomicMax": ("InterlockedMax", 1),
            "atomicAnd": ("InterlockedAnd", 1),
            "atomicOr": ("InterlockedOr", 1),
            "atomicXor": ("InterlockedXor", 1),
            "atomicExchange": ("InterlockedExchange", 1),
            "atomicCompSwap": ("InterlockedCompareExchange", 2),
            "atomicCompareExchange": ("InterlockedCompareExchange", 2),
        }

    def hlsl_typed_buffer_atomic_target_resource_type(
        self, target, resource_types=None
    ):
        if isinstance(target, PointerAccessNode):
            target = ArrayAccessNode(target.pointer_expr, 0)
        if isinstance(target, UnaryOpNode) and target.op == "*":
            target = ArrayAccessNode(target.operand, 0)
        load_access = self.hlsl_typed_buffer_atomic_load_access(target)
        if load_access is not None:
            target = load_access
        struct_buffer_type = self.hlsl_struct_buffer_expression_resource_type(target)
        if struct_buffer_type is not None:
            if (
                resource_types is None
                or self.hlsl_resource_type_name(struct_buffer_type) in resource_types
            ):
                return struct_buffer_type
            return None
        if isinstance(target, ArrayAccessNode) or (
            hasattr(target, "__class__") and "ArrayAccess" in str(target.__class__)
        ):
            array_expr = getattr(target, "array", getattr(target, "array_expr", None))
            array_type = self.expression_result_type(array_expr)
            binding = self.hlsl_resource_pointer_binding(array_expr)
            if binding is not None and binding.get("kind") != "workgroup-pointer":
                array_type = binding.get("resource_type") or array_type
            if (
                self.hlsl_typed_buffer_element_type(array_type, resource_types)
                is not None
            ):
                return array_type
            return self.hlsl_typed_buffer_atomic_target_resource_type(
                array_expr, resource_types
            )
        if isinstance(target, MemberAccessNode) or (
            hasattr(target, "__class__") and "MemberAccess" in str(target.__class__)
        ):
            object_expr = getattr(
                target, "object", getattr(target, "object_expr", None)
            )
            return self.hlsl_typed_buffer_atomic_target_resource_type(
                object_expr, resource_types
            )
        return None

    def hlsl_typed_buffer_atomic_load_access(self, target):
        if not isinstance(target, FunctionCallNode):
            return None
        name = self.function_call_name(target)
        args = getattr(target, "args", [])
        if (
            name == "buffer_load"
            and name not in self.function_return_types
            and len(args) == 2
            and self.hlsl_typed_buffer_element_type(
                self.hlsl_buffer_helper_resource_type(args[0]),
                {"Buffer", "StructuredBuffer", "RWBuffer", "RWStructuredBuffer"},
            )
            is not None
        ):
            return ArrayAccessNode(args[0], args[1])
        return None

    def hlsl_typed_buffer_atomic_lvalue(self, target):
        load_access = self.hlsl_typed_buffer_atomic_load_access(target)
        if load_access is not None:
            return load_access
        if isinstance(target, MemberAccessNode):
            return MemberAccessNode(
                self.hlsl_typed_buffer_atomic_lvalue(target.object), target.member
            )
        if isinstance(target, ArrayAccessNode):
            return ArrayAccessNode(
                self.hlsl_typed_buffer_atomic_lvalue(target.array), target.index
            )
        return target

    def hlsl_buffer_atomic_storage(self, target, *, allow_workgroup=False):
        if isinstance(target, PointerAccessNode):
            target = MemberAccessNode(
                ArrayAccessNode(target.pointer_expr, 0), target.member
            )
        if isinstance(target, UnaryOpNode) and target.op == "*":
            target = ArrayAccessNode(target.operand, 0)
        load_access = self.hlsl_typed_buffer_atomic_load_access(target)
        if load_access is not None:
            target = load_access
        if isinstance(target, MemberAccessNode):
            storage = self.hlsl_buffer_atomic_storage(
                target.object, allow_workgroup=allow_workgroup
            )
            if storage is not None:
                storage["suffix"] += f".{target.member}"
            return storage
        if isinstance(target, ArrayAccessNode):
            container, index = target.array, target.index
            if allow_workgroup:
                shared = self.hlsl_static_workgroup_pointer_binding(
                    target, self.current_hlsl_resource_pointer_aliases
                )
                if shared is not None:
                    return {
                        "resource_type": None,
                        "root": shared["root"],
                        "arguments": [
                            self.generate_expression(shared.get("offset", 0))
                        ],
                        "index_types": ["uint"],
                        "suffix": "[index0]",
                    }
            binding = self.hlsl_resource_pointer_binding(container)
            if isinstance(
                container, ArrayAccessNode
            ) and self.is_hlsl_buffer_resource_array_type(
                self.expression_result_type(container.array)
            ):
                # Descriptor selection is not an element offset into one buffer.
                binding = None
            resource_type = (
                binding.get("resource_type") if binding is not None else None
            ) or self.hlsl_buffer_helper_resource_type(container)
            if self.hlsl_typed_buffer_element_type(resource_type) is not None:
                word_storage = (
                    hlsl_word_view_atomic_storage(self, binding, index, resource_type)
                    if binding is not None
                    else None
                )
                if word_storage is not None:
                    return word_storage
                if binding is not None and (
                    binding.get("pointer_reinterpretation")
                    or str(binding.get("byte_offset", "0")) != "0"
                ):
                    raise DirectXRefusal(
                        "DirectX buffer atomic requires an unambiguous typed storage view"
                    )
                resource = (
                    binding["root"]
                    if binding is not None
                    else self.generate_expression(container)
                )
                rendered_index = self.generate_expression(index)
                if binding is not None:
                    rendered_index = self.hlsl_resource_pointer_offset_sum(
                        binding.get("offset"), rendered_index
                    )
                rendered_index = self.hlsl_resource_index_expression(
                    container,
                    index,
                    rendered_index,
                    rendered_resource=resource,
                    resource_type=resource_type,
                )
                return {
                    "resource_type": self.map_type(
                        self.directx_resource_declaration_type(resource_type)
                    ),
                    "arguments": [resource, rendered_index],
                    "index_types": ["uint"],
                    "suffix": "[index0]",
                }
            storage = self.hlsl_buffer_atomic_storage(
                container, allow_workgroup=allow_workgroup
            )
            if storage is not None:
                index_type = self.map_type(self.expression_result_type(index))
                if index_type not in {"int", "uint", "int64_t", "uint64_t"}:
                    raise DirectXRefusal(
                        "DirectX buffer atomic requires scalar integer member indices"
                    )
                storage["suffix"] += f"[index{len(storage['index_types'])}]"
                storage["index_types"].append(index_type)
                storage["arguments"].append(self.generate_expression(index))
            return storage
        name = self.expression_name(target)
        resource_type = self.hlsl_struct_buffer_resource_types.get(name)
        if resource_type is not None:
            return {
                "resource_type": self.map_type(resource_type),
                "arguments": [self.hlsl_identifier_name(name), "0u"],
                "index_types": ["uint"],
                "suffix": "[index0]",
            }
        return None

    def generate_hlsl_expected_compare_call(self, args):
        if len(args) != 3:
            raise DirectXRefusal(
                "DirectX atomicCompareExchangeWeak requires target, expected and desired"
            )
        kind = self.scalar_expression_kind(args[0])
        if (
            kind not in {"int", "uint", "float"}
            or self.scalar_expression_kind(args[1]) != kind
        ):
            raise DirectXRefusal(
                "DirectX atomicCompareExchangeWeak requires matching int, uint or float target and expected"
            )
        storage = self.hlsl_buffer_atomic_storage(args[0], allow_workgroup=True)
        if storage is None or (
            storage["resource_type"] and not storage["resource_type"].startswith("RW")
        ):
            raise DirectXRefusal(
                "DirectX atomicCompareExchangeWeak requires writable buffer or groupshared storage"
            )
        key = (
            kind,
            storage["resource_type"],
            storage.get("root"),
            tuple(storage["index_types"]),
            storage["suffix"],
        )
        helper = self.hlsl_expected_compare_helpers.get(key)
        if helper is None:
            digest = sha1(repr(key).encode("utf-8")).hexdigest()[:12]
            name = f"__crossgl_compare_expected_{digest}"
            reserved = (
                self.hlsl_float_atomic_reserved_names
                | set(self.function_return_types)
                | set(self.local_variable_types)
            )
            reserved.update(
                item["name"] for item in self.hlsl_expected_compare_helpers.values()
            )
            while name in reserved:
                name += "_"
            helper = {**storage, "name": name, "kind": kind}
            self.hlsl_expected_compare_helpers[key] = helper
        assignments, expected = self.hlsl_stabilize_inout_lvalue(
            args[1],
            args[1],
            kind,
            operation="atomic expected-value writeback",
            reason="atomic-expected-unstable-lvalue",
        )
        arguments = storage["arguments"] + [
            self.generate_expression(expected),
            self.generate_expression_with_expected(args[2], kind),
        ]
        call = f"{helper['name']}({', '.join(arguments)})"
        # The explicit result type distinguishes sequencing from a constructor
        # argument list in DXC's strict initializer diagnostics.
        return f"bool(({', '.join([*assignments, call])}))" if assignments else call

    def generate_hlsl_expected_compare_helpers(self):
        code = ""
        for helper in self.hlsl_expected_compare_helpers.values():
            kind = helper["kind"]
            roles = ["storage", "expected", "desired", "observed", "matched"] + [
                f"index{i}" for i in range(len(helper["index_types"]))
            ]
            prefix = "__crossgl_cas_"
            while any(
                prefix + role in self.hlsl_float_atomic_reserved_names for role in roles
            ):
                prefix += "_"
            names = {role: prefix + role for role in roles}
            parameters = (
                [f"{helper['resource_type']} {names['storage']}"]
                if helper["resource_type"]
                else []
            )
            parameters.extend(
                f"{index_kind} {names[f'index{i}']}"
                for i, index_kind in enumerate(helper["index_types"])
            )
            parameters.extend(
                [f"inout {kind} {names['expected']}", f"{kind} {names['desired']}"]
            )
            root = names["storage"] if helper["resource_type"] else helper["root"]
            suffix = helper["suffix"]
            for i in range(len(helper["index_types"])):
                suffix = suffix.replace(f"[index{i}]", f"[{names[f'index{i}']}]")
            intrinsic = "InterlockedCompareExchange"
            comparison = f"{names['observed']} == {names['expected']}"
            if kind == "float":
                intrinsic += "FloatBitwise"
                comparison = (
                    f"asuint({names['observed']}) == asuint({names['expected']})"
                )
            code += (
                f"bool {helper['name']}({', '.join(parameters)}) {{\n"
                f"    {kind} {names['observed']};\n"
                f"    {intrinsic}({root}{suffix}, {names['expected']}, {names['desired']}, {names['observed']});\n"
                f"    bool {names['matched']} = {comparison};\n"
                f"    if (!{names['matched']}) {names['expected']} = {names['observed']};\n"
                f"    return {names['matched']};\n"
                "}\n\n"
            )
        return code

    def hlsl_float_atomic_helper(self, func_name, storage):
        key = (
            func_name,
            storage["resource_type"],
            storage.get("root"),
            tuple(storage["index_types"]),
            storage["suffix"],
            storage.get("physical_kind", "float"),
        )
        helper = self.hlsl_float_atomic_helpers.get(key)
        if helper is None:
            digest = sha1(repr(key).encode("utf-8")).hexdigest()[:12]
            name = f"__crossgl_float_atomic_{digest}"
            reserved = (
                self.hlsl_float_atomic_reserved_names
                | set(self.function_return_types)
                | set(self.global_variable_types)
                | set(self.structs_by_name)
            )
            reserved.update(self.local_variable_types)
            reserved.update(
                item["name"] for item in self.hlsl_float_atomic_helpers.values()
            )
            while name in reserved:
                name += "_"
            helper = {**storage, "name": name, "operation": func_name}
            self.hlsl_float_atomic_helpers[key] = helper
        return helper["name"]

    def generate_hlsl_float_atomic_helpers(self):
        code = ""
        for helper in self.hlsl_float_atomic_helpers.values():
            roles = [
                "storage",
                "value",
                "original",
                "observed",
                "expected",
                "desired",
            ] + [f"index{i}" for i in range(len(helper["index_types"]))]
            prefix = ""
            if not helper["resource_type"]:
                while any(
                    prefix + role in self.hlsl_float_atomic_reserved_names
                    for role in roles
                ):
                    prefix += "__crossgl_float_"
            names = {role: prefix + role for role in roles}
            parameters = (
                [f"{helper['resource_type']} {names['storage']}"]
                if helper["resource_type"]
                else []
            )
            parameters.extend(
                f"{kind} {names[f'index{index}']}"
                for index, kind in enumerate(helper["index_types"])
            )
            if helper["operation"] != "atomicLoad":
                parameters.append(f"float {names['value']}")
            parameters.append(f"out float {names['original']}")
            root = names["storage"] if helper["resource_type"] else helper["root"]
            suffix = helper["suffix"]
            for index in range(len(helper["index_types"])):
                suffix = suffix.replace(
                    f"[index{index}]", f"[{names[f'index{index}']}]"
                )
            target = f"{root}{suffix}"
            code += f"void {helper['name']}({', '.join(parameters)}) {{\n"
            physical_kind = helper.get("physical_kind", "float")
            if physical_kind != "float":
                code += hlsl_word_float_atomic_body(
                    helper["operation"], target, names, physical_kind
                )
            elif helper["operation"] == "atomicLoad":
                # Equal compare/replacement bits make this a non-modifying atomic read.
                code += f"    InterlockedCompareExchangeFloatBitwise({target}, 0.0f, 0.0f, {names['original']});\n"
            elif helper["operation"] in {"atomicExchange", "atomicStore"}:
                code += f"    InterlockedExchange({target}, {names['value']}, {names['original']});\n"
            else:
                code += (
                    f"    float {names['observed']};\n"
                    f"    InterlockedCompareExchangeFloatBitwise({target}, 0.0f, 0.0f, {names['observed']});\n"
                    "    [allow_uav_condition]\n"
                    "    while (true) {\n"
                    f"        float {names['expected']} = {names['observed']};\n"
                    f"        precise float {names['desired']} = {names['expected']} + {names['value']};\n"
                    f"        InterlockedCompareExchangeFloatBitwise({target}, {names['expected']}, {names['desired']}, {names['observed']});\n"
                    f"        if (asuint({names['observed']}) == asuint({names['expected']})) {{\n"
                    f"            {names['original']} = {names['expected']};\n"
                    "            return;\n"
                    "        }\n"
                    "    }\n"
                )
            code += "}\n\n"
        return code

    def hlsl_typed_buffer_atomic_parts(self, func_name, args, *, render_operands=True):
        operation_info = self.hlsl_typed_buffer_atomic_operations().get(func_name)
        if (
            operation_info is None
            or func_name in getattr(self, "function_return_types", {})
            or not args
        ):
            return None

        target = args[0]
        resource_type = self.hlsl_typed_buffer_atomic_target_resource_type(target)
        if resource_type is None:
            shared = self.hlsl_static_workgroup_pointer_binding(
                target, self.current_hlsl_resource_pointer_aliases
            )
            if shared is not None or is_hlsl_groupshared_atomic_target(self, target):
                resource_type = "groupshared"
        if resource_type is None:
            readonly_resource_type = self.hlsl_typed_buffer_atomic_target_resource_type(
                target, {"Buffer", "StructuredBuffer"}
            )
            if readonly_resource_type is not None:
                raise DirectXRefusal(
                    f"DirectX typed buffer atomic '{func_name}' cannot write "
                    f"readonly {self.resource_base_type(readonly_resource_type)}"
                )
            if (
                func_name == "atomicLoad"
                and self.glsl_buffer_block_atomic_access(target)[0] is None
            ):
                raise DirectXRefusal(
                    "DirectX atomicLoad requires writable buffer or groupshared storage"
                )
            return None

        intrinsic, value_arg_count = operation_info
        min_args = 1 + value_arg_count
        max_args = min_args + 1
        if func_name in {"atomicLoad", "atomicStore"}:
            max_args = min_args
        if not min_args <= len(args) <= max_args:
            raise DirectXRefusal(
                f"DirectX typed buffer atomic '{func_name}' requires "
                f"{min_args} or {max_args} argument(s), got {len(args)}"
            )

        target_type = self.expression_result_type(target)
        target_kind = self.scalar_expression_kind(target)
        float_atomic = self.map_type(target_type) == "float" and func_name in {
            "atomicLoad",
            "atomicStore",
            "atomicAdd",
            "atomicExchange",
        }
        if target_kind not in {"int", "uint"} and not float_atomic:
            target_label = self.type_name_string(target_type) or str(resource_type)
            raise DirectXRefusal(
                f"DirectX typed buffer atomic '{func_name}' requires a scalar "
                f"int or uint target, got {target_label}"
            )

        value_args = args[1 : 1 + value_arg_count]
        for index, value_arg in enumerate(value_args, start=1):
            value_kind = self.scalar_expression_kind(value_arg)
            if value_kind is None or value_kind == target_kind:
                continue
            if {value_kind, target_kind} <= {"int", "uint"}:
                # The memory operand fixes the operation's signedness, as in
                # GLSL, SPIR-V and Metal; the operand is rendered as an
                # explicit bit-preserving int/uint conversion.
                continue
            role = "compare value" if value_arg_count == 2 and index == 1 else "value"
            if value_arg_count == 2 and index == 2:
                role = "replacement"
            raise DirectXRefusal(
                f"DirectX typed buffer atomic '{func_name}' {role} argument "
                f"must be scalar {target_kind}, got {value_kind}"
            )

        original_arg = args[min_args] if len(args) > min_args else None
        if original_arg is not None:
            original_kind = self.scalar_expression_kind(original_arg)
            if original_kind != target_kind:
                original_type = (
                    self.type_name_string(self.expression_result_type(original_arg))
                    or original_kind
                    or expression_debug_name(original_arg)
                )
                raise DirectXRefusal(
                    f"DirectX typed buffer atomic '{func_name}' original "
                    f"argument must be scalar {target_kind}, got "
                    f"{original_type}"
                )
            if not self.hlsl_typed_buffer_atomic_original_is_lvalue(original_arg):
                raise DirectXRefusal(
                    f"DirectX typed buffer atomic '{func_name}' original "
                    f"argument must be an assignable scalar {target_kind} target"
                )
        parts = {
            "func_name": func_name,
            "intrinsic": intrinsic,
            "float_atomic": float_atomic,
            "target_type": target_type,
            "target_kind": target_kind,
            "original_arg": original_arg,
        }
        # Discovery must not render operands: a nested atomic needs its own
        # statement context before the containing call can be emitted.
        if not render_operands:
            return parts
        rendered_target = (
            None if float_atomic else hlsl_word_view_atomic_lvalue(self, target)
        )
        if rendered_target is None:
            rendered_target = self.generate_expression(
                self.hlsl_typed_buffer_atomic_lvalue(target)
            )
        target_args = [rendered_target]
        if float_atomic:
            storage = self.hlsl_buffer_atomic_storage(
                target, allow_workgroup=func_name in {"atomicLoad", "atomicStore"}
            )
            if storage is None:
                raise DirectXRefusal(
                    "DirectX float atomic requires a writable typed buffer element"
                )
            parts["intrinsic"] = self.hlsl_float_atomic_helper(func_name, storage)
            target_args = storage["arguments"]
        code = ""
        rendered_values = []
        for value_arg in value_args:
            value_code, value = self.render_hlsl_typed_buffer_atomic_value_expression(
                value_arg, target_type, 0
            )
            value_kind = self.scalar_expression_kind(value_arg)
            if value_kind in {"int", "uint"} and value_kind != target_kind:
                value = f"{target_kind}({value})"
            code += value_code
            rendered_values.append(value)
        if func_name == "atomicLoad" and not float_atomic:
            # An unchanged-value RMW supplies an atomic read without a plain load.
            rendered_values = ["0u" if target_kind == "uint" else "0"]
        return {
            **parts,
            "target": rendered_target,
            "target_args": target_args,
            "values": rendered_values,
            "code": code,
        }

    def validate_hlsl_typed_buffer_atomic_result_context(self, parts, expected_type):
        if parts["func_name"] == "atomicStore":
            raise DirectXRefusal("DirectX atomicStore does not return a value")
        expected_label = self.hlsl_atomic_result_expected_label(expected_type)
        if expected_label is None:
            return
        expected_kind = image_atomic_result_kind_mismatch(
            expected_label,
            parts["target_kind"],
        )
        if expected_kind is None and expected_label == parts["target_kind"]:
            return

        target_type = (
            self.type_name_string(parts["target_type"]) or parts["target_kind"]
        )
        raise DirectXRefusal(
            f"DirectX typed buffer atomic '{parts['func_name']}' requires "
            f"{parts['target_kind']} result context for {target_type} target: "
            f"expected {expected_kind or expected_label}"
        )

    def hlsl_typed_buffer_atomic_original_is_lvalue(self, expr):
        if isinstance(expr, (IdentifierNode, VariableNode)):
            return True

        if isinstance(expr, MemberAccessNode):
            member = str(getattr(expr, "member", ""))
            if len(member) > 1 and all(ch in "xyzwrgba" for ch in member):
                return False
            object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
            return self.hlsl_typed_buffer_atomic_original_is_lvalue(object_expr)

        if isinstance(expr, PointerAccessNode):
            pointer_expr = getattr(expr, "pointer_expr", None)
            return self.hlsl_typed_buffer_atomic_original_is_lvalue(pointer_expr)

        if isinstance(expr, SwizzleNode):
            if len(str(getattr(expr, "components", ""))) != 1:
                return False
            return self.hlsl_typed_buffer_atomic_original_is_lvalue(
                getattr(expr, "vector_expr", None)
            )

        if isinstance(expr, ArrayAccessNode):
            array_expr = getattr(expr, "array", getattr(expr, "array_expr", None))
            array_type = self.expression_result_type(array_expr)
            if self.hlsl_typed_buffer_element_type(array_type) is not None:
                return False
            if self.is_storage_image_resource_type(array_type):
                return False
            return self.hlsl_typed_buffer_atomic_original_is_lvalue(array_expr)

        return False

    def generate_hlsl_typed_buffer_atomic_statement(
        self, expr, result_target=None, result_target_type=None
    ):
        if not (
            isinstance(expr, FunctionCallNode)
            or (hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__))
        ):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", []))
        parts = self.hlsl_typed_buffer_atomic_parts(func_name, args)
        if parts is None:
            return None

        original_arg = parts["original_arg"]
        if original_arg is not None:
            original = self.generate_expression(original_arg)
        elif result_target is not None:
            self.validate_hlsl_typed_buffer_atomic_result_context(
                parts, result_target_type
            )
            original = (
                result_target
                if isinstance(result_target, str)
                else self.generate_expression(result_target)
            )
        else:
            original = None

        call_args = [*parts["target_args"], *parts["values"]]
        if original is not None:
            call_args.append(original)
            return parts["code"] + f"{parts['intrinsic']}({', '.join(call_args)})"

        if (
            parts["intrinsic"]
            in {
                "InterlockedCompareExchange",
                "InterlockedExchange",
            }
            or parts["float_atomic"]
        ):
            temp_type = self.map_type(parts["target_type"])
            temp_name = self.next_hlsl_temp_variable("atomic_original")
            call_args.append(temp_name)
            return (
                parts["code"]
                + f"{temp_type} {temp_name}\n{parts['intrinsic']}({', '.join(call_args)})"
            )
        return parts["code"] + f"{parts['intrinsic']}({', '.join(call_args)})"

    def generate_hlsl_typed_buffer_atomic_return(self, expr, indent=0):
        if not (
            isinstance(expr, FunctionCallNode)
            or (hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__))
        ):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        func_name = getattr(func_expr, "name", func_expr)
        if not isinstance(func_name, str):
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", []))
        parts = self.hlsl_typed_buffer_atomic_parts(func_name, args)
        if parts is None:
            return None

        if func_name == "atomicStore":
            if self.map_type(self.current_function_return_type) != "void":
                raise DirectXRefusal("DirectX atomicStore does not return a value")
            call_args = [*parts["target_args"], *parts["values"]]
            indent_str = "    " * indent
            original = self.next_hlsl_temp_variable("atomic_discarded")
            call_args.append(original)
            return (
                self.generate_statement_code(parts["code"], indent)
                + f"{indent_str}{self.map_type(parts['target_type'])} {original};\n"
                f"{indent_str}{parts['intrinsic']}({', '.join(call_args)});\n"
                + self.generate_hlsl_parameter_epilogue(indent)
                + f"{indent_str}return;\n"
            )

        original_arg = parts["original_arg"]
        self.validate_hlsl_typed_buffer_atomic_result_context(
            parts, self.current_function_return_type
        )
        if original_arg is not None:
            original = self.generate_expression(original_arg)
            declaration = ""
        else:
            temp_type = self.map_type(parts["target_type"])
            original = self.next_hlsl_temp_variable("atomic_return")
            declaration = f"{temp_type} {original};\n"

        call_args = [*parts["target_args"], *parts["values"], original]
        indent_str = "    " * indent
        code = self.generate_statement_code(parts["code"], indent)
        if declaration:
            code += f"{indent_str}{declaration}"
        code += f"{indent_str}{parts['intrinsic']}({', '.join(call_args)});\n"
        code += self.generate_hlsl_return_value_statement(original, indent)
        return code

    def hlsl_is_ternary_expression(self, expr):
        return isinstance(expr, TernaryOpNode) or (
            hasattr(expr, "__class__") and "TernaryOp" in str(expr.__class__)
        )

    def hlsl_aggregate_value_type(self, type_name):
        type_name = self.type_name_string(type_name)
        if not type_name:
            return None
        mapped_type = self.map_type(type_name)
        if self.hlsl_struct_constructor_fields(mapped_type) is not None:
            return mapped_type
        if self.hlsl_outer_array_type(mapped_type) is not None:
            return mapped_type
        return None

    def hlsl_aggregate_conditional_type(self, expr, expected_type=None):
        if not self.hlsl_is_ternary_expression(expr):
            return None

        inferred_candidates = [
            self.expression_result_type(expr),
            self.expression_result_type(getattr(expr, "true_expr", None)),
            self.expression_result_type(getattr(expr, "false_expr", None)),
        ]
        for candidate in inferred_candidates:
            aggregate_type = self.hlsl_aggregate_value_type(candidate)
            if aggregate_type is not None:
                return aggregate_type

        inferred_types = [
            self.type_name_string(candidate)
            for candidate in inferred_candidates
            if self.type_name_string(candidate)
        ]
        if any(
            self.is_scalar_value_type(candidate)
            or self.is_vector_value_type(candidate)
            or self.hlsl_matrix_shape(self.map_type(candidate)) is not None
            for candidate in inferred_types
        ):
            return None
        return self.hlsl_aggregate_value_type(expected_type)

    def hlsl_first_aggregate_conditional(self, expr, expected_type=None):
        aggregate_type = self.hlsl_aggregate_conditional_type(
            expr, expected_type
        ) or self.hlsl_effectful_struct_constructor_type(expr)
        if aggregate_type is not None:
            return expr, aggregate_type
        for node in self.walk_ast(expr):
            if node is expr:
                continue
            aggregate_type = self.hlsl_aggregate_conditional_type(
                node
            ) or self.hlsl_effectful_struct_constructor_type(node)
            if aggregate_type is not None:
                return node, aggregate_type
        return None, None

    def hlsl_aggregate_conditional_error(
        self,
        expr,
        *,
        aggregate_type,
        context,
        reason,
        detail,
    ):
        type_name = self.map_type(aggregate_type) if aggregate_type else None
        display_type = type_name or "<unresolved>"
        raise DirectXAggregateConditionalError(
            "DirectX cannot lower aggregate conditional value "
            f"'{display_type}' in {context}: {detail}",
            aggregate_type=type_name,
            target="HLSL",
            context=context,
            reason=reason,
            source_location=getattr(expr, "source_location", None),
        )

    def hlsl_expression_has_observable_side_effects(
        self,
        expr,
        *,
        allow_integer_constructors=False,
        allow_bfloat_constructors=False,
        allow_numeric_constructors=False,
        allow_buffer_loads=False,
    ):
        for node in self.walk_ast(expr):
            if isinstance(node, AssignmentNode):
                return True
            if isinstance(node, UnaryOpNode) and getattr(node, "op", None) in {
                "++",
                "--",
            }:
                return True
            if isinstance(node, FunctionCallNode) or (
                hasattr(node, "__class__") and "FunctionCall" in str(node.__class__)
            ):
                function = getattr(node, "function", getattr(node, "name", None))
                arguments = getattr(node, "arguments", getattr(node, "args", [])) or []
                if allow_numeric_constructors and len(arguments) == 1:
                    numeric_type = numeric_trait_method_result_type(self, node)
                    if (
                        numeric_type is not None
                        and self.value_component_count(numeric_type) is not None
                        and self.value_component_count(
                            self.expression_result_type(arguments[0])
                        )
                        is not None
                    ):
                        # Numeric trait methods lower to operators, not calls.
                        # The walk still checks receiver and argument effects.
                        continue
                name = getattr(function, "name", function)
                definition = self.current_hlsl_available_functions.get(name)
                if getattr(definition, "resource_aggregate_nonmutating", False):
                    # Lowering proves handle reads and private-value helpers.
                    # The walk still checks all argument evaluation below them.
                    continue
                if (
                    allow_buffer_loads
                    and name == "buffer_load"
                    and len(arguments) == 2
                    and not self.hlsl_function_name_is_shadowed(name)
                ):
                    # A buffer load reads one element and writes nothing. The
                    # walk still checks the buffer and index expressions.
                    continue
                if (
                    isinstance(name, str)
                    and not self.hlsl_function_name_is_shadowed(name)
                    and (
                        (
                            allow_integer_constructors
                            and self.hlsl_integer_arithmetic_type_info(name) is not None
                        )
                        or (
                            allow_numeric_constructors
                            and self.value_component_count(name) is not None
                        )
                        or (
                            allow_bfloat_constructors
                            and self.is_hlsl_bfloat16_type(name)
                        )
                    )
                ):
                    continue
                return True
        return False

    def render_hlsl_statement_value_expression(
        self, expr, expected_type, indent, *, context
    ):
        aggregate_value = self.render_hlsl_aggregate_conditional_value_expression(
            expr,
            expected_type,
            indent,
            context=context,
        )
        if aggregate_value is not None:
            return aggregate_value

        if self.hlsl_expression_contains_typed_buffer_atomic(expr):
            return self.render_hlsl_typed_buffer_atomic_value_expression(
                expr, expected_type, indent
            )

        lifted_atomic = self.hlsl_typed_buffer_atomic_lifted_expression(
            expr, expected_type
        )
        if lifted_atomic is not None:
            statements, rendered = lifted_atomic
            return self.generate_statement_code("\n".join(statements), indent), rendered

        return "", self.generate_expression_with_expected(expr, expected_type)

    def hlsl_aggregate_conditional_declaration(self, type_name, variable_name):
        return format_c_style_array_declaration(self.map_type(type_name), variable_name)

    def render_hlsl_aggregate_array_copy(
        self,
        target,
        source,
        expected_type,
        root_expr,
        indent,
        *,
        context,
    ):
        try:
            _mapped_type, extent, element_type = (
                self.hlsl_fixed_array_aggregate_contract(root_expr, expected_type)
            )
        except DirectXAggregateInitializerError as error:
            self.hlsl_aggregate_conditional_error(
                root_expr,
                aggregate_type=expected_type,
                context=context,
                reason="array-shape-unresolved",
                detail=str(error),
            )

        code = ""
        for index in range(extent):
            child_target = f"{target}[{index}]"
            child_source = f"{source}[{index}]"
            if self.hlsl_outer_array_type(element_type) is not None:
                code += self.render_hlsl_aggregate_array_copy(
                    child_target,
                    child_source,
                    element_type,
                    root_expr,
                    indent,
                    context=context,
                )
            else:
                code += f"{'    ' * indent}{child_target} = {child_source};\n"
        return code

    def render_hlsl_aggregate_conditional_branch_assignment(
        self,
        expr,
        target,
        expected_type,
        indent,
        *,
        context,
    ):
        if self.hlsl_aggregate_conditional_type(expr, expected_type) is not None:
            return self.generate_hlsl_aggregate_conditional_assignment(
                expr,
                target,
                "=",
                expected_type,
                indent,
                context=context,
            )

        if self.hlsl_outer_array_type(expected_type) is not None:
            if isinstance(expr, ArrayLiteralNode):
                list_assignment = generate_hlsl_array_list_assignment(
                    self, target, expr, expected_type
                )
                if list_assignment is not None:
                    return self.generate_statement_code(list_assignment, indent)
                statements = self.hlsl_aggregate_assignment_statements(
                    target,
                    expr,
                    expected_type,
                    expr,
                )
                return self.generate_statement_code("\n".join(statements), indent)
            if not self.hlsl_repeatable_array_assignment_target(expr):
                self.hlsl_aggregate_conditional_error(
                    expr,
                    aggregate_type=expected_type,
                    context=context,
                    reason="array-branch-evaluation",
                    detail=(
                        "the selected array branch cannot be evaluated once and "
                        "copied element by element"
                    ),
                )
            source = self.generate_expression(expr)
            return self.render_hlsl_aggregate_array_copy(
                target,
                source,
                expected_type,
                expr,
                indent,
                context=context,
            )

        struct_initialization = self.render_hlsl_struct_value_initialization(
            target, expected_type, expr, indent
        )
        if struct_initialization is not None:
            return struct_initialization

        code, rendered = self.render_hlsl_statement_value_expression(
            expr,
            expected_type,
            indent,
            context=context,
        )
        return f"{code}{'    ' * indent}{target} = {rendered};\n"

    def generate_hlsl_aggregate_conditional_assignment(
        self,
        expr,
        target,
        operator,
        expected_type,
        indent,
        *,
        context,
    ):
        aggregate_type = self.hlsl_aggregate_conditional_type(expr, expected_type)
        if aggregate_type is None:
            return None
        if operator != "=":
            self.hlsl_aggregate_conditional_error(
                expr,
                aggregate_type=aggregate_type,
                context=context,
                reason="unsupported-assignment-operator",
                detail=f"operator '{operator}' has no aggregate assignment contract",
            )

        condition = getattr(expr, "condition", None)
        condition_code, rendered_condition = (
            self.render_hlsl_statement_value_expression(
                condition,
                "bool",
                indent,
                context="condition",
            )
        )
        indent_str = "    " * indent
        code = condition_code
        code += f"{indent_str}if ({self.hlsl_strip_wrapping_parentheses(rendered_condition)}) {{\n"
        code += self.render_hlsl_aggregate_conditional_branch_assignment(
            getattr(expr, "true_expr", None),
            target,
            aggregate_type,
            indent + 1,
            context=context,
        )
        code += f"{indent_str}}} else {{\n"
        code += self.render_hlsl_aggregate_conditional_branch_assignment(
            getattr(expr, "false_expr", None),
            target,
            aggregate_type,
            indent + 1,
            context=context,
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_hlsl_aggregate_conditional_initialization(
        self,
        expr,
        declaration,
        target,
        expected_type,
        indent,
    ):
        aggregate_type = self.hlsl_aggregate_conditional_type(expr, expected_type)
        if aggregate_type is None:
            return None
        indent_str = "    " * indent
        code = f"{indent_str}{declaration};\n"
        code += self.generate_hlsl_aggregate_conditional_assignment(
            expr,
            target,
            "=",
            aggregate_type,
            indent,
            context="initializer",
        )
        return code

    def generate_hlsl_aggregate_conditional_assignment_statement(self, stmt, indent):
        if hasattr(stmt, "target") and hasattr(stmt, "value"):
            target = stmt.target
            value = stmt.value
            operator = getattr(stmt, "operator", "=")
        else:
            target = stmt.left
            value = stmt.right
            operator = getattr(stmt, "operator", "=")

        record_transfer = self.generate_hlsl_workgroup_record_transfer(
            target, value, operator, statement_context=True
        )
        if record_transfer is not None:
            return self.generate_statement_code(record_transfer, indent)
        target_type = self.expression_result_type(target)
        if self.hlsl_aggregate_conditional_type(value, target_type) is None:
            materialized = self.render_hlsl_aggregate_conditional_value_expression(
                value, target_type, indent, context="assignment"
            )
            if materialized is None:
                return None
            if operator != "=" or self.hlsl_expression_has_observable_side_effects(
                target
            ):
                self.hlsl_aggregate_error(
                    value,
                    expected_type=target_type,
                    reason="field-evaluation-order-unproven",
                    detail="materializing constructor fields would reorder the assignment target",
                )
            code, rendered = materialized
            return (
                code
                + f"{'    ' * indent}{self.generate_expression(target)} = {rendered};\n"
            )
        return self.generate_hlsl_aggregate_conditional_assignment(
            value,
            self.generate_expression(target),
            operator,
            target_type,
            indent,
            context="assignment",
        )

    def generate_hlsl_aggregate_conditional_return(self, expr, indent=0):
        aggregate_type = self.hlsl_aggregate_conditional_type(
            expr, self.current_function_return_type
        )
        if aggregate_type is None:
            return None

        condition_code, rendered_condition = (
            self.render_hlsl_statement_value_expression(
                getattr(expr, "condition", None),
                "bool",
                indent,
                context="condition",
            )
        )
        indent_str = "    " * indent
        code = condition_code
        code += f"{indent_str}if ({self.hlsl_strip_wrapping_parentheses(rendered_condition)}) {{\n"
        code += self.generate_hlsl_aggregate_conditional_return_branch(
            getattr(expr, "true_expr", None), aggregate_type, indent + 1
        )
        code += f"{indent_str}}} else {{\n"
        code += self.generate_hlsl_aggregate_conditional_return_branch(
            getattr(expr, "false_expr", None), aggregate_type, indent + 1
        )
        code += f"{indent_str}}}\n"
        return code

    def generate_hlsl_aggregate_conditional_return_branch(
        self, expr, expected_type, indent
    ):
        nested = self.generate_hlsl_aggregate_conditional_return(expr, indent)
        if nested is not None:
            return nested

        struct_return = self.render_hlsl_struct_return(expr, indent)
        if struct_return is not None:
            return struct_return

        code, rendered = self.render_hlsl_statement_value_expression(
            expr,
            expected_type,
            indent,
            context="return",
        )
        return code + self.generate_hlsl_return_value_statement(rendered, indent)
