"""Function calls, HIP intrinsics, warp functions and cooperative groups in the HIP to CrossGL converter."""

import re

from ...CUDA.directed_rounding import refuse_directed_rounding
from ..HipAst import (
    FunctionCallNode,
    MemberAccessNode,
)


class HipIntrinsicMixin:
    """Intrinsic members of ``HipToCrossGLConverter``."""

    def visit_FunctionCallNode(self, node):
        self.validate_record_call_operator(node)
        self.reject_unsupported_record_method_call(node)
        self.validate_record_representation_query(node)
        scalar_representation = self.format_fixed_scalar_representation_query(node)
        if scalar_representation is not None:
            return scalar_representation
        self.validate_record_constructor_call(node)
        self.validate_record_function_arguments(node)
        self.validate_record_reference_return_call(node)
        if self.is_get_method_call(node):
            return self.visit(node.name.object)

        if hasattr(node, "name"):
            func_name = node.name
        else:
            func_name = str(node.function) if hasattr(node, "function") else "unknown"

        if not isinstance(func_name, str):
            func_name = self.visit(func_name)

        if func_name == "lambda":
            return self.format_lambda_call(getattr(node, "args", []))

        raw_args = []
        if hasattr(node, "args") and node.args:
            raw_args = node.args
        elif hasattr(node, "arguments") and node.arguments:
            raw_args = node.arguments

        if func_name == "__hip_pack_expand__":
            if raw_args:
                return f"{self.visit(raw_args[0])}..."
            return "..."

        cooperative_call = self.format_cooperative_group_call(node)
        if cooperative_call is not None:
            return cooperative_call

        args = [self.visit(arg) for arg in raw_args]
        args_str = ", ".join(args)

        device_heap_call = self.format_device_heap_call(func_name, args)
        if device_heap_call is not None:
            return device_heap_call

        make_unique = self.format_make_unique_call(func_name, args)
        if make_unique is not None:
            return make_unique

        unique_ptr_init = self.format_unique_ptr_constructor_call(func_name, args)
        if unique_ptr_init is not None:
            return unique_ptr_init

        numeric_limits_call = self.format_std_numeric_limits_call(func_name, args)
        if numeric_limits_call is not None:
            return numeric_limits_call

        hip_intrinsic = self.format_hip_intrinsic_call(func_name, args)
        if hip_intrinsic is not None:
            return hip_intrinsic

        user_defined_call_name = self.format_user_defined_function_call_name(func_name)
        if user_defined_call_name is not None:
            return f"{user_defined_call_name}({args_str})"
        if (
            self.is_simple_identifier(func_name)
            and func_name in self.record_definitions
        ):
            # ``let{1.0f}`` and ``let()`` construct the record under its declared name.
            return f"{self.convert_hip_type_to_crossgl(func_name)}({args_str})"

        timer_intrinsic = self.format_hip_timer_intrinsic_call(func_name, args)
        if timer_intrinsic is not None:
            return timer_intrinsic

        load_cache_intrinsic = self.format_hip_load_cache_intrinsic_call(
            func_name, raw_args, args
        )
        if load_cache_intrinsic is not None:
            return load_cache_intrinsic

        runtime_expression = self.format_hip_runtime_expression_call(node, args)
        if runtime_expression is not None:
            return runtime_expression

        resource_call = self.format_hip_resource_call(func_name, args, raw_args)
        if resource_call is not None:
            return resource_call

        warp_intrinsic = self.format_hip_warp_intrinsic_call(func_name, args)
        if warp_intrinsic is not None:
            return warp_intrinsic

        dim3_constructor = self.format_hip_dim3_constructor_call(func_name, args)
        if dim3_constructor is not None:
            return dim3_constructor

        alias_constructor = self.format_concrete_alias_constructor_call(
            func_name, args_str
        )
        if alias_constructor is not None:
            return alias_constructor

        # Convert HIP built-in functions
        crossgl_func = self.convert_hip_builtin_function(func_name)
        if (
            crossgl_func == func_name
            and isinstance(func_name, str)
            and func_name.startswith("::")
            and not self.is_simple_identifier(crossgl_func)
        ):
            crossgl_func = self.format_function_declaration_name(func_name)
        return f"{crossgl_func}({args_str})"

    def format_device_heap_call(self, function_name, args):
        """Refuse ``malloc``/``free`` in device code, like ``delete`` (#4684).

        CrossGL has no device heap, so a device-side heap call has no faithful
        form. Host code keeps the ordinary call; a user-defined ``malloc`` or
        ``free`` is not a heap call either.
        """
        if not self.in_device_code or not isinstance(function_name, str):
            return None
        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name not in {
            "malloc",
            "free",
            "std::malloc",
            "std::free",
        }:
            return None
        if self.is_user_defined_function(
            function_name
        ) or self.is_user_defined_function(normalized_name):
            return None
        return self.refuse_import(
            f"HIP {normalized_name}({', '.join(args)}): CrossGL has no device heap",
            "0",
        )

    def format_concrete_alias_constructor_call(self, func_name, args_str):
        """Write ``real(x)`` through an erased alias as a call of the aliased type.

        Concrete aliases are erased from the CrossGL program (each declaration
        names the aliased type instead), so a functional cast through the alias
        must name that type too or it calls an undeclared function (#3309).
        """
        if not isinstance(func_name, str) or not self.is_simple_identifier(func_name):
            return None
        resolved = self.resolve_concrete_type_alias(func_name)
        if resolved == func_name:
            return None
        return f"{resolved}({args_str})"

    def format_std_numeric_limits_call(self, function_name, args):
        parsed = self.parse_cpp_template_static_member(function_name)
        if parsed is None:
            return None

        base_name, template_args, member_name = parsed
        if base_name not in {
            "std::numeric_limits",
            "cuda::std::numeric_limits",
            "hip::std::numeric_limits",
        }:
            return None
        if len(template_args) != 1 or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", member_name
        ):
            return None

        value_type = self.convert_hip_type_to_crossgl(template_args[0])
        args_str = ", ".join(args)
        return f"numeric_limits_{member_name}<{value_type}>({args_str})"

    def parse_cpp_template_static_member(self, text):
        if not isinstance(text, str):
            return None

        normalized = text[2:] if text.startswith("::") else text
        normalized = self.resolve_namespace_alias_name(normalized)
        scope_index = self.find_last_scope_operator_outside_templates(normalized)
        if scope_index is None:
            return None

        template_name = normalized[:scope_index]
        member_name = normalized[scope_index + 2 :]
        if not member_name:
            return None

        base_name, template_args = self.parse_cpp_template(template_name)
        if not template_args:
            return None
        return base_name, template_args, member_name

    def find_last_scope_operator_outside_templates(self, text):
        depth = 0
        last_scope = None
        index = 0
        while index + 1 < len(text):
            char = text[index]
            if char == "<":
                depth += 1
            elif char == ">":
                depth = max(0, depth - 1)
            elif char == ":" and text[index + 1] == ":" and depth == 0:
                last_scope = index
                index += 1
            index += 1
        return last_scope

    def format_hip_intrinsic_call(self, function_name, args):
        refuse_directed_rounding(function_name)
        fp16_intrinsic = self.format_hip_fp16_intrinsic_call(function_name, args)
        if fp16_intrinsic is not None:
            return fp16_intrinsic

        complex_intrinsic = self.format_hip_complex_intrinsic_call(function_name, args)
        if complex_intrinsic is not None:
            return complex_intrinsic

        type_cast_intrinsic = self.format_hip_type_cast_intrinsic_call(
            function_name, args
        )
        if type_cast_intrinsic is not None:
            return type_cast_intrinsic

        integer_intrinsic = self.format_hip_integer_intrinsic_call(function_name, args)
        if integer_intrinsic is not None:
            return integer_intrinsic

        float_intrinsic = self.format_hip_float_intrinsic_call(function_name, args)
        if float_intrinsic is not None:
            return float_intrinsic

        sync_vote_intrinsic = self.format_hip_sync_vote_intrinsic_call(
            function_name, args
        )
        if sync_vote_intrinsic is not None:
            return sync_vote_intrinsic

        return None

    def is_full_warp_shuffle_width(self, width_args):
        return self.hip_warp_shuffle_width_limit(width_args) is not None

    def hip_warp_shuffle_width_limit(self, width_args):
        if not width_args:
            return "WaveGetLaneCount()"
        if len(width_args) != 1:
            return None
        normalized = self.normalize_warp_mask_expression(width_args[0])
        # HIP widths are powers of two up to warpSize; shuffles work inside width-lane segments.
        if normalized.rstrip("u") in {"1", "2", "4", "8", "16", "32", "64"}:
            return normalized.rstrip("u")
        # warpSize arrives already imported as int(gl_SubgroupSize) (visit_HipBuiltinNode).
        if normalized in {
            "warpsize",
            "warp_size",
            "warp_full_width",
            "gl_subgroupsize",
            "int(gl_subgroupsize)",
        }:
            return "WaveGetLaneCount()"
        return None

    def hip_warp_shuffle_segment_width(self, width_args):
        # HIP's mask-less shuffles take any power-of-two width up to warpSize
        # and work inside width-lane segments (HIP programming guide, "Warp
        # cross-lane functions"), so the narrow widths must import too. Kept
        # apart from hip_warp_shuffle_width_limit so widening it here does not
        # also accept those widths for the _sync family, whose lowering does
        # not segment yet.
        if not width_args:
            return "WaveGetLaneCount()"
        if len(width_args) != 1:
            return None
        normalized = self.normalize_warp_mask_expression(width_args[0])
        constant_width = normalized.rstrip("u")
        if constant_width in {"1", "2", "4", "8", "16", "32", "64"}:
            return constant_width
        if normalized in {
            "warpsize",
            "warp_size",
            "warp_full_width",
            "gl_subgroupsize",
            "int(gl_subgroupsize)",
        }:
            return "WaveGetLaneCount()"
        return None

    def is_full_or_active_warp_mask(self, mask):
        normalized = self.normalize_warp_mask_expression(mask)
        return normalized in {
            "0xffffffff",
            "0xffffffffu",
            "0xfffffffful",
            "0xffffffffffffffff",
            "0xffffffffffffffffu",
            "0xffffffffffffffffull",
            "uint_max",
            "ullong_max",
            "warp_full_mask",
            "full_mask",
            "waveactiveballot(true).x",
        }

    def normalize_warp_mask_expression(self, mask):
        text = str(mask).strip()
        while text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
            if not inner:
                break
            text = inner
        return text.replace(" ", "").lower()

    def format_wave_predicate(self, predicate):
        return f"({predicate} != 0)"

    def format_cooperative_group_call(self, node):
        if isinstance(node.name, MemberAccessNode):
            member = node.name.member
            group_metadata = self.resolve_cooperative_group_metadata(node.name.object)
            if group_metadata is None:
                return None
            return self.format_cooperative_group_member_call(
                group_metadata, member, node.args
            )

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if base_name in {"sync", "thread_rank", "size"} and len(node.args) == 1:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_member_call(
                    group_metadata, base_name, []
                )
        return None

    def format_thread_block_size_expression(self):
        return "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z)"

    def cooperative_group_declaration_metadata(self, node):
        declared_metadata = self.cooperative_group_metadata_from_type(node.vtype)
        factory_metadata = self.cooperative_group_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cooperative_group_metadata_from_type(self, type_name):
        type_name = self.normalize_cooperative_group_type_name(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(type_name)
        if base_name in {
            "thread_group",
            "thread_block",
            "grid_group",
            "multi_grid_group",
            "coalesced_group",
        }:
            return {"kind": base_name}
        base_name = self.cooperative_group_base_name(base_type)
        if base_name and base_name.startswith("thread_block_tile"):
            metadata = {"kind": "thread_block_tile"}
            if template_args:
                metadata["tile_size"] = template_args[0]
            return metadata
        return None

    def normalize_cooperative_group_type_name(self, type_name):
        text = str(type_name).strip()
        text = re.sub(
            r"\b(?:const|volatile|__restrict__|__restrict|restrict)\b", "", text
        )
        text = text.replace("&", " ").replace("*", " ")
        return " ".join(text.split())

    def cooperative_group_factory_metadata(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        raw_name = value.name if isinstance(value.name, str) else self.visit(value.name)
        base_call_name, template_args = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        factory_mapping = {
            "this_thread_block": "thread_block",
            "this_grid": "grid_group",
            "this_multi_grid": "multi_grid_group",
            "coalesced_threads": "coalesced_group",
            "tiled_partition": "thread_block_tile",
        }
        group_kind = factory_mapping.get(base_name)
        if group_kind is None:
            return None

        metadata = {"kind": group_kind}
        if group_kind == "thread_block_tile":
            if template_args:
                metadata["tile_size"] = template_args[0]
            if value.args:
                parent = self.resolve_cooperative_group_metadata(value.args[0])
                if parent is not None:
                    metadata["parent_kind"] = parent["kind"]
        return metadata

    def convert_cooperative_group_type(self, hip_type):
        metadata = self.cooperative_group_metadata_from_type(hip_type)
        if metadata is None:
            return None
        group_kind = metadata["kind"]
        if group_kind == "thread_block_tile" and metadata.get("tile_size"):
            return f"cooperative_groups_thread_block_tile_{metadata['tile_size']}"
        return f"cooperative_groups_{group_kind}"

    def resolve_cooperative_group_metadata(self, expression):
        name = self.simple_identifier(expression)
        group_metadata = self.lookup_cooperative_group_metadata(name)
        if group_metadata is not None:
            return group_metadata
        return self.cooperative_group_factory_metadata(expression)

    def simple_identifier(self, expression):
        if isinstance(expression, str):
            return expression
        return None

    def cooperative_group_base_name(self, name):
        if not isinstance(name, str):
            return None
        return name.rsplit("::", 1)[-1].split("<", 1)[0]
