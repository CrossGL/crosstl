"""Cooperative groups and asynchronous copies in the CUDA to CrossGL converter."""

from ..CudaAst import (
    CastNode,
    FunctionCallNode,
    MemberAccessNode,
    UnaryOpNode,
)


class CudaImportCooperativeGroupMixin:
    """Cooperative group members of ``CudaToCrossGLConverter``."""

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
        if base_name in {"memcpy_async", "wait", "wait_prior"} and node.args:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_member_call(
                    group_metadata, base_name, node.args[1:]
                )
        if base_name == "reduce" and node.args:
            group_metadata = self.resolve_cooperative_group_metadata(node.args[0])
            if group_metadata is not None:
                return self.format_cooperative_group_reduce_call(
                    group_metadata, node.args[1:]
                )
        return None

    def cooperative_group_reduce_operator(self, expression):
        if not isinstance(expression, FunctionCallNode) or expression.args:
            return None

        if not isinstance(expression.name, str):
            return None

        base_call_name, _ = self.parse_cpp_template(expression.name)
        return self.cooperative_group_base_name(base_call_name)

    def format_thread_block_size_expression(self):
        return "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z)"

    def format_unsupported_cooperative_group_statement(
        self, group_kind, member, args=None
    ):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        return self.refuse_import(
            f"CUDA cooperative_groups {group_kind}.{member}({formatted_args}) has no CrossGL form"
        )

    def format_unsupported_cooperative_group_expression(
        self, group_kind, member, fallback="0", args=None
    ):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        return self.refuse_import(
            f"CUDA cooperative_groups {group_kind}.{member}({formatted_args}) has no CrossGL form",
            fallback,
        )

    def format_cuda_async_sync_call(self, node):
        if isinstance(node.name, MemberAccessNode):
            metadata = self.resolve_cuda_async_sync_metadata(node.name.object)
            if metadata is None:
                return None
            return self.format_cuda_async_sync_member_call(
                metadata, node.name.member, node.args
            )

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if self.is_user_defined_function(base_name) or self.is_user_defined_function(
            raw_name
        ):
            return None

        if base_name == "memcpy_async":
            metadata = self.resolve_cuda_async_sync_metadata_from_args(node.args)
            if metadata is not None:
                return self.format_unsupported_cuda_async_statement(
                    metadata["kind"], base_name, node.args
                )

        if base_name == "init" and node.args:
            metadata = self.resolve_cuda_async_sync_metadata(
                self.unwrap_cuda_async_sync_target(node.args[0])
            )
            if metadata is not None and metadata["kind"] == "barrier":
                return self.format_unsupported_cuda_async_statement(
                    metadata["kind"], base_name, node.args
                )

        if base_name == "make_pipeline":
            return self.format_unsupported_cuda_async_expression("pipeline", base_name)

        primitive_member = self.cuda_pipeline_primitive_member_name(base_name)
        if primitive_member is not None:
            return self.format_unsupported_cuda_async_statement(
                "pipeline", primitive_member, node.args
            )

        return None

    def cuda_pipeline_primitive_member_name(self, base_name):
        primitive_mapping = {
            "__pipeline_memcpy_async": "memcpy_async",
            "__pipeline_commit": "commit",
            "__pipeline_wait_prior": "wait_prior",
            "__pipeline_arrive_on": "arrive_on",
        }
        return primitive_mapping.get(base_name)

    def format_cuda_async_sync_member_call(self, metadata, member, args):
        kind = metadata["kind"]
        member_base_name, _ = self.parse_cpp_template(member)
        member_name = self.cooperative_group_base_name(member_base_name) or member

        if kind == "barrier":
            if member_name in {"arrive", "arrival_token"}:
                return self.format_unsupported_cuda_async_expression(kind, member_name)
            if member_name in {"try_wait", "try_wait_parity"}:
                return self.format_unsupported_cuda_async_expression(
                    kind, member_name, "false"
                )
            return self.format_unsupported_cuda_async_statement(kind, member_name, args)

        if kind == "pipeline":
            return self.format_unsupported_cuda_async_statement(kind, member_name, args)

        return self.format_unsupported_cuda_async_statement(kind, member_name, args)

    def format_unsupported_cuda_async_statement(self, kind, member, args=None):
        args = args or []
        formatted_args = ", ".join(self.visit(arg) for arg in args)
        return self.refuse_import(
            f"CUDA {kind}.{member}({formatted_args}) has no CrossGL form"
        )

    def format_unsupported_cuda_async_expression(self, kind, member, fallback="0"):
        return self.refuse_import(f"CUDA {kind}.{member} has no CrossGL form", fallback)

    def format_cuda_async_sync_declaration(self, name, metadata):
        return self.refuse_import(
            f"CUDA {metadata['kind']} '{name}': CrossGL has no asynchronous copy object"
        )

    def cuda_async_sync_declaration_metadata(self, node):
        declared_metadata = self.cuda_async_sync_metadata_from_type(node.vtype)
        factory_metadata = self.cuda_async_sync_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cuda_async_sync_metadata_from_type(self, type_name):
        type_name = self.strip_cuda_async_sync_declarator(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(base_type)
        if base_name in {"barrier", "pipeline", "pipeline_shared_state"}:
            metadata = {"kind": base_name}
            if template_args:
                metadata["scope"] = template_args[0]
            return metadata
        return None

    def cuda_async_sync_factory_metadata(self, value):
        if not isinstance(value, FunctionCallNode):
            return None
        raw_name = value.name if isinstance(value.name, str) else self.visit(value.name)
        base_call_name, _ = self.parse_cpp_template(raw_name)
        base_name = self.cooperative_group_base_name(base_call_name)
        if base_name == "make_pipeline":
            return {"kind": "pipeline"}
        return None

    def resolve_cuda_async_sync_metadata_from_args(self, args):
        for arg in reversed(args):
            metadata = self.resolve_cuda_async_sync_metadata(arg)
            if metadata is not None and metadata["kind"] in {"barrier", "pipeline"}:
                return metadata
        return None

    def resolve_cuda_async_sync_metadata(self, expression):
        name = self.simple_identifier(expression)
        if name is None:
            return None
        return self.lookup_cuda_async_sync_metadata(name)

    def unwrap_cuda_async_sync_target(self, expression):
        if isinstance(expression, UnaryOpNode) and expression.op == "&":
            return expression.operand
        if isinstance(expression, CastNode):
            return self.unwrap_cuda_async_sync_target(expression.expression)
        return expression

    def strip_cuda_async_sync_declarator(self, type_name):
        type_name = self.strip_type_qualifiers(type_name).strip()
        if "[" in type_name:
            type_name = type_name.split("[", 1)[0].strip()

        while True:
            stripped = type_name.strip()
            for suffix in ("&&", "&", "*"):
                if stripped.endswith(suffix):
                    type_name = stripped[: -len(suffix)].strip()
                    break
            else:
                return stripped

    def cooperative_group_declaration_metadata(self, node):
        declared_metadata = self.cooperative_group_metadata_from_type(node.vtype)
        factory_metadata = self.cooperative_group_factory_metadata(node.value)
        return factory_metadata or declared_metadata

    def cooperative_group_kind_from_type(self, type_name):
        metadata = self.cooperative_group_metadata_from_type(type_name)
        return metadata["kind"] if metadata is not None else None

    def cooperative_group_metadata_from_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        base_type, template_args = self.parse_cpp_template(type_name)
        base_name = self.cooperative_group_base_name(type_name)
        if base_name in {
            "thread_group",
            "thread_block",
            "grid_group",
            "multi_grid_group",
            "coalesced_group",
            "cluster_group",
        }:
            return {"kind": base_name}
        base_name = self.cooperative_group_base_name(base_type)
        if base_name and base_name.startswith("thread_block_tile"):
            metadata = {"kind": "thread_block_tile"}
            if template_args:
                metadata["tile_size"] = template_args[0]
            return metadata
        return None

    def cooperative_group_factory_kind(self, value):
        metadata = self.cooperative_group_factory_metadata(value)
        return metadata["kind"] if metadata is not None else None

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
            "this_cluster": "cluster_group",
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

    def resolve_cooperative_group_expression(self, expression):
        metadata = self.resolve_cooperative_group_metadata(expression)
        return metadata["kind"] if metadata is not None else None

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
