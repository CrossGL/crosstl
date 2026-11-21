"""Module assembly in the SPIR-V generator: entry points, the SPIR-V module version, module and function body line order, SPIR-V 1.4 workgroup interfaces, generic function refusals, concrete overload checks and generate."""

import re
from typing import List

from ..ast import (
    FunctionCallNode,
    ShaderNode,
)
from ..refusals import (
    TranslatorArgumentError,
    supersedes_pending_refusals,
)
from ..semantics.constant_buffers import lower_constant_buffer_resources
from .call_graph import validate_recursion_free
from .fixed_sampler_contracts import (
    fixed_sampler_records,
    with_spirv_fixed_samplers,
)
from .generic_function_utils import (
    collect_unresolved_generic_function_call_names,
    generic_function_call_name,
    generic_function_definition_name,
    generic_function_parameters,
    iter_function_nodes,
)
from .relational_builtins import lower_relational_builtins
from .resource_aggregates import lower_resource_aggregates
from .spirv_builtin_overloads import separate_builtin_overloads
from .spirv_compute_derivative_groups import (
    declared_derivative_group,
    emit_compute_derivative_group,
    emit_declared_derivative_group,
)
from .spirv_dead_types import drop_unused_runtime_array_structs
from .spirv_entry_interfaces import complete_entry_point_interfaces
from .spirv_entry_workgroup_sizes import emit_entry_local_size_id
from .spirv_geometry_streams import (
    geometry_output_streams,
    stream_decorations,
)
from .spirv_per_vertex_builtins import lower_per_vertex_builtin_arrays
from .spirv_specialized_workgroup_size import declare_specialized_workgroup_size
from .spirv_union_storage import lower_union_storage
from .spirv_values import (
    SpirvId,
    UnsupportedSPIRVFeatureError,
)
from .stage_utils import requests_early_fragment_tests
from .workgroup_size_extents import fold_workgroup_size_extents


class SpirvModuleAssemblyMixin:
    """Module assembly members of ``VulkanSPIRVCodeGen``."""

    def emit_entry_point(
        self, execution_model: str, function_id: SpirvId, name: str, stage=None
    ):
        """Emit SPIR-V entry-point and execution-mode declarations."""
        interface_variables = list(
            self.function_interface_variables.get(function_id.id, [])
        )
        interface_variables.extend(self.entry_point_private_variables)
        if execution_model == "TaskEXT":
            task_payload = self.task_payload_interface_by_function.get(function_id.id)
            if task_payload is not None:
                interface_variables.append(task_payload)

        seen_interface_ids = set()
        interface_ids = []
        for variable in interface_variables:
            if variable.type.storage_class == "Workgroup":
                continue
            if variable.id in seen_interface_ids:
                continue
            seen_interface_ids.add(variable.id)
            interface_ids.append(f"%{variable.id}")

        interface_ids = " ".join(interface_ids)
        interface_suffix = f" {interface_ids}" if interface_ids else ""
        self.emit(
            f'OpEntryPoint {execution_model} %{function_id.id} "{name}"'
            f"{interface_suffix}"
        )
        if execution_model == "Fragment":
            self.emit(f"OpExecutionMode %{function_id.id} OriginUpperLeft")
            entry_function = getattr(stage, "entry_point", None)
            if requests_early_fragment_tests(entry_function, stage):
                self.emit(f"OpExecutionMode %{function_id.id} EarlyFragmentTests")
            # A stage-level output variable is created before its entry function, so the interface, not the
            # function being generated when the built-in was registered, says which outputs the entry writes.
            interface_builtins = {
                self.builtin_names_by_variable_id.get(variable.id)
                for variable in interface_variables
                if variable.type.storage_class == "Output"
            }
            if (
                function_id.id in self.fragment_depth_replacing_function_ids
                or "FragDepth" in interface_builtins
            ):
                self.emit(f"OpExecutionMode %{function_id.id} DepthReplacing")
            if (
                function_id.id in self.fragment_stencil_ref_replacing_function_ids
                or "FragStencilRefEXT" in interface_builtins
            ):
                self.emit(f"OpExecutionMode %{function_id.id} StencilRefReplacingEXT")
        elif execution_model == "Geometry":
            self.emit_geometry_execution_modes(function_id, stage)
        elif execution_model in {"TessellationControl", "TessellationEvaluation"}:
            self.emit_tessellation_execution_modes(execution_model, function_id, stage)
        elif execution_model in {"GLCompute", "MeshEXT", "TaskEXT"}:
            declared_group = declared_derivative_group(self, stage)
            declared = declared_group is not None
            size = None
            if not emit_entry_local_size_id(self, function_id, stage, declared):
                specialized = declare_specialized_workgroup_size(self, stage, declared)
                x, y, z = self.compute_local_size(stage)
                self.emit(f"OpExecutionMode %{function_id.id} LocalSize {x} {y} {z}")
                size = None if specialized else (x, y, z)
                if not declared and self.requires_compute_derivatives:
                    emit_compute_derivative_group(self, function_id, (x, y, z))
            if declared:
                emit_declared_derivative_group(self, function_id, declared_group, size)
            if execution_model in {"MeshEXT", "TaskEXT"}:
                self.require_capability("MeshShadingEXT")
                self.require_extension("SPV_EXT_mesh_shader")
            if execution_model == "MeshEXT":
                max_vertices, max_primitives = self.mesh_stage_output_limits(
                    function_id, stage
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} OutputVertices {max_vertices}"
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} "
                    f"OutputPrimitivesEXT {max_primitives}"
                )
                self.emit(
                    f"OpExecutionMode %{function_id.id} "
                    f"{self.mesh_stage_topology_mode(stage)}"
                )
        elif execution_model in {
            "RayGenerationKHR",
            "IntersectionKHR",
            "AnyHitKHR",
            "ClosestHitKHR",
            "MissKHR",
            "CallableKHR",
        }:
            self.require_capability("RayTracingKHR")
            self.require_extension("SPV_KHR_ray_tracing")

    def spirv_module_version(self) -> str:
        if self.include_resource_interface_variables:
            return "1.4"
        if (
            set(getattr(self, "program_execution_models", ()) or ())
            & self.SPIRV_1_4_EXECUTION_MODELS
        ):
            return "1.4"
        if "MeshShadingEXT" in self.required_capabilities:
            return "1.4"
        if "RayTracingKHR" in self.required_capabilities:
            return "1.4"
        if self.cooperative_matrix_khr_enabled:
            return "1.3"
        if any(
            capability.startswith("GroupNonUniform")
            for capability in self.required_capabilities
        ):
            return "1.3"
        if self.uses_local_size_id:
            return "1.2"
        return "1.0"

    def ordered_module_lines(self) -> List[str]:
        """Return SPIR-V assembly lines in logical module-layout order."""
        header_lines = self.code_lines[:3]
        if len(header_lines) > 1:
            header_lines[1] = f"; Version: {self.spirv_module_version()}"
        bound_line = f"; Bound: {self.next_id}"
        raw_lines = self.code_lines[4:]

        capabilities = ["OpCapability Shader"] + [
            f"OpCapability {capability}"
            for capability in sorted(self.required_capabilities)
        ]
        extensions = [
            f'OpExtension "{extension}"'
            for extension in sorted(self.required_extensions)
        ]
        imports = []
        memory_model = []
        entry_points = []
        execution_modes = []
        debug_names = list(getattr(self, "debug_strings", ())) + [
            f'OpSourceExtension "{extension}"'
            for extension in sorted(getattr(self, "source_extensions", ()))
        ]
        annotations = []
        declarations = []
        global_variables = []
        body = []

        for line in raw_lines:
            if line.startswith(self.POINTER_OFFSET_RESET_MARKER_PREFIX):
                continue
            if line.startswith("OpCapability "):
                if line not in capabilities:
                    capabilities.append(line)
            elif line.startswith("OpExtension "):
                if line not in extensions:
                    extensions.append(line)
            elif " = OpExtInstImport " in line:
                imports.append(line)
            elif line.startswith("OpMemoryModel "):
                if "VulkanMemoryModel" in self.required_capabilities:
                    memory_model.append("OpMemoryModel Logical VulkanKHR")
                else:
                    memory_model.append(line)
            elif line.startswith("OpEntryPoint "):
                entry_points.append(line)
            elif line.startswith("OpExecutionMode"):
                execution_modes.append(line)
            elif line.startswith(("OpName ", "OpMemberName ", "OpString ", "OpLine ")):
                debug_names.append(line)
            elif line.startswith(("OpDecorate ", "OpMemberDecorate ")):
                annotations.append(line)
            elif re.match(r"%\d+ = (OpType|OpConstant|OpSpecConstant|OpUndef)", line):
                declarations.append(line)
            elif re.match(r"%\d+ = OpVariable %\d+ (?!Function\b)", line):
                global_variables.append(line)
            else:
                body.append(line)

        if self.spirv_module_version() == "1.4":
            entry_points = complete_entry_point_interfaces(
                [
                    self.append_spirv_1_4_workgroup_interfaces(line)
                    for line in entry_points
                ],
                global_variables,
                body,
            )

        annotations.extend(self.decorations)
        if getattr(self, "geometry_streams_used", False):
            stream_annotations, unplaced = stream_decorations(
                global_variables, debug_names, self.geometry_output_streams
            )
            annotations.extend(stream_annotations)
            for name in unplaced:
                self.refuse_lowering(
                    f"SPIR-V geometry output '{name}' names a vertex stream, but no "
                    "output variable of that name was declared"
                )

        def unique(lines: List[str]) -> List[str]:
            seen = set()
            result = []
            for line in lines:
                if line in seen:
                    continue
                seen.add(line)
                result.append(line)
            return result

        body = self.ordered_function_body_lines(body)
        debug_names, annotations, declarations = drop_unused_runtime_array_structs(
            debug_names,
            annotations,
            declarations,
            entry_points + execution_modes + global_variables + body,
        )

        return (
            header_lines
            + [bound_line]
            + unique(capabilities)
            + unique(extensions)
            + imports
            + memory_model
            + entry_points
            + execution_modes
            + unique(debug_names)
            + unique(annotations)
            + declarations
            + global_variables
            + body
        )

    def append_spirv_1_4_workgroup_interfaces(self, entry_point: str) -> str:
        match = re.match(r"OpEntryPoint\s+\w+\s+%(\d+)\s+", entry_point)
        if match is None:
            return entry_point
        function_id = int(match.group(1))
        existing_ids = set(re.findall(r"%\d+", entry_point))
        workgroup_ids = [
            f"%{variable.id}"
            for variable in self.function_interface_variables.get(function_id, [])
            if variable.type.storage_class == "Workgroup"
            and f"%{variable.id}" not in existing_ids
        ]
        if not workgroup_ids:
            return entry_point
        return f"{entry_point} {' '.join(workgroup_ids)}"

    def ordered_function_body_lines(self, lines: List[str]) -> List[str]:
        """Move function-scope variables into each function's first block."""
        ordered = []
        in_function = False
        first_block_insert_at = None
        pending_variables = []

        def is_function_variable(line: str) -> bool:
            return re.match(r"%\d+ = OpVariable %\d+ Function\b", line) is not None

        def is_function_start(line: str) -> bool:
            return re.match(r"%\d+ = OpFunction\b", line) is not None

        def is_label(line: str) -> bool:
            return re.match(r"%\d+ = OpLabel\b", line) is not None

        for line in lines:
            if is_function_start(line):
                in_function = True
                first_block_insert_at = None
                pending_variables = []
                ordered.append(line)
                continue

            if in_function and line == "OpFunctionEnd":
                if pending_variables:
                    if first_block_insert_at is None:
                        ordered.extend(pending_variables)
                    else:
                        ordered[first_block_insert_at:first_block_insert_at] = (
                            pending_variables
                        )
                ordered.append(line)
                in_function = False
                first_block_insert_at = None
                pending_variables = []
                continue

            if in_function and is_function_variable(line):
                if first_block_insert_at is None:
                    pending_variables.append(line)
                else:
                    ordered.insert(first_block_insert_at, line)
                    first_block_insert_at += 1
                continue

            ordered.append(line)

            if in_function and first_block_insert_at is None and is_label(line):
                first_block_insert_at = len(ordered)
                if pending_variables:
                    ordered[first_block_insert_at:first_block_insert_at] = (
                        pending_variables
                    )
                    first_block_insert_at += len(pending_variables)
                    pending_variables = []

        return ordered

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions that have no concrete SPIR-V specialization."""
        functions = list(iter_function_nodes(ast_node)) + list(
            (self.generic_function_specializations or {}).values()
        )
        for func_name in collect_unresolved_generic_function_call_names(
            self, functions
        ):
            candidates = (self.generic_function_definitions or {}).get(func_name) or []
            if not candidates:
                continue
            if self._unresolved_generic_calls_are_inlineable(func_name, functions):
                continue
            raise self.unsupported_generic_function_error(candidates[0])

    def _unresolved_generic_calls_are_inlineable(self, func_name, functions):
        """Whether storage-buffer inlining resolves every unresolved call of ``func_name``.

        Such calls may carry extra arguments the inliner skips (generated scratch locals), so arity alone
        does not leave them unresolved: the inliner's subsequence matcher aligns them with an overload. A
        call it cannot align is still unresolved and must be refused.
        """
        candidates = (self.generic_function_definitions or {}).get(func_name) or []
        if not candidates:
            return False
        resolved_call_ids = getattr(self, "generic_function_resolved_call_ids", set())
        for func in functions:
            if generic_function_parameters(func):
                continue
            for node in self.walk_ast_nodes(func):
                if not isinstance(node, FunctionCallNode):
                    continue
                call_name = self.function_call_name(node)
                if call_name is None:
                    continue
                if generic_function_definition_name(self, call_name) != func_name:
                    continue
                if id(node) in resolved_call_ids:
                    continue
                args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
                if generic_function_call_name(self, call_name, args) is not None:
                    continue
                if not any(
                    self._storage_buffer_candidate_matches(candidate, args)
                    for candidate in candidates
                ):
                    return False
        return True

    def _storage_buffer_candidate_matches(self, candidate, args):
        try:
            match, _reason = self.storage_buffer_candidate_call_match(candidate, args)
        except Exception:
            return False
        return match is not None

    def has_concrete_function_overload(self, function_node, functions):
        function_name = getattr(function_node, "name", None)
        if not function_name:
            return False
        return any(
            other is not function_node
            and getattr(other, "name", None) == function_name
            and not generic_function_parameters(other)
            for other in functions
        )

    def generate(self, ast):
        """Generate SPIR-V code from a CrossGL AST.

        Raises :class:`SpirvLoweringRefused` listing every construct that has no faithful lowering.
        """
        if not isinstance(ast, ShaderNode):
            raise TranslatorArgumentError(
                f"SPIR-V generation needs a ShaderNode, got {type(ast).__name__}"
            )

        ast = lower_union_storage(ast, "vulkan")
        source_ast, fixed_samplers = ast, fixed_sampler_records(ast)
        validate_recursion_free(ast, "vulkan")
        separate_builtin_overloads(ast)
        lower_relational_builtins(ast)
        lower_per_vertex_builtin_arrays(ast, self)
        self.reset_generation_state()
        self.geometry_output_streams = geometry_output_streams(ast)
        ast = lower_constant_buffer_resources(
            ast,
            lambda name: self.refuse_lowering(
                f"array of arrays of constant buffers '{name}': a SPIR-V descriptor array has one dimension"
            ),
            block_arrays=True,
        )
        ast = lower_resource_aggregates(
            ast,
            storage_pointer_parameters=True,
            workgroup_access_assertions=getattr(
                self, "workgroup_access_assertions", ()
            ),
        )
        ast = fold_workgroup_size_extents(ast, "SPIR-V", self.refuse_lowering)
        self.spirv_program_ast = ast
        self.per_vertex_program_stages = list(
            (getattr(ast, "stages", None) or {}).items()
        )
        self.program_execution_models = {
            self.spirv_execution_model(self.stage_key(stage_type))
            for stage_type in getattr(ast, "stages", None) or {}
        }
        try:
            module = self._generate_module(ast)
        except Exception as error:
            if isinstance(
                error, UnsupportedSPIRVFeatureError
            ) or supersedes_pending_refusals(error):
                raise
            # A placeholder left by an earlier refusal can break later lowering; report the refusal.
            self.refusals.raise_pending(cause=error)
            raise
        self.refusals.raise_pending()
        return with_spirv_fixed_samplers(module, fixed_samplers, source_ast)
