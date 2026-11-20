"""Software-subgroup contract checks and uniformity proofs for ``HLSLCodeGen``.

When a program asks for a logical subgroup width, the DirectX backend emulates subgroup operations in software.
These methods check the compute entry point and its helpers against that contract, then prove helper by helper,
from callers to callees, the uniformity facts the emitted barriers rely on.
"""

import re
from dataclasses import dataclass

from ..ast import (
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    PointerType,
    ReferenceType,
    UnaryOpNode,
    VariableNode,
)
from .call_effects import builtin_argument_write_indices
from .directx_values import DirectXMappedOverloadError
from .hlsl_host_workgroup_size import hlsl_entry_local_size
from .lane_symmetry import SYMMETRIC, UNIFORM
from .loop_induction import metadata_loop_bindings
from .metadata_call_bounds import (
    call_parameter_bounds,
    inherited_metadata_bindings,
    join_parameter_bounds,
)
from .subgroup_reassignments import followed_offsets
from .uniform_returns import UniformReturnAnalysis, storage_load_call


@dataclass
class SoftwareSubgroupPropagation:
    """The call graph facts and the per-helper facts the uniformity proof passes from callers to callees."""

    entry_function: object
    functions_by_id: dict
    call_edges: dict
    concrete_workgroup_size: tuple
    dependent_calls: set
    call_mutations: dict
    uniform_semantics: set
    lane_symmetry: object
    symmetric_parameters: dict
    symmetric_entries: dict
    uniform_parameters: dict
    parameter_bounds: dict
    unreachable_calls: set
    called_ids: set


class DirectXSoftwareSubgroupContractMixin:
    """Contract checks and helper uniformity proofs behind ``prepare_hlsl_software_subgroup_contract``."""

    def reset_hlsl_software_subgroup_contract(self):
        """Clear every software-subgroup fact a previous program recorded."""
        self.required_hlsl_software_subgroup_helpers = set()
        self.hlsl_software_subgroup_helper_names = {}
        self.hlsl_software_subgroup_reserved_names = set()
        self.hlsl_software_subgroup_workgroup_size = None
        self.hlsl_software_subgroup_invocation_count = None
        self.hlsl_software_subgroup_function_names = set()
        self.hlsl_software_subgroup_invocation_expressions = {}
        self.hlsl_software_subgroup_invocation_variable = None
        self.hlsl_software_subgroup_uniform_calls = set()
        self.hlsl_software_subgroup_resource_names = set()
        self.hlsl_software_subgroup_uniform_returns = UniformReturnAnalysis({}, set())
        self.hlsl_software_subgroup_masked_if_plans = {}
        self.hlsl_software_subgroup_strided_for_plans = {}
        self.hlsl_software_subgroup_result_overrides = {}
        self.hlsl_software_subgroup_guard_names = set()
        self.hlsl_software_subgroup_guard_constant_names = set()
        self.hlsl_software_subgroup_counter_loops = {}
        self.hlsl_software_subgroup_fresh_counter_loops = set()
        self.hlsl_software_subgroup_proven_counter_loops = set()
        self.hlsl_software_subgroup_modular_stride_loops = set()
        self.hlsl_software_subgroup_symmetry = None

    def prepare_hlsl_software_subgroup_contract(self, ast, target_stage=None):
        """Check the software-subgroup contract of ``ast`` and record the facts its lowering relies on."""
        self.reset_hlsl_software_subgroup_contract()
        if self.software_subgroup_width is None:
            return

        entry_function, concrete_workgroup_size, invocation_count = (
            self.hlsl_software_subgroup_entry_contract(ast, target_stage)
        )
        all_records, lane_local_matrix_nodes, system_inputs_only = (
            self.hlsl_software_subgroup_operation_contract(
                ast, entry_function, concrete_workgroup_size, invocation_count
            )
        )
        if system_inputs_only:
            self.hlsl_software_subgroup_bind_lane_local_matrix_functions(
                ast, lane_local_matrix_nodes
            )
            return
        self.check_hlsl_software_subgroup_operations(
            all_records, entry_function, concrete_workgroup_size, invocation_count
        )
        (
            functions,
            functions_by_id,
            functions_by_name,
            direct_ids,
            resolved_calls,
            unresolved_calls,
            call_source_types,
            declarations,
            edges,
        ) = self.hlsl_software_subgroup_call_graph(ast)
        dependent_ids, call_edges, ordered_ids = self.hlsl_software_subgroup_call_order(
            entry_function,
            functions_by_id,
            functions_by_name,
            direct_ids,
            unresolved_calls,
            edges,
            concrete_workgroup_size,
        )
        dependent_calls = self.bind_hlsl_software_subgroup_names(
            ast,
            functions,
            functions_by_id,
            direct_ids,
            dependent_ids,
            call_edges,
            lane_local_matrix_nodes,
        )
        call_mutations = self.hlsl_software_subgroup_call_mutations(
            ast, functions_by_name, resolved_calls, call_source_types, declarations
        )
        propagation = self.hlsl_software_subgroup_propagation(
            entry_function,
            functions_by_id,
            call_edges,
            concrete_workgroup_size,
            invocation_count,
            resolved_calls,
            call_mutations,
            dependent_calls,
        )
        for function_id in ordered_ids:
            self.analyze_hlsl_software_subgroup_function(propagation, function_id)

    def hlsl_software_subgroup_entry_contract(self, ast, target_stage):
        """Return the one compute entry point, its concrete local size and its invocation count."""
        entries = self.hlsl_software_subgroup_stage_entries(ast, target_stage)
        if len(entries) != 1 or entries[0][0] != "compute":
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering requires exactly one compute "
                "entry point and no other emitted stages",
                reason="entry-point-contract-invalid",
            )
        _stage_name, entry_function, execution_config = entries[0]
        raw_workgroup_size = tuple(hlsl_entry_local_size(execution_config))
        concrete_workgroup_size = []
        for value in raw_workgroup_size:
            text = str(value).strip()
            if not re.fullmatch(r"[0-9]+", text):
                concrete_workgroup_size = []
                break
            concrete_workgroup_size.append(int(text))
        concrete_workgroup_size = tuple(concrete_workgroup_size)
        invocation_count = (
            concrete_workgroup_size[0]
            * concrete_workgroup_size[1]
            * concrete_workgroup_size[2]
            if len(concrete_workgroup_size) == 3
            else 0
        )
        if not (
            len(concrete_workgroup_size) == 3
            and all(value > 0 for value in concrete_workgroup_size)
            and invocation_count <= 1024
        ):
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering requires concrete positive "
                "local dimensions and at most 1024 invocations",
                workgroup_size=raw_workgroup_size,
                reason="workgroup-size-mismatch",
                source_location=getattr(entry_function, "source_location", None),
            )
        self.hlsl_software_subgroup_workgroup_size = concrete_workgroup_size
        self.hlsl_software_subgroup_invocation_count = invocation_count
        return entry_function, concrete_workgroup_size, invocation_count

    def hlsl_software_subgroup_operation_contract(
        self, ast, entry_function, concrete_workgroup_size, invocation_count
    ):
        """Check the matrix, shuffle-policy and WaveSize contracts of the entry point.

        Returns the subgroup operation records, the lane-local matrix nodes and whether only the subgroup system
        inputs need lowering.
        """
        all_records = self.hlsl_software_subgroup_operation_records(ast)
        lane_local_matrix_nodes = self.hlsl_software_subgroup_lane_local_matrix_nodes(
            ast
        )
        if lane_local_matrix_nodes and invocation_count % self.software_subgroup_width:
            raise self.hlsl_software_subgroup_error(
                "DirectX matrix fills and transfers require complete logical subgroups",
                workgroup_size=concrete_workgroup_size,
                operation="CooperativeMatrixTransfer",
                reason="incomplete-matrix-subgroup",
                source_location=getattr(
                    lane_local_matrix_nodes[0], "source_location", None
                ),
            )
        # Subgroup system inputs, matrix fills and matrix transfers without
        # collective operations need only the logical split of SV_GroupIndex
        # into subgroup and lane IDs. That split is exact under any control
        # flow, so no shuffle policy applies.
        system_inputs_only = not all_records and (
            bool(lane_local_matrix_nodes)
            or self.hlsl_software_subgroup_uses_system_inputs(ast)
        )
        if self.relative_wave_shuffle_out_of_range != "self" and not system_inputs_only:
            raise self.hlsl_software_subgroup_error(
                "DirectX software relative shuffles require the explicit "
                "calling-lane out-of-range policy",
                workgroup_size=concrete_workgroup_size,
                reason="out-of-range-policy-unproven",
                source_location=getattr(entry_function, "source_location", None),
            )
        if not system_inputs_only:
            self.prepare_hlsl_software_subgroup_masked_if_plans(entry_function)

        wave_attributes = [
            attribute
            for attribute in getattr(entry_function, "attributes", []) or []
            if self.hlsl_wave_size_attribute(attribute)
        ]
        for attribute in wave_attributes:
            arguments = list(getattr(attribute, "arguments", []) or [])
            width = (
                self.hlsl_int_literal_value(arguments[0])
                if len(arguments) == 1
                else None
            )
            if width != self.software_subgroup_width:
                raise self.hlsl_software_subgroup_error(
                    "DirectX software subgroup width conflicts with the source "
                    "WaveSize contract",
                    workgroup_size=concrete_workgroup_size,
                    reason="hardware-contract-conflict",
                    source_location=getattr(attribute, "source_location", None),
                )
        return all_records, lane_local_matrix_nodes, system_inputs_only

    def check_hlsl_software_subgroup_operations(
        self, all_records, entry_function, concrete_workgroup_size, invocation_count
    ):
        """Refuse an empty operation set and every operation the software lowering cannot express."""
        if not all_records:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup lowering was requested but no subgroup "
                "operation is present",
                workgroup_size=concrete_workgroup_size,
                reason="operation-set-empty",
                source_location=getattr(entry_function, "source_location", None),
            )
        for operation, node in all_records:
            if operation == "CooperativeMatrixMultiplyAccumulate":
                if invocation_count % self.software_subgroup_width:
                    raise self.hlsl_software_subgroup_error(
                        "DirectX cooperative matrices require complete logical subgroups",
                        workgroup_size=concrete_workgroup_size,
                        operation=operation,
                        reason="incomplete-matrix-subgroup",
                        source_location=getattr(node, "source_location", None),
                    )
                continue
            if operation not in self.HLSL_SOFTWARE_SUBGROUP_OPERATIONS:
                raise self.hlsl_software_subgroup_error(
                    f"DirectX software subgroup lowering does not support "
                    f"'{operation}'",
                    workgroup_size=concrete_workgroup_size,
                    operation=operation,
                    reason="operation-unsupported",
                    source_location=getattr(node, "source_location", None),
                )

    def hlsl_software_subgroup_call_graph(self, ast):
        """Resolve every helper call to one source overload and collect each helper's lexical type facts."""
        functions = self.collect_functions(ast)
        definitions = {}
        for function in functions:
            name = getattr(function, "name", None)
            if not name:
                continue
            key = (name, self.hlsl_source_function_signature(function))
            if key not in definitions or getattr(function, "body", None) is not None:
                definitions[key] = function
        functions = list(definitions.values())
        functions_by_id = {id(function): function for function in functions}
        functions_by_name = {}
        for function in functions:
            functions_by_name.setdefault(function.name, []).append(function)
        direct_ids = {
            id(function)
            for function in functions
            if self.hlsl_software_subgroup_operation_records(
                getattr(function, "body", None)
            )
        }
        resolved_calls = {}
        unresolved_calls = []
        call_source_types = {}
        declarations = {}
        constant_nodes = list(ast.constants) + [
            node
            for node in ast.global_variables
            if isinstance(node, VariableNode)
            and set(node.qualifiers) & {"const", "constant"}
            and "volatile" not in node.qualifiers
        ]
        edges = {id(function): [] for function in functions}
        for function in functions:
            lexical = self.hlsl_function_lexical_type_bindings(function)
            call_source_types.update(lexical["call_source_types"])
            declared_names = {
                binding["name"] for binding in lexical["declarations"].values()
            }
            constant_types = {
                node.name: self.directx_specialization_constant_type(node)
                for node in constant_nodes
                if node.name not in declared_names
            }
            for binding in lexical["declarations"].values():
                binding["initializer_source_types"] = {
                    **constant_types,
                    **binding.get("initializer_source_types", {}),
                }
            declarations.update(lexical["declarations"])
            for call in self.walk_ast(getattr(function, "body", None)):
                if not isinstance(call, FunctionCallNode):
                    continue
                name = self.function_call_name(call)
                if name not in functions_by_name:
                    continue
                previous_source_types = self.local_variable_source_types
                previous_types = self.local_variable_types
                source_types = lexical["call_source_types"].get(id(call), {})
                self.local_variable_source_types = source_types
                self.local_variable_types = source_types
                try:
                    callee = self.resolve_hlsl_function_overload(
                        name, call.arguments, call_node=call
                    )
                except DirectXMappedOverloadError:
                    callee = None
                finally:
                    self.local_variable_source_types = previous_source_types
                    self.local_variable_types = previous_types
                if callee is None:
                    unresolved_calls.append((function, call))
                    continue
                key = (callee.name, self.hlsl_source_function_signature(callee))
                callee = definitions[key]
                resolved_calls[id(call)] = callee
                edges[id(function)].append((call, id(callee)))
        return (
            functions,
            functions_by_id,
            functions_by_name,
            direct_ids,
            resolved_calls,
            unresolved_calls,
            call_source_types,
            declarations,
            edges,
        )

    def hlsl_software_subgroup_call_order(
        self,
        entry_function,
        functions_by_id,
        functions_by_name,
        direct_ids,
        unresolved_calls,
        edges,
        concrete_workgroup_size,
    ):
        """Return the helpers that reach a subgroup operation, the calls among them and a callers-first order."""
        dependent_ids = set(direct_ids)
        changed = True
        while changed:
            changed = False
            for function_id, calls in edges.items():
                if function_id in dependent_ids:
                    continue
                if any(target in dependent_ids for _call, target in calls):
                    dependent_ids.add(function_id)
                    changed = True
        for _function, call in unresolved_calls:
            name = self.function_call_name(call)
            if any(
                id(candidate) in dependent_ids for candidate in functions_by_name[name]
            ):
                raise self.hlsl_software_subgroup_error(
                    "DirectX software subgroup lowering cannot resolve one source "
                    f"overload for helper '{name}'",
                    workgroup_size=concrete_workgroup_size,
                    reason="helper-identity-ambiguous",
                    source_location=getattr(call, "source_location", None),
                )
        if id(entry_function) not in dependent_ids:
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup operations are not reached from the "
                "compute entry point",
                workgroup_size=concrete_workgroup_size,
                reason="helper-call-unproven",
                source_location=getattr(entry_function, "source_location", None),
            )

        # Process callers before callees, intersecting facts from every call.
        # Recursive collective graphs cannot establish a finite barrier schedule.
        call_edges = {}
        incoming = {function_id: 0 for function_id in dependent_ids}
        for function_id in functions_by_id:
            if function_id not in dependent_ids:
                continue
            call_edges[function_id] = [
                (call, target)
                for call, target in edges[function_id]
                if target in dependent_ids
            ]
            for _call, target in call_edges[function_id]:
                incoming[target] += 1
        pending = [key for key in functions_by_id if incoming.get(key) == 0]
        ordered_ids = []
        while pending:
            function_id = pending.pop(0)
            ordered_ids.append(function_id)
            for _call, target in call_edges[function_id]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    pending.append(target)
        if len(ordered_ids) != len(dependent_ids):
            raise self.hlsl_software_subgroup_error(
                "DirectX software subgroup helpers cannot use recursive calls",
                workgroup_size=concrete_workgroup_size,
                reason="helper-call-recursive",
                source_location=getattr(entry_function, "source_location", None),
            )
        return dependent_ids, call_edges, ordered_ids

    def bind_hlsl_software_subgroup_names(
        self,
        ast,
        functions,
        functions_by_id,
        direct_ids,
        dependent_ids,
        call_edges,
        lane_local_matrix_nodes,
    ):
        """Record the dependent helpers and a free invocation variable; return the calls between those helpers."""
        dependent_calls = {
            id(call) for calls in call_edges.values() for call, _target in calls
        }
        self.hlsl_software_subgroup_function_names = {
            functions_by_id[key].name for key in dependent_ids
        }
        reserved_names = self.hlsl_helper_reserved_names(functions)
        self.hlsl_software_subgroup_reserved_names = reserved_names
        variable = "__crossgl_software_subgroup_invocation"
        while variable in reserved_names:
            variable += "_"
        self.hlsl_software_subgroup_invocation_variable = variable
        for function_id in direct_ids:
            self.hlsl_software_subgroup_invocation_expressions[
                functions_by_id[function_id].name
            ] = variable
        self.hlsl_software_subgroup_bind_lane_local_matrix_functions(
            ast, lane_local_matrix_nodes
        )
        return dependent_calls

    def hlsl_software_subgroup_call_mutations(
        self, ast, functions_by_name, resolved_calls, call_source_types, declarations
    ):
        """Record the uniform calls and return analysis; return the names each call or pointer may write."""
        calls = [
            node for node in self.walk_ast(ast) if isinstance(node, FunctionCallNode)
        ]
        storage_loads = set()
        for node in calls:
            name = self.function_call_name(node)
            if name in functions_by_name or not isinstance(
                node.function, (str, IdentifierNode)
            ):
                continue
            if (
                (name in {"min", "max"} and len(node.arguments) == 2)
                or (name == "clamp" and len(node.arguments) == 3)
                or self.directx_compile_time_constructor_type(name) is not None
                # A bit reinterpretation, such as the binary16 storage
                # lowering's, writes nothing and keeps its operand's value.
                or (
                    len(node.arguments) == 1
                    and self.hlsl_metal_as_type_target_type(name) is not None
                )
            ):
                self.hlsl_software_subgroup_uniform_calls.add(id(node))
            elif storage_load_call(name, node.arguments):
                storage_loads.add(id(node))
        self.hlsl_software_subgroup_uniform_returns = UniformReturnAnalysis(
            resolved_calls,
            self.hlsl_software_subgroup_uniform_calls,
            self.structs_by_name,
            storage_loads,
        )
        call_mutations = {}
        for binding in declarations.values():
            declaration = binding["node"]
            if not isinstance(getattr(declaration, "var_type", None), PointerType):
                continue
            roots = self.hlsl_software_subgroup_uniform_returns.pointer_argument_write_roots(
                declaration.initial_value,
                binding.get("initializer_source_types", {}),
            )
            if roots is not None:
                call_mutations[id(declaration)] = roots
        for node in calls:
            call_name = self.function_call_name(node)
            arguments = list(getattr(node, "arguments", []) or [])
            callee = resolved_calls.get(id(node))
            builtin_writes = (
                builtin_argument_write_indices(call_name, len(arguments))
                if call_name not in functions_by_name
                and isinstance(node.function, (str, IdentifierNode))
                else None
            )
            pure_call = id(node) in self.hlsl_software_subgroup_uniform_calls
            written_roots = set()
            if (
                self.hlsl_software_subgroup_uniform_returns.dependencies(node)
                is not None
            ):
                mutations = []
            elif callee is not None:
                parameters = getattr(callee, "parameters", []) or []
                mutations = []
                for index, argument in enumerate(arguments):
                    if self.hlsl_software_subgroup_uniform_returns.read_only_resource_parameter(
                        callee, index
                    ):
                        continue
                    if index >= len(parameters):
                        mutations.append(argument)
                    elif isinstance(parameters[index].param_type, PointerType):
                        roots = self.hlsl_software_subgroup_uniform_returns.pointer_argument_write_roots(
                            argument, call_source_types.get(id(node), {})
                        )
                        if roots is None:
                            mutations.append(argument)
                        else:
                            written_roots.update(roots)
                    elif isinstance(parameters[index].param_type, ReferenceType) or set(
                        self.hlsl_parameter_qualifiers(parameters[index])
                    ) & {"out", "inout"}:
                        # A reference writes its selected lvalue, not its indices.
                        # Argument evaluation effects are visited independently.
                        root = self.hlsl_texture_offset_write_root_name(argument)
                        if root is None:
                            mutations.append(argument)
                        else:
                            written_roots.add(root)
            elif builtin_writes is not None:
                mutations = [arguments[index] for index in builtin_writes]
            elif call_name not in functions_by_name and (
                call_name in self.HLSL_WAVE_INTRINSIC_ARITIES
                or self.hlsl_metal_simd_shuffle_name(call_name) is not None
                or pure_call
            ):
                mutations = []
            else:
                mutations = arguments
            call_mutations[id(node)] = written_roots.union(
                *(
                    self.hlsl_software_subgroup_expression_identifier_names(argument)
                    for argument in mutations
                )
            )
        return call_mutations

    def hlsl_software_subgroup_propagation(
        self,
        entry_function,
        functions_by_id,
        call_edges,
        concrete_workgroup_size,
        invocation_count,
        resolved_calls,
        call_mutations,
        dependent_calls,
    ):
        """Start the caller-to-callee uniformity proof with no facts about any helper."""
        uniform_semantics = {
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "gl_WorkGroupID",
            "gl_SubgroupSize",
            "gl_NumSubgroups",
            "SV_GroupID",
        }
        if invocation_count <= self.software_subgroup_width:
            uniform_semantics.add("gl_SubgroupID")
        lane_symmetry = self.hlsl_software_subgroup_lane_symmetry_analysis(
            resolved_calls, call_mutations, invocation_count
        )
        # Lane-symmetric parameters, and whether every subgroup enters a
        # helper with the same lanes, from every reachable call.
        symmetric_parameters = {}
        symmetric_entries = {}
        uniform_parameters = {}
        # Checked metadata bounds every call passes to a helper parameter.
        parameter_bounds = {}
        # Calls in branches a constant guard never selects, and the helpers
        # only such calls reach: neither runs, so neither constrains a proof.
        unreachable_calls = set()
        called_ids = {target for edges in call_edges.values() for _c, target in edges}
        return SoftwareSubgroupPropagation(
            entry_function=entry_function,
            functions_by_id=functions_by_id,
            call_edges=call_edges,
            concrete_workgroup_size=concrete_workgroup_size,
            dependent_calls=dependent_calls,
            call_mutations=call_mutations,
            uniform_semantics=uniform_semantics,
            lane_symmetry=lane_symmetry,
            symmetric_parameters=symmetric_parameters,
            symmetric_entries=symmetric_entries,
            uniform_parameters=uniform_parameters,
            parameter_bounds=parameter_bounds,
            unreachable_calls=unreachable_calls,
            called_ids=called_ids,
        )

    def analyze_hlsl_software_subgroup_function(self, propagation, function_id):
        """Validate one helper's control flow from its callers' facts, then pass facts on to its callees."""
        entry_function = propagation.entry_function
        functions_by_id = propagation.functions_by_id
        call_edges = propagation.call_edges
        called_ids = propagation.called_ids
        call_mutations = propagation.call_mutations
        dependent_calls = propagation.dependent_calls
        lane_symmetry = propagation.lane_symmetry
        symmetric_parameters = propagation.symmetric_parameters
        symmetric_entries = propagation.symmetric_entries
        uniform_parameters = propagation.uniform_parameters
        parameter_bounds = propagation.parameter_bounds
        unreachable_calls = propagation.unreachable_calls
        function = functions_by_id[function_id]
        if (
            function is not entry_function
            and function_id in called_ids
            and function_id not in uniform_parameters
        ):
            unreachable_calls.update(
                id(call) for call, _target in call_edges[function_id]
            )
            return
        parameters, uniform_names, uniform_components = (
            self.hlsl_software_subgroup_function_uniform_names(
                propagation, function, function_id
            )
        )
        # A name alone is not a uniformity proof after writes, shadowing,
        # or a call that could receive it through an output parameter.
        mutable_names = self.hlsl_software_subgroup_function_mutable_names(
            function, uniform_names, call_mutations
        )
        # Immutable entry storage moved only by offsets outside loops keeps
        # its uniform view where the flow can follow those offsets.
        offset_names = (
            followed_offsets(
                function,
                {
                    parameter.name
                    for parameter in parameters
                    if parameter.name in uniform_names
                    and parameter.name in self.hlsl_software_subgroup_resource_names
                },
                call_mutations,
                walk=self.walk_ast,
                map_operator=self.map_operator,
                identifier_names=(
                    self.hlsl_software_subgroup_expression_identifier_names
                ),
                write_roots=lambda target: {
                    self.hlsl_texture_offset_write_root_name(target)
                }
                - {None},
            )
            if function is entry_function
            else set()
        )
        uniform_names.difference_update(mutable_names - offset_names)
        uniform_components = {
            name: components
            for name, components in uniform_components.items()
            if name not in mutable_names
        }
        uniform_call_arguments = {}
        metadata_bounds = (
            metadata_loop_bindings(
                function, self.metadata_preconditions, mutable_names, self.walk_ast
            )
            if function is entry_function
            else inherited_metadata_bindings(
                function,
                parameter_bounds.get(function_id),
                mutable_names,
                self.walk_ast,
            )
        )
        seeds = {name: UNIFORM for name in uniform_names}
        for index, parameter in enumerate(parameters):
            if parameter.name not in seeds and (
                index in symmetric_parameters.get(function_id, set())
                or (
                    function is entry_function
                    and self.semantic_from_node(parameter) == "gl_SubgroupInvocationID"
                )
            ):
                seeds[parameter.name] = SYMMETRIC
        symmetry = lane_symmetry.analyze(
            function,
            seeds,
            function is entry_function or symmetric_entries.get(function_id, False),
        )
        self.hlsl_software_subgroup_symmetry = symmetry
        try:
            self.validate_hlsl_software_subgroup_control_flow(
                function,
                dependent_calls,
                uniform_names,
                call_mutations,
                mutable_names,
                uniform_components,
                uniform_call_arguments,
                entry=function is entry_function,
                offset_names=offset_names,
                metadata_bounds=metadata_bounds,
                unreachable_calls=unreachable_calls,
            )
        finally:
            self.hlsl_software_subgroup_symmetry = None
        self.propagate_hlsl_software_subgroup_call_facts(
            propagation,
            function_id,
            symmetry,
            uniform_names,
            uniform_components,
            uniform_call_arguments,
            metadata_bounds,
        )

    def hlsl_software_subgroup_function_uniform_names(
        self, propagation, function, function_id
    ):
        """Return a helper's parameters, the names uniform on entry and the uniform components of its inputs."""
        entry_function = propagation.entry_function
        concrete_workgroup_size = propagation.concrete_workgroup_size
        uniform_semantics = propagation.uniform_semantics
        uniform_parameters = propagation.uniform_parameters
        parameters = getattr(function, "parameters", []) or []
        analysis = self.hlsl_software_subgroup_uniform_returns
        declarations = [
            *parameters,
            *(
                node
                for node in self.walk_ast(function.body)
                if isinstance(node, VariableNode)
            ),
        ]
        counts = {}
        for declaration in declarations:
            counts[declaration.name] = counts.get(declaration.name, 0) + 1
        self.hlsl_software_subgroup_resource_names = {
            declaration.name
            for declaration in declarations
            if counts[declaration.name] == 1
            and analysis.resource_type(
                getattr(
                    declaration,
                    "param_type",
                    getattr(declaration, "var_type", None),
                ),
                declaration,
            )
        }
        uniform_names = set(self.literal_int_constants) - {
            parameter.name for parameter in parameters
        }
        uniform_names.update(
            parameter.name
            for index, parameter in enumerate(parameters)
            if index in uniform_parameters.get(function_id, set())
            and not isinstance(parameter.param_type, ReferenceType)
            and not set(self.hlsl_parameter_qualifiers(parameter)) & {"out", "inout"}
            and (
                analysis.value_type(parameter.param_type)
                or parameter.name in self.hlsl_software_subgroup_resource_names
            )
        )
        uniform_components = {}
        if function is entry_function:
            unit_dimensions = {
                index for index, size in enumerate(concrete_workgroup_size) if size == 1
            }
            uniform_components = {
                parameter.name: unit_dimensions
                for parameter in parameters
                if self.hlsl_canonical_semantic(self.semantic_from_node(parameter))
                in {"SV_DispatchThreadID", "SV_GroupThreadID"}
            }
            uniform_names.update(
                parameter.name
                for parameter in parameters
                if self.semantic_from_node(parameter) in uniform_semantics
                or set(getattr(parameter, "qualifiers", []) or [])
                & {"uniform", "constant"}
                or (
                    parameter.name in self.hlsl_software_subgroup_resource_names
                    and (
                        analysis._qualified(parameter, {"constant"})
                        # A checked contract: no writable binding aliases it.
                        or parameter.name in self.hlsl_immutable_storage_names
                    )
                )
            )
        return parameters, uniform_names, uniform_components

    def hlsl_software_subgroup_function_mutable_names(
        self, function, uniform_names, call_mutations
    ):
        """Return the names a helper may write, and drop every name it declares from ``uniform_names``."""
        mutable_names = set()
        for node in self.walk_ast(getattr(function, "body", None)):
            if isinstance(node, VariableNode):
                uniform_names.discard(node.name)
                if isinstance(node.var_type, ReferenceType):
                    mutable_names.update(
                        self.hlsl_software_subgroup_expression_identifier_names(
                            node.initial_value
                        )
                    )
            elif isinstance(node, AssignmentNode):
                mutable_names.add(
                    self.hlsl_software_subgroup_assignment_target_name(node)
                )
            elif isinstance(node, UnaryOpNode) and self.map_operator(
                getattr(node, "op", getattr(node, "operator", None))
            ) in {"++", "--", "&"}:
                mutable_names.add(
                    self.hlsl_texture_offset_write_root_name(node.operand)
                )
            elif isinstance(node, FunctionCallNode):
                mutable_names.update(call_mutations[id(node)])
        return mutable_names

    def propagate_hlsl_software_subgroup_call_facts(
        self,
        propagation,
        function_id,
        symmetry,
        uniform_names,
        uniform_components,
        uniform_call_arguments,
        metadata_bounds,
    ):
        """Intersect the symmetry, uniformity and bound facts each reachable call passes to its callee."""
        functions_by_id = propagation.functions_by_id
        call_edges = propagation.call_edges
        symmetric_parameters = propagation.symmetric_parameters
        symmetric_entries = propagation.symmetric_entries
        uniform_parameters = propagation.uniform_parameters
        parameter_bounds = propagation.parameter_bounds
        unreachable_calls = propagation.unreachable_calls
        for call, target in call_edges[function_id]:
            if id(call) in unreachable_calls:
                continue
            levels = [symmetry.level(argument) for argument in call.arguments]
            symmetric = {
                index for index, level in enumerate(levels) if level >= SYMMETRIC
            }
            if target not in symmetric_parameters:
                symmetric_parameters[target] = symmetric
            else:
                symmetric_parameters[target].intersection_update(symmetric)
            symmetric_entries[target] = symmetric_entries.get(
                target, True
            ) and symmetry.entered_symmetrically(call)
            # A uniformly reached call also passes the uniform values the
            # lane facts prove: collective results, and names the checks
            # below accept where the existing call-site proof stops, such
            # as the right operand of a uniform short-circuit condition.
            facts = uniform_call_arguments.get(id(call), set())
            if symmetry.reach(call) == UNIFORM:
                facts = facts | {
                    index
                    for index, level in enumerate(levels)
                    if level == UNIFORM
                    and (
                        symmetry.collective_derived(call.arguments[index])
                        or self.hlsl_software_subgroup_uniform_expression(
                            call.arguments[index],
                            uniform_names,
                            uniform_components,
                        )
                    )
                }
            if target not in uniform_parameters:
                uniform_parameters[target] = set(facts)
            else:
                uniform_parameters[target].intersection_update(facts)
            parameter_bounds[target] = join_parameter_bounds(
                parameter_bounds.get(target),
                call_parameter_bounds(
                    getattr(functions_by_id[target], "parameters", ()) or (),
                    getattr(call, "arguments", ()) or (),
                    metadata_bounds,
                    self.type_name_string,
                ),
            )
