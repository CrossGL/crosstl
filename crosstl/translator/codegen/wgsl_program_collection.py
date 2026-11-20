"""Identifier scopes and program collection in the WGSL generator: structs, constant buffers, globals, stage resources, helpers, stages and deduplication."""

from ..host_workgroup_size import function_entry_execution_config
from .global_specialization_constants import split_global_specialization_constants
from .stage_utils import (
    STAGE_QUALIFIER_NAMES,
    normalize_stage_name,
)


class WGSLProgramCollectionMixin:
    """Program collection members of ``WGSLCodeGen``."""

    def push_identifier_scope(self, names=()):
        names = [name for name in names if name]
        self._identifier_scopes.append(set(names))
        self._identifier_alias_scopes.append(self.wgsl_identifier_map(names))
        self._value_type_scopes.append({})
        self._resource_alias_scopes.append({})

    def pop_identifier_scope(self):
        self._identifier_scopes.pop()
        self._identifier_alias_scopes.pop()
        self._value_type_scopes.pop()
        self._resource_alias_scopes.pop()

    def register_local_identifier(self, name):
        if not self._identifier_scopes:
            return
        if name:
            self._identifier_scopes[-1].add(name)

    def is_local_identifier(self, name):
        return any(name in scope for scope in reversed(self._identifier_scopes))

    def push_pointer_identifier_scope(self, names=()):
        self._pointer_identifier_scopes.append({name for name in names if name})

    def pop_pointer_identifier_scope(self):
        self._pointer_identifier_scopes.pop()

    def is_pointer_identifier(self, name):
        return any(name in scope for scope in reversed(self._pointer_identifier_scopes))

    def _collect_structs(self, ast, target_stage):
        structs = list(getattr(ast, "structs", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            structs.extend(getattr(stage_node, "local_structs", []) or [])
        return self._dedupe_by_name(structs)

    def _collect_cbuffers(self, ast, target_stage):
        cbuffers = list(getattr(ast, "cbuffers", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            cbuffers.extend(getattr(stage_node, "local_cbuffers", []) or [])
        return self._dedupe_by_name(cbuffers)

    def _collect_global_variables(self, ast, target_stage):
        # Specialization-id globals are declared with the constants.
        _, variables = split_global_specialization_constants(ast)
        for stage_node in self._stage_nodes(ast, target_stage):
            lowered_output_name = self._stage_output_lowerings.get(
                id(stage_node), {}
            ).get("source_name")
            variables.extend(
                variable
                for variable in getattr(stage_node, "local_variables", []) or []
                if getattr(variable, "name", None) != lowered_output_name
            )
        return self._dedupe_by_name(variables)

    def _collect_stage_resource_parameters(self, ast, target_stage):
        parameters = []
        for stage_node in self._stage_nodes(ast, target_stage):
            entry_point = getattr(stage_node, "entry_point", None)
            if entry_point is None:
                continue
            parameters.extend(
                parameter
                for parameter in getattr(entry_point, "parameters", []) or []
                if self.is_stage_resource_parameter(parameter)
            )
        return self._dedupe_by_name(parameters)

    def _helper_functions(self, ast, target_stage):
        stage_entries = {
            id(stage_node.entry_point)
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        }
        helpers = []
        for func in getattr(ast, "functions", []) or []:
            if id(func) not in stage_entries and not self._function_stage_name(func):
                helpers.append(func)
        for stage_node in self._stage_nodes(ast, target_stage):
            helpers.extend(getattr(stage_node, "local_functions", []) or [])
        # WGSL permits forward calls, but not declarations without a body.
        # Discard prototypes before deduplication so they cannot hide definitions.
        return self._dedupe_functions(
            func for func in helpers if getattr(func, "body", None) is not None
        )

    def _function_stage_name(self, func):
        qualifiers = getattr(func, "qualifiers", []) or []
        for qualifier in qualifiers:
            stage_name = normalize_stage_name(qualifier)
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", ""))
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        return None

    def _function_execution_config(self, func):
        config = function_entry_execution_config(func)
        for attr in getattr(func, "attributes", []) or []:
            key = str(getattr(attr, "name", "")).lower()
            if key not in {"numthreads", "workgroup_size"}:
                continue
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 3:
                continue
            config["numthreads"] = [
                self.generate_attribute_argument(argument) for argument in arguments
            ]
        return config

    def _dedupe_by_name(self, nodes):
        seen = set()
        deduped = []
        for node in nodes:
            name = getattr(node, "name", None)
            if not name or name in seen:
                continue
            seen.add(name)
            deduped.append(node)
        return deduped

    def _dedupe_functions(self, funcs):
        seen = set()
        deduped = []
        for func in funcs:
            key = self.function_signature_key(func)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(func)
        return deduped
