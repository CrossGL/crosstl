"""Program options of the HLSL generator: target profiles, entry selection, built-in options, narrow storage types, software subgroup setup, groupshared values and matrix layouts."""

from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    ArrayType,
    BinaryOpNode,
    CastNode,
    ConstructorNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PointerType,
    ReferenceType,
    StageMap,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..dispatch_region_contract import (
    codegen_dispatch_region,
    with_dispatch_region_header,
)
from ..explicit_specializations import functions_by_called_name
from ..immutable_storage import (
    checked_immutable_storage,
    codegen_immutable_storage_names,
    with_immutable_storage_header,
)
from ..metadata_preconditions import (
    checked_metadata_preconditions,
    validate_codegen_metadata_preconditions,
    with_metadata_precondition_header,
)
from ..refusals import (
    DirectXRefusal,
    TranslatorArgumentError,
)
from ..semantics.declared_types import declared_type_names
from ..semantics.narrowing_constructors import select_narrowing_constructor_components
from .array_utils import evaluate_literal_int_expression
from .barrier_peephole import merge_adjacent_hlsl_barriers
from .directx_values import (
    DirectXEntryPointSelectionError,
    DirectXResourcePointerParameterError,
    DirectXSoftwareSubgroupError,
)
from .enum_utils import generic_type_parts
from .fixed_sampler_contracts import (
    fixed_sampler_records,
    with_hlsl_fixed_samplers,
)
from .generic_pointer_spaces import (
    assign_generic_pointer_spaces,
    refuse_written_generic_pointers,
)
from .hlsl_builtin_input_members import lower_hlsl_builtin_input_members
from .hlsl_entry_arguments import (
    drop_other_entry_arguments,
    separate_entry_argument_names,
)
from .hlsl_intrinsic_functions import rename_hlsl_intrinsic_functions
from .hlsl_keyword_names import legalize_hlsl_keyword_names
from .hlsl_work_graph_nodes import pin_hlsl_node_names
from .lane_symmetry import (
    UNIFORM,
    LaneSymmetryAnalysis,
    SymmetryRules,
)
from .loop_induction import (
    fixed_stride_loop_terminates,
    geometric_loop_terminates,
)
from .pointer_reinterpret import scalar_storage_layout
from .relational_builtins import lower_relational_builtins
from .stage_utils import (
    collect_stage_entry_records,
    function_stage_name,
    normalize_stage_name,
)
from .subgroup_control_flow import has_subgroup_early_exits
from .workgroup_access_contracts import parse_workgroup_access_assertions
from .workgroup_size_extents import fold_workgroup_size_extents
from .written_function_names import (
    HLSL_CALLABLE_FUNCTION_NAMES,
    HLSL_PAYLOAD_ACCESS_QUALIFIER,
    refuse_written_foreign_calls,
)
from .written_type_names import refuse_written_foreign_types


class DirectXProgramOptionMixin:
    """Program option members of ``HLSLCodeGen``."""

    def normalize_directx_target_profile(self, target_profile):
        if target_profile is None:
            return None

        normalized = str(target_profile).strip().lower().replace("_", "-")
        # Every accepted spelling, including the canonical one, so that an unknown profile is refused.
        aliases = {
            "dx10": "dx10",
            "dx11": "dx11",
            "dx12": "dx12",
            "dx-10": "dx10",
            "d3d-10": "dx10",
            "d3d10": "dx10",
            "direct3d-10": "dx10",
            "directx-10": "dx10",
            "dx-11": "dx11",
            "d3d-11": "dx11",
            "d3d11": "dx11",
            "direct3d-11": "dx11",
            "directx-11": "dx11",
            "dx-12": "dx12",
            "d3d-12": "dx12",
            "d3d12": "dx12",
            "direct3d-12": "dx12",
            "directx-12": "dx12",
        }
        canonical = aliases.get(normalized)
        if canonical is None:
            raise DirectXRefusal(
                "DirectX target profile must be one of: dx10, dx11, dx12, "
                "directx-10, directx-11, directx-12"
            )
        return canonical

    def set_relative_wave_shuffle_out_of_range(self, policy):
        """Configure relative wave-shuffle behavior for invalid source lanes."""
        if not isinstance(policy, str):
            raise TranslatorArgumentError(
                "relative_wave_shuffle_out_of_range must be a string"
            )
        normalized = policy.strip().lower()
        if normalized not in self.HLSL_RELATIVE_WAVE_SHUFFLE_OUT_OF_RANGE_POLICIES:
            choices = ", ".join(
                sorted(self.HLSL_RELATIVE_WAVE_SHUFFLE_OUT_OF_RANGE_POLICIES)
            )
            raise DirectXRefusal(
                "relative_wave_shuffle_out_of_range must be one of: " f"{choices}"
            )
        self.relative_wave_shuffle_out_of_range = normalized

    def set_widen_native_float16(self, enabled):
        if not isinstance(enabled, bool):
            raise TranslatorArgumentError("widen_native_float16 must be a boolean")
        self.widen_native_float16 = enabled

    HLSL_GROUPSHARED_BOOL_STORAGES = frozenset({"bool", "uint"})

    def set_groupshared_bool_storage(self, storage):
        """Configure the word type of ``groupshared`` bool scalars and arrays.

        DXC loads a ``groupshared bool`` word with ``!range [0, 2)``, so a
        word the program reads before writing it, as MLX's contiguous scan
        reads all 32 ``simdgroup_sums`` slots after writing one per
        SIMD-group, makes the load undefined rather than an unspecified
        value. ``"uint"`` declares those variables as ``uint`` words and
        leaves every access to HLSL's implicit conversions: a store writes
        ``bool`` as 0 or 1 and a ``bool`` read tests ``word != 0``, so written
        words keep their values and unwritten ones read as an unspecified
        value. ``"bool"`` keeps the declared type.
        """
        if not isinstance(storage, str):
            raise TranslatorArgumentError("groupshared_bool_storage must be a string")
        normalized = storage.strip().lower()
        if normalized not in self.HLSL_GROUPSHARED_BOOL_STORAGES:
            choices = ", ".join(sorted(self.HLSL_GROUPSHARED_BOOL_STORAGES))
            raise DirectXRefusal(f"groupshared_bool_storage must be one of: {choices}")
        self.groupshared_bool_storage = normalized

    def set_cooperative_matrix_software_lowering(self, enabled):
        """Configure checked lane-local cooperative-matrix operations."""
        if not isinstance(enabled, bool):
            raise TranslatorArgumentError(
                "cooperative_matrix_software_lowering must be a boolean"
            )
        self.cooperative_matrix_software_lowering = enabled
        return self

    def set_cooperative_matrix_mma_profile(self, profile):
        """Select explicit binary32 arithmetic for software matrix products."""
        if profile is not None and (
            not isinstance(profile, str)
            or profile not in {
                "rne-gradual",
                "rne-flush",
                "fma-rne-gradual",
                "fma-rne-flush",
            }
        ):
            raise DirectXRefusal(
                "cooperative_matrix_mma_profile must be 'rne-gradual', 'rne-flush', "
                "'fma-rne-gradual', 'fma-rne-flush', or None"
            )
        self.cooperative_matrix_mma_profile = profile
        return self

    def set_workgroup_access_assertions(self, assertions):
        """Configure absolute shared-storage ranges for aggregate record copies."""
        self.workgroup_access_assertions = parse_workgroup_access_assertions(assertions)

    def set_software_subgroup_width(self, width):
        """Configure fail-closed barrier-backed DirectX subgroup lowering."""
        if width is not None and (
            isinstance(width, bool) or not isinstance(width, int)
        ):
            raise TranslatorArgumentError(
                "software_subgroup_width must be an integer or None"
            )
        if width not in {None, self.HLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH}:
            raise DirectXRefusal(
                "software_subgroup_width currently supports only an exact width "
                f"of {self.HLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH}"
            )
        self.software_subgroup_width = width
        return self

    def generate(self, ast):
        """Generate complete HLSL source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_entry(self, ast, entry_point):
        """Generate one independently loadable HLSL stage entry artifact."""
        scoped_ast = self.entry_scoped_ast(ast, entry_point)
        return self.generate_program(scoped_ast)

    def entry_scoped_ast(self, ast, entry_point):
        if not isinstance(entry_point, str) or not entry_point.strip():
            raise DirectXEntryPointSelectionError(
                "DirectX entry-point selection requires a non-empty source entry "
                "name",
                entry_point=entry_point,
                reason="invalid-name",
            )
        entry_point = entry_point.strip()
        stage_entry_types = self.stage_entry_types()
        entries = collect_stage_entry_records(ast, None, stage_entry_types)
        available = sorted(
            {
                str(getattr(function, "name", ""))
                for _entry_id, _stage_name, function in entries
                if getattr(function, "name", None)
            }
        )
        matches = [
            (entry_id, stage_name, function)
            for entry_id, stage_name, function in entries
            if getattr(function, "name", None) == entry_point
        ]
        if not matches:
            raise DirectXEntryPointSelectionError(
                f"DirectX source entry point '{entry_point}' was not found",
                entry_point=entry_point,
                available_entry_points=available,
                reason="not-found",
            )
        if len(matches) != 1:
            raise DirectXEntryPointSelectionError(
                f"DirectX source entry point '{entry_point}' is ambiguous",
                entry_point=entry_point,
                available_entry_points=available,
                reason="ambiguous",
            )

        _selected_id, selected_stage_name, selected_function = matches[0]
        selected_stage_name = normalize_stage_name(selected_stage_name)
        if selected_stage_name != "compute":
            raise DirectXEntryPointSelectionError(
                f"DirectX standalone entry point '{entry_point}' must be a compute "
                f"entry; found stage '{selected_stage_name}'",
                entry_point=entry_point,
                available_entry_points=available,
                reason="unsupported-stage",
                stage=selected_stage_name,
                source_location=getattr(selected_function, "source_location", None),
            )
        selected_stage = next(
            (
                (stage_type, stage)
                for stage_type, stage in getattr(ast, "stages", {}).items()
                if getattr(stage, "entry_point", None) is selected_function
            ),
            None,
        )
        scoped_ast = deepcopy(ast)
        selected_copy_function = None
        scoped_ast.functions = [
            function
            for function in getattr(scoped_ast, "functions", []) or []
            if function_stage_name(function) not in stage_entry_types
            or (
                selected_stage is None
                and function_stage_name(function) == selected_stage_name
                and getattr(function, "name", None) == entry_point
            )
        ]
        selected_stages = StageMap()
        if selected_stage is not None:
            stage_type, stage = selected_stage
            selected_stage_copy = deepcopy(stage)
            selected_stages.append(stage_type, selected_stage_copy)
            selected_copy_function = selected_stage_copy.entry_point
        else:
            selected_copy_function = next(
                (
                    function
                    for function in scoped_ast.functions
                    if function_stage_name(function) == selected_stage_name
                    and getattr(function, "name", None) == entry_point
                ),
                None,
            )
        scoped_ast.stages = selected_stages
        if selected_copy_function is None:
            raise DirectXEntryPointSelectionError(
                f"DirectX source entry point '{entry_point}' could not be copied",
                entry_point=entry_point,
                available_entry_points=available,
                reason="copy-failed",
            )

        reachable_names = self.entry_reachable_function_names(
            scoped_ast,
            selected_copy_function,
        )
        scoped_ast.functions = [
            function
            for function in getattr(scoped_ast, "functions", []) or []
            if function is selected_copy_function
            or getattr(function, "name", None) in reachable_names
        ]
        for _stage_type, stage in scoped_ast.stages.items():
            stage.local_functions = [
                function
                for function in getattr(stage, "local_functions", []) or []
                if getattr(function, "name", None) in reachable_names
            ]
        drop_other_entry_arguments(scoped_ast, entry_point)
        self.prune_entry_scoped_declarations(scoped_ast, selected_copy_function)
        return scoped_ast

    def prune_entry_scoped_declarations(self, ast, entry_function):
        """Retain only module declarations reachable from one compute entry."""
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        functions.append(entry_function)

        function_names = {
            getattr(function, "name", None)
            for function in functions
            if getattr(function, "name", None)
        }
        referenced_names = set()
        for function in functions:
            local_names = {
                getattr(parameter, "name", None)
                for parameter in (
                    getattr(
                        function,
                        "parameters",
                        getattr(function, "params", []),
                    )
                    or []
                )
                if getattr(parameter, "name", None)
            }
            local_names.update(
                getattr(node, "name", None)
                for node in self.entry_ast_nodes(getattr(function, "body", None))
                if isinstance(node, VariableNode) and getattr(node, "name", None)
            )
            referenced_names.update(
                self.entry_identifier_references(function) - local_names
            )
        referenced_names.difference_update(function_names)

        declaration_groups = [
            (ast, "global_variables", False),
            (ast, "constants", False),
            (ast, "cbuffers", True),
        ]
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            declaration_groups.extend(
                (
                    (stage, "local_variables", False),
                    (stage, "local_cbuffers", True),
                )
            )

        declarations = []
        for owner, attribute, include_members in declaration_groups:
            for declaration in getattr(owner, attribute, []) or []:
                names = self.entry_declaration_names(
                    declaration,
                    include_members=include_members,
                )
                declarations.append((declaration, names))

        retained_ids = set()
        changed = True
        while changed:
            changed = False
            for declaration, names in declarations:
                declaration_id = id(declaration)
                if declaration_id in retained_ids or not names & referenced_names:
                    continue
                retained_ids.add(declaration_id)
                referenced_names.update(
                    self.entry_identifier_references(declaration) - names
                )
                changed = True

        for owner, attribute, _include_members in declaration_groups:
            setattr(
                owner,
                attribute,
                [
                    declaration
                    for declaration in getattr(owner, attribute, []) or []
                    if id(declaration) in retained_ids
                    or getattr(declaration, "is_type_alias", False)
                ],
            )

    def entry_ast_nodes(self, root):
        walk = getattr(root, "walk", None)
        return walk() if callable(walk) else self.walk_ast(root)

    def entry_identifier_references(self, root):
        return {
            str(node.name)
            for node in self.entry_ast_nodes(root)
            if isinstance(node, IdentifierNode) and getattr(node, "name", None)
        }

    def entry_declaration_names(self, declaration, *, include_members):
        name = getattr(
            declaration,
            "name",
            getattr(declaration, "variable_name", None),
        )
        names = {str(name)} if name else set()
        if include_members:
            names.update(
                str(member.name)
                for member in getattr(declaration, "members", []) or []
                if getattr(member, "name", None)
            )
        return names

    def entry_reachable_function_names(self, ast, entry_function):
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        functions_by_name = functions_by_called_name(functions)

        reachable_names = set()
        pending = [entry_function]
        visited = set()
        while pending:
            function = pending.pop()
            if id(function) in visited:
                continue
            visited.add(id(function))
            for node in self.walk_ast(getattr(function, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                name = self.function_call_name(node)
                if not name or name in reachable_names:
                    continue
                candidates = functions_by_name.get(name, ())
                if not candidates:
                    continue
                reachable_names.add(name)
                reachable_names.update(
                    candidate.name for candidate in candidates if candidate.name
                )
                pending.extend(candidates)
        return reachable_names

    def generate_stage(self, ast, shader_type):
        """Generate HLSL source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def with_hlsl_builtin_option_prelude(self, ast):
        if not self.hlsl_ast_needs_builtin_option(ast):
            return ast
        if self.hlsl_ast_declares_type(ast, "Option"):
            return ast

        self.hlsl_builtin_option_available = True
        ast = deepcopy(ast)
        prelude_ast = self.parse_hlsl_builtin_option_prelude()
        ast.structs = list(getattr(prelude_ast, "structs", []) or []) + list(
            getattr(ast, "structs", []) or []
        )
        ast.functions = list(getattr(prelude_ast, "functions", []) or []) + list(
            getattr(ast, "functions", []) or []
        )
        return ast

    def parse_hlsl_builtin_option_prelude(self):
        from ..lexer import Lexer
        from ..parser import Parser

        source = """
        shader DirectXBuiltinOption {
            generic<T> struct Option {
                enum OptionType { Some(T), Absent }
                OptionType variant;
            }

            generic<T> bool is_Some(Option<T> item) {
                return item.variant == Option::Some;
            }

            generic<T> bool is_None(Option<T> item) {
                return item.variant == Option::Absent;
            }

            generic<T> T unwrap_Some(Option<T> item) {
                return item.Some_0;
            }
        }
        """
        return Parser(Lexer(source).tokens).parse()

    def hlsl_ast_declares_type(self, ast, type_name):
        return any(
            getattr(struct, "name", None) == type_name
            for struct in getattr(ast, "structs", []) or []
        )

    def hlsl_ast_needs_builtin_option(self, ast):
        option_helpers = {"is_Some", "is_None", "unwrap_Some"}
        for node in self.walk_ast(ast):
            if self.hlsl_node_type_uses_builtin_option(node):
                return True
            if self.function_call_name(node) in option_helpers:
                return True
        return False

    def hlsl_node_type_uses_builtin_option(self, node):
        for attr in (
            "return_type",
            "param_type",
            "var_type",
            "vtype",
            "member_type",
            "const_type",
            "constructor_type",
        ):
            if self.hlsl_type_uses_builtin_option(getattr(node, attr, None)):
                return True
        return False

    def hlsl_type_uses_builtin_option(self, type_value):
        type_name = self.type_name_string(type_value)
        if not isinstance(type_name, str):
            return False
        base_name, generic_args = generic_type_parts(type_name)
        return base_name.rsplit("::", 1)[-1] == "Option" and bool(generic_args)

    def hlsl_builtin_option_payload_type(self, type_value):
        type_name = self.type_name_string(type_value)
        if not isinstance(type_name, str):
            return None

        base_name, generic_args = generic_type_parts(type_name)
        if base_name.rsplit("::", 1)[-1] == "Option" and len(generic_args) == 1:
            return generic_args[0]

        for specialization in getattr(
            self, "generic_enum_specializations", {}
        ).values():
            if (
                specialization.get("struct_name") == type_name
                and str(specialization.get("base_name")).rsplit("::", 1)[-1] == "Option"
            ):
                generic_params = specialization.get("definition", {}).get(
                    "generic_params", []
                )
                if not generic_params:
                    return None
                return specialization.get("substitutions", {}).get(generic_params[0])
        return None

    def hlsl_builtin_option_call_type(self, node, func_name):
        if not getattr(self, "hlsl_builtin_option_available", False):
            return None

        base_name = str(func_name).rsplit("::", 1)[-1]
        if base_name == "Some":
            return self.current_expression_expected_type
        if base_name in {"is_Some", "is_None"}:
            return "bool"
        if base_name == "unwrap_Some":
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            if len(args) != 1:
                return None
            option_type = self.expression_result_type(args[0])
            return self.hlsl_builtin_option_payload_type(option_type)
        return None

    def set_metadata_preconditions(self, records):
        self.metadata_preconditions = checked_metadata_preconditions(records)

    def set_immutable_storage(self, records):
        """Name read-only storage parameters that no writable binding aliases."""
        self.immutable_storage = checked_immutable_storage(records)

    def generate_program(self, ast, target_stage=None):
        """Render an AST to HLSL, optionally filtering stage entry points.

        Raises :class:`DirectXLoweringRefused` listing every construct that has no faithful HLSL lowering.
        """
        ast = separate_entry_argument_names(ast)
        pin_hlsl_node_names(ast)
        ast = legalize_hlsl_keyword_names(ast)
        ast = rename_hlsl_intrinsic_functions(ast)
        ast = lower_hlsl_builtin_input_members(ast)
        lower_relational_builtins(ast)
        select_narrowing_constructor_components(ast)
        return merge_adjacent_hlsl_barriers(
            self.refusals.run(self.render_program, ast, target_stage)
        )

    def render_program(self, ast, target_stage=None):
        ast = fold_workgroup_size_extents(ast, "HLSL", self.refuse_lowering)
        ast = assign_generic_pointer_spaces(ast, self.refuse_lowering)
        self.program_type_names = declared_type_names(ast)
        self.written_foreign_type_names = set()
        try:
            code = self.render_hlsl_program(ast, target_stage)
            refuse_written_foreign_types(
                code,
                self.written_foreign_type_names,
                self.LOWERING_REFUSAL_TYPE.label,
            )
            refuse_written_foreign_calls(
                code,
                HLSL_CALLABLE_FUNCTION_NAMES,
                self.LOWERING_REFUSAL_TYPE.label,
                (HLSL_PAYLOAD_ACCESS_QUALIFIER,),
            )
            refuse_written_generic_pointers(
                code, self.refuse_lowering, self.LOWERING_REFUSAL_TYPE.label
            )
        finally:
            self.written_foreign_type_names = None
        return code

    def render_hlsl_program(self, ast, target_stage=None):
        region = codegen_dispatch_region(
            ast, self.hlsl_software_subgroup_stage_entries(ast, target_stage)
        )
        records = validate_codegen_metadata_preconditions(
            self.hlsl_software_subgroup_stage_entries(ast, target_stage),
            self.metadata_preconditions,
            getattr(ast, "structs", ()),
        )
        fixed_samplers = fixed_sampler_records(ast)
        immutable, self.hlsl_immutable_storage_names = codegen_immutable_storage_names(
            self.hlsl_software_subgroup_stage_entries(ast, target_stage),
            self.immutable_storage,
            ast,
        )
        try:
            code = self.generate_hlsl_program_with_resource_access_variants(
                ast, target_stage
            )
        finally:
            self.hlsl_immutable_storage_names = frozenset()
        code = with_hlsl_fixed_samplers(code, fixed_samplers, ast)
        if records:
            code = with_metadata_precondition_header(code, records)
        if immutable:
            code = with_immutable_storage_header(code, immutable)
        return with_dispatch_region_header(code, region)

    HLSL_NARROW_POINTER_LAYOUT_PASS_LIMIT = 4

    def generate_hlsl_program_with_narrow_pointer_layouts(
        self, ast, target_stage, narrow_parameters=None
    ):
        """Generate HLSL whose helper pointers keep their callers' byte layout.

        A DirectX root buffer stores each 8- or 16-bit source element in its own
        32-bit carrier, while a narrow view cast from a 32-bit buffer keeps
        several elements in each word. A helper pointer parameter cannot tell
        these layouts apart from its declaration (#2145). Every generation pass
        records the layout of each call-site argument; helpers whose callers all
        pass widened storage are regenerated with element-indexed reads, and a
        helper that receives both layouts is rejected. Output is returned only
        from a complete pass whose recorded layouts reproduce its assumptions.

        Every pass renders a copy, so ``ast`` itself is left unchanged
        exactly when it has such parameters; ``narrow_parameters`` is that
        answer when the caller has already computed it.
        """
        self.hlsl_widened_narrow_pointer_parameters = frozenset()
        if narrow_parameters is None:
            narrow_parameters = self.hlsl_ast_has_narrow_storage_pointer_parameters(ast)
        if not narrow_parameters:
            self.hlsl_narrow_pointer_argument_layouts = None
            return self._generate_program(ast, target_stage)
        widened = frozenset()
        hypotheses = set()
        attempts = 0
        try:
            while attempts < self.HLSL_NARROW_POINTER_LAYOUT_PASS_LIMIT + len(
                hypotheses
            ):
                attempts += 1
                self.hlsl_widened_narrow_pointer_parameters = widened
                self.hlsl_narrow_pointer_argument_layouts = {}
                try:
                    code = self._generate_program(
                        deepcopy(ast), target_stage, owned=True
                    )
                except Exception as error:
                    # An incomplete pass can still prove widened callers for
                    # helpers emitted before the failure. Retry only when that
                    # adds information; otherwise keep the original error.
                    solved = self.solve_hlsl_narrow_pointer_parameter_layouts(
                        self.hlsl_narrow_pointer_argument_layouts
                    )
                    origin = getattr(error, "hlsl_narrow_storage_origin", None)
                    if (
                        origin is not None
                        and origin not in widened
                        and origin not in hypotheses
                    ):
                        # Helpers precede their callers, so a store through a
                        # helper's default packed view can stop the pass before
                        # any call site is recorded. Assume the widened layout
                        # once: output still requires a complete pass whose
                        # recorded call sites prove every assumed layout.
                        hypotheses.add(origin)
                        widened = widened | solved | {origin}
                        continue
                    if solved <= widened:
                        raise
                    widened = widened | solved
                    continue
                solved = self.solve_hlsl_narrow_pointer_parameter_layouts(
                    self.hlsl_narrow_pointer_argument_layouts
                )
                if solved == widened:
                    return code
                widened = solved
        finally:
            self.hlsl_narrow_pointer_argument_layouts = None
        raise DirectXResourcePointerParameterError(
            "DirectX narrow storage pointer layouts did not converge across "
            "helper call sites",
            reason="narrow-storage-layout-unresolved",
        )

    def hlsl_ast_has_narrow_storage_pointer_parameters(self, ast):
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        for function in functions:
            parameters = getattr(
                function, "parameters", getattr(function, "params", [])
            )
            for parameter in parameters or []:
                type_node = getattr(parameter, "param_type", None) or getattr(
                    parameter, "vtype", None
                )
                while isinstance(type_node, ArrayType):
                    type_node = type_node.element_type
                if not isinstance(type_node, PointerType):
                    continue
                layout = scalar_storage_layout(
                    self.type_name_string(type_node.pointee_type)
                )
                if self.hlsl_narrow_storage_layout(layout):
                    return True
        return False

    @staticmethod
    def hlsl_narrow_storage_layout(layout):
        return (
            layout is not None
            and layout.kind == "integer"
            and layout.bit_width in {8, 16}
        )

    def hlsl_narrow_pointer_parameter_key(self, function_name, parameter_name):
        return (str(function_name), str(parameter_name))

    def hlsl_narrow_pointer_parameter_is_widened(self, function_name, parameter_name):
        return self.hlsl_narrow_pointer_parameter_key(
            function_name, parameter_name
        ) in getattr(self, "hlsl_widened_narrow_pointer_parameters", frozenset())

    def hlsl_narrow_pointer_parameter_alias_binding(
        self, contract, *, function_name, parameter_name, root, offset
    ):
        """Bind a helper storage pointer with its call-site-proven layout."""

        key = self.hlsl_narrow_pointer_parameter_key(function_name, parameter_name)
        binding = self.hlsl_resource_pointer_parameter_alias_binding(
            contract,
            root=root,
            offset=offset,
            logical_storage=self.hlsl_narrow_pointer_parameter_is_widened(
                function_name, parameter_name
            ),
        )
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is not None and reinterpretation.get(
            "physical_storage_mapping"
        ):
            # Forwarded arguments resolve to this parameter's eventual layout.
            binding["narrow_storage_origin"] = key
        return binding

    def hlsl_narrow_pointer_argument_layout(self, element_type, binding):
        view_layout = scalar_storage_layout(element_type)
        if not self.hlsl_narrow_storage_layout(view_layout):
            return None
        physical_layout = scalar_storage_layout(
            self.hlsl_resource_pointer_element_type(binding.get("resource_type"))
        )
        if physical_layout is None or physical_layout.bit_width != 32:
            return None
        reinterpretation = binding.get("pointer_reinterpretation")
        if reinterpretation is None:
            return ("widened", None)
        origin = binding.get("narrow_storage_origin")
        if reinterpretation.get("physical_storage_mapping") and origin is not None:
            return ("parameter", origin)
        return ("packed", None)

    def record_hlsl_narrow_pointer_argument(
        self, function_name, parameter_name, element_type, binding, argument
    ):
        records = getattr(self, "hlsl_narrow_pointer_argument_layouts", None)
        if records is None:
            return
        layout = self.hlsl_narrow_pointer_argument_layout(element_type, binding)
        if layout is None:
            return
        key = self.hlsl_narrow_pointer_parameter_key(function_name, parameter_name)
        observations = records.setdefault(key, {})
        observations.setdefault(layout, getattr(argument, "source_location", None))

    def solve_hlsl_narrow_pointer_parameter_layouts(self, records):
        """Return helper parameters whose every caller passes widened storage."""

        records = records or {}
        state = {}
        changed = True
        while changed:
            changed = False
            for key, observations in records.items():
                kinds = set()
                for kind, origin in observations:
                    if kind != "parameter":
                        kinds.add(kind)
                    elif origin not in records:
                        # An origin without recorded callers keeps the
                        # default packed interpretation of its own reads.
                        kinds.add("packed")
                    elif origin in state:
                        kinds.add(state[origin])
                if not kinds:
                    continue
                resolved = "mixed" if len(kinds) > 1 else next(iter(kinds))
                if state.get(key) != resolved:
                    state[key] = resolved
                    changed = True
        for key, resolved in state.items():
            if resolved != "mixed":
                continue
            function_name, parameter_name = key
            locations = [
                location for location in records[key].values() if location is not None
            ]
            raise DirectXResourcePointerParameterError(
                "DirectX storage pointer parameter "
                f"'{function_name}.{parameter_name}' receives both widened "
                "narrow-element storage and packed words; one helper cannot "
                "preserve both layouts",
                function_name=function_name,
                parameter_name=parameter_name,
                address_space="storage",
                reason="mixed-narrow-storage-layout",
                source_location=locations[0] if locations else None,
            )
        return frozenset(
            key for key, resolved in state.items() if resolved == "widened"
        )

    HLSL_EARLY_EXIT_RETRY_REASONS = frozenset(
        {"early-return-unproven", "potentially-divergent-control-flow"}
    )

    def _generate_program(self, ast, target_stage=None, owned=False):
        """Render an AST to HLSL, optionally filtering stage entry points.

        Software subgroups need every invocation at their barriers. A program
        whose early returns cannot be proven uniform is rendered once more
        with ``converge_subgroup_early_exits``; programs that validate as
        written keep their output, and a failed retry reports the original
        error. ``owned`` says that the caller made ``ast`` for this call alone,
        so a pass may rewrite it in place instead of copying it first.
        """
        retry = self.software_subgroup_width is not None and has_subgroup_early_exits(
            ast, self.walk_ast
        )
        pristine = deepcopy(ast) if retry else None
        try:
            return self._generate_program_attempt(ast, target_stage, False, owned=owned)
        except DirectXSoftwareSubgroupError as error:
            if not retry or error.reason not in self.HLSL_EARLY_EXIT_RETRY_REASONS:
                raise
            try:
                return self._generate_program_attempt(
                    pristine, target_stage, True, owned=True
                )
            except DirectXSoftwareSubgroupError:
                raise error from None
        finally:
            self.hlsl_program_functions = None

    def generate_hlsl_function_declarations(self, functions):
        declarations = ""
        seen_declarations = set()
        for function in functions:
            if getattr(function, "body", None) is None:
                continue
            if getattr(function, "name", None) in (
                self.unsupported_glsl_buffer_block_functions
            ):
                continue

            prototype = deepcopy(function)
            prototype.body = None
            # The copy's parameters are new objects; carry the view analysis.
            for original, copied in zip(
                getattr(function, "parameters", None) or [],
                getattr(prototype, "parameters", None) or [],
            ):
                self.storage_parameter_texels.copy_parameter(original, copied)
                view = self.hlsl_view_storage_array_parameters.get(id(original))
                if view is not None and view[0] is original:
                    self.hlsl_view_storage_array_parameters[id(copied)] = (
                        copied,
                        view[1],
                    )
            declaration = self.generate_function(prototype)
            if declaration in seen_declarations:
                continue
            seen_declarations.add(declaration)
            declarations += declaration
        return declarations

    def hlsl_helper_reserved_names(self, functions):
        used_names = set(self.global_variable_types)
        used_names.update(self.structs_by_name)
        used_names.update(getattr(self, "enum_type_names", set()))
        used_names.update(getattr(self, "enum_struct_type_names", set()))
        used_names.update(
            getattr(func, "name", None)
            for func in functions or []
            if getattr(func, "name", None)
        )
        used_names.update(self.hlsl_function_name_aliases.values())
        for func in functions or []:
            used_names.update(self.collect_hlsl_function_identifier_names(func))
        return used_names

    def prepare_hlsl_trailing_zero_helper_names(self, functions):
        used_names = self.hlsl_helper_reserved_names(functions)

        self.hlsl_trailing_zero_helper_names = {}
        for operand_type in ("uint", "uint64_t"):
            base_name = f"__crossgl_ctz_{operand_type}"
            helper_name = base_name
            while helper_name in used_names:
                helper_name += "_"
            self.hlsl_trailing_zero_helper_names[operand_type] = helper_name
            used_names.add(helper_name)

    def hlsl_software_subgroup_error(
        self,
        message,
        *,
        workgroup_size=None,
        operation=None,
        reason,
        source_location=None,
    ):
        return DirectXSoftwareSubgroupError(
            message,
            software_subgroup_width=self.software_subgroup_width,
            workgroup_size=workgroup_size,
            operation=operation,
            reason=reason,
            source_location=source_location,
        )

    @staticmethod
    def hlsl_software_subgroup_statement_list(body):
        if hasattr(body, "statements"):
            return list(body.statements or [])
        if isinstance(body, list):
            return list(body)
        if body is None:
            return []
        return [body]

    HLSL_METAL_SIMD_SHUFFLE_OPERATIONS = {
        "simd_broadcast": "WaveReadLaneAt",
        "simd_shuffle": "WaveReadLaneAt",
        "simd_shuffle_down": "WaveShuffleDown",
        "simd_shuffle_up": "WaveShuffleUp",
        "simd_shuffle_and_fill_up": "WaveShuffleAndFillUp",
    }

    def hlsl_software_subgroup_collective_operation(self, node):
        """The subgroup value operation ``node`` performs, or None."""
        if isinstance(node, WaveOpNode):
            operation = node.operation
        elif isinstance(node, FunctionCallNode):
            function_name = self.function_call_name(node)
            if function_name in self.function_return_types:
                return None
            metal_name = self.hlsl_metal_simd_shuffle_name(function_name)
            operation = (
                self.HLSL_METAL_SIMD_SHUFFLE_OPERATIONS[metal_name]
                if metal_name is not None
                else function_name
            )
        else:
            return None
        return operation if operation in self.HLSL_WAVE_INTRINSIC_ARITIES else None

    def hlsl_software_subgroup_uniform_exit(self, node):
        """Whether every invocation takes the exit ``node`` together.

        The lane-symmetry facts of the function under validation prove it
        when every condition guarding the exit, and every exit before it, is
        workgroup-uniform.
        """
        symmetry = getattr(self, "hlsl_software_subgroup_symmetry", None)
        return symmetry is not None and symmetry.reach(node) == UNIFORM

    def hlsl_software_subgroup_lane_symmetry_analysis(
        self, resolved_calls, call_mutations, invocation_count
    ):
        """Lane-symmetry rules for the software subgroups of this program."""
        analysis = self.hlsl_software_subgroup_uniform_returns
        width = self.software_subgroup_width

        def call_writes(call):
            if id(call) in call_mutations:
                return call_mutations[id(call)]
            return self.hlsl_software_subgroup_expression_identifier_names(
                list(call.arguments or [])
            )

        def storage_parameter(parameter):
            parameter_type = getattr(parameter, "param_type", None)
            if isinstance(parameter_type, PointerType):
                return True
            if isinstance(parameter_type, ReferenceType):
                return parameter_type.address_space not in {None, "", "thread"}
            return bool(
                analysis.resource_type(parameter_type, parameter, read_only=False)
            )

        def private_reference(parameter):
            parameter_type = getattr(parameter, "param_type", None)
            if isinstance(parameter_type, ReferenceType):
                return not storage_parameter(parameter)
            return bool(
                set(self.hlsl_parameter_qualifiers(parameter)) & {"out", "inout"}
            ) and bool(analysis.value_type(parameter_type))

        def seeded_parameter(parameter):
            parameter_type = getattr(parameter, "param_type", None)
            return (
                not isinstance(parameter_type, ReferenceType)
                and not set(self.hlsl_parameter_qualifiers(parameter))
                & {"out", "inout"}
                and bool(
                    analysis.value_type(parameter_type)
                    or analysis.resource_type(parameter_type, parameter)
                )
            )

        return LaneSymmetryAnalysis(
            SymmetryRules(
                walk=self.walk_ast,
                map_operator=self.map_operator,
                collective=self.hlsl_software_subgroup_collective_operation,
                call_target=lambda call: resolved_calls.get(id(call)),
                pure_call=lambda call: id(call)
                in self.hlsl_software_subgroup_uniform_calls,
                call_dependencies=analysis.dependencies,
                volatile=lambda declaration: any(
                    analysis._qualified(node, {"volatile"})
                    for node in (declaration, getattr(declaration, "var_type", None))
                ),
                storage_parameter=storage_parameter,
                private_reference=private_reference,
                storage_load=analysis.storage_load,
                constant_int=lambda expression: evaluate_literal_int_expression(
                    expression, self.literal_int_constants
                ),
                call_writes=call_writes,
                seeded_parameter=seeded_parameter,
                width=width,
                full_subgroups=bool(invocation_count) and invocation_count % width == 0,
                reductions=frozenset(self.HLSL_SOFTWARE_SUBGROUP_REDUCTIONS),
                broadcasts=frozenset({"WaveReadLaneAt"}),
                lane_preserving=frozenset(
                    {
                        *self.HLSL_SOFTWARE_SUBGROUP_SCANS,
                        *self.HLSL_SOFTWARE_SUBGROUP_RELATIVE_SHUFFLES,
                    }
                ),
            )
        )

    SOFTWARE_SUBGROUP_SYSTEM_INPUTS = frozenset(
        {
            "gl_SubgroupInvocationID",
            "gl_SubgroupSize",
            "gl_SubgroupID",
            "gl_NumSubgroups",
        }
    )

    def hlsl_software_subgroup_bind_lane_local_matrix_functions(self, ast, nodes):
        """Give every function holding a transfer the logical invocation index.

        Transfers address each lane's fragment elements and exchange nothing,
        so any function may perform them. The entry stores ``SV_GroupIndex``
        in a static global, which HLSL keeps private to each invocation.
        """
        if not nodes:
            return
        node_ids = {id(node) for node in nodes}
        functions = [
            function
            for function in self.collect_functions(ast)
            if getattr(function, "name", None)
            and any(
                id(node) in node_ids
                for node in self.walk_ast(getattr(function, "body", None))
            )
        ]
        variable = self.hlsl_software_subgroup_invocation_variable
        if variable is None:
            reserved_names = self.hlsl_helper_reserved_names(
                self.collect_functions(ast)
            )
            self.hlsl_software_subgroup_reserved_names |= reserved_names
            variable = "__crossgl_software_subgroup_invocation"
            while variable in self.hlsl_software_subgroup_reserved_names:
                variable += "_"
            self.hlsl_software_subgroup_invocation_variable = variable
        for function in functions:
            self.hlsl_software_subgroup_invocation_expressions.setdefault(
                function.name, variable
            )

    def hlsl_software_subgroup_stage_entries(self, ast, target_stage=None):
        entries = []
        seen = set()
        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if target_stage is not None and stage_name != target_stage:
                continue
            function = getattr(stage, "entry_point", None)
            if function is None or id(function) in seen:
                continue
            seen.add(id(function))
            entries.append(
                (stage_name, function, getattr(stage, "execution_config", None))
            )
        for function in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(function)
            if stage_name is None:
                continue
            if target_stage is not None and stage_name != target_stage:
                continue
            if id(function) in seen:
                continue
            seen.add(id(function))
            entries.append(
                (
                    stage_name,
                    function,
                    getattr(function, "execution_config", None),
                )
            )
        return entries

    def hlsl_software_subgroup_expression_identifier_names(self, expression):
        return {
            node.name
            for node in self.walk_ast(expression)
            if isinstance(node, IdentifierNode) and getattr(node, "name", None)
        }

    def hlsl_software_subgroup_assignment_target_name(self, node):
        target = getattr(node, "target", getattr(node, "left", None))
        return self.hlsl_texture_offset_write_root_name(target)

    def hlsl_software_subgroup_uniform_expression(
        self, expression, uniform_names, uniform_components
    ):
        if isinstance(expression, LiteralNode):
            return True
        symmetry = getattr(self, "hlsl_software_subgroup_symmetry", None)
        if (
            symmetry is not None
            and symmetry.collective_derived(expression)
            and symmetry.level(expression) == UNIFORM
        ):
            # A collective of lane-symmetric values, or a name or helper result
            # built from one, that every invocation holding it agrees on.
            # Values without a collective keep the proofs below.
            return True
        if isinstance(expression, IdentifierNode):
            return expression.name in uniform_names
        load = self.hlsl_software_subgroup_uniform_returns.storage_load(expression)
        if isinstance(expression, ArrayAccessNode) or load is not None:
            base, index = load or (expression.array_expr, expression.index_expr)
            if (
                not isinstance(base, IdentifierNode)
                or base.name not in self.hlsl_software_subgroup_resource_names
            ):
                return False
            children = [base, index]
        elif isinstance(expression, (MemberAccessNode, SwizzleNode)):
            base = (
                expression.object_expr
                if isinstance(expression, MemberAccessNode)
                else expression.vector_expr
            )
            member = (
                expression.member
                if isinstance(expression, MemberAccessNode)
                else expression.components
            )
            if isinstance(base, IdentifierNode) and base.name in uniform_components:
                # A uniform component does not make the rest of the vector uniform.
                positions = {
                    letter: index
                    for alphabet in ("xyzw", "rgba")
                    for index, letter in enumerate(alphabet)
                }
                return bool(member) and all(
                    positions.get(letter) in uniform_components[base.name]
                    for letter in member
                )
            children = [base]
        elif isinstance(expression, FunctionCallNode):
            children = self.hlsl_software_subgroup_uniform_returns.uniform_arguments(
                expression, self.hlsl_software_subgroup_resource_names
            )
            if children is None:
                return False
        elif isinstance(expression, ConstructorNode):
            children = [*expression.arguments, *expression.named_arguments.values()]
        elif isinstance(expression, CastNode):
            children = [expression.expression]
        elif isinstance(expression, BinaryOpNode):
            children = [expression.left, expression.right]
        elif isinstance(expression, UnaryOpNode):
            if self.map_operator(expression.operator) not in {"+", "-", "!", "~"}:
                return False
            children = [expression.operand]
        elif isinstance(expression, TernaryOpNode):
            children = [
                expression.condition,
                expression.true_expr,
                expression.false_expr,
            ]
        else:
            return False
        return all(
            self.hlsl_software_subgroup_uniform_expression(
                child, uniform_names, uniform_components
            )
            for child in children
        )

    def hlsl_software_subgroup_fixed_stride(
        self,
        initializer,
        bound,
        update,
        constants,
        operator,
        metadata_bounds=None,
        any_start=False,
    ):
        if self.hlsl_software_subgroup_uniform_returns._qualified(
            initializer, {"volatile"}
        ) or self.hlsl_software_subgroup_uniform_returns._qualified(
            initializer.var_type, {"volatile"}
        ):
            return False
        proof = (
            geometric_loop_terminates
            if update.operator in {"*=", "<<="}
            else fixed_stride_loop_terminates
        )
        return proof(
            self.type_name_string(initializer.var_type),
            initializer.initial_value,
            bound,
            update,
            constants,
            operator,
            **(
                {"metadata_bounds": metadata_bounds, "any_start": any_start}
                if proof is fixed_stride_loop_terminates
                else {}
            ),
        )
