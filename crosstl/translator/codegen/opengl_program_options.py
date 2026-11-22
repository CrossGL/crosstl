"""Program options of the GLSL generator: software subgroup widths, entry reachability, built-in options, version and ES requirements, and software subgroup reachable nodes."""

import re
from copy import (
    copy,
    deepcopy,
)

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    CastNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IdentifierPatternNode,
    IfNode,
    LiteralNode,
    MatchNode,
    PreprocessorNode,
    StageMap,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    WaveOpNode,
)
from ..explicit_specializations import functions_by_called_name
from ..refusals import (
    OpenGLRefusal,
    TranslatorArgumentError,
)
from .apple_sdk_includes import is_apple_sdk_include
from .array_utils import evaluate_literal_int_expression
from .entry_selection import entry_type_references
from .glsl_compute_derivative_groups import compute_derivative_group_extension_lines
from .glsl_fragment_interlock import fragment_interlock_extension_lines
from .glsl_scalar_block_layout import scalar_block_layout_extension_line
from .glsl_tessellation_modes import remember_scoped_control_sources
from .glsl_version_requirements import (
    generated_program_requirements,
    program_extensions,
    resolve_version_line,
    with_es_default_precisions,
)
from .glsl_vulkan_descriptor_sets import descriptor_binding_qualifier
from .opengl_values import OpenGLEntryPointSelectionError
from .stage_utils import (
    collect_stage_entry_records,
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)
from .texture_lod_clamp import (
    GLSL_LOD_CLAMP_EXTENSION,
    TEXTURE_LOD_CLAMP_INTRINSIC_NAMES,
)
from .texture_residency import (
    GLSL_RESIDENCY_EXTENSION,
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENT_OPERATION,
)


class OpenGLProgramOptionMixin:
    """Program option members of ``GLSLCodeGen``."""

    def set_software_subgroup_width(self, width):
        """Configure the explicit, fail-closed OpenGL software subgroup mode."""
        if width is not None and (
            isinstance(width, bool) or not isinstance(width, int)
        ):
            raise TranslatorArgumentError(
                "software_subgroup_width must be an integer or None"
            )
        if width not in {None, self.GLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH}:
            raise OpenGLRefusal(
                "software_subgroup_width currently supports only an exact width "
                f"of {self.GLSL_SOFTWARE_SUBGROUP_SUPPORTED_WIDTH}"
            )
        self.software_subgroup_width = width
        return self

    def generate(self, ast):
        """Generate complete GLSL source for a CrossGL AST."""
        return self.generate_program(ast)

    def generate_entry(self, ast, entry_point):
        """Generate one independently loadable GLSL stage entry artifact."""
        scoped_ast = self.entry_scoped_ast(ast, entry_point)
        with self.opengl_descriptor_set_plan_for(ast):
            return self.generate_program(scoped_ast)

    def entry_scoped_ast(self, ast, entry_point):
        if not isinstance(entry_point, str) or not entry_point.strip():
            raise OpenGLEntryPointSelectionError(
                "OpenGL entry-point selection requires a non-empty source entry name",
                entry_point=entry_point,
                reason="invalid-name",
            )
        entry_point = entry_point.strip()
        stage_entry_types = self.combined_stage_entry_types()
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
            raise OpenGLEntryPointSelectionError(
                f"OpenGL source entry point '{entry_point}' was not found",
                entry_point=entry_point,
                available_entry_points=available,
                reason="not-found",
            )
        if len(matches) != 1:
            raise OpenGLEntryPointSelectionError(
                f"OpenGL source entry point '{entry_point}' is ambiguous",
                entry_point=entry_point,
                available_entry_points=available,
                reason="ambiguous",
            )

        _selected_id, selected_stage_name, selected_function = matches[0]
        selected_stage = next(
            (
                (stage_type, stage)
                for stage_type, stage in getattr(ast, "stages", {}).items()
                if getattr(stage, "entry_point", None) is selected_function
            ),
            None,
        )
        # Copy only what the entry can use: another entry's function or stage is never reachable from it, and
        # copying every entry for every entry made translating a file entry by entry quadratic.
        scoped_ast = copy(ast)
        scoped_ast.functions = [
            function
            for function in getattr(ast, "functions", []) or []
            if function_stage_name(function) not in stage_entry_types
            or function is selected_function
        ]
        scoped_ast.stages = StageMap()
        scoped_ast = deepcopy(scoped_ast)
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
            raise OpenGLEntryPointSelectionError(
                f"OpenGL source entry point '{entry_point}' could not be copied",
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
        self.prune_entry_scoped_declarations(scoped_ast, selected_copy_function)
        remember_scoped_control_sources(self, ast, scoped_ast)
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
        referenced_types = set()
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
            referenced_types.update(
                entry_type_references(function, self.entry_ast_nodes)
            )
        referenced_names.difference_update(function_names)
        referenced_names.update(referenced_types)

        # GLSL keeps struct declarations, so their field types must remain available.
        structs = list(getattr(ast, "structs", []) or [])
        for stage in getattr(ast, "stages", {}).values():
            structs.extend(getattr(stage, "local_structs", []) or [])
        for declaration in structs:
            referenced_names.update(
                entry_type_references(declaration, self.entry_ast_nodes)
            )

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
                referenced_names.update(
                    entry_type_references(declaration, self.entry_ast_nodes)
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
        return self.reachable_function_names(ast, [entry_function])

    def reachable_function_names(self, ast, entry_functions):
        functions = list(getattr(ast, "functions", []) or [])
        for _stage_type, stage in getattr(ast, "stages", {}).items():
            functions.extend(getattr(stage, "local_functions", []) or [])
        functions_by_name = functions_by_called_name(functions)

        reachable_names = set()
        pending = list(entry_functions)
        visited = set()
        while pending:
            function = pending.pop()
            if id(function) in visited:
                continue
            visited.add(id(function))
            for node in self.walk_ast(getattr(function, "body", [])):
                name = None
                if isinstance(node, FunctionCallNode):
                    name = self.function_call_name(node)
                elif isinstance(node, IdentifierNode):
                    name = getattr(node, "name", None)
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

    def stage_reachable_function_names(self, ast, target_stage=None):
        entries = collect_stage_entry_records(
            ast,
            target_stage,
            self.combined_stage_entry_types(),
        )
        if not entries:
            return None
        return self.reachable_function_names(
            ast,
            [function for _id, _stage, function in entries],
        )

    def generate_stage(self, ast, shader_type):
        """Generate GLSL source for a single requested shader stage."""
        return self.generate_program(ast, target_stage=shader_type)

    def with_glsl_builtin_option_prelude(self, ast):
        if not self.glsl_ast_needs_builtin_option(ast):
            return ast
        if self.glsl_ast_declares_type(ast, "Option"):
            return ast

        ast = deepcopy(ast)
        prelude_ast = self.parse_glsl_builtin_option_prelude()
        ast.structs = list(getattr(prelude_ast, "structs", []) or []) + list(
            getattr(ast, "structs", []) or []
        )
        ast.functions = list(getattr(prelude_ast, "functions", []) or []) + list(
            getattr(ast, "functions", []) or []
        )
        return ast

    def parse_glsl_builtin_option_prelude(self):
        from ..lexer import Lexer
        from ..parser import Parser

        source = """
        shader OpenGLBuiltinOption {
            generic<T> struct Option {
                enum OptionType { Some(T), None }
                OptionType variant;
            }

            generic<T> bool is_Some(Option<T> item) {
                return item.variant == Option::Some;
            }

            generic<T> bool is_None(Option<T> item) {
                return item.variant == Option::None;
            }

            generic<T> T unwrap_Some(Option<T> item) {
                return item.Some_0;
            }
        }
        """
        return Parser(Lexer(source).tokens).parse()

    def glsl_ast_declares_type(self, ast, type_name):
        return any(
            getattr(struct, "name", None) == type_name
            for struct in getattr(ast, "structs", []) or []
        )

    def glsl_ast_needs_builtin_option(self, ast):
        option_helpers = {"is_Some", "is_None", "unwrap_Some"}
        for node in self.walk_ast(ast):
            if self.glsl_node_type_uses_builtin_option(node):
                return True
            if self.function_call_name(node) in option_helpers:
                return True
        return False

    def glsl_node_type_uses_builtin_option(self, node):
        for attr in (
            "return_type",
            "param_type",
            "var_type",
            "vtype",
            "member_type",
            "const_type",
            "constructor_type",
        ):
            if self.glsl_type_uses_builtin_option(getattr(node, attr, None)):
                return True
        return False

    def glsl_type_uses_builtin_option(self, type_value):
        type_name = self.type_name_string(type_value)
        return isinstance(type_name, str) and re.search(
            r"(?:^|::|\b)Option\s*<", type_name
        )

    def glsl_stage_names(self, ast, target_stage=None):
        stage_names = set()

        for func in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(func)
            if should_emit_qualified_function(target_stage, stage_name):
                stage_names.add(stage_name)

        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_matches(target_stage, stage_name):
                stage_names.add(stage_name)

        return stage_names

    def default_glsl_version_line(self, ast, target_stage=None):
        if self.glsl_stage_names(
            ast, target_stage
        ) & self.RAY_STAGE_NAMES or self.uses_ray_extension_type(ast, target_stage):
            return "#version 460 core"
        return "#version 450 core"

    def glsl_version_number(self, version_line):
        parts = str(version_line or "").strip().split()
        if len(parts) < 2 or parts[0] != "#version":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def glsl_resource_binding_layouts_supported(self, version_line):
        if self.glsl_extension_enabled("GL_ARB_shading_language_420pack"):
            return True

        version_number = self.glsl_version_number(version_line)
        if version_number is None:
            return True
        # layout(binding) arrived in GLSL 4.20 and GLSL ES 3.10 (section 4.4.5 and 4.4.6).
        return version_number >= (310 if self.glsl_version_is_es(version_line) else 420)

    def glsl_version_line_with_minimum(self, version_line, minimum_version):
        version_number = self.glsl_version_number(version_line)
        if version_number is None or version_number >= minimum_version:
            return version_line

        parts = str(version_line or "").strip().split()
        profile_parts = parts[2:] if len(parts) > 2 else []
        if "es" in profile_parts:
            es_minimum = 300 if minimum_version <= 330 else 320
            if version_number >= es_minimum:
                return version_line
            return f"#version {es_minimum} es"

        profile = next(
            (part for part in profile_parts if part in {"core", "compatibility"}),
            "core",
        )
        return f"#version {minimum_version} {profile}"

    def glsl_version_line_is_fixed(self):
        """Whether the target profile pins the ``#version`` line, so a construct above it is refused."""
        return False

    def ensure_glsl_version_supports_generated_layouts(self, code):
        """Raise the ``#version`` line to the lowest one the rendered program's constructs need.

        Constructs that no version of the profile has, and any raise when the version is fixed, are refused. GLSL
        ES output then declares ``highp`` defaults for every type that lacks one.
        """
        lines = str(code).splitlines(keepends=True)
        version_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().startswith("#version")
            ),
            None,
        )
        if version_index is None:
            return code
        version_line = lines[version_index].strip()
        requirements = generated_program_requirements(
            code,
            sorted(getattr(self, "current_glsl_stage_names", ()) or ()),
            self.GLSL_STAGE_GUARD_MACROS,
        )
        for requirement in getattr(self, "glsl_emission_requirements", ()):
            if requirement not in requirements:
                requirements.append(requirement)
        extensions = set(self.current_glsl_extensions) | program_extensions(code)
        upgraded, refusals = resolve_version_line(
            version_line,
            requirements,
            extensions,
            can_raise=not self.glsl_version_line_is_fixed(),
        )
        for reason in refusals:
            self.refuse_lowering(f"GLSL version requirement: {reason}")
        if upgraded != version_line:
            line_end = "\n" if lines[version_index].endswith("\n") else ""
            lines[version_index] = f"{upgraded}{line_end}"
            self.current_glsl_version_line = upgraded
            self.glsl_raised_version_line = upgraded
            code = "".join(lines)
        if self.glsl_version_is_es(upgraded):
            code = with_es_default_precisions(code)
        return code

    def should_emit_resource_binding_layouts(self):
        return getattr(
            self,
            "current_glsl_resource_binding_layouts_supported",
            True,
        )

    def glsl_resource_layout_prefix(self, *layout_parts, binding=None, node=None):
        parts = [
            str(part).strip()
            for part in layout_parts
            if part is not None and str(part).strip()
        ]
        if binding is not None and self.should_emit_resource_binding_layouts():
            parts.append(self.glsl_binding_qualifier(binding, node))
        if not parts:
            return ""
        return f"layout({', '.join(parts)})"

    def glsl_binding_qualifier(self, binding, node=None):
        """``binding = N``, or the Vulkan/OpenGL binding macro for a declaration in a descriptor set other than 0."""
        descriptor_set = (
            self.resource_descriptor_set_index(node) if node is not None else 0
        )
        vulkan_binding = (
            self.raw_descriptor_binding_index(node) if descriptor_set else None
        )
        if vulkan_binding is None:
            return f"binding = {binding}"
        self.glsl_descriptor_binding_macro_used = True
        return descriptor_binding_qualifier(descriptor_set, vulkan_binding, binding)

    def glsl_preprocessor_extensions(self, ast):
        extensions = set()
        for directive in getattr(ast, "preprocessors", []) or []:
            if not isinstance(directive, PreprocessorNode):
                continue
            if directive.directive != "extension":
                continue

            content = str(getattr(directive, "content", "") or "").strip()
            if not content:
                continue

            extension_name, separator, behavior_text = content.partition(":")
            extension_name_parts = extension_name.strip().split()
            if not extension_name_parts:
                continue
            extension_name = extension_name_parts[0]
            behavior = behavior_text.strip().split()[0] if separator else ""
            if behavior in {"", "enable", "require"}:
                extensions.add(extension_name)
        return extensions

    def should_emit_glsl_preprocessor_line(self, line):
        stripped = line.strip()
        if stripped.startswith("#pragma METAL"):
            return False
        if not stripped.startswith("#include"):
            return True

        include_target = stripped[len("#include") :].strip()
        if include_target.startswith("<") and is_apple_sdk_include(include_target):
            return False
        return not include_target.lower().startswith(("<metal", '"metal'))

    def should_emit_glsl_preprocessor_directive(self, directive):
        if not isinstance(directive, PreprocessorNode):
            return True
        if str(directive.directive).lstrip("#") != "include":
            return True

        content = str(getattr(directive, "content", "") or "").strip()
        if not content:
            return True
        match = re.match(r'^[<"]([^>"]+)[>"]', content)
        include_name = match.group(1) if match else content.split()[0]
        normalized = include_name.strip().replace("\\", "/").lower()
        if content.startswith("<") and is_apple_sdk_include(normalized):
            return False
        return normalized not in self.METAL_ONLY_SYSTEM_INCLUDES

    def glsl_extension_enabled(self, extension_name):
        return extension_name in self.current_glsl_extensions

    def glsl_version_is_es(self, version_line):
        return "es" in str(version_line or "").strip().split()[2:]

    def glsl_version_supports_bit_casts(self):
        """Return whether floatBitsToUint exists: GLSL 3.30 or GLSL ES 3.00."""
        version_line = getattr(self, "current_glsl_version_line", None)
        version_number = self.glsl_version_number(version_line)
        if version_number is None:
            return True
        return version_number >= (300 if self.glsl_version_is_es(version_line) else 330)

    def ensure_glsl_int64_support(self, code):
        # Narrowing can retain a wide literal without any wide declarations.
        uses_int64 = re.search(
            r"\b(?:int64_t|uint64_t|i64vec[234]|u64vec[234]|[0-9]+u?l)\b",
            str(code),
        )
        if uses_int64 is None:
            return code

        version_line = self.current_glsl_version_line
        if self.glsl_version_is_es(version_line):
            raise OpenGLRefusal(
                f"OpenGL target profile '{version_line}' does not support "
                f"{self.GLSL_INT64_EXTENSION}; select a desktop GLSL profile "
                "for 64-bit integer values"
            )

        disabled_extension = re.search(
            rf"^\s*#extension\s+{re.escape(self.GLSL_INT64_EXTENSION)}\s*:\s*"
            r"disable\b",
            str(code),
            flags=re.MULTILINE,
        )
        if disabled_extension is not None:
            raise OpenGLRefusal(
                f"OpenGL output requires {self.GLSL_INT64_EXTENSION}, but the "
                "source profile disables that extension"
            )

        lines = str(code).splitlines(keepends=True)
        if (self.glsl_version_number(version_line) or 0) < 400:
            upgraded_version = self.glsl_version_line_with_minimum(version_line, 400)
            for index, line in enumerate(lines):
                if not line.strip().startswith("#version"):
                    continue
                line_end = "\n" if line.endswith("\n") else ""
                lines[index] = f"{upgraded_version}{line_end}"
                break
            self.current_glsl_version_line = upgraded_version
            self.glsl_raised_version_line = upgraded_version

        has_extension = any(
            re.match(
                rf"^\s*#extension\s+{re.escape(self.GLSL_INT64_EXTENSION)}\s*:\s*"
                r"(?:enable|require|warn)\b",
                line,
            )
            for line in lines
        )
        if not has_extension:
            version_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.strip().startswith("#version")
                ),
                -1,
            )
            lines.insert(version_index + 1, f"{self.GLSL_INT64_EXTENSION_LINE}\n")
        self.current_glsl_extensions.add(self.GLSL_INT64_EXTENSION)
        return "".join(lines)

    def glsl_texture_offset_non_const_enabled(self):
        return self.glsl_extension_enabled("GL_EXT_texture_offset_non_const")

    def glsl_stage_extension_lines(self, ast, target_stage=None):
        stage_names = self.glsl_stage_names(ast, target_stage)

        lines = []
        if self.uses_nonuniform_ext(ast):
            lines.append("#extension GL_EXT_nonuniform_qualifier : require")
        # A program written against GL_NV_mesh_shader keeps it: glslang refuses both extensions together.
        if stage_names & self.MESH_STAGE_NAMES and not self.glsl_extension_enabled(
            "GL_NV_mesh_shader"
        ):
            lines.append("#extension GL_EXT_mesh_shader : require")
        uses_ray_stage = bool(stage_names & self.RAY_STAGE_NAMES)
        if uses_ray_stage:
            lines.append("#extension GL_EXT_ray_tracing : require")
        if self.uses_ray_query(ast, target_stage) or (
            self.uses_acceleration_structure(ast, target_stage) and not uses_ray_stage
        ):
            lines.append("#extension GL_EXT_ray_query : require")
        if self.uses_ray_tracing_position_fetch(ast, target_stage):
            lines.append("#extension GL_EXT_ray_tracing_position_fetch : require")
        if self.uses_fragment_stencil_export(ast, target_stage):
            lines.append("#extension GL_ARB_shader_stencil_export : require")
        if self.uses_texture_lod_clamp(ast, target_stage) and not (
            self.glsl_version_is_es(getattr(self, "current_glsl_version_line", None))
        ):
            # GLSL ES has no such extension; the version requirements refuse the clamp functions there.
            lines.append(f"#extension {GLSL_LOD_CLAMP_EXTENSION} : require")
        if self.uses_texture_residency(ast, target_stage) and not (
            self.glsl_version_is_es(getattr(self, "current_glsl_version_line", None))
        ):
            # GLSL ES has no such extension; the version requirements refuse the sparse functions there.
            lines.append(f"#extension {GLSL_RESIDENCY_EXTENSION} : require")
        lines.extend(self.glsl_wave_extension_lines(ast, target_stage))
        if self.uses_glsl_debug_printf(ast):
            lines.append("#extension GL_EXT_debug_printf : require")
        lines.extend(self.glsl_shader_clock_extension_lines(ast))
        lines.extend(fragment_interlock_extension_lines(self, ast))
        lines.extend(compute_derivative_group_extension_lines(self, ast, target_stage))
        scalar_layout = scalar_block_layout_extension_line(self, ast)
        if scalar_layout is not None:
            lines.append(scalar_layout)
        return lines

    def glsl_shader_clock_extension_lines(self, ast):
        """The shader clock built-ins are only declared once their extension is requested."""
        enabled = {
            extension
            for extension in set(self.GLSL_SHADER_CLOCK_EXTENSIONS.values())
            if any(
                extension in str(getattr(directive, "content", "") or "")
                for directive in getattr(ast, "preprocessors", []) or []
            )
        }
        needed = {
            self.GLSL_SHADER_CLOCK_EXTENSIONS[name]
            for node in self.walk_ast(ast)
            if isinstance(node, FunctionCallNode)
            for name in [self.function_call_name(node)]
            if name in self.GLSL_SHADER_CLOCK_EXTENSIONS
        }
        return [
            f"#extension {extension} : require"
            for extension in sorted(needed - enabled)
        ]

    def uses_glsl_debug_printf(self, ast):
        """HLSL's ``printf`` (a SPIR-V-only DXC feature) is GLSL's ``debugPrintfEXT``."""
        for directive in getattr(ast, "preprocessors", []) or []:
            if "GL_EXT_debug_printf" in str(getattr(directive, "content", "") or ""):
                return False
        return any(
            isinstance(node, FunctionCallNode)
            and self.function_call_name(node) in {"printf", "debugPrintfEXT"}
            for node in self.walk_ast(ast)
        )

    def uses_texture_residency(self, ast, target_stage=None):
        for directive in getattr(ast, "preprocessors", []) or []:
            if GLSL_RESIDENCY_EXTENSION in str(getattr(directive, "content", "") or ""):
                return False  # The source enables the extension itself.
        return any(
            isinstance(node, FunctionCallNode)
            and self.function_call_name(node)
            in {TEXTURE_RESIDENCY_OPERATION, TEXTURE_RESIDENT_OPERATION}
            for root in self.ray_query_search_roots(ast, target_stage)
            for node in self.walk_ast(root)
        )

    def uses_texture_lod_clamp(self, ast, target_stage=None):
        return any(
            isinstance(node, FunctionCallNode)
            and self.function_call_name(node) in TEXTURE_LOD_CLAMP_INTRINSIC_NAMES
            for root in self.ray_query_search_roots(ast, target_stage)
            for node in self.walk_ast(root)
        )

    def uses_fragment_stencil_export(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        if target_stage is not None and target_stage != "fragment":
            return False
        if "fragment" not in self.glsl_stage_names(ast, target_stage):
            return False

        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                semantic = self.semantic_from_node(node)
                if semantic is None:
                    continue
                mapped_semantic = self.map_semantic(semantic)
                if mapped_semantic in {
                    "gl_FragStencilRefARB",
                    "gl_FragStencilRefEXT",
                }:
                    return True
        return False

    def glsl_wave_operations(self, ast, target_stage=None):
        operations = set()
        for root in self.ray_query_search_roots(ast, target_stage):
            for node in self.walk_ast(root):
                if isinstance(node, WaveOpNode):
                    operations.add(node.operation)
                elif isinstance(node, FunctionCallNode):
                    operation = self.glsl_wave_operation_name(
                        self.function_call_name(node)
                    )
                    if operation in self.GLSL_WAVE_INTRINSIC_ARITIES:
                        operations.add(operation)
        return operations

    def uses_glsl_subgroup_basic_builtin(self, ast, target_stage=None):
        for root in self.ray_query_search_roots(ast, target_stage):
            referenced_names = set()
            subgroup_semantic_names = set()
            for node in self.walk_ast(root):
                if isinstance(node, IdentifierNode):
                    referenced_names.add(node.name)
                source_name = getattr(node, "name", None)
                if not isinstance(source_name, str):
                    continue
                semantic = self.semantic_from_node(node)
                if semantic is None:
                    continue
                if self.map_semantic(semantic) in self.GLSL_SUBGROUP_BASIC_BUILTINS:
                    subgroup_semantic_names.add(source_name)
            if referenced_names & (
                self.GLSL_SUBGROUP_BASIC_BUILTINS | subgroup_semantic_names
            ):
                return True
            for scope in self.glsl_call_scopes(root):
                for node in self.walk_ast(scope):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    function_name = self.function_call_name(node)
                    if (
                        function_name in self.GLSL_SUBGROUP_BARRIER_FUNCTIONS
                        and not self.glsl_function_defined_for_scope(
                            function_name, scope
                        )
                    ):
                        return True
        return False

    @staticmethod
    def glsl_call_scopes(root):
        """The parts of ``root`` whose calls see different helper functions."""
        stages = getattr(root, "stages", None)
        if not stages:
            return [root]
        scopes = list(stages.values())
        scopes.extend(getattr(root, "functions", []) or [])
        return scopes

    def glsl_function_defined_for_scope(self, function_name, scope):
        """Whether a call to ``function_name`` inside ``scope`` names a user function.

        A helper declared inside one stage block is emitted only in that stage, so
        a call from any other stage still names the built-in.
        """
        if function_name not in self.function_return_types:
            return False
        if function_name in self.glsl_global_function_names:
            return True
        defined_in_a_stage = any(
            function_name in names
            for names in self.glsl_stage_local_function_names.values()
        )
        if not defined_in_a_stage:
            return True
        return any(
            getattr(function, "name", None) == function_name
            for function in getattr(scope, "local_functions", []) or []
        )

    def uses_nonuniform_ext(self, ast):
        for node in self.walk_ast(ast):
            if isinstance(node, FunctionCallNode) and self.function_call_name(node) in {
                "nonuniformEXT",
                "NonUniformResourceIndex",
            }:
                return True
        return False

    def glsl_software_subgroup_proof_int_type(self, value_type):
        """Return whether ``value_type`` has proof-safe signed-int semantics."""
        type_name = (
            value_type
            if isinstance(value_type, str)
            else self.type_name_string(value_type)
        )
        normalized = "".join(str(type_name or "").strip().lower().split())
        return normalized in {"int", "int32", "int32_t", "signed", "signedint"}

    def glsl_software_subgroup_integral_variable(self, variable):
        if not isinstance(variable, VariableNode):
            return False
        # Unsigned and narrow integer declarations need source-width conversion
        # modeling before their comparisons can prove reachability.  Keep this
        # optimization fail-closed on the signed 32-bit domain used by GLSL int.
        return self.glsl_software_subgroup_proof_int_type(
            getattr(variable, "var_type", None)
        )

    def glsl_software_subgroup_safe_int_value(self, expression, constants):
        """Evaluate a signed-int expression without admitting wrap or coercion."""
        value = evaluate_literal_int_expression(expression, constants)
        int_min = -(1 << 31)
        int_max = (1 << 31) - 1
        # ``type(...) is int`` deliberately rejects the evaluator's private
        # unsigned-provenance marker, which subclasses int.
        if type(value) is not int or not int_min <= value <= int_max:
            return None

        if isinstance(expression, int) and not isinstance(expression, bool):
            return value
        if isinstance(expression, str):
            return value if expression in constants else None
        if isinstance(expression, IdentifierNode):
            return value if getattr(expression, "name", None) in constants else None
        if isinstance(expression, LiteralNode):
            return (
                value
                if self.glsl_software_subgroup_proof_int_type(
                    getattr(expression, "literal_type", None)
                )
                else None
            )
        if isinstance(expression, UnaryOpNode):
            operator = self.map_operator(
                getattr(expression, "op", getattr(expression, "operator", None))
            )
            if operator not in {"+", "-", "~", "!"}:
                return None
            return (
                value
                if self.glsl_software_subgroup_safe_int_value(
                    getattr(expression, "operand", None), constants
                )
                is not None
                else None
            )
        if isinstance(expression, BinaryOpNode):
            operator = self.map_operator(
                getattr(expression, "op", getattr(expression, "operator", None))
            )
            if operator not in {
                "+",
                "-",
                "*",
                "/",
                "%",
                "&",
                "|",
                "^",
                "&&",
                "||",
                "<",
                "<=",
                ">",
                ">=",
                "==",
                "!=",
            }:
                return None
            left = self.glsl_software_subgroup_safe_int_value(
                getattr(expression, "left", None), constants
            )
            right = self.glsl_software_subgroup_safe_int_value(
                getattr(expression, "right", None), constants
            )
            if left is None or right is None:
                return None
            # Signed INT_MIN / -1 and INT_MIN % -1 are undefined/overflowing in
            # the source integer model even though Python can represent them.
            if operator in {"/", "%"} and left == int_min and right == -1:
                return None
            return value
        if isinstance(expression, TernaryOpNode):
            children = (
                getattr(expression, "condition", None),
                getattr(expression, "true_expr", None),
                getattr(expression, "false_expr", None),
            )
            return (
                value
                if all(
                    self.glsl_software_subgroup_safe_int_value(child, constants)
                    is not None
                    for child in children
                )
                else None
            )
        if isinstance(expression, CastNode):
            target_type = getattr(expression, "target_type", None)
            operand = getattr(expression, "expression", None)
        elif isinstance(expression, ConstructorNode):
            target_type = getattr(expression, "constructor_type", None)
            arguments = list(getattr(expression, "arguments", []) or [])
            operand = arguments[0] if len(arguments) == 1 else None
        elif isinstance(expression, FunctionCallNode):
            target_type = self.function_call_name(expression)
            arguments = list(getattr(expression, "arguments", []) or [])
            operand = arguments[0] if len(arguments) == 1 else None
        else:
            return None
        if (
            operand is None
            or not self.glsl_software_subgroup_proof_int_type(target_type)
            or self.glsl_software_subgroup_safe_int_value(operand, constants) is None
        ):
            return None
        return value

    def glsl_software_subgroup_const_int_value(self, variable, constants):
        if not self.glsl_software_subgroup_integral_variable(
            variable
        ) or "const" not in {
            str(qualifier).strip().lower()
            for qualifier in getattr(variable, "qualifiers", []) or []
        }:
            return None
        visible_constants = dict(constants)
        visible_constants.pop(getattr(variable, "name", None), None)
        return self.glsl_software_subgroup_safe_int_value(
            getattr(variable, "initial_value", None),
            visible_constants,
        )

    def glsl_software_subgroup_initial_int_constants(self, constants):
        """Filter global constants to declarations with signed-int provenance."""
        constant_types = getattr(self, "glsl_constant_types", {})
        return {
            name: value
            for name, value in dict(constants or {}).items()
            if self.glsl_software_subgroup_proof_int_type(constant_types.get(name))
            and type(value) is int
            and -(1 << 31) <= value <= (1 << 31) - 1
        }

    def glsl_software_subgroup_module_constant_shadow_names(self, root):
        """Return non-constant module names that block name-only proofs.

        Stage-local declarations share the emitted entry's source namespace but
        are stored beside the entry in ``StageNode`` rather than in its body.
        Likewise, cbuffer members and imports may be referenced without an
        explicit owner.  If any such declaration reuses a global constant's
        name, declaration identity is ambiguous to this intentionally narrow
        analysis, so retain the loop rather than selecting either declaration.
        """
        shadows = set()

        def add_declarations(declarations):
            for declaration in declarations or []:
                name = getattr(
                    declaration,
                    "name",
                    getattr(declaration, "variable_name", None),
                )
                if name:
                    shadows.add(str(name))

        def add_cbuffers(cbuffers):
            for cbuffer in cbuffers or []:
                name = getattr(cbuffer, "name", None)
                if name:
                    shadows.add(str(name))
                add_declarations(getattr(cbuffer, "members", []) or [])

        add_declarations(getattr(root, "global_variables", []) or [])
        add_cbuffers(getattr(root, "cbuffers", []) or [])
        for import_node in getattr(root, "imports", []) or []:
            alias = getattr(import_node, "alias", None)
            if alias:
                shadows.add(str(alias))
            shadows.update(
                str(item) for item in getattr(import_node, "items", []) or [] if item
            )
        shadows.update(self.glsl_preprocessor_identifier_names(root))

        stages = getattr(root, "stages", {}) or {}
        stage_values = stages.values() if hasattr(stages, "values") else stages
        for stage in stage_values:
            add_declarations(getattr(stage, "local_variables", []) or [])
            add_cbuffers(getattr(stage, "local_cbuffers", []) or [])
        return shadows

    def glsl_software_subgroup_proven_zero_trip_for(self, node, constants=None):
        """Prove that a source ``for`` condition is false before its first body.

        The proof is intentionally narrow.  It accepts only a signed 32-bit
        loop declaration with a concrete, non-wrapping initializer and a
        comparison whose loop variable occurs on exactly one side.  The
        opposite bound and full initial condition must both be evaluable from
        explicitly supplied signed lexical constants.  Calls, mutable aliases,
        narrowing/wrap, unsigned coercions, and unsupported expressions remain
        reachable.
        """
        if not isinstance(node, ForNode):
            return False
        initializer = getattr(node, "init", None)
        if not self.glsl_software_subgroup_integral_variable(initializer):
            return False
        loop_name = getattr(initializer, "name", None)
        if not loop_name:
            return False

        visible_constants = dict(
            self.glsl_active_literal_int_constants() if constants is None else constants
        )
        # A loop declaration shadows any outer alias before its initializer is
        # interpreted.  Never let a same-named outer constant prove the loop.
        visible_constants.pop(loop_name, None)
        initial_value = self.glsl_software_subgroup_safe_int_value(
            getattr(initializer, "initial_value", None),
            visible_constants,
        )
        if initial_value is None:
            return False

        condition = getattr(node, "condition", None)
        if not isinstance(condition, BinaryOpNode):
            return False
        operator = self.map_operator(
            getattr(condition, "op", getattr(condition, "operator", None))
        )
        if operator not in {"<", "<=", ">", ">=", "==", "!="}:
            return False

        def contains_loop_name(expression):
            return any(
                isinstance(child, IdentifierNode)
                and getattr(child, "name", None) == loop_name
                for child in self.walk_ast(expression)
            )

        left_contains_loop = contains_loop_name(condition.left)
        right_contains_loop = contains_loop_name(condition.right)
        if left_contains_loop == right_contains_loop:
            return False
        loop_expression = condition.left if left_contains_loop else condition.right
        if self.expression_name(loop_expression) != loop_name:
            return False
        bound = condition.right if left_contains_loop else condition.left
        if self.glsl_software_subgroup_safe_int_value(bound, visible_constants) is None:
            return False

        initial_constants = dict(visible_constants)
        initial_constants[loop_name] = initial_value
        return (
            self.glsl_software_subgroup_safe_int_value(
                condition,
                initial_constants,
            )
            == 0
        )

    def glsl_software_subgroup_reachable_nodes(
        self,
        root,
        constants=None,
        *,
        discover_zero_trip=False,
    ):
        """Walk subgroup-relevant AST nodes with lexical dead-loop pruning."""
        visited = set()
        supplied_constants = (
            self.glsl_active_literal_int_constants() if constants is None else constants
        )
        initial_constants = self.glsl_software_subgroup_initial_int_constants(
            supplied_constants
        )
        for name in self.glsl_software_subgroup_module_constant_shadow_names(root):
            initial_constants.pop(name, None)
        zero_trip_ids = getattr(
            self,
            "glsl_software_subgroup_zero_trip_for_node_ids",
            set(),
        )

        def walk_sequence(values, constants_in_scope):
            lexical_constants = dict(constants_in_scope)
            for statement in values or []:
                name = (
                    getattr(statement, "name", None)
                    if isinstance(statement, VariableNode)
                    else None
                )
                if name:
                    lexical_constants.pop(name, None)
                yield from walk(statement, lexical_constants)
                if name:
                    value = self.glsl_software_subgroup_const_int_value(
                        statement,
                        lexical_constants,
                    )
                    if value is not None:
                        lexical_constants[name] = value
                mutation = statement
                if isinstance(statement, ExpressionStatementNode):
                    mutation = getattr(statement, "expression", None)
                if isinstance(mutation, AssignmentNode):
                    target_name = self.glsl_software_subgroup_assignment_target_name(
                        mutation
                    )
                    if target_name:
                        lexical_constants.pop(target_name, None)
                elif isinstance(mutation, UnaryOpNode) and self.map_operator(
                    getattr(mutation, "op", getattr(mutation, "operator", None))
                ) in {"++", "--"}:
                    target_name = self.expression_name(
                        getattr(mutation, "operand", None)
                    )
                    if target_name:
                        lexical_constants.pop(target_name, None)

        def walk(value, constants_in_scope):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            if isinstance(value, dict):
                for child in value.values():
                    yield from walk(child, dict(constants_in_scope))
                return
            if isinstance(value, (list, tuple)):
                yield from walk_sequence(value, constants_in_scope)
                return
            if isinstance(value, (set, frozenset)):
                # Sets have no lexical order.  Visit every sibling from the same
                # incoming scope rather than transporting declarations between
                # an arbitrary iteration order.
                for child in value:
                    yield from walk(child, dict(constants_in_scope))
                return

            value_id = id(value)
            if value_id in visited:
                return
            visited.add(value_id)
            yield value

            if isinstance(value, BlockNode) or hasattr(value, "statements"):
                yield from walk_sequence(
                    getattr(value, "statements", []) or [],
                    constants_in_scope,
                )
                return
            if isinstance(value, VariableNode):
                yield from walk(
                    getattr(value, "initial_value", None),
                    constants_in_scope,
                )
                return
            if isinstance(value, ForNode):
                loop_constants = dict(constants_in_scope)
                initializer = getattr(value, "init", None)
                loop_name = getattr(initializer, "name", None)
                if loop_name:
                    loop_constants.pop(loop_name, None)
                yield from walk(initializer, loop_constants)
                yield from walk(getattr(value, "condition", None), loop_constants)
                zero_trip = value_id in zero_trip_ids
                if (
                    not zero_trip
                    and discover_zero_trip
                    and self.glsl_software_subgroup_proven_zero_trip_for(
                        value,
                        loop_constants,
                    )
                ):
                    zero_trip_ids.add(value_id)
                    zero_trip = True
                if not zero_trip:
                    yield from walk(
                        getattr(value, "body", None),
                        dict(loop_constants),
                    )
                # Keep the update conservative even though a zero-trip loop
                # cannot execute it; only the body is removed from discovery.
                yield from walk(getattr(value, "update", None), loop_constants)
                return
            if isinstance(value, ForInNode):
                yield from walk(
                    getattr(value, "iterable", None),
                    constants_in_scope,
                )
                body_constants = dict(constants_in_scope)
                pattern = getattr(value, "pattern", None)
                pattern_name = (
                    pattern
                    if isinstance(pattern, str)
                    else getattr(pattern, "name", getattr(pattern, "identifier", None))
                )
                if pattern_name:
                    body_constants.pop(pattern_name, None)
                yield from walk(getattr(value, "body", None), body_constants)
                return
            if isinstance(value, MatchNode):
                yield from walk(
                    getattr(value, "expression", None),
                    constants_in_scope,
                )
                for arm in getattr(value, "arms", []) or []:
                    arm_id = id(arm)
                    if arm_id not in visited:
                        visited.add(arm_id)
                        yield arm
                    pattern = getattr(arm, "pattern", None)
                    yield from walk(pattern, constants_in_scope)
                    arm_constants = dict(constants_in_scope)
                    for pattern_node in self.walk_ast(pattern):
                        if not isinstance(pattern_node, IdentifierPatternNode):
                            continue
                        pattern_name = getattr(pattern_node, "name", None)
                        if (
                            pattern_name
                            and pattern_name != ".."
                            and "::" not in pattern_name
                        ):
                            arm_constants.pop(pattern_name, None)
                    yield from walk(getattr(arm, "guard", None), arm_constants)
                    yield from walk(getattr(arm, "body", None), arm_constants)
                return
            if isinstance(value, IfNode):
                yield from walk(
                    getattr(value, "condition", getattr(value, "if_condition", None)),
                    constants_in_scope,
                )
                yield from walk(
                    getattr(value, "if_body", None),
                    dict(constants_in_scope),
                )
                for condition, body in zip(
                    getattr(value, "else_if_conditions", []) or [],
                    getattr(value, "else_if_bodies", []) or [],
                ):
                    yield from walk(condition, constants_in_scope)
                    yield from walk(body, dict(constants_in_scope))
                yield from walk(
                    getattr(value, "else_body", None),
                    dict(constants_in_scope),
                )
                return
            if hasattr(value, "body") and hasattr(value, "parameters"):
                function_constants = dict(constants_in_scope)
                for parameter in getattr(value, "parameters", []) or []:
                    function_constants.pop(getattr(parameter, "name", None), None)
                yield from walk(getattr(value, "body", None), function_constants)
                return
            if hasattr(value, "__dict__"):
                for name, child in vars(value).items():
                    if name == "parent":
                        continue
                    yield from walk(child, dict(constants_in_scope))

        yield from walk(root, initial_constants)
