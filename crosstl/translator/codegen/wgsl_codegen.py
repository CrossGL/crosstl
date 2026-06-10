"""CrossGL-to-WGSL code generator."""

from __future__ import annotations

import re

from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    GenericType,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MatchNode,
    MatrixType,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    RangeNode,
    ReferenceType,
    ReturnNode,
    SwitchNode,
    SwizzleNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WhileNode,
)
from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name


class WGSLCodeGen:
    """Generate WebGPU WGSL output from CrossGL ASTs."""

    SUPPORTED_STAGE_NAMES = {"vertex", "fragment", "compute"}
    UNSUPPORTED_STAGE_NAMES = {
        "geometry",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "amplification",
        "object",
        "ray_generation",
        "raygen",
        "ray_intersection",
        "intersection",
        "ray_any_hit",
        "any_hit",
        "ray_closest_hit",
        "closest_hit",
        "ray_miss",
        "miss",
        "ray_callable",
        "callable",
    }
    VECTOR_TYPE_RE = re.compile(r"^(?:vec|float|int|uint|bool)([234])$")
    MATRIX_TYPE_RE = re.compile(r"^(?:mat|float)([234])(?:x([234]))?$")
    TYPE_CONSTRUCTOR_RE = re.compile(
        r"^(?:vec[234]|float[234]|int[234]|uint[234]|bool[234]|mat[234](?:x[234])?|float[234]x[234])$"
    )

    PRIMITIVE_TYPE_MAP = {
        "void": "void",
        "bool": "bool",
        "boolean": "bool",
        "int": "i32",
        "i32": "i32",
        "short": "i32",
        "long": "i32",
        "uint": "u32",
        "u32": "u32",
        "unsigned": "u32",
        "float": "f32",
        "f32": "f32",
        "half": "f32",
        "double": "f32",
        "f64": "f32",
    }
    BUILTIN_SEMANTICS = {
        "gl_position": "position",
        "position_builtin": "position",
        "sv_position": "position",
        "frag_depth": "frag_depth",
        "sv_depth": "frag_depth",
        "vertex_index": "vertex_index",
        "vertexid": "vertex_index",
        "sv_vertexid": "vertex_index",
        "instance_index": "instance_index",
        "instanceid": "instance_index",
        "sv_instanceid": "instance_index",
        "global_invocation_id": "global_invocation_id",
        "gl_globalinvocationid": "global_invocation_id",
        "sv_dispatchthreadid": "global_invocation_id",
        "local_invocation_id": "local_invocation_id",
        "gl_localinvocationid": "local_invocation_id",
        "sv_groupthreadid": "local_invocation_id",
        "local_invocation_index": "local_invocation_index",
        "gl_localinvocationindex": "local_invocation_index",
        "sv_groupindex": "local_invocation_index",
        "workgroup_id": "workgroup_id",
        "gl_workgroupid": "workgroup_id",
        "sv_groupid": "workgroup_id",
        "num_workgroups": "num_workgroups",
    }
    BUILTIN_IDENTIFIER_ALIASES = {
        "gl_Position": "position",
        "SV_Position": "position",
        "gl_GlobalInvocationID": "global_invocation_id",
        "SV_DispatchThreadID": "global_invocation_id",
        "gl_LocalInvocationID": "local_invocation_id",
        "SV_GroupThreadID": "local_invocation_id",
        "gl_LocalInvocationIndex": "local_invocation_index",
        "SV_GroupIndex": "local_invocation_index",
        "gl_WorkGroupID": "workgroup_id",
        "SV_GroupID": "workgroup_id",
        "gl_NumWorkGroups": "num_workgroups",
    }
    WORKGROUP_SIZE_IDENTIFIER_ALIASES = {"gl_WorkGroupSize"}
    INPUT_BUILTIN_TYPE_MAP = {
        "vertex_index": "u32",
        "instance_index": "u32",
        "global_invocation_id": "vec3<u32>",
        "local_invocation_id": "vec3<u32>",
        "local_invocation_index": "u32",
        "workgroup_id": "vec3<u32>",
        "num_workgroups": "vec3<u32>",
    }
    FUNCTION_NAME_MAP = {
        "atan2": "atan2",
        "fract": "fract",
        "inversesqrt": "inverseSqrt",
        "inverseSqrt": "inverseSqrt",
        "lerp": "mix",
        "mix": "mix",
        "mod": "mod",
        "rsqrt": "inverseSqrt",
        "saturate": "saturate",
    }
    STAGE_INPUT_BUILTINS = {
        "vertex": {"instance_index", "vertex_index"},
        "fragment": set(),
        "compute": {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "num_workgroups",
            "workgroup_id",
        },
    }
    RESOURCE_TYPE_NAMES = {
        "image1d",
        "image1darray",
        "image2d",
        "image2darray",
        "image2dms",
        "image2dmsarray",
        "image3d",
        "imagebuffer",
        "imagecube",
        "imagecubearray",
        "iimage1d",
        "iimage1darray",
        "iimage2d",
        "iimage2darray",
        "iimage2dms",
        "iimage2dmsarray",
        "iimage3d",
        "iimagebuffer",
        "iimagecube",
        "iimagecubearray",
        "sampler",
        "sampler1d",
        "sampler1darray",
        "sampler2d",
        "sampler2darray",
        "sampler2darrayshadow",
        "sampler2dms",
        "sampler2dmsarray",
        "sampler2dshadow",
        "sampler3d",
        "samplercube",
        "samplercubearray",
        "samplerstate",
        "samplercomparisonstate",
        "texture1d",
        "texture1darray",
        "texture2d",
        "texture2darray",
        "texture2dms",
        "texture2dmsarray",
        "texture3d",
        "texturecube",
        "texturecubearray",
        "uimage1d",
        "uimage1darray",
        "uimage2d",
        "uimage2darray",
        "uimage2dms",
        "uimage2dmsarray",
        "uimage3d",
        "uimagebuffer",
        "uimagecube",
        "uimagecubearray",
    }
    STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
        "structuredbuffer",
    }
    WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES = {
        "rwstructuredbuffer",
    }
    SAMPLED_TEXTURE_TYPE_MAP = {
        "sampler1d": "texture_1d<f32>",
        "sampler2d": "texture_2d<f32>",
        "sampler2darray": "texture_2d_array<f32>",
        "sampler3d": "texture_3d<f32>",
        "samplercube": "texture_cube<f32>",
        "samplercubearray": "texture_cube_array<f32>",
        "texture1d": "texture_1d<f32>",
        "texture2d": "texture_2d<f32>",
        "texture2darray": "texture_2d_array<f32>",
        "texture3d": "texture_3d<f32>",
        "texturecube": "texture_cube<f32>",
        "texturecubearray": "texture_cube_array<f32>",
    }
    SAMPLER_TYPE_NAMES = {
        "sampler",
        "samplerstate",
    }
    TEXTURE_FUNCTION_NAMES = {
        "texture",
        "texturecompare",
        "texturecomparegrad",
        "texturecompareoffset",
        "texturecomparelod",
        "texturecomparelodoffset",
        "texturecompareproj",
        "texturecompareprojgrad",
        "texturecompareprojoffset",
        "texturegrad",
        "texturegradoffset",
        "texturegather",
        "texturegathercompare",
        "texturegathercompareoffset",
        "texturegatheroffset",
        "texturegatheroffsets",
        "texturelod",
        "texturelodoffset",
        "textureoffset",
        "textureproj",
        "textureprojgrad",
        "textureprojgradoffset",
        "textureprojlod",
        "textureprojlodoffset",
        "textureprojoffset",
        "texturequerylevels",
        "texturequerylod",
        "texturesize",
    }
    BARRIER_FUNCTION_NAMES = {
        "barrier",
        "groupmemorybarrierwithgroupsync",
        "workgroupbarrier",
    }

    def __init__(self):
        self._current_stage_name = None
        self._current_workgroup_size = None
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._global_binding_index = 0
        self._cbuffer_member_accesses = {}
        self._function_texture_parameters = {}
        self._function_pointer_parameters = {}
        self._identifier_scopes = []
        self._pointer_identifier_scopes = []

    def generate(self, ast):
        return self.generate_program(ast)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        self.validate_wgsl_stage_support(ast, target_stage)
        self._location_counters = {"in": 0, "out": 0, "generic": 0}
        self._global_binding_index = 0
        self._identifier_scopes = []
        self._pointer_identifier_scopes = []

        lines = ["// Generated by CrossGL for WebGPU WGSL"]
        emitted_sections = []

        cbuffers = self._collect_cbuffers(ast, target_stage)
        self._cbuffer_member_accesses = self.cbuffer_member_accesses(cbuffers)
        self._function_texture_parameters = self.function_texture_parameters(
            ast, target_stage
        )
        self._function_pointer_parameters = self.function_buffer_pointer_parameters(
            ast, target_stage
        )

        structs = self._collect_structs(ast, target_stage)
        if structs:
            emitted_sections.append(
                "\n\n".join(self.generate_struct(node) for node in structs)
            )

        constants = [
            self.generate_constant(node) for node in getattr(ast, "constants", []) or []
        ]
        if constants:
            emitted_sections.append("\n".join(constants))

        if cbuffers:
            emitted_sections.append(
                "\n\n".join(self.generate_cbuffer(node) for node in cbuffers)
            )

        global_variables = [
            self.generate_global_variable(node)
            for node in self._collect_global_variables(ast, target_stage)
        ]
        if global_variables:
            emitted_sections.append("\n".join(global_variables))

        helper_functions = [
            self.generate_function(func)
            for func in self._helper_functions(ast, target_stage)
        ]
        if helper_functions:
            emitted_sections.append("\n\n".join(helper_functions))

        stage_functions = []
        stage_name_counts = {}
        for stage_node in self._stage_nodes(ast, target_stage):
            stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
            stage_index = stage_name_counts.get(stage_name, 0)
            stage_name_counts[stage_name] = stage_index + 1
            entry_name = f"{stage_name}_main"
            if stage_index:
                entry_name = f"{entry_name}_{stage_index}"
            stage_functions.append(self.generate_stage(stage_node, entry_name))
        if stage_functions:
            emitted_sections.append("\n\n".join(stage_functions))

        if emitted_sections:
            lines.append("")
            lines.append(
                "\n\n".join(section for section in emitted_sections if section)
            )
        return "\n".join(lines).rstrip() + "\n"

    def validate_wgsl_stage_support(self, ast, target_stage=None):
        stages = set()
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages.add(normalized_target_stage)

        for func in getattr(ast, "functions", []) or []:
            qualifier = (
                func.qualifiers[0]
                if getattr(func, "qualifiers", None)
                else getattr(func, "qualifier", None)
            )
            stage_name = normalize_stage_name(qualifier)
            if stage_name:
                stages.add(stage_name)
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)
            entry_point = getattr(stage_node, "entry_point", None)
            qualifier = (
                entry_point.qualifiers[0]
                if getattr(entry_point, "qualifiers", None)
                else None
            )
            entry_stage = normalize_stage_name(qualifier)
            if entry_stage:
                stages.add(entry_stage)

        unsupported = sorted(
            stage
            for stage in stages
            if stage in self.UNSUPPORTED_STAGE_NAMES
            or (stage and stage not in self.SUPPORTED_STAGE_NAMES)
        )
        if unsupported:
            raise ValueError(
                "WGSL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

    def generate_stage(self, stage_node, entry_name=None):
        stage_name = normalize_stage_name(getattr(stage_node, "stage", None))
        if stage_name not in self.SUPPORTED_STAGE_NAMES:
            raise ValueError(f"WGSL target does not support shader stage: {stage_name}")

        entry_point = stage_node.entry_point
        previous_stage = self._current_stage_name
        previous_workgroup_size = self._current_workgroup_size
        self._current_stage_name = stage_name
        self._current_workgroup_size = (
            self.stage_workgroup_size_values(stage_node)
            if stage_name == "compute"
            else None
        )
        try:
            attributes = [f"@{stage_name}"]
            if stage_name == "compute":
                attributes.append(self.generate_workgroup_size_attribute(stage_node))
            signature = self.generate_function_signature(
                entry_point,
                name=entry_name or f"{stage_name}_main",
                return_attributes=self.stage_return_attributes(stage_name, entry_point),
                leading_parameters=self.stage_implicit_builtin_parameters(
                    entry_point, stage_name
                ),
            )
            self.push_identifier_scope(
                getattr(param, "name", "")
                for param in getattr(entry_point, "parameters", [])
            )
            self.push_pointer_identifier_scope(
                self.buffer_pointer_parameter_names(entry_point)
            )
            try:
                body = self.generate_block(entry_point.body, indent=0)
            finally:
                self.pop_pointer_identifier_scope()
                self.pop_identifier_scope()
            return "\n".join(attributes + [f"{signature} {body}"])
        finally:
            self._current_stage_name = previous_stage
            self._current_workgroup_size = previous_workgroup_size

    def generate_workgroup_size_attribute(self, stage_node):
        return (
            "@workgroup_size("
            + ", ".join(self.stage_workgroup_size_values(stage_node))
            + ")"
        )

    def stage_workgroup_size_values(self, stage_node):
        config = getattr(stage_node, "execution_config", {}) or {}
        values = config.get("numthreads") or [
            config.get("local_size_x", "1"),
            config.get("local_size_y", "1"),
            config.get("local_size_z", "1"),
        ]
        if isinstance(values, str):
            values = [part.strip() for part in values.split(",")]
        values = list(values)[:3]
        while len(values) < 3:
            values.append("1")
        return tuple(str(value) for value in values)

    def generate_function(self, func):
        self.validate_helper_function_references(func)
        signature = self.generate_function_signature(func)
        self.push_identifier_scope(
            getattr(param, "name", "") for param in getattr(func, "parameters", [])
        )
        self.push_pointer_identifier_scope(self.buffer_pointer_parameter_names(func))
        try:
            body = self.generate_block(func.body, indent=0)
        finally:
            self.pop_pointer_identifier_scope()
            self.pop_identifier_scope()
        return f"{signature} {body}"

    def generate_function_signature(
        self, func, name=None, return_attributes=(), leading_parameters=()
    ):
        function_name = name or func.name
        parameters = ", ".join(
            list(leading_parameters)
            + [self.generate_parameter(param) for param in func.parameters]
        )
        return_type = self.type_name_string(func.return_type)
        if return_type == "void":
            return f"fn {function_name}({parameters})"
        return_prefix = " ".join(return_attributes)
        if return_prefix:
            return f"fn {function_name}({parameters}) -> {return_prefix} {return_type}"
        return f"fn {function_name}({parameters}) -> {return_type}"

    def generate_parameter(self, node):
        attributes = self.wgsl_attributes(node.attributes, direction="in")
        prefix = f"{attributes} " if attributes else ""
        if self.structured_buffer_element_type(node.param_type) is not None:
            raise ValueError(
                "WGSL target does not support StructuredBuffer parameters yet; "
                "declare them as module-scope storage resources"
            )
        sampled_texture_type = self.sampled_texture_type(node.param_type)
        if sampled_texture_type is not None:
            return (
                f"{prefix}{node.name}: {sampled_texture_type}, "
                f"{self.texture_sampler_name(node.name)}: sampler"
            )
        if self.is_sampler_type(node.param_type):
            return f"{prefix}{node.name}: sampler"
        if self.is_buffer_pointer_type(node.param_type, node.qualifiers):
            return f"{prefix}{node.name}: {self.buffer_pointer_parameter_type(node.param_type)}"
        return f"{prefix}{node.name}: {self.type_name_string(node.param_type)}"

    def stage_implicit_builtin_parameters(self, function, stage_name):
        referenced = self.stage_direct_builtin_references(function)
        workgroup_size_references = self.direct_workgroup_size_references(function)
        if not referenced and not workgroup_size_references:
            return ()

        if workgroup_size_references and stage_name != "compute":
            raise ValueError(
                "WGSL target does not support gl_WorkGroupSize outside compute stages"
            )

        existing = self.existing_parameter_builtin_names(function)

        unsupported = sorted(
            original
            for original, builtin in referenced.items()
            if builtin not in self.INPUT_BUILTIN_TYPE_MAP
        )
        if unsupported:
            raise ValueError(
                "WGSL target does not support implicit builtin identifier(s): "
                + ", ".join(unsupported)
                + "; model them as entry-point parameters or return values"
            )

        stage_allowed = self.STAGE_INPUT_BUILTINS.get(stage_name, set())
        invalid_for_stage = sorted(
            original
            for original, builtin in referenced.items()
            if builtin in self.INPUT_BUILTIN_TYPE_MAP and builtin not in stage_allowed
        )
        if invalid_for_stage:
            raise ValueError(
                "WGSL target does not support implicit builtin identifier(s) "
                f"in {stage_name} stage: "
                + ", ".join(invalid_for_stage)
                + "; model them as entry-point parameters or return values"
            )

        parameter_names = {
            getattr(param, "name", "") for param in getattr(function, "parameters", [])
        }
        colliding_builtins = sorted(
            builtin
            for builtin in set(referenced.values())
            if builtin in parameter_names and builtin not in existing
        )
        if colliding_builtins:
            builtin = colliding_builtins[0]
            raise ValueError(
                "WGSL target cannot inject builtin "
                f"{builtin} because an entry parameter already uses that name "
                f"without @builtin({builtin})"
            )

        parameters = []
        for builtin, builtin_type in self.INPUT_BUILTIN_TYPE_MAP.items():
            if builtin not in referenced.values() or builtin in existing:
                continue
            parameters.append(f"@builtin({builtin}) {builtin}: {builtin_type}")
        return tuple(parameters)

    def validate_helper_function_references(self, function):
        builtin_references = self.stage_direct_builtin_references(function)
        if builtin_references:
            raise ValueError(
                "WGSL target does not support direct builtin identifier(s) "
                f"in helper function {function.name}: "
                + ", ".join(sorted(builtin_references))
                + "; pass builtin values through entry-point parameters instead"
            )

        workgroup_size_references = self.direct_workgroup_size_references(function)
        if workgroup_size_references:
            raise ValueError(
                "WGSL target does not support gl_WorkGroupSize inside helper "
                f"function {function.name}; keep it directly in compute entry points"
            )

        barrier_calls = self.function_barrier_call_names(function)
        if barrier_calls:
            raise ValueError(
                "WGSL target does not support barrier() inside helper function "
                f"{function.name} yet; keep barriers directly in compute entry points"
            )

    def direct_workgroup_size_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        references = []
        for node in body.walk():
            if (
                isinstance(node, IdentifierNode)
                and node.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES
            ):
                references.append(node.name)
        return tuple(references)

    def function_barrier_call_names(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return ()
        calls = []
        for node in body.walk():
            if not isinstance(node, FunctionCallNode):
                continue
            function_name = self.expression_name(node.function)
            if self.semantic_key(function_name) in self.BARRIER_FUNCTION_NAMES:
                calls.append(function_name)
        return tuple(calls)

    def stage_direct_builtin_references(self, function):
        body = getattr(function, "body", None)
        if body is None or not hasattr(body, "walk"):
            return {}
        referenced = {}
        for node in body.walk():
            if not isinstance(node, IdentifierNode):
                continue
            builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(node.name)
            if builtin:
                referenced[node.name] = builtin
        return referenced

    def existing_parameter_builtin_names(self, function):
        builtin_names = set()
        for param in function.parameters:
            for attr in getattr(param, "attributes", []) or []:
                key = self.semantic_key(str(getattr(attr, "name", attr)))
                builtin = self.BUILTIN_SEMANTICS.get(key)
                if builtin:
                    builtin_names.add(builtin)
        return builtin_names

    def generate_struct(self, node):
        if getattr(node, "generic_params", None):
            raise ValueError("WGSL target does not support generic structs yet")
        lines = [f"struct {node.name} {{"]
        for member in node.members:
            resource_type_name = self.struct_member_resource_type_name(member)
            if resource_type_name:
                raise ValueError(
                    "WGSL target does not support resource member "
                    f"{node.name}.{member.name} of type {resource_type_name}; "
                    "declare textures, samplers, and storage resources as "
                    "module-scope bindings instead of user-struct fields"
                )
            attributes = self.wgsl_attributes(member.attributes, direction="generic")
            prefix = f"{attributes} " if attributes else ""
            lines.append(
                f"    {prefix}{member.name}: {self.type_name_string(member.member_type)},"
            )
        lines.append("};")
        return "\n".join(lines)

    def generate_cbuffer(self, node):
        lines = [f"struct {node.name} {{"]
        for member in getattr(node, "members", []) or []:
            lines.append(
                f"    {member.name}: {self.type_name_string(member.member_type)},"
            )
        lines.append("};")

        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        instance_name = self.cbuffer_instance_name(node)
        lines.append(f"{attributes}\nvar<uniform> {instance_name}: {node.name};")
        return "\n".join(lines)

    def generate_constant(self, node):
        value = self.generate_expression(node.value)
        return f"const {node.name}: {self.type_name_string(node.const_type)} = {value};"

    def generate_global_variable(self, node):
        sampled_texture_type = self.sampled_texture_type(node.var_type)
        if sampled_texture_type is not None:
            return self.generate_sampled_texture_global_variable(
                node, sampled_texture_type
            )
        if self.is_sampler_type(node.var_type):
            return self.generate_sampler_global_variable(node)
        if self.is_buffer_pointer_type(node.var_type, node.qualifiers):
            return self.generate_buffer_pointer_global_variable(node)

        qualifier_names = {str(qualifier).lower() for qualifier in node.qualifiers}
        address_space = "private"
        access = ""
        attributes = ""
        storage_buffer_access = self.structured_buffer_access(node.var_type)
        if storage_buffer_access:
            address_space = "storage"
            access = f", {storage_buffer_access}"
            attributes = (
                self.explicit_binding_attributes(node) or self.next_binding_attributes()
            )
        elif "uniform" in qualifier_names:
            address_space = "uniform"
            attributes = self.next_binding_attributes()
        elif "buffer" in qualifier_names or "storage" in qualifier_names:
            address_space = "storage"
            access = ", read_write"
            attributes = self.next_binding_attributes()
        elif "workgroup" in qualifier_names or "shared" in qualifier_names:
            address_space = "workgroup"

        initializer = ""
        if node.initial_value is not None and address_space == "private":
            initializer = f" = {self.generate_expression(node.initial_value)}"
        prefix = f"{attributes}\n" if attributes else ""
        return (
            f"{prefix}var<{address_space}{access}> {node.name}: "
            f"{self.type_name_string(node.var_type, allow_storage_resources=True)}"
            f"{initializer};"
        )

    def generate_sampled_texture_global_variable(self, node, texture_type):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for sampled texture "
                f"resource {node.name}"
            )
        texture_attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        sampler_attributes = self.next_binding_attributes()
        sampler_name = self.texture_sampler_name(node.name)
        return (
            f"{texture_attributes}\nvar {node.name}: {texture_type};\n"
            f"{sampler_attributes}\nvar {sampler_name}: sampler;"
        )

    def generate_sampler_global_variable(self, node):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for sampler resource "
                f"{node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return f"{attributes}\nvar {node.name}: sampler;"

    def generate_buffer_pointer_global_variable(self, node):
        if node.initial_value is not None:
            raise ValueError(
                "WGSL target does not support initializers for buffer pointer "
                f"resource {node.name}"
            )
        attributes = (
            self.explicit_binding_attributes(node) or self.next_binding_attributes()
        )
        return (
            f"{attributes}\nvar<storage, read_write> {node.name}: "
            f"{self.buffer_pointer_storage_type(node.var_type)};"
        )

    def generate_statement(self, stmt, indent=0):
        pad = "    " * indent
        if isinstance(stmt, BlockNode):
            return self.generate_block(stmt, indent)
        if isinstance(stmt, VariableNode):
            mutable_keyword = "var" if stmt.is_mutable else "let"
            initializer = ""
            if stmt.initial_value is not None:
                initializer = f" = {self.generate_expression(stmt.initial_value)}"
            line = (
                f"{pad}{mutable_keyword} {stmt.name}: "
                f"{self.type_name_string(stmt.var_type)}{initializer};"
            )
            self.register_local_identifier(stmt.name)
            return line
        if isinstance(stmt, ExpressionStatementNode):
            return f"{pad}{self.generate_expression(stmt.expression)};"
        if isinstance(stmt, AssignmentNode):
            return f"{pad}{self.generate_assignment(stmt)};"
        if isinstance(stmt, ReturnNode):
            if stmt.value is None:
                return f"{pad}return;"
            return f"{pad}return {self.generate_expression(stmt.value)};"
        if isinstance(stmt, IfNode):
            return self.generate_if(stmt, indent)
        if isinstance(stmt, ForNode):
            return self.generate_for(stmt, indent)
        if isinstance(stmt, WhileNode):
            return f"{pad}while ({self.generate_expression(stmt.condition)}) {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, LoopNode):
            return f"{pad}loop {self.generate_block(stmt.body, indent)}"
        if isinstance(stmt, BreakNode):
            return f"{pad}break;"
        if isinstance(stmt, ContinueNode):
            return f"{pad}continue;"
        if isinstance(stmt, SwitchNode):
            return self.generate_switch(stmt, indent)
        if isinstance(stmt, DoWhileNode):
            raise ValueError("WGSL target does not support do-while statements")
        if isinstance(stmt, ForInNode):
            raise ValueError("WGSL target does not support for-in statements")
        if isinstance(stmt, MatchNode):
            raise ValueError("WGSL target does not support match statements")
        raise ValueError(
            f"WGSL target does not support statement {type(stmt).__name__}"
        )

    def generate_block(self, block, indent=0):
        if block is None:
            return "{}"
        statements = getattr(block, "statements", [])
        if not statements:
            return "{}"
        pad = "    " * indent
        lines = [f"{pad}{{"]
        self.push_identifier_scope()
        try:
            for stmt in statements:
                lines.append(self.generate_statement(stmt, indent + 1))
        finally:
            self.pop_identifier_scope()
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_if(self, node, indent):
        pad = "    " * indent
        code = (
            f"{pad}if ({self.generate_expression(node.condition)}) "
            f"{self.generate_block(node.then_branch, indent)}"
        )
        if node.else_branch is not None:
            code += f" else {self.generate_block(node.else_branch, indent)}"
        return code

    def generate_for(self, node, indent):
        pad = "    " * indent
        self.push_identifier_scope()
        try:
            init = self.generate_for_initializer(node.init)
            condition = (
                self.generate_expression(node.condition) if node.condition else ""
            )
            update = self.generate_expression(node.update) if node.update else ""
            return (
                f"{pad}for ({init}; {condition}; {update}) "
                f"{self.generate_block(node.body, indent)}"
            )
        finally:
            self.pop_identifier_scope()

    def generate_for_initializer(self, init):
        if init is None:
            return ""
        if isinstance(init, VariableNode):
            initializer = ""
            if init.initial_value is not None:
                initializer = f" = {self.generate_expression(init.initial_value)}"
            line = (
                f"var {init.name}: {self.type_name_string(init.var_type)}"
                f"{initializer}"
            )
            self.register_local_identifier(init.name)
            return line
        if isinstance(init, AssignmentNode):
            return self.generate_assignment(init)
        if isinstance(init, ExpressionStatementNode):
            return self.generate_expression(init.expression)
        return self.generate_expression(init)

    def generate_switch(self, node, indent):
        pad = "    " * indent
        lines = [f"{pad}switch ({self.generate_expression(node.expression)}) {{"]
        for case in node.cases:
            lines.append(f"{pad}    case {self.generate_expression(case.value)}: {{")
            for stmt in case.statements:
                lines.append(self.generate_statement(stmt, indent + 2))
            lines.append(f"{pad}    }}")
        if node.default_case is not None:
            lines.append(
                f"{pad}    default: {self.generate_block(node.default_case, indent + 1)}"
            )
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    def generate_assignment(self, node):
        return (
            f"{self.generate_expression(node.target)} {node.operator} "
            f"{self.generate_expression(node.value)}"
        )

    def generate_expression(self, expr):
        if expr is None:
            return ""
        if isinstance(expr, LiteralNode):
            return self.generate_literal(expr)
        if isinstance(expr, IdentifierNode):
            if expr.name in self.WORKGROUP_SIZE_IDENTIFIER_ALIASES:
                return self.generate_workgroup_size_literal()
            mapped_builtin = self.BUILTIN_IDENTIFIER_ALIASES.get(expr.name)
            if mapped_builtin:
                return mapped_builtin
            if not self.is_local_identifier(expr.name):
                cbuffer_access = self._cbuffer_member_accesses.get(expr.name)
                if cbuffer_access:
                    return cbuffer_access
            return expr.name
        if isinstance(expr, BinaryOpNode):
            return (
                f"({self.generate_expression(expr.left)} {expr.operator} "
                f"{self.generate_expression(expr.right)})"
            )
        if isinstance(expr, UnaryOpNode):
            operand = self.generate_expression(expr.operand)
            if expr.is_postfix:
                if expr.operator == "++":
                    return f"{operand} += 1"
                if expr.operator == "--":
                    return f"{operand} -= 1"
                return f"{operand}{expr.operator}"
            return f"{expr.operator}{operand}"
        if isinstance(expr, TernaryOpNode):
            return (
                f"select({self.generate_expression(expr.false_expr)}, "
                f"{self.generate_expression(expr.true_expr)}, "
                f"{self.generate_expression(expr.condition)})"
            )
        if isinstance(expr, FunctionCallNode):
            return self.generate_function_call(expr)
        if isinstance(expr, ConstructorNode):
            return self.generate_constructor(expr)
        if isinstance(expr, MemberAccessNode):
            return f"{self.generate_expression(expr.object_expr)}." f"{expr.member}"
        if isinstance(expr, SwizzleNode):
            return f"{self.generate_expression(expr.vector_expr)}." f"{expr.components}"
        if isinstance(expr, ArrayAccessNode):
            if isinstance(
                expr.array_expr, IdentifierNode
            ) and self.is_pointer_identifier(expr.array_expr.name):
                return (
                    f"(*{expr.array_expr.name})"
                    f"[{self.generate_expression(expr.index_expr)}]"
                )
            return (
                f"{self.generate_expression(expr.array_expr)}"
                f"[{self.generate_expression(expr.index_expr)}]"
            )
        if isinstance(expr, ArrayLiteralNode):
            return (
                "array("
                + ", ".join(
                    self.generate_expression(element) for element in expr.elements
                )
                + ")"
            )
        if isinstance(expr, CastNode):
            return (
                f"{self.type_name_string(expr.target_type)}"
                f"({self.generate_expression(expr.expression)})"
            )
        if isinstance(expr, RangeNode):
            raise ValueError("WGSL target does not support range expressions")
        if isinstance(expr, AssignmentNode):
            return self.generate_assignment(expr)
        raise ValueError(
            f"WGSL target does not support expression {type(expr).__name__}"
        )

    def generate_literal(self, node):
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        text = str(value)
        literal_type = self.type_name_string(getattr(node, "literal_type", ""))
        if literal_type == "f32" and re.fullmatch(r"[-+]?\d+", text):
            return f"{text}.0"
        return text

    def generate_workgroup_size_literal(self):
        values = self._current_workgroup_size or ("1", "1", "1")
        return (
            "vec3<u32>(" + ", ".join(self.u32_literal(value) for value in values) + ")"
        )

    def u32_literal(self, value):
        text = str(value)
        if re.fullmatch(r"\d+", text):
            return f"{text}u"
        return text

    def generate_function_call(self, node):
        function_name = self.expression_name(node.function)
        normalized_name = self.semantic_key(function_name)
        if normalized_name in self.TEXTURE_FUNCTION_NAMES:
            return self.generate_texture_function_call(node, function_name)
        if normalized_name in self.BARRIER_FUNCTION_NAMES:
            return self.generate_barrier_call(node, function_name)

        args = self.generate_call_arguments(function_name, node.arguments)
        if self.is_type_constructor_name(function_name):
            return f"{self.type_name_string(function_name)}({args})"
        mapped_name = self.FUNCTION_NAME_MAP.get(function_name, function_name)
        return f"{mapped_name}({args})"

    def generate_call_arguments(self, function_name, arguments):
        texture_parameter_indices = self._function_texture_parameters.get(
            function_name, ()
        )
        pointer_parameter_indices = self._function_pointer_parameters.get(
            function_name, ()
        )
        if not texture_parameter_indices and not pointer_parameter_indices:
            return ", ".join(self.generate_expression(arg) for arg in arguments)

        texture_parameter_indices = set(texture_parameter_indices)
        pointer_parameter_indices = set(pointer_parameter_indices)
        rendered = []
        for index, arg in enumerate(arguments):
            if index in pointer_parameter_indices:
                rendered.append(self.pointer_argument_expression(arg))
            else:
                rendered.append(self.generate_expression(arg))
            if index in texture_parameter_indices:
                rendered.append(self.texture_sampler_expression(arg))
        return ", ".join(rendered)

    def generate_texture_function_call(self, node, function_name):
        normalized_name = self.semantic_key(function_name)
        args = list(node.arguments)
        if normalized_name == "texture":
            return self.generate_texture_sample_call(function_name, args)
        if normalized_name == "texturelod":
            return self.generate_texture_sample_level_call(function_name, args)
        if normalized_name == "texturesize":
            return self.generate_texture_dimensions_call(args)
        raise ValueError(
            "WGSL target does not support CrossGL texture function "
            f"{function_name} yet"
        )

    def generate_texture_sample_call(self, function_name, args):
        if len(args) == 2:
            texture, coords = args
            return (
                f"textureSample({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.generate_expression(coords)})"
            )
        if len(args) == 3:
            texture, sampler, coords = args
            return (
                f"textureSample({self.generate_expression(texture)}, "
                f"{self.generate_expression(sampler)}, "
                f"{self.generate_expression(coords)})"
            )
        raise ValueError(
            "WGSL target supports texture() calls with texture/coords or "
            "texture/sampler/coords arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_sample_level_call(self, function_name, args):
        if len(args) == 3:
            texture, coords, level = args
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.texture_sampler_expression(texture)}, "
                f"{self.generate_expression(coords)}, "
                f"{self.generate_expression(level)})"
            )
        if len(args) == 4:
            texture, sampler, coords, level = args
            return (
                f"textureSampleLevel({self.generate_expression(texture)}, "
                f"{self.generate_expression(sampler)}, "
                f"{self.generate_expression(coords)}, "
                f"{self.generate_expression(level)})"
            )
        raise ValueError(
            "WGSL target supports textureLod() calls with texture/coords/lod or "
            "texture/sampler/coords/lod arguments; got "
            f"{len(args)} argument(s) for {function_name}"
        )

    def generate_texture_dimensions_call(self, args):
        if len(args) not in {1, 2}:
            raise ValueError(
                "WGSL target supports textureSize() calls with texture or "
                f"texture/lod arguments; got {len(args)} argument(s)"
            )
        return (
            "textureDimensions("
            + ", ".join(self.generate_expression(arg) for arg in args)
            + ")"
        )

    def texture_sampler_expression(self, texture_expr):
        if isinstance(texture_expr, IdentifierNode):
            return self.texture_sampler_name(texture_expr.name)
        raise ValueError(
            "WGSL target cannot infer a companion sampler for texture expression "
            f"{self.generate_expression(texture_expr)}; pass an explicit sampler"
        )

    def pointer_argument_expression(self, pointer_expr):
        if isinstance(pointer_expr, IdentifierNode) and self.is_pointer_identifier(
            pointer_expr.name
        ):
            return pointer_expr.name
        return f"&{self.generate_expression(pointer_expr)}"

    def generate_barrier_call(self, node, function_name):
        if node.arguments:
            raise ValueError(
                f"WGSL target does not support arguments for barrier function {function_name}"
            )
        if self._current_stage_name != "compute":
            raise ValueError(
                "WGSL target only supports barrier() inside compute stages"
            )
        return "workgroupBarrier()"

    def generate_constructor(self, node):
        args = ", ".join(self.generate_expression(arg) for arg in node.arguments)
        return f"{self.type_name_string(node.constructor_type)}({args})"

    def type_name_string(self, vtype, allow_storage_resources=False):
        if vtype is None:
            return "void"
        if isinstance(vtype, PrimitiveType):
            return self.PRIMITIVE_TYPE_MAP.get(vtype.name.lower(), vtype.name)
        if isinstance(vtype, VectorType):
            return f"vec{vtype.size}<{self.type_name_string(vtype.element_type)}>"
        if isinstance(vtype, MatrixType):
            return (
                f"mat{vtype.cols}x{vtype.rows}<"
                f"{self.type_name_string(vtype.element_type)}>"
            )
        if isinstance(vtype, ArrayType):
            element = self.type_name_string(vtype.element_type)
            if vtype.size is None:
                return f"array<{element}>"
            size = (
                self.generate_expression(vtype.size)
                if hasattr(vtype.size, "__class__")
                and not isinstance(vtype.size, (str, int))
                else str(vtype.size)
            )
            return f"array<{element}, {size}>"
        if isinstance(vtype, NamedType):
            storage_element = self.structured_buffer_element_type(vtype)
            if storage_element is not None:
                if not allow_storage_resources:
                    raise ValueError(
                        "WGSL target only supports StructuredBuffer resources as "
                        "module-scope storage bindings"
                    )
                return f"array<{self.type_name_string(storage_element)}>"
            if vtype.generic_args:
                raise ValueError("WGSL target does not support generic named types yet")
            return self.type_name_string(vtype.name)
        if isinstance(vtype, GenericType):
            raise ValueError("WGSL target does not support generic types yet")
        if isinstance(vtype, PointerType):
            raise ValueError(
                "WGSL target only supports pointer types for buffer/storage "
                "resources and helper parameters"
            )
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)
        if isinstance(vtype, str):
            return self.map_type_name(vtype)
        return str(vtype)

    def map_type_name(self, type_name):
        normalized = type_name.strip()
        lower = normalized.lower()
        if lower in self.PRIMITIVE_TYPE_MAP:
            return self.PRIMITIVE_TYPE_MAP[lower]
        if self.is_resource_type_name(lower):
            raise ValueError(
                "WGSL target does not support CrossGL resource type "
                f"{normalized} yet; split texture/sampler/storage bindings are required"
            )

        vector_match = self.VECTOR_TYPE_RE.match(lower)
        if vector_match:
            size = vector_match.group(1)
            element = "f32"
            if lower.startswith("int"):
                element = "i32"
            elif lower.startswith("uint"):
                element = "u32"
            elif lower.startswith("bool"):
                element = "bool"
            return f"vec{size}<{element}>"

        matrix_match = self.MATRIX_TYPE_RE.match(lower)
        if matrix_match:
            columns = matrix_match.group(1)
            rows = matrix_match.group(2) or columns
            return f"mat{columns}x{rows}<f32>"

        return normalized

    def is_resource_type_name(self, lower_type_name):
        return lower_type_name in self.RESOURCE_TYPE_NAMES

    def structured_buffer_element_type(self, vtype):
        if not isinstance(vtype, NamedType):
            return None
        base_name = str(vtype.name).lower()
        if base_name not in self.STRUCTURED_BUFFER_TYPE_NAMES:
            return None
        if len(vtype.generic_args) != 1:
            raise ValueError(
                "WGSL target requires StructuredBuffer resources to declare one "
                "element type"
            )
        return vtype.generic_args[0]

    def structured_buffer_access(self, vtype):
        if self.structured_buffer_element_type(vtype) is None:
            return None
        base_name = str(vtype.name).lower()
        if base_name in self.WRITABLE_STRUCTURED_BUFFER_TYPE_NAMES:
            return "read_write"
        return "read"

    def is_buffer_pointer_type(self, vtype, qualifiers=()):
        if not isinstance(vtype, PointerType):
            return False
        qualifier_names = {str(qualifier).lower() for qualifier in qualifiers or []}
        return bool(qualifier_names.intersection({"buffer", "storage"}))

    def buffer_pointer_element_type(self, vtype):
        if not isinstance(vtype, PointerType):
            raise ValueError("WGSL target expected a buffer pointer type")
        return vtype.pointee_type

    def buffer_pointer_storage_type(self, vtype):
        element_type = self.type_name_string(self.buffer_pointer_element_type(vtype))
        return f"array<{element_type}>"

    def buffer_pointer_parameter_type(self, vtype):
        return f"ptr<storage, {self.buffer_pointer_storage_type(vtype)}, read_write>"

    def sampled_texture_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        if type_name is None:
            return None
        return self.SAMPLED_TEXTURE_TYPE_MAP.get(type_name)

    def is_sampler_type(self, vtype):
        type_name = self.resource_type_name(vtype)
        return type_name in self.SAMPLER_TYPE_NAMES

    def resource_type_name(self, vtype):
        if isinstance(vtype, NamedType) and not vtype.generic_args:
            return str(vtype.name).lower()
        if isinstance(vtype, str):
            return vtype.strip().lower()
        return None

    def struct_member_resource_type_name(self, member):
        member_type = getattr(member, "member_type", None)
        type_name = self.resource_type_name(member_type)
        if type_name is None or not self.is_resource_type_name(type_name):
            return None
        if isinstance(member_type, NamedType):
            return str(member_type.name)
        return str(member_type)

    def texture_sampler_name(self, texture_name):
        return f"{texture_name}_sampler"

    def is_type_constructor_name(self, name):
        lower = str(name).lower()
        return lower in self.PRIMITIVE_TYPE_MAP or self.TYPE_CONSTRUCTOR_RE.match(lower)

    def expression_name(self, expr):
        if isinstance(expr, IdentifierNode):
            return expr.name
        if isinstance(expr, MemberAccessNode):
            return f"{self.expression_name(expr.object_expr)}.{expr.member}"
        return self.generate_expression(expr)

    def stage_return_attributes(self, stage_name, function):
        attributes = getattr(function, "attributes", []) or []
        if not attributes and stage_name == "vertex":
            return ()
        return_attributes = self.wgsl_attributes(attributes, direction="out")
        return (return_attributes,) if return_attributes else ()

    def wgsl_attributes(self, attributes, direction="generic"):
        rendered = []
        for attr in attributes or []:
            semantic = str(getattr(attr, "name", attr))
            explicit_attribute = self.explicit_wgsl_attribute(attr, semantic)
            if explicit_attribute:
                rendered.append(explicit_attribute)
                continue

            key = self.semantic_key(semantic)
            builtin = self.BUILTIN_SEMANTICS.get(key)
            if builtin:
                rendered.append(f"@builtin({builtin})")
                continue

            location = self.semantic_location(semantic, direction)
            if location is not None:
                rendered.append(f"@location({location})")

        return " ".join(rendered)

    def explicit_wgsl_attribute(self, attr, name):
        key = self.semantic_key(name)
        if key not in {
            "builtin",
            "invariant",
            "interpolate",
            "location",
        }:
            return None

        arguments = getattr(attr, "arguments", []) or []
        if not arguments:
            if key in {"builtin", "interpolate", "location"}:
                return None
            return f"@{key}"

        rendered_args = ", ".join(
            self.generate_attribute_argument(arg) for arg in arguments
        )
        return f"@{key}({rendered_args})"

    def generate_attribute_argument(self, argument):
        if isinstance(argument, IdentifierNode):
            return argument.name
        return self.generate_expression(argument)

    def semantic_key(self, semantic):
        return re.sub(r"[^a-z0-9_]+", "", semantic.lower())

    def semantic_location(self, semantic, direction):
        key = self.semantic_key(semantic)
        if key in {"gl_fragcolor", "sv_target", "color"}:
            return 0
        semantic_bases = {
            "position": 0,
            "normal": 1,
            "texcoord": 2,
            "uv": 2,
            "tangent": 6,
            "bitangent": 7,
        }
        for base, offset in semantic_bases.items():
            if key == base:
                return offset
            if key.startswith(base):
                suffix = key[len(base) :]
                if suffix.isdigit():
                    return offset + int(suffix)
        numeric_suffix = re.search(r"(\d+)$", key)
        if numeric_suffix:
            return int(numeric_suffix.group(1))
        return None

    def next_binding_attributes(self):
        binding = self._global_binding_index
        self._global_binding_index += 1
        return f"@group(0) @binding({binding})"

    def explicit_binding_attributes(self, node):
        group = "0"
        binding = None
        for attr in getattr(node, "attributes", []) or []:
            key = self.semantic_key(str(getattr(attr, "name", attr)))
            arguments = getattr(attr, "arguments", []) or []
            if key == "group" and arguments:
                group = self.generate_attribute_argument(arguments[0])
            elif key == "binding" and arguments:
                binding = self.generate_attribute_argument(arguments[0])
            elif key == "register" and arguments:
                binding, group = self.register_attribute_binding(arguments, group)

        if binding is None:
            return ""
        try:
            next_binding = int(str(binding), 0) + 1
        except ValueError:
            next_binding = self._global_binding_index
        self._global_binding_index = max(self._global_binding_index, next_binding)
        return f"@group({group}) @binding({binding})"

    def register_attribute_binding(self, arguments, default_group):
        binding_text = self.generate_attribute_argument(arguments[0])
        group = default_group
        binding_match = re.search(r"\d+", str(binding_text))
        binding = binding_match.group(0) if binding_match else str(binding_text)
        if len(arguments) > 1:
            space_text = self.generate_attribute_argument(arguments[1])
            group_match = re.search(r"\d+", str(space_text))
            group = group_match.group(0) if group_match else str(space_text)
        return binding, group

    def cbuffer_instance_name(self, node):
        return f"_{node.name}"

    def cbuffer_member_accesses(self, cbuffers):
        accesses = {}
        for cbuffer in cbuffers:
            instance_name = self.cbuffer_instance_name(cbuffer)
            for member in getattr(cbuffer, "members", []) or []:
                member_name = getattr(member, "name", "")
                if not member_name:
                    continue
                if member_name in accesses:
                    raise ValueError(
                        "WGSL target cannot flatten duplicate cbuffer member "
                        f"name: {member_name}"
                    )
                accesses[member_name] = f"{instance_name}.{member_name}"
        return accesses

    def function_texture_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        texture_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.sampled_texture_type(parameter.param_type) is not None
            ]
            if indices:
                texture_parameters[function.name] = tuple(indices)
        return texture_parameters

    def function_buffer_pointer_parameters(self, ast, target_stage):
        functions = list(self._helper_functions(ast, target_stage))
        functions.extend(
            stage_node.entry_point
            for stage_node in self._stage_nodes(ast, target_stage)
            if getattr(stage_node, "entry_point", None) is not None
        )
        pointer_parameters = {}
        for function in functions:
            indices = [
                index
                for index, parameter in enumerate(
                    getattr(function, "parameters", []) or []
                )
                if self.is_buffer_pointer_type(
                    parameter.param_type, getattr(parameter, "qualifiers", [])
                )
            ]
            if indices:
                pointer_parameters[function.name] = tuple(indices)
        return pointer_parameters

    def buffer_pointer_parameter_names(self, function):
        return tuple(
            getattr(parameter, "name", "")
            for parameter in getattr(function, "parameters", []) or []
            if self.is_buffer_pointer_type(
                parameter.param_type, getattr(parameter, "qualifiers", [])
            )
        )

    def push_identifier_scope(self, names=()):
        self._identifier_scopes.append({name for name in names if name})

    def pop_identifier_scope(self):
        self._identifier_scopes.pop()

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
        variables = list(getattr(ast, "global_variables", []) or [])
        for stage_node in self._stage_nodes(ast, target_stage):
            variables.extend(getattr(stage_node, "local_variables", []) or [])
        return self._dedupe_by_name(variables)

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
        return self._dedupe_functions(helpers)

    def _stage_nodes(self, ast, target_stage):
        nodes = []
        for stage_type, stage_node in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(stage_node)
        if nodes:
            return nodes

        for func in getattr(ast, "functions", []) or []:
            stage_name = self._function_stage_name(func)
            if not stage_name:
                continue
            if target_stage is not None and stage_name != target_stage:
                continue
            nodes.append(
                _FunctionStageNode(
                    stage_name,
                    func,
                    execution_config=self._function_execution_config(func),
                )
            )
        return nodes

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
        config = {}
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
            key = (getattr(func, "name", None), len(getattr(func, "parameters", [])))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(func)
        return deduped


class _FunctionStageNode:
    def __init__(self, stage, entry_point, execution_config=None):
        self.stage = stage
        self.entry_point = entry_point
        self.execution_config = execution_config or {}
        self.layout_qualifiers = []
        self.local_structs = []
        self.local_functions = []
