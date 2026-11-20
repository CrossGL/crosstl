"""Semantic mapping and validation of the CUDA generator, including the compute built-in parameters."""

from ..refusals import CudaRefusal
from .array_utils import split_array_type_suffix


class CudaSemanticMixin:
    """Semantic members of ``CudaCodeGen``."""

    def semantic_from_node(self, node):
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if self.is_semantic_name(attr_name):
                return attr_name
        return None

    def is_semantic_name(self, name):
        if name is None:
            return False
        name = str(name)
        upper_name = name.upper()
        if name.startswith("gl_"):
            return True
        if upper_name in {
            "BINORMAL",
            "CALLABLE_DATA",
            "CALLABLEDATAEXT",
            "CALLABLEDATAINEXT",
            "COLOR",
            "HIT_ATTRIBUTE",
            "HITATTRIBUTEEXT",
            "NORMAL",
            "PAYLOAD",
            "POSITION",
            "RAYPAYLOADEXT",
            "RAYPAYLOADINEXT",
            "SV_DEPTH",
            "SV_DISPATCHRAYSDIMENSIONS",
            "SV_DISPATCHRAYSINDEX",
            "SV_DISPATCHTHREADID",
            "SV_DRAWID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_POSITION",
            "SV_PRIMITIVEID",
            "SV_SAMPLEINDEX",
            "SV_STARTINSTANCELOCATION",
            "SV_STARTVERTEXLOCATION",
            "SV_TARGET",
            "SV_VERTEXID",
            "TANGENT",
            "TEXCOORD",
        }:
            return True
        for prefix in ("COLOR", "SV_TARGET", "TEXCOORD"):
            if upper_name.startswith(prefix) and upper_name[len(prefix) :].isdigit():
                return True
        return False

    def map_semantic(self, semantic):
        if semantic is None:
            return ""
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth(any)"
        if lower_name == "gl_fragcolor":
            return "target(0)"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "SV_TARGET":
            return "target(0)"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix.isdigit():
                return f"target({suffix})"
        if upper_name == "TEXCOORD":
            return "texcoord"
        if (
            upper_name.startswith("TEXCOORD")
            and upper_name[len("TEXCOORD") :].isdigit()
        ):
            return f"texcoord({upper_name[len('TEXCOORD'):]})"
        if upper_name == "COLOR":
            return "color"
        if upper_name.startswith("COLOR") and upper_name[len("COLOR") :].isdigit():
            return f"color({upper_name[len('COLOR'):]})"
        generic_semantics = {
            "BINORMAL": "binormal",
            "NORMAL": "normal",
            "POSITION": "position",
            "TANGENT": "tangent",
            "gl_FragCoord": "position",
            "gl_FrontFacing": "front_facing",
            "gl_PointCoord": "point_coord",
            "gl_VertexID": "vertex_id",
            "gl_InstanceID": "instance_id",
            "gl_BaseVertex": "start_vertex_location",
            "gl_BaseInstance": "start_instance_location",
            "gl_DrawID": "draw_id",
            "gl_WorkGroupID": "workgroup_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "gl_GlobalInvocationID": "global_invocation_id",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "payload": "ray_payload",
            "rayPayloadEXT": "ray_payload",
            "rayPayloadInEXT": "ray_payload",
            "hit_attribute": "hit_attribute",
            "hitAttributeEXT": "hit_attribute",
            "callable_data": "callable_data",
            "callableDataEXT": "callable_data",
            "callableDataInEXT": "callable_data",
            "gl_LaunchIDEXT": "launch_id",
            "gl_LaunchSizeEXT": "launch_size",
            "gl_HitTEXT": "hit_t",
            "gl_HitKindEXT": "hit_kind",
            "SV_DispatchRaysIndex": "launch_id",
            "SV_DispatchRaysDimensions": "launch_size",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DrawID": "draw_id",
            "SV_GroupID": "workgroup_id",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_InstanceID": "instance_id",
            "SV_IsFrontFace": "front_facing",
            "SV_PrimitiveID": "primitive_id",
            "SV_SampleIndex": "sample_index",
            "SV_StartInstanceLocation": "start_instance_location",
            "SV_StartVertexLocation": "start_vertex_location",
            "SV_VertexID": "vertex_id",
        }
        if semantic_name in generic_semantics:
            return generic_semantics[semantic_name]
        # Only stage metadata comments record the semantic; a user semantic is recorded as written.
        return semantic_name

    def cuda_semantic_output_kind(self, semantic):
        if semantic is None:
            return None
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        upper_name = semantic_name.upper()
        if lower_name in {
            "gl_fragcoord",
            "gl_frontfacing",
            "gl_globalinvocationid",
            "gl_instanceid",
            "gl_localinvocationid",
            "gl_localinvocationindex",
            "gl_pointcoord",
            "gl_vertexid",
            "gl_workgroupid",
        } or upper_name in {
            "SV_DISPATCHTHREADID",
            "SV_GROUPID",
            "SV_GROUPINDEX",
            "SV_GROUPTHREADID",
            "SV_INSTANCEID",
            "SV_ISFRONTFACE",
            "SV_VERTEXID",
        }:
            return "input_only"
        if lower_name == "gl_position" or upper_name == "SV_POSITION":
            return "position"
        if lower_name == "gl_fragdepth" or upper_name == "SV_DEPTH":
            return "depth"
        if lower_name.startswith("gl_fragcolor"):
            suffix = lower_name[len("gl_fragcolor") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        if upper_name.startswith("SV_TARGET"):
            suffix = upper_name[len("SV_TARGET") :]
            if suffix == "" or suffix.isdigit():
                return "color"
        return None

    def is_cuda_float4_type(self, type_name):
        return self.map_type(type_name) == "float4"

    def is_cuda_float_scalar_type(self, type_name):
        return self.map_type(type_name) == "float"

    def validate_cuda_builtin_semantic_type(self, semantic, type_name, context):
        kind = self.cuda_semantic_output_kind(semantic)
        if kind is None or kind == "input_only":
            return
        if kind in {"position", "color"}:
            if self.is_cuda_float4_type(type_name):
                return
            raise CudaRefusal(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "expected vec4-compatible type"
            )
        if kind == "depth" and not self.is_cuda_float_scalar_type(type_name):
            raise CudaRefusal(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "expected float type"
            )

    def validate_cuda_output_semantic_stage(self, stage_name, semantic, context):
        kind = self.cuda_semantic_output_kind(semantic)
        if kind is None:
            return
        if kind == "input_only":
            raise CudaRefusal(
                f"Unsupported {semantic} {context} for CUDA codegen; "
                "input-only builtin semantics cannot be used as outputs"
            )
        if stage_name is None:
            return
        allowed_stages = {
            "position": {"vertex"},
            "color": {"fragment"},
            "depth": {"fragment"},
        }[kind]
        if stage_name not in allowed_stages:
            allowed = ", ".join(sorted(allowed_stages))
            raise CudaRefusal(
                f"Unsupported {semantic} {context} for CUDA {stage_name} stage; "
                f"valid stage is {allowed}"
            )

    def validate_cuda_return_semantic(self, stage_name, return_type, semantic):
        if semantic is None:
            return
        if self.map_type(return_type) == "void":
            raise CudaRefusal(
                f"Unsupported {semantic} return semantic for CUDA codegen; "
                "void return type"
            )
        self.validate_cuda_output_semantic_stage(
            stage_name, semantic, "return semantic"
        )
        self.validate_cuda_builtin_semantic_type(
            semantic, return_type, "return semantic"
        )

    def validate_cuda_struct_return_semantics(self, stage_name, return_type):
        if stage_name is None:
            return
        base_type = self.type_name_string(return_type)
        if not base_type:
            return
        base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
        member_semantics = self.struct_member_semantics.get(base_type, {})
        member_types = self.struct_member_types.get(base_type, {})
        for member_name, semantic in member_semantics.items():
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_cuda_output_semantic_stage(stage_name, semantic, context)
            self.validate_cuda_builtin_semantic_type(
                semantic, member_types.get(member_name, "float"), context
            )

    def cuda_stage_parameter_semantic_key(self, semantic):
        semantic_name = str(semantic)
        lower_name = semantic_name.lower()
        if lower_name.startswith("gl_"):
            return lower_name
        return semantic_name.upper()

    def cuda_stage_parameter_semantic_rules(self):
        ray_stages = self.cuda_ray_stage_names()
        ray_hit_stages = {"ray_any_hit", "ray_closest_hit", "ray_intersection"}
        return {
            "gl_vertexid": ("vertex_id", "uint", {"vertex"}),
            "gl_instanceid": ("instance_id", "uint", {"vertex"}),
            "gl_basevertex": ("start_vertex_location", "int", {"vertex"}),
            "gl_baseinstance": ("start_instance_location", "uint", {"vertex"}),
            "gl_drawid": ("draw_id", "uint", {"vertex"}),
            "SV_VERTEXID": ("vertex_id", "uint", {"vertex"}),
            "SV_INSTANCEID": ("instance_id", "uint", {"vertex"}),
            "SV_STARTVERTEXLOCATION": ("start_vertex_location", "int", {"vertex"}),
            "SV_STARTINSTANCELOCATION": ("start_instance_location", "uint", {"vertex"}),
            "SV_DRAWID": ("draw_id", "uint", {"vertex"}),
            "gl_position": ("position", "float4", {"fragment"}),
            "gl_fragcoord": ("position", "float4", {"fragment"}),
            "gl_frontfacing": ("front_facing", "bool", {"fragment"}),
            "gl_pointcoord": ("point_coord", "float2", {"fragment"}),
            "gl_primitiveid": (
                "primitive_id",
                "uint",
                {"fragment"} | ray_hit_stages,
            ),
            "gl_sampleid": ("sample_index", "uint", {"fragment"}),
            "SV_POSITION": ("position", "float4", {"fragment"}),
            "SV_ISFRONTFACE": ("front_facing", "bool", {"fragment"}),
            "SV_PRIMITIVEID": (
                "primitive_id",
                "uint",
                {"fragment"} | ray_hit_stages,
            ),
            "SV_SAMPLEINDEX": ("sample_index", "uint", {"fragment"}),
            "gl_workgroupid": ("workgroup_id", "uint3", {"compute"}),
            "gl_localinvocationid": ("local_invocation_id", "uint3", {"compute"}),
            "gl_globalinvocationid": ("global_invocation_id", "uint3", {"compute"}),
            "gl_localinvocationindex": ("local_invocation_index", "uint", {"compute"}),
            "SV_GROUPID": ("workgroup_id", "uint3", {"compute"}),
            "SV_GROUPTHREADID": ("local_invocation_id", "uint3", {"compute"}),
            "SV_DISPATCHTHREADID": ("global_invocation_id", "uint3", {"compute"}),
            "SV_GROUPINDEX": ("local_invocation_index", "uint", {"compute"}),
            "gl_launchidext": ("launch_id", "uint3", ray_stages),
            "gl_launchsizeext": ("launch_size", "uint3", ray_stages),
            "gl_hittext": ("hit_t", "float", ray_hit_stages),
            "gl_hitkindext": ("hit_kind", "uint", {"ray_any_hit"}),
            "SV_DISPATCHRAYSINDEX": ("launch_id", "uint3", ray_stages),
            "SV_DISPATCHRAYSDIMENSIONS": ("launch_size", "uint3", ray_stages),
        }

    def cuda_compute_builtin_parameter_role(self, stage_name, param):
        """Return the compute built-in role for a semantic kernel parameter."""
        if stage_name != "compute":
            return None

        semantic = self.semantic_from_node(param)
        if semantic is None:
            return None

        rule = self.cuda_stage_parameter_semantic_rules().get(
            self.cuda_stage_parameter_semantic_key(semantic)
        )
        if rule is None:
            return None

        role, _expected_type, allowed_stages = rule
        if "compute" not in allowed_stages:
            return None
        if role in {
            "global_invocation_id",
            "local_invocation_id",
            "local_invocation_index",
            "workgroup_id",
        }:
            return role
        return None

    def cuda_stage_local_builtin_role(self, stage_name, var):
        """Return the compute built-in role for a stage-local input variable.

        A stage-local ``@input`` variable carrying a compute built-in semantic
        (``uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;``, the
        shape the SPIR-V importer writes for a GLSL compute shader that reads
        ``gl_GlobalInvocationID``) is the built-in itself, not a variable to
        declare: CUDA has no such global, and a file-scope declaration would
        be an uninitialised host global that the kernel must not read. A
        variable whose name is a compute built-in and which is declared as a
        stage input is the same built-in even when the semantic spelling is
        missing.
        """
        if stage_name != "compute":
            return None
        role = self.cuda_compute_builtin_parameter_role(stage_name, var)
        if role is not None:
            return role
        if self.cuda_compute_builtin_role_for_name(getattr(var, "name", None)) is None:
            return None
        if any(
            str(getattr(attr, "name", "")).lower() == "input"
            for attr in getattr(var, "attributes", []) or []
        ):
            return self.cuda_compute_builtin_role_for_name(var.name)
        return None

    def cuda_compute_builtin_expression(self, role, component=None):
        """Return the CUDA expression for a CrossGL compute built-in role."""
        local_index = self.cuda_thread_block_linear_index_expression()
        component_expressions = {
            "global_invocation_id": {
                "x": "(blockIdx.x * blockDim.x + threadIdx.x)",
                "y": "(blockIdx.y * blockDim.y + threadIdx.y)",
                "z": "(blockIdx.z * blockDim.z + threadIdx.z)",
            },
            "local_invocation_id": {
                "x": "threadIdx.x",
                "y": "threadIdx.y",
                "z": "threadIdx.z",
            },
            "workgroup_id": {
                "x": "blockIdx.x",
                "y": "blockIdx.y",
                "z": "blockIdx.z",
            },
            "workgroup_size": {
                "x": "blockDim.x",
                "y": "blockDim.y",
                "z": "blockDim.z",
            },
            "num_workgroups": {
                "x": "gridDim.x",
                "y": "gridDim.y",
                "z": "gridDim.z",
            },
        }

        if role == "local_invocation_index":
            return local_index if component is None else None

        role_components = component_expressions.get(role)
        if role_components is None:
            return None
        if component is not None:
            if component in role_components:
                return role_components[component]
            swizzle_components = tuple(str(component))
            if all(part in role_components for part in swizzle_components):
                constructor = {
                    2: "make_uint2",
                    3: "make_uint3",
                    4: "make_uint4",
                }.get(len(swizzle_components))
                if constructor is not None:
                    values = ", ".join(
                        role_components[part] for part in swizzle_components
                    )
                    return f"{constructor}({values})"
            return None
        return "make_uint3({x}, {y}, {z})".format(**role_components)

    def cuda_thread_block_linear_index_expression(self):
        """Return CUDA's row-major linear thread ID within the current block."""
        return (
            "(threadIdx.z * blockDim.y * blockDim.x + "
            "threadIdx.y * blockDim.x + threadIdx.x)"
        )

    def cuda_warp_lane_index_expression(self):
        """Return the lane index for the current thread in its CUDA warp."""
        return (
            f"({self.cuda_thread_block_linear_index_expression()} " "& (warpSize - 1))"
        )

    def cuda_compute_builtin_role_for_name(self, name):
        """Return the compute built-in role represented by a source identifier."""
        return {
            "gl_GlobalInvocationID": "global_invocation_id",
            "SV_DispatchThreadID": "global_invocation_id",
            "SV_DISPATCHTHREADID": "global_invocation_id",
            "gl_LocalInvocationID": "local_invocation_id",
            "SV_GroupThreadID": "local_invocation_id",
            "SV_GROUPTHREADID": "local_invocation_id",
            "gl_WorkGroupID": "workgroup_id",
            "SV_GroupID": "workgroup_id",
            "SV_GROUPID": "workgroup_id",
            "gl_WorkGroupSize": "workgroup_size",
            "gl_NumWorkGroups": "num_workgroups",
            "gl_LocalInvocationIndex": "local_invocation_index",
            "SV_GroupIndex": "local_invocation_index",
            "SV_GROUPINDEX": "local_invocation_index",
        }.get(name)

    def cuda_identifier_shadows_compute_builtin(self, name):
        """Return whether a user symbol hides a CUDA-lowered compute built-in."""
        if name is None:
            return False
        root_name = str(name).split(".", 1)[0]
        return (
            root_name in self.variable_types
            and root_name not in self.stage_builtin_aliases
            and self.cuda_compute_builtin_role_for_name(root_name) is not None
        )

    def cuda_compute_builtin_type(self, name):
        """Return the CUDA value type for a compute built-in identifier."""
        role = self.cuda_compute_builtin_role_for_name(name)
        if role is None:
            return None
        if role == "local_invocation_index":
            return "uint"
        return "uint3"

    def cuda_compute_builtin_member_type(self, name, member):
        """Return the type of a direct compute built-in member/swizzle."""
        root_type = self.cuda_compute_builtin_type(name)
        vector_info = self.vector_type_info(root_type)
        if vector_info is None:
            return None
        component_aliases = {
            "x": "x",
            "y": "y",
            "z": "z",
            "w": "w",
            "r": "x",
            "g": "y",
            "b": "z",
            "a": "w",
        }
        components = [component_aliases.get(component) for component in str(member)]
        if not components or any(component is None for component in components):
            return None
        if any(component not in vector_info["components"] for component in components):
            return None
        if len(components) == 1:
            return vector_info["component_type"]
        return self.swizzle_vector_type(vector_info, len(components))

    def cuda_stage_builtin_alias_expression(self, name, component=None):
        role = self.stage_builtin_aliases.get(name)
        if role is None:
            return None
        expression = self.cuda_compute_builtin_expression(role, component)
        if component is not None or expression is None:
            return expression

        alias_type = self.stage_builtin_alias_types.get(name)
        if alias_type is None or role != "global_invocation_id":
            return expression
        mapped_type = self.convert_crossgl_type_to_cuda(alias_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return expression
        if base_type in {"uint", "unsigned int"}:
            return self.cuda_compute_builtin_expression(role, "x")
        if base_type == "uint2":
            x = self.cuda_compute_builtin_expression(role, "x")
            y = self.cuda_compute_builtin_expression(role, "y")
            if x is not None and y is not None:
                return f"make_uint2({x}, {y})"
        return expression

    def validate_cuda_stage_parameter_semantic_type(
        self, param, semantic, expected_type
    ):
        param_type = self.resource_type_with_access(
            self.get_parameter_type(param), param
        )
        actual_type = self.convert_crossgl_type_to_cuda(param_type)
        if actual_type == expected_type:
            return
        semantic_key = self.cuda_stage_parameter_semantic_key(semantic)
        if (
            expected_type == "uint3"
            and semantic_key in {"gl_globalinvocationid", "SV_DISPATCHTHREADID"}
            and actual_type in {"uint", "unsigned int", "uint2"}
        ):
            return
        raise CudaRefusal(
            f"Unsupported {semantic} stage parameter semantic for CUDA codegen; "
            f"expected {expected_type} type"
        )

    def validate_cuda_stage_parameter_semantics(self, stage_name, parameters):
        if stage_name is None:
            return

        rules = self.cuda_stage_parameter_semantic_rules()
        seen_system_semantics = {}
        for param in parameters or []:
            semantic = self.semantic_from_node(param)
            if semantic is None:
                continue

            semantic_key = self.cuda_stage_parameter_semantic_key(semantic)
            rule = rules.get(semantic_key)
            if rule is not None:
                mapped_semantic, expected_type, allowed_stages = rule
                if stage_name not in allowed_stages:
                    allowed = ", ".join(sorted(allowed_stages))
                    raise CudaRefusal(
                        f"Unsupported {semantic} stage parameter semantic for CUDA "
                        f"{stage_name} stage; valid stage is {allowed}"
                    )
                param_name = getattr(param, "name", getattr(param, "param_name", None))
                previous_name = seen_system_semantics.get(mapped_semantic)
                if previous_name is not None:
                    raise CudaRefusal(
                        f"Duplicate CUDA stage parameter semantic {mapped_semantic} "
                        f"on '{previous_name}' and '{param_name}'"
                    )
                seen_system_semantics[mapped_semantic] = param_name
                self.validate_cuda_stage_parameter_semantic_type(
                    param, semantic, expected_type
                )
                continue

            kind = self.cuda_semantic_output_kind(semantic)
            if kind in {"color", "depth"}:
                raise CudaRefusal(
                    f"Unsupported {semantic} stage parameter semantic for CUDA "
                    f"{stage_name} stage; output-only builtin semantics cannot "
                    "be used as inputs"
                )
