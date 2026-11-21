"""Stage naming and stage attributes of the Slang generator: entry names, tessellation shapes, attribute arguments and their validation."""

from ..refusals import SlangRefusal
from .slang_exact_lookups import slang_shader_stage
from .stage_utils import function_stage_name


class SlangStageAttributesMixin:
    """Stage-attribute members of ``SlangCodeGen``."""

    def get_stage_name(self, stage_type):
        if hasattr(stage_type, "value"):
            return stage_type.value
        return str(stage_type).split(".")[-1].lower()

    def get_function_stage(self, function):
        """The stage a top-level entry declares, as a qualifier or as an ``@ compute``-style attribute."""
        stage_name = function_stage_name(function)
        if stage_name in {"vertex", "fragment", "compute"}:
            return stage_name
        return None

    def slang_shader_stage_name(self, stage_name):
        """Return the Slang [shader(...)] spelling for a CrossGL stage name; refuse a name that is no stage."""
        return slang_shader_stage(stage_name)

    def collect_stage_entry_name_overrides(self, stages):
        """Return replacement entry names for stage blocks with duplicate names."""
        entries_by_name = {}
        for stage_type, stage in stages.items():
            entry_point = getattr(stage, "entry_point", None)
            entry_name = getattr(entry_point, "name", None)
            if entry_name is None:
                continue
            stage_name = self.get_stage_name(stage_type)
            entries_by_name.setdefault(entry_name, []).append((stage_name, entry_point))

        used_names = set(self.user_function_names)
        overrides = {}
        for entries in entries_by_name.values():
            if len(entries) < 2:
                continue
            for stage_name, entry_point in entries:
                candidate = self.slang_stage_entry_function_name(stage_name)
                unique_name = self.unique_stage_entry_name(candidate, used_names)
                used_names.add(unique_name)
                overrides[id(entry_point)] = unique_name
        return overrides

    def slang_stage_entry_function_name(self, stage_name):
        stage_name = self.slang_shader_stage_name(stage_name)
        stage_map = {
            "vertex": "VSMain",
            "fragment": "PSMain",
            "compute": "CSMain",
            "geometry": "GSMain",
            "hull": "HSMain",
            "domain": "DSMain",
            "mesh": "MSMain",
            "amplification": "ASMain",
            "raygeneration": "RayGenMain",
            "intersection": "IntersectionMain",
            "closesthit": "ClosestHitMain",
            "anyhit": "AnyHitMain",
            "miss": "MissMain",
            "callable": "CallableMain",
        }
        return stage_map.get(stage_name, f"{stage_name}_main")

    def unique_stage_entry_name(self, candidate, used_names):
        if candidate not in used_names:
            return candidate

        suffix = 1
        while f"{candidate}_{suffix}" in used_names:
            suffix += 1
        return f"{candidate}_{suffix}"

    def slang_stage_by_shader_name(self, stages, expected_stage_name):
        for stage_type, stage in (stages or {}).items():
            stage_name = self.get_stage_name(stage_type)
            if self.slang_shader_stage_name(stage_name) == expected_stage_name:
                return stage
        return None

    def validate_slang_tessellation_stage_shapes(self, stages):
        if not isinstance(stages, dict):
            return

        hull_stage = self.slang_stage_by_shader_name(stages, "hull")
        domain_stage = self.slang_stage_by_shader_name(stages, "domain")
        if hull_stage is None:
            return

        hull_entry = getattr(hull_stage, "entry_point", None)
        if hull_entry is None:
            return

        hull_shape = self.slang_hull_output_patch_shape(hull_entry)
        self.validate_slang_hull_output_patch_shape(hull_entry, hull_shape)

        if domain_stage is None:
            return

        domain_entry = getattr(domain_stage, "entry_point", None)
        if domain_entry is None:
            return

        self.validate_slang_tessellation_domain_attributes_match(
            hull_entry, domain_entry
        )
        self.validate_slang_domain_output_patch_shape(domain_entry, hull_shape)

    def validate_slang_tessellation_domain_attributes_match(
        self, hull_entry, domain_entry
    ):
        hull_domain = self.normalized_slang_stage_attribute_argument(
            hull_entry, "domain"
        )
        domain_domain = self.normalized_slang_stage_attribute_argument(
            domain_entry, "domain"
        )
        if hull_domain and not domain_domain:
            raise SlangRefusal(
                "Slang tessellation_evaluation stage requires a domain "
                "attribute matching tessellation_control"
            )
        if domain_domain and not hull_domain:
            raise SlangRefusal(
                "Slang tessellation_control stage requires a domain "
                "attribute matching tessellation_evaluation"
            )
        if not hull_domain or not domain_domain:
            return

        hull_domain = self.canonical_slang_tessellation_domain(hull_domain)
        domain_domain = self.canonical_slang_tessellation_domain(domain_domain)
        if hull_domain != domain_domain:
            raise SlangRefusal(
                "Slang tessellation_evaluation stage domain "
                f"'{domain_domain}' must match tessellation_control domain "
                f"'{hull_domain}'"
            )

    def validate_slang_hull_output_patch_shape(self, hull_entry, hull_shape):
        output_patches = self.slang_patch_parameters(
            getattr(hull_entry, "parameters", getattr(hull_entry, "params", [])),
            "OutputPatch",
        )
        if len(output_patches) > 1:
            raise SlangRefusal(
                "Slang tessellation_control stage requires at most one "
                "OutputPatch<..., N> parameter"
            )

        output_control_points = self.slang_stage_attribute_int_argument(
            hull_entry, "outputcontrolpoints"
        )
        if output_control_points is None or hull_shape is None:
            return

        _element_type, patch_size = hull_shape
        if patch_size is None or str(output_control_points) == str(patch_size):
            return

        output_patch = self.format_slang_patch_shape("OutputPatch", hull_shape)
        raise SlangRefusal(
            "Slang tessellation_control stage "
            f"{output_patch} must match outputcontrolpoints({output_control_points})"
        )

    def validate_slang_domain_output_patch_shape(self, domain_entry, hull_shape):
        domain_output_patches = self.slang_patch_parameters(
            getattr(domain_entry, "parameters", getattr(domain_entry, "params", [])),
            "OutputPatch",
        )
        if len(domain_output_patches) > 1:
            raise SlangRefusal(
                "Slang tessellation_evaluation stage requires at most one "
                "OutputPatch<..., N> parameter"
            )
        if not domain_output_patches or hull_shape is None:
            return

        _domain_param, domain_shape = domain_output_patches[0]
        hull_type, hull_size = hull_shape
        domain_type, domain_size = domain_shape
        if hull_type and domain_type and hull_type != domain_type:
            raise SlangRefusal(
                "Slang tessellation_evaluation stage "
                f"{self.format_slang_patch_shape('OutputPatch', domain_shape)} "
                "must match tessellation_control output "
                f"{self.format_slang_patch_shape('OutputPatch', hull_shape)}"
            )
        if (
            hull_size is not None
            and domain_size is not None
            and hull_size != domain_size
        ):
            raise SlangRefusal(
                "Slang tessellation_evaluation stage "
                f"{self.format_slang_patch_shape('OutputPatch', domain_shape)} "
                "must match tessellation_control output "
                f"{self.format_slang_patch_shape('OutputPatch', hull_shape)}"
            )

    def slang_hull_output_patch_shape(self, hull_entry):
        param_list = getattr(
            hull_entry, "parameters", getattr(hull_entry, "params", [])
        )
        output_patches = self.slang_patch_parameters(param_list, "OutputPatch")
        output_control_points = self.slang_stage_attribute_int_argument(
            hull_entry, "outputcontrolpoints"
        )

        return_type_name = self.convert_type_node_to_string(
            getattr(hull_entry, "return_type", "void")
        )
        if self.convert_type(return_type_name) != "void":
            return return_type_name, (
                str(output_control_points)
                if output_control_points is not None
                else None
            )

        if output_patches:
            _param, patch_shape = output_patches[0]
            element_type, patch_size = patch_shape
            return element_type, patch_size

        if output_control_points is not None:
            return None, str(output_control_points)
        return None

    def slang_stage_uses_numthreads(self, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        return shader_stage in {"compute", "mesh", "amplification"}

    def slang_stage_attribute_name(self, attr):
        attr_name = getattr(attr, "name", None)
        if not attr_name:
            return None

        normalized = str(attr_name).lower()
        if normalized.startswith("slang_"):
            normalized = normalized[len("slang_") :]
        elif normalized.startswith("hlsl_"):
            normalized = normalized[len("hlsl_") :]

        valid_names = {
            "domain",
            "instance",
            "maxvertexcount",
            "maxtessfactor",
            "numthreads",
            "outputcontrolpoints",
            "outputtopology",
            "partitioning",
            "patchconstantfunc",
        }
        if normalized in valid_names:
            return normalized
        return None

    def slang_stage_attribute_arguments(self, func, expected_name):
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name == expected_name:
                return getattr(attr, "arguments", []) or []
        return []

    def slang_stage_attribute_names(self, func):
        names = set()
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if attr_name:
                names.add(attr_name)
        return names

    def slang_stage_attribute_duplicate_names(self, func):
        seen = set()
        duplicates = []
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.slang_stage_attribute_name(attr)
            if not attr_name:
                continue
            if attr_name in seen and attr_name not in duplicates:
                duplicates.append(attr_name)
            seen.add(attr_name)
        return duplicates

    def slang_stage_attribute_value_to_string(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "value"):
            return str(value.value).strip('"')
        if hasattr(value, "name"):
            return str(value.name)
        return str(value)

    def canonical_slang_tessellation_domain(self, domain):
        if domain is None:
            return None
        normalized = str(domain).strip('"').lower()
        if normalized == "triangle":
            return "tri"
        return normalized

    def normalized_slang_stage_attribute_argument(self, func, expected_name):
        arguments = self.slang_stage_attribute_arguments(func, expected_name)
        if not arguments:
            return None
        value = self.slang_stage_attribute_value_to_string(arguments[0])
        if value is None:
            return None
        return str(value).strip('"').lower()

    def slang_int_literal_value(self, value):
        if value is None:
            return None
        if hasattr(value, "value"):
            value = value.value
        elif hasattr(value, "name"):
            value = value.name
        if isinstance(value, int) and not isinstance(value, bool):
            return value

        text = str(value).strip().strip('"').replace("_", "")
        if not text:
            return None
        while text and text[-1] in {"u", "U", "l", "L"}:
            text = text[:-1]
        try:
            return int(text, 0)
        except ValueError:
            return None

    def slang_stage_attribute_int_argument(self, func, expected_name):
        arguments = self.slang_stage_attribute_arguments(func, expected_name)
        if not arguments:
            return None
        return self.slang_int_literal_value(arguments[0])

    def validate_positive_slang_stage_attribute(self, func, stage_name, attr_name):
        value = self.slang_stage_attribute_int_argument(func, attr_name)
        if value is None:
            return
        if value <= 0:
            raise SlangRefusal(
                f"Slang {stage_name} stage {attr_name} ({value}) must be positive"
            )

    def validate_slang_stage_attribute_applicability(self, func, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        allowed_by_stage = {
            "compute": {"numthreads"},
            # [instance(N)] runs the geometry stage N times per primitive, as GLSL's invocations = N.
            "geometry": {"maxvertexcount", "instance"},
            "hull": {
                "domain",
                "maxtessfactor",
                "outputcontrolpoints",
                "outputtopology",
                "partitioning",
                "patchconstantfunc",
            },
            # A CrossGL domain stage may state the partitioning and output topology it expects.
            # Slang, like HLSL, takes both from the hull, so they are validated and not written.
            "domain": {"domain", "outputtopology", "partitioning"},
            "mesh": {"numthreads", "outputtopology"},
            "amplification": {"numthreads"},
        }
        allowed_attributes = allowed_by_stage.get(shader_stage, set())
        for attr_name in self.slang_stage_attribute_names(func):
            if attr_name not in allowed_attributes:
                raise SlangRefusal(
                    f"Slang {stage_name} stage does not support {attr_name} attribute"
                )

    def validate_slang_stage_attribute_uniqueness(self, func, stage_name):
        for attr_name in self.slang_stage_attribute_duplicate_names(func):
            raise SlangRefusal(
                f"Slang {stage_name} stage {attr_name} attribute "
                "must appear at most once"
            )

    def validate_slang_tessellation_domain(self, func, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage not in {"hull", "domain"}:
            return

        domain = self.normalized_slang_stage_attribute_argument(func, "domain")
        if not domain:
            return

        canonical_domain = self.canonical_slang_tessellation_domain(domain)
        valid_domains = {"tri", "quad", "isoline"}
        if canonical_domain not in valid_domains:
            valid_values = ", ".join(sorted(valid_domains))
            raise SlangRefusal(
                f"Slang {stage_name} stage domain '{domain}' must be one of: "
                f"{valid_values}"
            )

    def validate_slang_tessellation_output_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) not in {"hull", "domain"}:
            return

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not topology:
            return

        valid_topologies = {
            "point",
            "line",
            "triangle_cw",
            "triangle_ccw",
        }
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise SlangRefusal(
                f"Slang {stage_name} stage outputtopology "
                f"'{topology}' must be one of: {valid_values}"
            )

    def validate_slang_tessellation_domain_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) not in {"hull", "domain"}:
            return

        domain = self.normalized_slang_stage_attribute_argument(func, "domain")
        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not domain or not topology:
            return

        domain = self.canonical_slang_tessellation_domain(domain)
        if domain in {"tri", "quad"} and topology == "line":
            raise SlangRefusal(
                f"Slang {stage_name} stage domain "
                f"'{domain}' requires outputtopology triangle_cw or triangle_ccw"
            )
        if domain == "isoline" and topology in {"triangle_cw", "triangle_ccw"}:
            raise SlangRefusal(
                f"Slang {stage_name} stage domain 'isoline' requires "
                "outputtopology line"
            )

    def validate_slang_mesh_output_topology(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "mesh":
            return

        topology = self.normalized_slang_stage_attribute_argument(
            func, "outputtopology"
        )
        if not topology:
            return

        valid_topologies = {"point", "line", "triangle"}
        if topology not in valid_topologies:
            valid_values = ", ".join(sorted(valid_topologies))
            raise SlangRefusal(
                f"Slang mesh stage outputtopology '{topology}' must be one of: "
                f"{valid_values}"
            )

    def validate_slang_tessellation_partitioning(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) not in {"hull", "domain"}:
            return

        partitioning = self.normalized_slang_stage_attribute_argument(
            func, "partitioning"
        )
        if not partitioning:
            return

        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if partitioning not in valid_partitioning:
            valid_values = ", ".join(sorted(valid_partitioning))
            raise SlangRefusal(
                f"Slang {stage_name} stage partitioning "
                f"'{partitioning}' must be one of: {valid_values}"
            )

    def validate_required_slang_tessellation_attributes(self, func, stage_name):
        shader_stage = self.slang_shader_stage_name(stage_name)
        if shader_stage == "hull":
            arguments = self.slang_stage_attribute_arguments(
                func, "outputcontrolpoints"
            )
            if not arguments:
                raise SlangRefusal(
                    "Slang tessellation_control stage requires outputcontrolpoints"
                )
            if (
                self.slang_stage_attribute_int_argument(func, "outputcontrolpoints")
                is None
            ):
                raise SlangRefusal(
                    "Slang tessellation_control stage outputcontrolpoints "
                    "requires an integer value"
                )
            return

        if shader_stage == "domain":
            domain = self.normalized_slang_stage_attribute_argument(func, "domain")
            if not domain:
                raise SlangRefusal(
                    "Slang tessellation_evaluation stage requires a domain attribute"
                )

    def validate_slang_patch_constant_function(self, func, stage_name):
        if self.slang_shader_stage_name(stage_name) != "hull":
            return

        arguments = self.slang_stage_attribute_arguments(func, "patchconstantfunc")
        if not arguments:
            return
        if len(arguments) != 1:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc requires "
                "exactly one function name"
            )

        function_name = self.slang_stage_attribute_value_to_string(arguments[0])
        if not function_name:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc requires "
                "a function name"
            )
        if function_name not in self.user_function_names:
            raise SlangRefusal(
                "Slang tessellation_control stage patchconstantfunc "
                f"'{function_name}' does not reference a generated function"
            )

    def validate_slang_numthreads(self, func, stage_name):
        if not self.slang_stage_uses_numthreads(stage_name):
            return

        arguments = self.slang_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return
        if len(arguments) > 3:
            raise SlangRefusal(
                f"Slang {stage_name} stage numthreads requires at most three arguments"
            )

        for argument in arguments:
            value = self.slang_int_literal_value(argument)
            if value is not None and value <= 0:
                raise SlangRefusal(
                    f"Slang {stage_name} stage numthreads values must be positive"
                )

    def validate_slang_stage_attributes(self, func, stage_name):
        if stage_name is None:
            return

        self.validate_slang_stage_attribute_applicability(func, stage_name)
        self.validate_slang_stage_attribute_uniqueness(func, stage_name)
        self.validate_positive_slang_stage_attribute(func, stage_name, "maxvertexcount")
        self.validate_positive_slang_stage_attribute(func, stage_name, "instance")
        self.validate_positive_slang_stage_attribute(
            func, stage_name, "outputcontrolpoints"
        )
        self.validate_slang_tessellation_domain(func, stage_name)
        self.validate_slang_tessellation_output_topology(func, stage_name)
        self.validate_slang_tessellation_domain_topology(func, stage_name)
        self.validate_slang_tessellation_partitioning(func, stage_name)
        self.validate_slang_patch_constant_function(func, stage_name)
        self.validate_required_slang_tessellation_attributes(func, stage_name)
        self.validate_slang_mesh_output_topology(func, stage_name)
        self.validate_slang_numthreads(func, stage_name)
