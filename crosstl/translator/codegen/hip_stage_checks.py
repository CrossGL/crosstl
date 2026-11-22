"""Stage, enum and generic-function checks of the HIP generator: unsupported stages, ray, geometry, tessellation and mesh stage detection, the emulation helpers those stages need, and the matrix type helpers."""

from ..ast import StructNode
from .cuda_subscripts import matrix_column_members, matrix_column_view_helpers
from .enum_utils import (
    collect_enum_struct_variant_fields,
    collect_enum_type_names,
    collect_enum_variant_constants,
    collect_enum_variant_constructor_fields,
    collect_enum_variant_constructors,
    collect_generic_enum_specialization_member_types,
    collect_generic_enum_specializations,
    collect_generic_enum_struct_definitions,
    collect_generic_enum_variant_constants,
    collect_plain_enums,
    collect_struct_payload_enums,
    enum_struct_fields,
)
from .generic_function_utils import (
    collect_unresolved_generic_function_call_names,
    iter_function_nodes,
    raise_unresolved_generic_function_call,
)
from .generic_struct_utils import (
    collect_generic_struct_definitions,
    collect_generic_struct_specialization_member_types,
    collect_generic_struct_specializations,
    generic_struct_member_type_name,
)
from .kernel_only_stages import (
    NON_KERNEL_STAGES,
    refuse_non_kernel_stages,
)
from .stage_utils import (
    normalize_stage_name,
    stage_matches,
)


class HipStageCheckMixin:
    """Stage check members of ``HipCodeGen``."""

    def fused_multiply_add_result_type(self, raw_args):
        if len(raw_args) != 3:
            return None

        scalar_result_type = None
        for raw_arg in raw_args:
            arg_type = self.expression_result_type(raw_arg)
            if self.vector_type_info(arg_type) is not None:
                return arg_type
            component_type = self.scalar_component_type(arg_type)
            if component_type == "double":
                scalar_result_type = "double"
            elif component_type == "float" and scalar_result_type is None:
                scalar_result_type = "float"
        return scalar_result_type

    def reject_unsupported_generic_functions(self, ast_node):
        """Reject generic functions before emitting non-compilable HIP code."""
        functions = list(iter_function_nodes(ast_node)) + list(
            (self.generic_function_specializations or {}).values()
        )
        unresolved = collect_unresolved_generic_function_call_names(self, functions)
        if unresolved:
            raise_unresolved_generic_function_call(self, unresolved[0], "HIP")

    def setup_enum_and_generic_metadata(self, ast_node):
        """Collect enum and concrete generic type metadata used by HIP lowering."""
        structs = list(getattr(ast_node, "structs", []) or [])
        for stage in (getattr(ast_node, "stages", {}) or {}).values():
            structs.extend(getattr(stage, "local_structs", []) or [])
        self.structs_by_name = {
            node.name: node
            for node in structs
            if isinstance(node, StructNode) and getattr(node, "name", None)
        }
        self.generic_enum_struct_definitions = collect_generic_enum_struct_definitions(
            structs
        )
        self.generic_struct_definitions = collect_generic_struct_definitions(
            structs,
            excluded_names=set(self.generic_enum_struct_definitions),
        )
        self.generic_enum_specializations = collect_generic_enum_specializations(
            ast_node,
            self.generic_enum_struct_definitions,
            self.type_name_string,
        )
        self.generic_struct_specializations = collect_generic_struct_specializations(
            ast_node,
            self.generic_struct_definitions,
            self.type_name_string,
        )
        self.plain_enums = collect_plain_enums(structs)
        self.struct_payload_enums = collect_struct_payload_enums(structs)
        self.enum_type_names = collect_enum_type_names(self.plain_enums)
        self.enum_struct_type_names = (
            collect_enum_type_names(self.struct_payload_enums)
            | set(self.generic_enum_struct_definitions)
            | {
                specialization["struct_name"]
                for specialization in self.generic_enum_specializations.values()
            }
        )
        self.enum_struct_variant_fields = collect_enum_struct_variant_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constructors = collect_enum_variant_constructors(
            self.struct_payload_enums
        )
        self.enum_variant_constructor_fields = collect_enum_variant_constructor_fields(
            self.struct_payload_enums
        )
        self.enum_variant_constants = {
            **collect_enum_variant_constants(
                self.plain_enums + self.struct_payload_enums
            ),
            **collect_generic_enum_variant_constants(
                self.generic_enum_struct_definitions
            ),
        }
        self.struct_member_types.update(
            self.generic_struct_definition_member_types(self.generic_struct_definitions)
        )
        self.struct_member_types.update(
            self.enum_payload_struct_member_types(self.struct_payload_enums)
        )
        self.struct_member_types.update(
            collect_generic_enum_specialization_member_types(
                self,
                self.generic_enum_specializations,
            )
        )
        self.struct_member_types.update(
            collect_generic_struct_specialization_member_types(
                self,
                self.generic_struct_specializations,
            )
        )

    def enum_payload_struct_member_types(self, enums):
        member_types = {}
        for enum in enums or []:
            fields = {"variant": "int"}
            for field_name, field_type in enum_struct_fields(enum) or []:
                fields[field_name] = self.type_name_string(field_type)
            member_types[enum.name] = fields
        return member_types

    def generic_struct_definition_member_types(self, definitions):
        member_types = {}
        for name, definition in (definitions or {}).items():
            member_types[name] = {
                member.name: generic_struct_member_type_name(
                    member,
                    self.type_name_string,
                )
                for member in definition["members"]
            }
        return member_types

    def unsupported_stage_types(self):
        return NON_KERNEL_STAGES

    def hip_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def hip_ray_stage_metadata_name(self, stage_name):
        return {
            "ray_any_hit": "any_hit",
            "ray_callable": "callable",
            "ray_closest_hit": "closest_hit",
            "ray_generation": "ray_generation",
            "ray_intersection": "intersection",
            "ray_miss": "miss",
        }[
            stage_name
        ]  # only called for hip_ray_stage_names(), which this table covers

    def has_geometry_stage(self, ast_node, target_stage=None):
        if not stage_matches(target_stage, "geometry"):
            return False

        return self.has_stage(ast_node, "geometry")

    def has_tessellation_stage(self, ast_node, target_stage=None):
        if not (
            stage_matches(target_stage, "tessellation_control")
            or stage_matches(target_stage, "tessellation_evaluation")
        ):
            return False

        return self.has_stage(ast_node, "tessellation_control") or self.has_stage(
            ast_node, "tessellation_evaluation"
        )

    def hip_mesh_task_stage_names(self):
        return {"mesh", "task", "object", "amplification"}

    def has_mesh_task_stage(self, ast_node, target_stage=None):
        stage_names = self.hip_mesh_task_stage_names()
        if target_stage is not None and not any(
            stage_matches(target_stage, stage_name) for stage_name in stage_names
        ):
            return False

        return any(self.has_stage(ast_node, stage_name) for stage_name in stage_names)

    def has_stage(self, ast_node, stage_name):
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == stage_name:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == stage_name
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_hip_geometry_stream_helpers(self):
        return (
            "template <typename T>\n"
            "struct CglHipPointStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipLineStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n\n"
            "template <typename T>\n"
            "struct CglHipTriangleStream {\n"
            "    __device__ void Append(const T& value) { (void)value; }\n"
            "    __device__ void RestartStrip() { }\n"
            "};\n"
        )

    def generate_hip_tessellation_patch_helpers(self):
        return (
            "template <typename T, int N>\n"
            "struct CglHipInputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n\n"
            "template <typename T, int N>\n"
            "struct CglHipOutputPatch {\n"
            "    T data[N];\n"
            "    __device__ const T& operator[](int index) const { return data[index]; }\n"
            "    __device__ T& operator[](int index) { return data[index]; }\n"
            "};\n"
        )

    def generate_hip_mesh_task_helpers(self):
        return (
            "__device__ inline void cgl_hip_set_mesh_output_counts(\n"
            "    unsigned int vertex_count, unsigned int primitive_count)\n"
            "{\n"
            "    (void)vertex_count;\n"
            "    (void)primitive_count;\n"
            "}\n\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "}\n\n"
            "template <typename Payload>\n"
            "__device__ inline void cgl_hip_dispatch_mesh(\n"
            "    unsigned int group_count_x,\n"
            "    unsigned int group_count_y,\n"
            "    unsigned int group_count_z,\n"
            "    const Payload& payload)\n"
            "{\n"
            "    (void)group_count_x;\n"
            "    (void)group_count_y;\n"
            "    (void)group_count_z;\n"
            "    (void)payload;\n"
            "}\n"
        )

    def generate_hip_matrix_type_helpers(self):
        components = ("x", "y", "z", "w")

        def matrix_type(scalar, columns, rows):
            return f"{scalar}{columns}x{rows}"

        def vector_type(scalar, size):
            return f"{scalar}{size}"

        def vector_constructor(scalar, size, args):
            return f"make_{scalar}{size}({', '.join(args)})"

        lines = matrix_column_view_helpers()
        lines.append("// CrossGL matrix value helpers")
        for scalar in ("float", "double"):
            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    count = columns * rows
                    params = ", ".join(f"{scalar} v{i}" for i in range(count))
                    component_values = ", ".join(f"v{i}" for i in range(count))
                    zero = f"{scalar}(0)"
                    add_args = ", ".join(f"m[{i}] + rhs.m[{i}]" for i in range(count))
                    sub_args = ", ".join(f"m[{i}] - rhs.m[{i}]" for i in range(count))
                    neg_args = ", ".join(f"-m[{i}]" for i in range(count))
                    mul_args = ", ".join(f"m[{i}] * rhs" for i in range(count))
                    div_args = ", ".join(f"m[{i}] / rhs" for i in range(count))
                    column_params = ", ".join(
                        f"{vector_type(scalar, rows)} c{column}"
                        for column in range(columns)
                    )
                    column_assignments = " ".join(
                        (f"m[{column * rows + row}] = " f"c{column}.{components[row]};")
                        for column in range(columns)
                        for row in range(rows)
                    )
                    one = f"{scalar}(1)"
                    lines.extend(
                        [
                            f"struct {type_name} {{",
                            f"    {scalar} m[{count}];",
                            f"    static const int CGL_COLUMNS = {columns};",
                            f"    static const int CGL_ROWS = {rows};",
                            f"    __host__ __device__ {type_name}() {{ }}",
                            f"    __host__ __device__ explicit {type_name}({scalar} diagonal) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                        ]
                    )
                    for index in range(min(columns, rows)):
                        lines.append(f"        m[{index * rows + index}] = diagonal;")
                    lines.extend(
                        [
                            "    }",
                            f"    __host__ __device__ {type_name}({column_params}) {{ {column_assignments} }}",
                            (
                                "    template <typename Matrix, "
                                "typename = decltype(Matrix::CGL_COLUMNS), "
                                "typename = decltype(Matrix::CGL_ROWS)>"
                            ),
                            f"    __host__ __device__ explicit {type_name}(const Matrix& source) {{",
                            f"        for (int i = 0; i < {count}; ++i) {{ m[i] = {zero}; }}",
                            f"        for (int column = 0; column < {columns}; ++column) {{",
                            f"            for (int row = 0; row < {rows}; ++row) {{",
                            "                if (column < Matrix::CGL_COLUMNS && row < Matrix::CGL_ROWS) {",
                            f"                    m[column * {rows} + row] = source.m[column * Matrix::CGL_ROWS + row];",
                            "                } else if (column == row) {",
                            f"                    m[column * {rows} + row] = {one};",
                            "                }",
                            "            }",
                            "        }",
                            "    }",
                            # constexpr, so that a file-scope __device__ matrix constant is statically initialized.
                            (
                                f"    __host__ __device__ constexpr {type_name}({params})"
                                f" : m{{{component_values}}} {{ }}"
                            ),
                            *matrix_column_members(scalar, columns, rows),
                            f"    __host__ __device__ {type_name} operator+(const {type_name}& rhs) const {{",
                            f"        return {type_name}({add_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-(const {type_name}& rhs) const {{",
                            f"        return {type_name}({sub_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator-() const {{",
                            f"        return {type_name}({neg_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator*({scalar} rhs) const {{",
                            f"        return {type_name}({mul_args});",
                            "    }",
                            f"    __host__ __device__ {type_name} operator/({scalar} rhs) const {{",
                            f"        return {type_name}({div_args});",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator+=(const {type_name}& rhs) {{",
                            "        *this = *this + rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator-=(const {type_name}& rhs) {{",
                            "        *this = *this - rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator*=({scalar} rhs) {{",
                            "        *this = *this * rhs;",
                            "        return *this;",
                            "    }",
                            f"    __host__ __device__ {type_name}& operator/=({scalar} rhs) {{",
                            "        *this = *this / rhs;",
                            "        return *this;",
                            "    }",
                            "};",
                            "",
                            f"__host__ __device__ inline {type_name} operator*({scalar} lhs, const {type_name}& rhs) {{",
                            "    return rhs * lhs;",
                            "}",
                            "",
                        ]
                    )

            for columns in range(2, 5):
                for rows in range(2, 5):
                    type_name = matrix_type(scalar, columns, rows)
                    result_type = matrix_type(scalar, rows, columns)
                    transpose_values = [
                        f"value.m[{row * rows + column}]"
                        for column in range(rows)
                        for row in range(columns)
                    ]
                    lines.extend(
                        [
                            f"__host__ __device__ inline {result_type} transpose(const {type_name}& value) {{",
                            f"    return {result_type}({', '.join(transpose_values)});",
                            "}",
                            "",
                        ]
                    )

            for size in range(2, 5):
                type_name = matrix_type(scalar, size, size)
                zero = f"{scalar}(0)"
                one = f"{scalar}(1)"
                lines.extend(
                    [
                        f"__host__ __device__ inline {type_name} inverse(const {type_name}& value) {{",
                        f"    {type_name} a(value);",
                        f"    {type_name} result({one});",
                        f"    for (int column = 0; column < {size}; ++column) {{",
                        "        int pivot_row = column;",
                        f"        {scalar} pivot_value = a.m[column * {size} + column];",
                        (
                            f"        {scalar} pivot_abs = pivot_value < {zero} ? "
                            "-pivot_value : pivot_value;"
                        ),
                        f"        for (int row = column + 1; row < {size}; ++row) {{",
                        f"            {scalar} candidate_value = a.m[column * {size} + row];",
                        (
                            f"            {scalar} candidate_abs = "
                            f"candidate_value < {zero} ? "
                            "-candidate_value : candidate_value;"
                        ),
                        "            if (candidate_abs > pivot_abs) {",
                        "                pivot_abs = candidate_abs;",
                        "                pivot_row = row;",
                        "            }",
                        "        }",
                        f"        if (pivot_abs == {zero}) {{",
                        "            return result;",
                        "        }",
                        "        if (pivot_row != column) {",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                {scalar} tmp = a.m[c * {size} + column];",
                        f"                a.m[c * {size} + column] = a.m[c * {size} + pivot_row];",
                        f"                a.m[c * {size} + pivot_row] = tmp;",
                        f"                tmp = result.m[c * {size} + column];",
                        f"                result.m[c * {size} + column] = result.m[c * {size} + pivot_row];",
                        f"                result.m[c * {size} + pivot_row] = tmp;",
                        "            }",
                        "        }",
                        f"        {scalar} pivot = a.m[column * {size} + column];",
                        f"        for (int c = 0; c < {size}; ++c) {{",
                        f"            a.m[c * {size} + column] /= pivot;",
                        f"            result.m[c * {size} + column] /= pivot;",
                        "        }",
                        f"        for (int row = 0; row < {size}; ++row) {{",
                        "            if (row == column) {",
                        "                continue;",
                        "            }",
                        f"            {scalar} factor = a.m[column * {size} + row];",
                        f"            for (int c = 0; c < {size}; ++c) {{",
                        f"                a.m[c * {size} + row] -= factor * a.m[c * {size} + column];",
                        f"                result.m[c * {size} + row] -= factor * result.m[c * {size} + column];",
                        "            }",
                        "        }",
                        "    }",
                        "    return result;",
                        "}",
                        "",
                    ]
                )

            for left_columns in range(2, 5):
                for left_rows in range(2, 5):
                    left_type = matrix_type(scalar, left_columns, left_rows)
                    in_vector = vector_type(scalar, left_columns)
                    out_vector = vector_type(scalar, left_rows)
                    values = []
                    for row in range(left_rows):
                        terms = [
                            f"lhs.m[{column * left_rows + row}] * rhs.{components[column]}"
                            for column in range(left_columns)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {left_type}& lhs, const {in_vector}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_rows, values)};",
                            "}",
                            "",
                        ]
                    )

                    in_vector = vector_type(scalar, left_rows)
                    out_vector = vector_type(scalar, left_columns)
                    values = []
                    for column in range(left_columns):
                        terms = [
                            f"lhs.{components[row]} * rhs.m[{column * left_rows + row}]"
                            for row in range(left_rows)
                        ]
                        values.append(" + ".join(terms))
                    lines.extend(
                        [
                            f"__host__ __device__ inline {out_vector} operator*(const {in_vector}& lhs, const {left_type}& rhs) {{",
                            f"    return {vector_constructor(scalar, left_columns, values)};",
                            "}",
                            "",
                        ]
                    )

                    for right_columns in range(2, 5):
                        right_type = matrix_type(scalar, right_columns, left_columns)
                        result_type = matrix_type(scalar, right_columns, left_rows)
                        values = []
                        for column in range(right_columns):
                            for row in range(left_rows):
                                terms = [
                                    (
                                        f"lhs.m[{shared * left_rows + row}] * "
                                        f"rhs.m[{column * left_columns + shared}]"
                                    )
                                    for shared in range(left_columns)
                                ]
                                values.append(" + ".join(terms))
                        lines.extend(
                            [
                                f"__host__ __device__ inline {result_type} operator*(const {left_type}& lhs, const {right_type}& rhs) {{",
                                f"    return {result_type}({', '.join(values)});",
                                "}",
                                "",
                            ]
                        )
        return "\n".join(lines)

    def validate_supported_stage_types(self, ast_node, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast_node, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast_node, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        refuse_non_kernel_stages("hip", "HIP", unsupported_stages)
