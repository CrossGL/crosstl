"""Bitcasts and built-in calls in the SPIR-V generator: bitcast targets, bfloat16 bitcasts, integer bit built-ins, derivative availability, operand validation for geometric and trigonometric built-ins, bool mix, faceforward, normalize and unsigned sign calls, branches, return types, block terminators and vector components."""

import re
from typing import (
    List,
    Optional,
    Tuple,
)

from .spirv_exact_lookups import (
    SPIRV_DERIVATIVE_OPCODES,
    SPIRV_INTEGER_BIT_BUILTINS,
    spirv_scalar_key,
)
from .spirv_values import (
    SPIRVBuiltinLoweringError,
    SpirvId,
    UnsupportedSPIRVFeatureError,
)


class SpirvBuiltinCallMixin:
    """Builtin call members of ``VulkanSPIRVCodeGen``."""

    def bitcast_builtin_target_component_type(
        self, function_name: str
    ) -> Optional[str]:
        return {
            "asfloat": "float",
            "asint": "int",
            "asuint": "uint",
            "floatBitsToInt": "int",
            "floatBitsToUint": "uint",
            "intBitsToFloat": "float",
            "uintBitsToFloat": "float",
        }.get(function_name)

    def bitcast_builtin_result_type(
        self, function_name: str, source_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None

        target_component_type = self.register_primitive_type(target_component_name)
        if source_type is None:
            return target_component_type

        source_vector = self.vector_component_type_and_count(source_type.type.base_type)
        if source_vector is None:
            return target_component_type

        return self.register_vector_type(target_component_type, source_vector[1])

    def bitcast_builtin_source_is_valid(self, source_type: SpirvId) -> bool:
        source_vector = self.vector_component_type_and_count(source_type.type.base_type)
        if source_vector is not None:
            source_component_name = source_vector[0]
        else:
            source_component_name = self.normalize_primitive_name(
                source_type.type.base_type
            )
        return source_component_name in {"float", "int", "uint"}

    def explicit_bitcast_target_name(self, function_name: str) -> Optional[str]:
        target_match = re.fullmatch(
            r"(?:metal::)?as_type<(.+)>", str(function_name).strip()
        )
        if target_match is None:
            return None
        return target_match.group(1).strip()

    def explicit_bitcast_target_type(self, function_name: str) -> Optional[SpirvId]:
        target_name = self.explicit_bitcast_target_name(function_name)
        if target_name is None:
            return None

        normalized_name = self.normalize_generic_vector_type(target_name)
        primitive_name = self.normalize_primitive_name(normalized_name)
        if primitive_name in {
            "float",
            "double",
            *self.INTEGER_TYPE_NAMES,
            self.BFLOAT16_TYPE_NAME,
        }:
            return self.register_primitive_type(primitive_name)

        vector_info = self.vector_component_type_and_count(normalized_name)
        if vector_info is None:
            return None
        component_name, component_count = vector_info
        if component_name not in {
            "float",
            "double",
            *self.INTEGER_TYPE_NAMES,
            self.BFLOAT16_TYPE_NAME,
        }:
            return None
        return self.register_vector_type(
            self.register_primitive_type(component_name), component_count
        )

    def numeric_bitcast_type_width(self, value_type: SpirvId) -> Optional[int]:
        registered_type = self.ensure_registered_type(value_type)
        type_name = registered_type.type.base_type
        component_count = 1
        vector_info = self.vector_component_type_and_count(type_name)
        if vector_info is not None:
            type_name, component_count = vector_info

        primitive_name = self.normalize_primitive_name(type_name)
        component_width = {
            "float": 32,
            "double": 64,
            "int": 32,
            "uint": 32,
            "i64": 64,
            "u64": 64,
            self.BFLOAT16_TYPE_NAME: 16,
        }.get(primitive_name)
        if component_width is None:
            return None
        return component_width * component_count

    def call_explicit_bitcast_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        target_name = self.explicit_bitcast_target_name(function_name)
        if target_name is None:
            return None

        target_type = self.explicit_bitcast_target_type(function_name)
        if target_type is None:
            raise UnsupportedSPIRVFeatureError(
                "shape-preserving-bitcast",
                f"SPIR-V cannot bitcast to non-numeric type '{target_name}'",
            )
        if len(args) != 1:
            raise UnsupportedSPIRVFeatureError(
                "shape-preserving-bitcast",
                f"{function_name} requires exactly one operand",
            )

        source_type = self.registered_value_type(
            args[0]
        ) or self.ensure_registered_type(args[0].type)
        source_width = self.numeric_bitcast_type_width(source_type)
        target_width = self.numeric_bitcast_type_width(target_type)
        if source_width is None:
            raise UnsupportedSPIRVFeatureError(
                "shape-preserving-bitcast",
                "SPIR-V explicit bitcasts require a numeric scalar or vector "
                f"operand, got '{source_type.type.base_type}'",
            )
        if source_width != target_width:
            raise UnsupportedSPIRVFeatureError(
                "shape-preserving-bitcast",
                f"SPIR-V cannot bitcast {source_type.type.base_type} "
                f"({source_width} bits) to {target_type.type.base_type} "
                f"({target_width} bits); total bit widths must match",
            )
        if source_type.id == target_type.id:
            return args[0]

        result_id = self.get_id()
        self.emit(f"%{result_id} = OpBitcast %{target_type.id} %{args[0].id}")
        result = SpirvId(result_id, target_type.type)
        self.value_types[result_id] = target_type
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(result)
        return result

    def call_bfloat16_bitcast_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Lower Metal's 16-bit ``as_type`` operations without numeric conversion."""
        operation = self.bfloat16_bitcast_operation(function_name, args)
        if operation is None:
            return None

        result_type = self.register_primitive_type(
            self.BFLOAT16_TYPE_NAME if operation == "asbfloat16" else "uint"
        )
        if len(args) != 1:
            self.refuse_lowering(f"{function_name} requires exactly one operand")
            return self.default_value_for_type(result_type)

        source_type_name = self.normalize_primitive_name(args[0].type.base_type)
        if operation == "asbfloat16":
            if source_type_name == self.BFLOAT16_TYPE_NAME:
                return args[0]
            if source_type_name not in self.INTEGER_TYPE_NAMES:
                self.refuse_lowering(
                    "asbfloat16 requires a 16-bit integer bit-pattern " "operand"
                )
                return self.default_value_for_type(result_type)
            opcode = (
                "OpUConvert"
                if source_type_name in self.UNSIGNED_INTEGER_TYPES
                else "OpSConvert"
            )
        else:
            if source_type_name != self.BFLOAT16_TYPE_NAME:
                self.refuse_lowering("asuint16 requires a bfloat16 bit-pattern operand")
                return self.default_value_for_type(result_type)
            opcode = "OpUConvert"

        result_id = self.get_id()
        self.emit(f"%{result_id} = {opcode} %{result_type.id} %{args[0].id}")
        self.value_types[result_id] = result_type
        result = SpirvId(result_id, result_type.type)
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(result)
        return result

    def bfloat16_bitcast_operation(
        self, function_name: str, args: Optional[List[SpirvId]] = None
    ) -> Optional[str]:
        if function_name in {"asbfloat16", "asuint16"}:
            return function_name

        target_match = re.fullmatch(r"(?:metal::)?as_type<(.+)>", function_name)
        if target_match is not None:
            target_type = self.normalize_primitive_name(target_match.group(1).strip())
            if target_type == self.BFLOAT16_TYPE_NAME:
                return "asbfloat16"
            if function_name and args and len(args) == 1:
                source_type = self.normalize_primitive_name(args[0].type.base_type)
                if target_type == "uint" and source_type == self.BFLOAT16_TYPE_NAME:
                    return "asuint16"

        if function_name == "asuint" and args and len(args) == 1:
            source_type = self.normalize_primitive_name(args[0].type.base_type)
            if source_type == self.BFLOAT16_TYPE_NAME:
                return "asuint16"
        return None

    def infer_bfloat16_bitcast_result_type(
        self, function_name: str
    ) -> Optional[SpirvId]:
        operation = self.bfloat16_bitcast_operation(function_name)
        if operation == "asbfloat16":
            return self.register_primitive_type(self.BFLOAT16_TYPE_NAME)
        if operation == "asuint16":
            return self.register_primitive_type("uint")
        return None

    def call_bitcast_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if function_name in self.struct_types:
            return None

        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None

        if len(args) != 1:
            self.refuse_lowering(f"{function_name} requires exactly one operand")
            result_type = self.register_primitive_type(target_component_name)
            return self.default_value_for_type(result_type)

        source_type = self.registered_value_type(
            args[0]
        ) or self.ensure_registered_type(args[0].type)
        result_type = self.bitcast_builtin_result_type(function_name, source_type)
        if result_type is None:
            return None

        if not self.bitcast_builtin_source_is_valid(source_type):
            self.refuse_lowering(
                f"{function_name} requires a 32-bit numeric scalar " "or vector operand"
            )
            return self.default_value_for_type(result_type)

        if source_type.type.base_type == result_type.type.base_type:
            return args[0]

        id_value = self.get_id()
        self.emit(f"%{id_value} = OpBitcast %{result_type.id} %{args[0].id}")
        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(spirv_id)
        return spirv_id

    def infer_bitcast_builtin_result_type(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        target_component_name = self.bitcast_builtin_target_component_type(
            function_name
        )
        if target_component_name is None:
            return None
        if len(args) != 1:
            return self.register_primitive_type(target_component_name)

        source_type = self.infer_expression_result_type(args[0])
        return self.bitcast_builtin_result_type(function_name, source_type)

    def normalize_integer_bit_builtin_name(self, function_name: str) -> Optional[str]:
        """The GLSL name of integer bit built-in ``function_name``, or None when it is not one."""
        return SPIRV_INTEGER_BIT_BUILTINS.get(function_name)

    def integer_bit_builtin_result_type(
        self, function_name: str, source_type: Optional[SpirvId]
    ) -> Optional[SpirvId]:
        function_name = self.normalize_integer_bit_builtin_name(function_name)
        if function_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None

        if source_type is None:
            return self.register_primitive_type("int")

        source_type = self.ensure_registered_type(source_type)
        vector_info = self.vector_component_type_and_count(source_type.type.base_type)
        if vector_info is not None:
            component_type, component_count = vector_info
            if component_type in {"int", "uint"}:
                return source_type
            return self.register_vector_type(
                self.register_primitive_type("int"), component_count
            )

        component_type = self.normalize_primitive_name(source_type.type.base_type)
        if component_type in {"int", "uint"}:
            return source_type
        return self.register_primitive_type("int")

    def integer_bit_builtin_operand_is_valid(self, operand: SpirvId) -> bool:
        operand_type = self.registered_value_type(
            operand
        ) or self.ensure_registered_type(operand.type)
        component_type = self.scalar_or_vector_component_type(operand_type.type)
        return self.normalize_primitive_name(component_type) in {"int", "uint"}

    def call_integer_bit_builtin_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        if function_name in self.struct_types:
            return None

        normalized_name = self.normalize_integer_bit_builtin_name(function_name)
        if normalized_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None

        if len(args) != 1:
            self.refuse_lowering(f"{function_name} requires exactly one operand")
            result_type = self.integer_bit_builtin_result_type(normalized_name, None)
            return self.default_value_for_type(result_type)

        source_type = self.registered_value_type(
            args[0]
        ) or self.ensure_registered_type(args[0].type)
        result_type = self.integer_bit_builtin_result_type(normalized_name, source_type)
        if result_type is None:
            return None

        if not self.integer_bit_builtin_operand_is_valid(args[0]):
            self.refuse_lowering(
                f"{function_name} requires an integer scalar or vector " "operand"
            )
            return self.default_value_for_type(result_type)

        if normalized_name == "bitCount":
            opcode = "OpBitCount"
        elif normalized_name == "bitfieldReverse":
            opcode = "OpBitReverse"
        else:
            component_type = self.scalar_or_vector_component_type(source_type.type)
            if normalized_name == "findLSB":
                instruction = "FindILsb"
            elif self.normalize_primitive_name(component_type) == "uint":
                instruction = "FindUMsb"
            else:
                instruction = "FindSMsb"
            return self.emit_glsl_std450_instruction(instruction, result_type, args)

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{args[0].id}")
        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        if self.is_non_uniform_value(args[0]):
            self.mark_non_uniform_result(spirv_id)
        return spirv_id

    def infer_integer_bit_builtin_result_type(
        self, function_name: str, args: List
    ) -> Optional[SpirvId]:
        normalized_name = self.normalize_integer_bit_builtin_name(function_name)
        if normalized_name not in {"bitCount", "bitfieldReverse", "findLSB", "findMSB"}:
            return None
        if len(args) != 1:
            return self.integer_bit_builtin_result_type(normalized_name, None)
        return self.integer_bit_builtin_result_type(
            normalized_name, self.infer_expression_result_type(args[0])
        )

    def derivative_function_opcode(self, function_name: str) -> Optional[str]:
        return SPIRV_DERIVATIVE_OPCODES.get(function_name)

    def derivative_operand_is_valid(self, value: SpirvId) -> bool:
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return False

        type_name = self.normalize_primitive_name(result_type.type.base_type)
        return (
            type_name in {"float", "double"}
            or self.vector_component_type_and_count(result_type.type.base_type)
            is not None
        )

    def current_derivative_execution_models(self) -> set:
        return self.current_synchronization_execution_models()

    def can_emit_derivative_function(self) -> bool:
        execution_models = self.current_derivative_execution_models()
        derivative_execution_models = {"Fragment", "GLCompute", "MeshEXT", "TaskEXT"}
        return bool(execution_models) and execution_models.issubset(
            derivative_execution_models
        )

    def call_derivative_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        opcode = self.derivative_function_opcode(function_name)
        if opcode is None:
            return None

        if len(args) != 1:
            self.refuse_lowering(f"{function_name} requires 1 operand")
            return self.register_constant(0.0, self.register_primitive_type("float"))

        result_type = self.ensure_registered_type(args[0].type)
        if not self.derivative_operand_is_valid(args[0]):
            self.refuse_lowering(
                f"{function_name} requires floating-point scalar " "or vector operand"
            )
            return self.default_value_for_type(result_type)

        if not self.can_emit_derivative_function():
            self.refuse_lowering(
                f"derivative builtin '{function_name}' requires Fragment "
                "or compute-derivative-capable execution model; current execution "
                f"model: {self.current_execution_model_label()}"
            )
            return self.default_value_for_type(result_type)

        compute_derivative_models = {"GLCompute", "MeshEXT", "TaskEXT"}
        if self.current_derivative_execution_models() & compute_derivative_models:
            self.require_compute_derivatives()

        if opcode.endswith("Fine") or opcode.endswith("Coarse"):
            self.require_capability("DerivativeControl")

        id_value = self.get_id()
        self.emit(f"%{id_value} = {opcode} %{result_type.id} %{args[0].id}")

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_scalar_constructor(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        primitive_name = self.normalize_primitive_name(function_name)
        if (
            primitive_name not in {
                "bool",
                "float",
                "double",
                self.BFLOAT16_TYPE_NAME,
            } | self.INTEGER_TYPE_NAMES
        ):
            return None

        target_type = self.register_primitive_type(primitive_name)
        if not args:
            return self.default_value_for_type(target_type)
        if len(args) != 1:
            self.refuse_lowering(
                f"Constructor {function_name} expected 1 component "
                f"but got {len(args)}"
            )

        value = args[0]
        if self.vector_component_type_and_count(value.type.base_type) is not None:
            self.refuse_lowering(
                f"Constructor {function_name} requires a scalar operand"
            )
            return self.default_value_for_type(target_type)

        converted = self.convert_scalar_to_type(value, target_type)
        if self.normalize_primitive_name(
            converted.type.base_type
        ) == self.normalize_primitive_name(target_type.type.base_type):
            return converted

        self.refuse_lowering(
            f"Constructor {function_name} cannot convert " f"{value.type.base_type}"
        )
        return self.default_value_for_type(target_type)

    def dot_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is not None and vector_info[0] in {"float", "double"}:
                return self.register_primitive_type(vector_info[0])
        return self.register_primitive_type("float")

    def dot_operands_are_valid(self, args: List[SpirvId]) -> bool:
        left_vector = self.vector_component_type_and_count(args[0].type.base_type)
        right_vector = self.vector_component_type_and_count(args[1].type.base_type)
        if left_vector is None or right_vector is None:
            return False
        if left_vector != right_vector:
            return False
        return left_vector[0] in {"float", "double"}

    def vector_math_operand_diagnostic(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[str]:
        if self.is_float32_unary_std450_function(
            function_name
        ) and not self.float32_scalar_or_vector_operand_is_valid(args[0]):
            return (
                f"{function_name} requires 32-bit floating-point "
                "scalar or vector operand"
            )

        if function_name == "length" and not self.metric_operand_is_valid(args[0]):
            return "length requires floating-point scalar or vector operand"

        if function_name == "distance" and not self.distance_operands_are_valid(args):
            return (
                "distance requires matching floating-point scalar " "or vector operands"
            )

        if function_name == "normalize" and not self.metric_operand_is_valid(args[0]):
            return "normalize requires floating-point scalar or vector operand"

        if function_name == "atan2" and not self.atan2_operands_are_valid(args):
            return (
                "atan2 requires matching 32-bit floating-point "
                "scalar or vector operands"
            )

        if function_name == "cross" and not self.cross_operands_are_valid(args):
            return (
                "cross requires matching 3-component " "floating-point vector operands"
            )

        if function_name == "reflect" and not self.reflect_operands_are_valid(args):
            return (
                "reflect requires matching floating-point scalar " "or vector operands"
            )

        if function_name == "refract" and not self.refract_operands_are_valid(args):
            return (
                "refract requires matching floating-point incident "
                "and normal operands plus scalar eta"
            )

        return None

    def builtin_lowering_error(
        self, function_name: str, detail: str, *, reason: str = "invalid-operands"
    ) -> SPIRVBuiltinLoweringError:
        """Return the typed refusal for a call with no meaning-preserving lowering."""
        return SPIRVBuiltinLoweringError(function_name, detail, reason=reason)

    def is_float32_unary_std450_function(self, function_name: str) -> bool:
        return function_name in {
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "exp",
            "log",
            "exp2",
            "log2",
            "sqrt",
            "inversesqrt",
            "floor",
            "ceil",
            "fract",
            "trunc",
            "round",
            "roundEven",
            "degrees",
            "radians",
        }

    def float32_scalar_or_vector_operand_is_valid(self, arg: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(arg.type)
        if component_type != "float":
            return False
        scalar_type = self.normalize_primitive_name(arg.type.base_type)
        return (
            scalar_type == "float"
            or self.vector_component_type_and_count(arg.type.base_type) is not None
        )

    def metric_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            component_type = self.scalar_or_vector_component_type(arg.type)
            if component_type in {"float", "double"}:
                return self.register_primitive_type(component_type)
        return self.register_primitive_type("float")

    def metric_operand_is_valid(self, arg: SpirvId) -> bool:
        component_type = self.scalar_or_vector_component_type(arg.type)
        if component_type not in {"float", "double"}:
            return False
        scalar_type = self.normalize_primitive_name(arg.type.base_type)
        return (
            scalar_type in {"float", "double"}
            or self.vector_component_type_and_count(arg.type.base_type) is not None
        )

    def distance_operands_are_valid(self, args: List[SpirvId]) -> bool:
        return (
            self.matching_floating_scalar_or_vector_operands(args[0], args[1])
            is not None
        )

    def atan2_operands_are_valid(self, args: List[SpirvId]) -> bool:
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return False
        component_type, _ = operand_shape
        return component_type == "float"

    def cross_result_type(self, args: List[SpirvId]) -> SpirvId:
        for arg in args:
            vector_info = self.vector_component_type_and_count(arg.type.base_type)
            if vector_info is not None and vector_info[0] in {"float", "double"}:
                component_type = self.register_primitive_type(vector_info[0])
                return self.register_vector_type(component_type, 3)
        return self.register_vector_type(self.register_primitive_type("float"), 3)

    def cross_operands_are_valid(self, args: List[SpirvId]) -> bool:
        left_vector = self.vector_component_type_and_count(args[0].type.base_type)
        right_vector = self.vector_component_type_and_count(args[1].type.base_type)
        return (
            left_vector is not None
            and right_vector is not None
            and left_vector == right_vector
            and left_vector[0] in {"float", "double"}
            and left_vector[1] == 3
        )

    def matching_floating_scalar_or_vector_operands(
        self, left: SpirvId, right: SpirvId
    ) -> Optional[Tuple[str, Optional[int]]]:
        left_vector = self.vector_component_type_and_count(left.type.base_type)
        right_vector = self.vector_component_type_and_count(right.type.base_type)
        left_component = self.scalar_or_vector_component_type(left.type)
        right_component = self.scalar_or_vector_component_type(right.type)
        if left_component not in {"float", "double"}:
            return None
        if left_component != right_component:
            return None
        if left_vector is None and right_vector is None:
            return left_component, None
        if left_vector is None or right_vector is None:
            return None
        if left_vector != right_vector:
            return None
        return left_component, left_vector[1]

    def reflect_operands_are_valid(self, args: List[SpirvId]) -> bool:
        return (
            self.matching_floating_scalar_or_vector_operands(args[0], args[1])
            is not None
        )

    def refract_operands_are_valid(self, args: List[SpirvId]) -> bool:
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return False
        eta_vector = self.vector_component_type_and_count(args[2].type.base_type)
        eta_component = self.scalar_or_vector_component_type(args[2].type)
        return eta_vector is None and eta_component in {"float", "double"}

    def struct_constructor_args(
        self, struct_name: str, args: List[SpirvId]
    ) -> List[SpirvId]:
        members = self.current_struct_members.get(struct_name, [])
        constructor_args = []
        for index, (member_type, _) in enumerate(members):
            if index < len(args):
                constructor_args.append(
                    self.convert_value_to_type(args[index], member_type)
                )
            else:
                constructor_args.append(self.default_value_for_type(member_type))

        if len(args) < len(members):
            self.refuse_lowering(
                f"Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}"
            )
        elif len(args) > len(members):
            self.refuse_lowering(
                f"Constructor {struct_name} expected {len(members)} "
                f"members but got {len(args)}"
            )

        return constructor_args

    def call_numeric_mix_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower floating mix with result-typed operands."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FMix {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_bool_mix_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower mix(x, y, bool/bvec) to selector semantics."""
        false_value, true_value, condition = args
        result_type = self.ensure_registered_type(false_value.type)
        true_type = self.ensure_registered_type(true_value.type)
        if result_type.type.base_type != true_type.type.base_type:
            return None

        condition_vector = self.vector_component_type_and_count(
            condition.type.base_type
        )
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        if condition_vector is not None:
            if condition_vector[0] != "bool":
                return None
            if result_vector is None or result_vector[1] != condition_vector[1]:
                return None
        elif condition.type.base_type != "bool":
            return None

        return self.select_operation(result_type, condition, true_value, false_value)

    def call_abs_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower abs to typed GLSL.std.450 floating or signed-integer ops."""
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        glsl_function = {
            "float": "FAbs",
            "double": "FAbs",
            "int": "SAbs",
        }.get(component_type)
        if glsl_function is None:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{glsl_function} %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_pow_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower pow to GLSL.std.450 with validator-compatible operand types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type != "float":
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Pow {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_fma_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower fma to GLSL.std.450 with validator-compatible operand types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Fma {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_faceforward_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower faceforward to GLSL.std.450 with validator-compatible types."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FaceForward {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_refract_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower refract to GLSL.std.450 with result-typed scalar eta."""
        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None
        operand_component, _ = operand_shape
        if operand_component != component_type:
            return None

        eta_vector = self.vector_component_type_and_count(args[2].type.base_type)
        eta_component = self.scalar_or_vector_component_type(args[2].type)
        if eta_vector is not None or eta_component not in {"float", "double"}:
            return None

        eta_type = self.register_primitive_type(component_type)
        eta = self.convert_scalar_to_type(args[2], eta_type)
        if self.normalize_primitive_name(eta.type.base_type) != component_type:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Refract %{args[0].id} %{args[1].id} %{eta.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_length_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower length to GLSL.std.450 with scalar component result type."""
        if not self.metric_operand_is_valid(value):
            return None

        result_type = self.metric_result_type([value])
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Length %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_distance_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower distance to GLSL.std.450 with matching scalar/vector operands."""
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None

        component_type, _ = operand_shape
        result_type = self.register_primitive_type(component_type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Distance %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_normalize_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower normalize to GLSL.std.450 with the operand result type."""
        if not self.metric_operand_is_valid(value):
            return None

        result_type = self.ensure_registered_type(value.type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Normalize %{value.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_reflect_function(self, args: List[SpirvId]) -> Optional[SpirvId]:
        """Lower reflect to GLSL.std.450 with matching scalar/vector operands."""
        operand_shape = self.matching_floating_scalar_or_vector_operands(
            args[0], args[1]
        )
        if operand_shape is None:
            return None

        result_type = self.ensure_registered_type(args[0].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"Reflect %{args[0].id} %{args[1].id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_step_smoothstep_function(
        self, glsl_function: str, args: List[SpirvId], value_index: int
    ) -> Optional[SpirvId]:
        """Lower step/smoothstep using the value operand's scalar/vector type."""
        result_type = self.ensure_registered_type(args[value_index].type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type not in {"float", "double"}:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{glsl_function} {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def call_sign_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower sign to typed GLSL.std.450 or unsigned select operations."""
        result_type = self.ensure_registered_type(value.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        if component_type in {"float", "double", "int"}:
            glsl_function = "SSign" if component_type == "int" else "FSign"
            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
                f"{glsl_function} %{value.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if component_type == "uint":
            return self.call_unsigned_sign_function(value, result_type)

        return value if component_type == "bool" else None

    def call_unsigned_sign_function(
        self, value: SpirvId, result_type: SpirvId
    ) -> SpirvId:
        """Lower unsigned sign as value > 0 ? 1 : 0."""
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        uint_type = self.register_primitive_type("uint")
        bool_type = self.register_primitive_type("bool")
        zero = self.register_constant(0, uint_type)
        one = self.register_constant(1, uint_type)

        if vector_info is not None:
            _, component_count = vector_info
            zero_value = self.register_vector_constant(
                result_type, [zero] * component_count
            )
            one_value = self.register_vector_constant(
                result_type, [one] * component_count
            )
        else:
            zero_value = zero
            one_value = one

        condition = self.binary_operation(">", bool_type, value, zero_value)
        return self.select_operation(result_type, condition, one_value, zero_value)

    def call_min_max_clamp_function(
        self, function_name: str, args: List[SpirvId]
    ) -> Optional[SpirvId]:
        """Lower min/max/clamp to typed GLSL.std.450 extinsts."""
        # A scalar operand next to vector operands is splatted to their width, as HLSL's intrinsics promote it
        # (``max(dot(n, l), (0.0).xxx)`` is a float3), so the first vector operand gives the result type.
        shaped_arg = next(
            (
                arg
                for arg in args
                if self.vector_component_type_and_count(arg.type.base_type) is not None
            ),
            args[0],
        )
        result_type = self.ensure_registered_type(shaped_arg.type)
        component_type = self.scalar_or_vector_component_type(result_type.type)
        prefix = {
            "float": "F",
            "double": "F",
            "int": "S",
            "uint": "U",
        }.get(component_type)
        if prefix is None:
            return None

        operands = self.match_extinst_operands_to_result_type(result_type, args)
        if operands is None:
            return None

        op_suffix = {
            "min": "Min",
            "max": "Max",
            "clamp": "Clamp",
        }[function_name]
        id_value = self.get_id()
        arg_list = " ".join(f"%{arg.id}" for arg in operands)
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"{prefix}{op_suffix} {arg_list}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def match_extinst_operands_to_result_type(
        self, result_type: SpirvId, args: List[SpirvId]
    ) -> Optional[List[SpirvId]]:
        """Return operands shaped to the given GLSL.std.450 result type."""
        result_vector = self.vector_component_type_and_count(result_type.type.base_type)
        result_component_type = self.scalar_or_vector_component_type(result_type.type)

        operands = []
        for arg in args:
            arg_vector = self.vector_component_type_and_count(arg.type.base_type)
            if result_vector is None:
                if arg_vector is not None:
                    return None
                scalar_arg = self.convert_scalar_to_type(arg, result_type)
                if (
                    self.normalize_primitive_name(scalar_arg.type.base_type)
                    != result_component_type
                ):
                    return None
                operands.append(scalar_arg)
                continue

            if arg_vector is not None:
                if arg_vector != result_vector:
                    return None
                operands.append(arg)
                continue

            component_type = self.register_primitive_type(result_component_type)
            scalar_arg = self.convert_scalar_to_type(arg, component_type)
            if (
                self.normalize_primitive_name(scalar_arg.type.base_type)
                != result_component_type
            ):
                return None
            operands.append(self.splat_scalar_to_vector(scalar_arg, result_type))

        return operands

    def call_saturate_function(self, value: SpirvId) -> Optional[SpirvId]:
        """Lower floating saturate(x) to GLSL.std.450 FClamp."""
        result_type = self.ensure_registered_type(value.type)
        vector_info = self.vector_component_type_and_count(result_type.type.base_type)
        if vector_info is not None:
            component_type_name, component_count = vector_info
            if component_type_name not in {"float", "double"}:
                return None

            component_type = self.register_primitive_type(component_type_name)
            zero = self.register_constant(0.0, component_type)
            one = self.register_constant(1.0, component_type)
            zero_vector = self.register_vector_constant(
                result_type, [zero] * component_count
            )
            one_vector = self.register_vector_constant(
                result_type, [one] * component_count
            )

            id_value = self.get_id()
            self.emit(
                f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
                f"FClamp %{value.id} %{zero_vector.id} %{one_vector.id}"
            )

            spirv_id = SpirvId(id_value, result_type.type)
            self.value_types[id_value] = result_type
            return spirv_id

        if result_type.type.base_type not in {"float", "double"}:
            return None

        zero = self.register_constant(0.0, result_type)
        one = self.register_constant(1.0, result_type)
        id_value = self.get_id()
        self.emit(
            f"%{id_value} = OpExtInst %{result_type.id} %{self.glsl_std450_id} "
            f"FClamp %{value.id} %{zero.id} %{one.id}"
        )

        spirv_id = SpirvId(id_value, result_type.type)
        self.value_types[id_value] = result_type
        return spirv_id

    def create_branch(self, target_label: SpirvId):
        """Create an unconditional branch."""
        self.emit(f"OpBranch %{target_label.id}")

    def create_conditional_branch(
        self, condition: SpirvId, true_label: SpirvId, false_label: SpirvId
    ):
        """Create a conditional branch."""
        self.emit(
            f"OpBranchConditional %{condition.id} %{true_label.id} %{false_label.id}"
        )

    def create_selection_merge(
        self, merge_label: SpirvId, selection_control: str = "None"
    ):
        """Create a selection merge instruction for if/switch statements."""
        self.emit(f"OpSelectionMerge %{merge_label.id} {selection_control}")

    def create_loop_merge(
        self, merge_label: SpirvId, continue_label: SpirvId, loop_control: str = "None"
    ):
        """Create a loop merge instruction for loops."""
        self.emit(f"OpLoopMerge %{merge_label.id} %{continue_label.id} {loop_control}")

    def current_spirv_return_type(self) -> Optional[SpirvId]:
        """Return the actual SPIR-V return type for the current function."""
        if self.current_function_name is not None:
            signature = self.resolve_function_signature(self.current_function_name)
            if signature is not None:
                return signature[0]
        return self.current_return_type

    def create_return(self):
        """Create a return instruction."""
        return_type = self.current_spirv_return_type()
        if return_type is not None and return_type.type.base_type != "void":
            function_name = self.current_function_name or "<unknown>"
            self.refuse_lowering(f"Bare return in non-void function {function_name}")
            default_value = self.default_value_for_type(return_type)
            self.create_return_value(default_value)
            return

        self.emit("OpReturn")

    def create_return_value(self, value: SpirvId):
        """Create a return value instruction."""
        if self.store_entry_point_return_value(value):
            return

        if self.current_return_semantic_output is not None:
            target_type = self.pointer_pointee_type(self.current_return_semantic_output)
            if target_type is not None:
                value = self.convert_value_to_type(value, target_type)
            self.store_to_variable(self.current_return_semantic_output, value)
            self.create_return()
            return

        if self.current_return_type is not None:
            value = self.convert_value_to_type(value, self.current_return_type)
        self.emit(f"OpReturnValue %{value.id}")

    def create_unreachable(self):
        """Create an unreachable terminator for impossible fallthrough paths."""
        self.emit("OpUnreachable")

    def current_block_has_terminator(self) -> bool:
        """Return whether the current block already ends in a terminator."""
        for line in reversed(self.code_lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue
            if re.match(r"%\d+ = OpLabel$", stripped):
                return False
            return stripped.startswith(
                (
                    "OpBranch",
                    "OpReturn",
                    "OpKill",
                    "OpUnreachable",
                    "OpEmitMeshTasksEXT",
                    "OpIgnoreIntersectionKHR",
                    "OpTerminateRayKHR",
                )
            )
        return False

    def normalize_primitive_name(self, type_name: str) -> str:
        """The scalar spelling ``type_name`` is classified by; see :func:`.spirv_exact_lookups.spirv_scalar_key`."""
        return spirv_scalar_key(type_name, self.BFLOAT16_TYPE_NAME)

    def normalize_generic_vector_type(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        generic_prefix = re.match(r"(?:::)?(?:metal::)?vec<", compact)
        generic_parts = None
        if generic_prefix is not None and compact.endswith(">"):
            body = compact[generic_prefix.end() : -1]
            if "," in body:
                generic_parts = body.split(",", 1)
        if generic_parts is not None:
            source_element_type, source_width = generic_parts
            element_type = self.normalize_primitive_name(source_element_type)
            width_text = source_width.strip()
            while width_text.startswith("(") and width_text.endswith(")"):
                width_text = width_text[1:-1].strip()
            try:
                width = int(width_text, 0)
            except ValueError as exc:
                raise UnsupportedSPIRVFeatureError(
                    "unresolved-generic-vector-width",
                    "SPIR-V requires a concrete generic-vector width; could not "
                    f"resolve '{source_width}' in '{compact}'",
                    missing_capabilities=("spirv.generic_vector_width",),
                ) from exc
            if width not in {2, 3, 4}:
                raise UnsupportedSPIRVFeatureError(
                    "unsupported-generic-vector-width",
                    f"SPIR-V generic vector '{compact}' has unsupported width "
                    f"{width}; lower it to a supported aggregate representation",
                    missing_capabilities=("spirv.generic_vector_width",),
                )
            supported_elements = {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "i64",
                "u64",
                self.BFLOAT16_TYPE_NAME,
            }
            if element_type not in supported_elements:
                raise UnsupportedSPIRVFeatureError(
                    "unsupported-generic-vector-element-type",
                    f"SPIR-V generic vector '{compact}' has unsupported element "
                    f"type '{source_element_type}'",
                    missing_capabilities=("spirv.generic_vector_element_type",),
                )
            compact = f"vec{width}<{element_type}>"
        half_vector_match = re.fullmatch(
            r"(?:f16vec|half|min16float|float16_t)([234])", compact
        )
        if half_vector_match:
            return f"vec{half_vector_match.group(1)}"
        bfloat_vector_match = re.fullmatch(
            r"(?:bfloat|bfloat16|bfloat16_t)(?:vec)?([234])", compact
        )
        if bfloat_vector_match:
            return f"v{bfloat_vector_match.group(1)}{self.BFLOAT16_TYPE_NAME}"
        narrow_vector_match = re.fullmatch(r"([iu])(?:8|16)vec([234])", compact)
        if narrow_vector_match:
            signedness, size = narrow_vector_match.groups()
            return f"{'i' if signedness == 'i' else 'u'}vec{size}"
        wide_vector_match = re.fullmatch(r"([iu])64vec([234])", compact)
        if wide_vector_match:
            signedness, size = wide_vector_match.groups()
            return f"v{size}{signedness}64"
        match = re.fullmatch(r"vec([234])<([^>]+)>", compact)
        if match:
            size, element_type = match.groups()
            element_type = self.normalize_primitive_name(element_type)
        else:
            match = re.fullmatch(
                r"(float|double|int|uint|bool|i64|u64|int64|int64_t|long|"
                r"uint64|uint64_t|ulong)([234])",
                compact,
            )
            if not match:
                return compact
            element_type, size = match.groups()
            element_type = self.normalize_primitive_name(element_type)

        if element_type in {"i64", "u64"}:
            return f"v{size}{element_type}"
        if element_type == self.BFLOAT16_TYPE_NAME:
            return f"v{size}{self.BFLOAT16_TYPE_NAME}"
        prefixes = {
            "float": "vec",
            "double": "dvec",
            "int": "ivec",
            "uint": "uvec",
            "bool": "bvec",
        }
        return f"{prefixes.get(element_type, 'vec')}{size}"

    def normalize_hlsl_matrix_type(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        match = re.fullmatch(
            r"(float|double|half|min16float|float16_t)([234])x([234])",
            compact,
        )
        if not match:
            return str(type_str)

        component_type, rows, cols = match.groups()
        prefix = "dmat" if component_type == "double" else "mat"
        rows = int(rows)
        cols = int(cols)
        if rows == cols:
            return f"{prefix}{cols}"
        return f"{prefix}{cols}x{rows}"

    def vector_component_type_and_count(
        self, type_str: str
    ) -> Optional[Tuple[str, int]]:
        type_str = self.normalize_generic_vector_type(type_str)
        internal_match = re.fullmatch(
            r"v([234])(float|double|int|uint|bool|i64|u64|bfloat16)", type_str
        )
        if internal_match:
            size, component_type = internal_match.groups()
            return component_type, int(size)

        vector_prefixes = (
            ("dvec", "double"),
            ("ivec", "int"),
            ("uvec", "uint"),
            ("bvec", "bool"),
            ("vec", "float"),
        )
        for prefix, component_type in vector_prefixes:
            if type_str.startswith(prefix) and type_str[len(prefix) :].isdigit():
                return component_type, int(type_str[len(prefix) :])
        return None

    def normalize_resource_type_name(self, type_str: str) -> str:
        compact = re.sub(r"\s+", "", str(type_str))
        direct_aliases = {
            "Texture1D": "sampler1D",
            "Texture1DArray": "sampler1DArray",
            "Texture2D": "sampler2D",
            "Texture2DArray": "sampler2DArray",
            "Texture2DMS": "sampler2DMS",
            "Texture2DMSArray": "sampler2DMSArray",
            "Texture3D": "sampler3D",
            "TextureCube": "samplerCube",
            "TextureCubeArray": "samplerCubeArray",
        }
        if compact in direct_aliases:
            return direct_aliases[compact]

        resource_match = re.fullmatch(
            r"([iu]?(?:sampler|image)|texture)"
            r"(1d|2d|3d|cube|buffer)"
            r"(ms)?(array)?(shadow)?",
            compact,
            re.IGNORECASE,
        )
        if resource_match:
            prefix, dim, ms_suffix, array_suffix, shadow_suffix = (
                resource_match.groups()
            )
            dim = {
                "1d": "1D",
                "2d": "2D",
                "3d": "3D",
                "cube": "Cube",
                "buffer": "Buffer",
            }[dim.lower()]
            return (
                f"{prefix.lower()}{dim}"
                f"{'MS' if ms_suffix else ''}"
                f"{'Array' if array_suffix else ''}"
                f"{'Shadow' if shadow_suffix else ''}"
            )

        texture_match = re.fullmatch(
            r"Texture(1D|2D|3D|Cube)(MS)?(Array)?<([^>]+)>", compact
        )
        if texture_match:
            dim, ms_suffix, array_suffix, component_type = texture_match.groups()
            prefix = {
                "int": "i",
                "uint": "u",
            }.get(self.normalize_primitive_name(component_type), "")
            return f"{prefix}sampler{dim}" f"{ms_suffix or ''}{array_suffix or ''}"

        storage_match = re.fullmatch(
            r"RWTexture(1D|2D|3D|Cube)(MS)?(Array)?(?:<([^>]+)>)?", compact
        )
        if storage_match:
            dim, ms_suffix, array_suffix, component_type = storage_match.groups()
            prefix = {
                "int": "i",
                "uint": "u",
            }.get(self.normalize_primitive_name(component_type or "float"), "")
            return f"{prefix}image{dim}{ms_suffix or ''}{array_suffix or ''}"

        return compact
