"""Assignments, operators, function calls and intrinsics in the CUDA to CrossGL converter: time, sleep, cache-load, fence, warp, half-precision, conversion, integer and floating-point intrinsics."""

from ...assignment_operands import grouped_operand
from ...bfloat16_conversions import format_bfloat16_conversion
from ...thread_hierarchy import global_invocation_axis
from ..bit_intrinsics import format_bit_intrinsic
from ..float_classification import format_float_classification_intrinsic
from ..CudaAst import UnaryOpNode
from ..directed_rounding import directed_rounding_call, refuse_directed_rounding
from ..function_templates import generic_call_name
from ..pointer_math_import import format_pointer_output_call
from ..sizeof_operands import sizeof_type_operand


class CudaImportIntrinsicMixin:
    """Intrinsic members of ``CudaToCrossGLConverter``."""

    def visit_AssignmentNode(self, node):
        left = self.visit(node.left)
        operator = getattr(node, "operator", "=")
        runtime_expression = (
            self.format_cuda_runtime_expression(node.right) if operator == "=" else None
        )
        if runtime_expression is not None:
            comments, value = runtime_expression
            for comment in comments:
                self.emit(comment)
            self.register_warp_mask_assignment(node.left, left, operator, value)
            self.emit(f"{left} = {value};")
            return None

        right = self.visit(node.right)
        self.register_warp_mask_assignment(node.left, left, operator, right)
        return f"{left} {operator} {right}"

    def register_warp_mask_assignment(self, raw_left, output_left, operator, value):
        raw_name = raw_left if isinstance(raw_left, str) else None
        if raw_name is None:
            return

        assigned_value = value if operator == "=" else None
        self.register_warp_mask_value(raw_name, output_left, assigned_value)

    def visit_BinaryOpNode(self, node):
        axis = global_invocation_axis(node)
        if axis is not None:
            return f"gl_GlobalInvocationID.{axis}"
        left = grouped_operand(self, node.left)
        right = grouped_operand(self, node.right)
        return f"({left} {node.op} {right})"

    def visit_UnaryOpNode(self, node):
        operand = grouped_operand(self, node.operand)
        if node.op.startswith("post"):
            return f"({operand}{node.op[4:]})"
        else:
            return f"({node.op}{operand})"

    def visit_FunctionCallNode(self, node):
        if self.is_get_method_call(node):
            return self.visit(node.name.object)

        cuda_async_sync_call = self.format_cuda_async_sync_call(node)
        if cuda_async_sync_call is not None:
            return cuda_async_sync_call

        cooperative_call = self.format_cooperative_group_call(node)
        if cooperative_call is not None:
            return cooperative_call

        raw_name = node.name if isinstance(node.name, str) else self.visit(node.name)
        runtime_expression = self.format_cuda_runtime_inline_expression(node)
        if runtime_expression is not None:
            return runtime_expression

        if raw_name == "lambda":
            return self.format_lambda_call(node.args)
        if raw_name == "cuda_fold_expression":
            return self.format_cuda_fold_expression(node.args)
        if raw_name == "cuda_comma_expression":
            return self.format_cuda_comma_expression(node.args)

        if not self.is_user_defined_function(raw_name):
            pointer_output_call = format_pointer_output_call(
                self, raw_name, node.args, UnaryOpNode
            )
            if pointer_output_call is not None:
                return pointer_output_call

        args = [self.visit(arg) for arg in node.args]
        # Names with a double underscore are reserved to the implementation (C++ [lex.name]), so these are the
        # intrinsics even where a program function shares the name.
        bit_intrinsic = format_bit_intrinsic(self, raw_name, node.args, args, "CUDA")
        if bit_intrinsic is not None:
            return bit_intrinsic
        classification = format_float_classification_intrinsic(raw_name, args)
        if classification is not None:
            return classification
        if raw_name == "sizeof" and len(args) == 1 and isinstance(node.args[0], str):
            # The type operand of ``sizeof(let)`` names the struct as its declaration does.
            args = [
                sizeof_type_operand(node.args[0], self.convert_cuda_type_to_crossgl)
                or self.rename_keyword_record_names(args[0])
            ]
        args_str = ", ".join(args)
        device_heap_call = self.format_device_heap_call(raw_name, args)
        if device_heap_call is not None:
            return device_heap_call
        make_unique = self.format_make_unique_call(raw_name, args)
        if make_unique is not None:
            return make_unique

        unique_ptr_init = self.format_unique_ptr_constructor_call(raw_name, args)
        if unique_ptr_init is not None:
            return unique_ptr_init

        if self.is_cuda_intrinsic_override_function(raw_name):
            integer_intrinsic = self.format_cuda_integer_intrinsic_call(raw_name, args)
            if integer_intrinsic is not None:
                return integer_intrinsic

        generic_call = generic_call_name(
            raw_name, self.function_templates, self.convert_cuda_type_to_crossgl
        )
        if generic_call is not None:
            return f"{generic_call}({args_str})"
        if self.is_user_defined_function(raw_name):
            # A call names the function as its declaration does, ``let_`` for ``let``.
            func_name = (
                self.convert_cuda_function_name_to_crossgl(raw_name)
                if raw_name.startswith("::") or self.is_simple_identifier(raw_name)
                else raw_name
            )
            return f"{func_name}({args_str})"
        if self.is_simple_identifier(raw_name) and raw_name in self.cuda_record_names:
            # ``let{1.0f}`` and ``let()`` construct the record under its declared name.
            return f"{self.convert_cuda_type_to_crossgl(raw_name)}({args_str})"

        load_cache_intrinsic = self.format_cuda_load_cache_intrinsic_call(
            raw_name, node.args, args
        )
        if load_cache_intrinsic is not None:
            return load_cache_intrinsic

        memory_fence = self.format_cuda_memory_fence_call(raw_name, args)
        if memory_fence is not None:
            return memory_fence

        time_function = self.format_cuda_time_function_call(raw_name, args)
        if time_function is not None:
            return time_function

        nanosleep = self.format_cuda_nanosleep_call(raw_name, args)
        if nanosleep is not None:
            return nanosleep

        warp_intrinsic = self.format_cuda_warp_intrinsic_call(raw_name, args)
        if warp_intrinsic is not None:
            return warp_intrinsic

        directed_rounding = directed_rounding_call(
            self.directed_rounding, raw_name, args
        )
        if directed_rounding is not None:
            return directed_rounding
        refuse_directed_rounding(raw_name)
        bfloat16_conversion = format_bfloat16_conversion(raw_name, args)
        if bfloat16_conversion is not None:
            return bfloat16_conversion
        fp16_intrinsic = self.format_cuda_fp16_intrinsic_call(raw_name, args)
        if fp16_intrinsic is not None:
            return fp16_intrinsic

        type_cast_intrinsic = self.format_cuda_type_cast_intrinsic_call(raw_name, args)
        if type_cast_intrinsic is not None:
            return type_cast_intrinsic

        integer_intrinsic = self.format_cuda_integer_intrinsic_call(raw_name, args)
        if integer_intrinsic is not None:
            return integer_intrinsic

        float_intrinsic = self.format_cuda_float_intrinsic_call(raw_name, args)
        if float_intrinsic is not None:
            return float_intrinsic

        block_sync_vote = self.format_cuda_block_sync_vote_call(raw_name, args)
        if block_sync_vote is not None:
            return block_sync_vote

        resource_call = self.format_cuda_resource_call(raw_name, args, node.args)
        if resource_call is not None:
            return resource_call

        func_name = self.convert_cuda_builtin_function(raw_name)
        if (
            func_name == raw_name
            and isinstance(raw_name, str)
            and raw_name.startswith("::")
            and not self.is_simple_identifier(func_name)
        ):
            func_name = self.convert_cuda_function_name_to_crossgl(raw_name)
        return f"{func_name}({args_str})"

    def format_device_heap_call(self, function_name, args):
        """Refuse ``malloc``/``free`` in device code, like ``delete`` (#4684).

        CrossGL has no device heap, so a device-side heap call has no faithful
        form. Host code keeps the ordinary call; a user-defined ``malloc`` or
        ``free`` is not a heap call either.
        """
        if not self.in_device_code or not isinstance(function_name, str):
            return None
        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name not in {
            "malloc",
            "free",
            "std::malloc",
            "std::free",
        }:
            return None
        if self.is_user_defined_function(
            function_name
        ) or self.is_user_defined_function(normalized_name):
            return None
        return self.refuse_import(
            f"CUDA {normalized_name}({', '.join(args)}): CrossGL has no device heap",
            "0",
        )

    def format_cuda_time_function_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name not in {"clock", "clock64", "cuda::std::clock"}:
            return None

        display_name = normalized_name.rsplit("::", 1)[-1]
        if args:
            return self.format_cuda_time_function_diagnostic(display_name, args)
        # clock64() reads the multiprocessor's 64-bit cycle counter, PTX %clock64, as a long long; CrossGL's
        # clockARB() is that counter (CUDA output writes it as clock64()). clock() reads %clock, the low 32 bits of
        # %clock64, as an int and returns it widened to clock_t, a long (CUDA C++ Programming Guide, "Time
        # Function"; PTX ISA, "Special Registers: %clock, %clock_hi, %clock64").
        if display_name == "clock64":
            return "i64(clockARB())"
        return "i64(i32(clock2x32ARB().x))"

    def format_cuda_time_function_diagnostic(self, function_name, args):
        return self.refuse_import(
            f"CUDA time function {function_name}({', '.join(args)}) takes no arguments",
            "0",
        )

    def format_cuda_comma_expression(self, args):
        # CrossGL has the C comma operator: ``(a, b)`` evaluates ``a``, then yields ``b``.
        return f"({', '.join(self.visit(arg) for arg in args)})"

    def format_cuda_nanosleep_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = (
            function_name[2:] if function_name.startswith("::") else function_name
        )
        if normalized_name != "__nanosleep":
            return None

        # The sleep is a scheduling hint whose duration lies in [0, 2 * ns] (CUDA C++ Programming Guide,
        # "Nanosleep Function"), so a sleep of zero is one of its executions; the argument is still evaluated.
        if len(args) == 1:
            return f"({args[0]})"
        return self.refuse_import(
            f"CUDA __nanosleep({', '.join(args)}) takes one argument", "0"
        )

    def format_cuda_load_cache_intrinsic_call(
        self, function_name, raw_args, formatted_args
    ):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__ldca",
            "__ldcg",
            "__ldcs",
            "__ldcv",
            "__ldg",
            "__ldlu",
        }:
            return None

        if (
            len(raw_args) == 1
            and isinstance(raw_args[0], UnaryOpNode)
            and raw_args[0].op == "&"
        ):
            return self.visit(raw_args[0].operand)

        if len(raw_args) == 1:
            return f"(*{formatted_args[0]})"

        return self.refuse_import(
            f"CUDA load cache intrinsic {function_name}({', '.join(formatted_args)}): CrossGL loads take one "
            "address",
            "0",
        )

    def format_cuda_memory_fence_call(self, function_name, args):
        if not isinstance(function_name, str):
            return None

        normalized_name = self.resolve_namespace_alias_name(function_name)
        if normalized_name.startswith("::"):
            normalized_name = normalized_name[2:]

        if normalized_name != "cuda::atomic_thread_fence" or len(args) != 2:
            return None

        return "memoryBarrier()"

    def is_full_or_active_warp_mask(self, mask):
        normalized = self.normalize_warp_mask_expression(
            self.resolve_warp_mask_expression(mask)
        )
        return normalized in {
            "0xffffffff",
            "0xffffffffu",
            "0xfffffffful",
            "0xffffffffffffffff",
            "0xffffffffffffffffu",
            "0xffffffffffffffffull",
            "uint_max",
            "ullong_max",
            "warp_full_mask",
            "full_mask",
            "waveactiveballot(true).x",
        }

    def is_full_warp_width(self, width):
        normalized = self.normalize_warp_mask_expression(width)
        return normalized in {"32", "32u", "warpsize", "warp_size", "warp_full_width"}

    def normalize_warp_mask_expression(self, mask):
        text = str(mask).strip()
        while text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
            if not inner:
                break
            text = inner
        return text.replace(" ", "").lower()

    def format_wave_predicate(self, predicate):
        return f"({predicate} != 0)"

    def format_unsupported_cuda_warp_intrinsic(self, function_name, args):
        # A warp intrinsic with no wave-intrinsic equivalent stays a CUDA call in CrossGL: the CUDA target emits
        # it natively, and crosstl.translator.semantics.warp emulates or refuses it for every other target. It was
        # replaced by the constant 0.
        return f"{function_name}({', '.join(args)})"

    def format_cuda_block_sync_vote_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name not in {
            "__syncthreads_count",
            "__syncthreads_and",
            "__syncthreads_or",
        }:
            return None

        if len(args) != 1:
            return self.refuse_import(
                f"CUDA thread block sync vote {function_name}({', '.join(args)}) takes one predicate",
                "0",
            )
        return self.block_votes.call(function_name, self.format_wave_predicate(args[0]))

    def format_cuda_fp16_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if (
            function_name in {
                "__float2half",
                "__float2half_rn",
                "__float2half_rd",
                "__float2half_ru",
                "__float2half_rz",
            }
            and len(args) == 1
        ):
            return f"f16({args[0]})"
        if function_name == "__float2half2_rn" and len(args) == 1:
            return self.format_vector_constructor("vec2", [args[0], args[0]], "f16")
        if function_name == "__float22half2_rn" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[0], "y"),
                ],
                "f16",
            )
        if function_name == "__floats2half2_rn" and len(args) == 2:
            return self.format_vector_constructor("vec2", args, "f16")
        if function_name == "__half22float2" and len(args) == 1:
            return self.format_vector_constructor(
                "vec2",
                [
                    self.format_vector_component_access(args[0], "x"),
                    self.format_vector_component_access(args[0], "y"),
                ],
                "f32",
            )
        if function_name == "__half2float" and len(args) == 1:
            return f"f32({args[0]})"
        if function_name == "__low2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'x')})"
        if function_name == "__high2float" and len(args) == 1:
            return f"f32({self.format_vector_component_access(args[0], 'y')})"
        if function_name == "__hadd2" and len(args) == 2:
            return f"({args[0]} + {args[1]})"
        if function_name == "__hmul2" and len(args) == 2:
            return f"({args[0]} * {args[1]})"
        if function_name == "__hfma2" and len(args) == 3:
            return f"fma({args[0]}, {args[1]}, {args[2]})"
        return None

    def format_cuda_type_cast_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        bit_reinterpret_intrinsics = {
            "__double_as_longlong": "doubleBitsToLong",
            "__float_as_int": "floatBitsToInt",
            "__float_as_uint": "floatBitsToUint",
            "__int_as_float": "intBitsToFloat",
            "__longlong_as_double": "longBitsToDouble",
            "__uint_as_float": "uintBitsToFloat",
        }
        mapped_name = bit_reinterpret_intrinsics.get(function_name)
        if mapped_name is not None and len(args) == 1:
            return f"{mapped_name}({args[0]})"

        integer_conversion = self.format_cuda_float_to_integer_conversion_intrinsic(
            function_name, args
        )
        if integer_conversion is not None:
            return integer_conversion

        return None

    def format_cuda_float_to_integer_conversion_intrinsic(self, function_name, args):
        if len(args) != 1:
            return None

        conversion_intrinsics = {
            "__float2int_rd": ("i32", "floor"),
            "__float2int_rn": ("i32", "roundEven"),
            "__float2int_ru": ("i32", "ceil"),
            "__float2int_rz": ("i32", None),
            "__float2uint_rd": ("u32", "floor"),
            "__float2uint_rn": ("u32", "roundEven"),
            "__float2uint_ru": ("u32", "ceil"),
            "__float2uint_rz": ("u32", None),
            "__float2ll_rd": ("i64", "floor"),
            "__float2ll_rn": ("i64", "roundEven"),
            "__float2ll_ru": ("i64", "ceil"),
            "__float2ll_rz": ("i64", None),
            "__float2ull_rd": ("u64", "floor"),
            "__float2ull_rn": ("u64", "roundEven"),
            "__float2ull_ru": ("u64", "ceil"),
            "__float2ull_rz": ("u64", None),
            "__double2int_rd": ("i32", "floor"),
            "__double2int_rn": ("i32", "roundEven"),
            "__double2int_ru": ("i32", "ceil"),
            "__double2int_rz": ("i32", None),
            "__double2uint_rd": ("u32", "floor"),
            "__double2uint_rn": ("u32", "roundEven"),
            "__double2uint_ru": ("u32", "ceil"),
            "__double2uint_rz": ("u32", None),
            "__double2ll_rd": ("i64", "floor"),
            "__double2ll_rn": ("i64", "roundEven"),
            "__double2ll_ru": ("i64", "ceil"),
            "__double2ll_rz": ("i64", None),
            "__double2ull_rd": ("u64", "floor"),
            "__double2ull_rn": ("u64", "roundEven"),
            "__double2ull_ru": ("u64", "ceil"),
            "__double2ull_rz": ("u64", None),
        }
        conversion = conversion_intrinsics.get(function_name)
        if conversion is None:
            return None

        target_type, rounding_function = conversion
        value = args[0]
        if rounding_function is None:
            return f"{target_type}({value})"
        return f"{target_type}({rounding_function}({value}))"

    def format_cuda_integer_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        if function_name == "__mul24" and len(args) == 2:
            left = self.format_cuda_signed_24_bit_operand(args[0])
            right = self.format_cuda_signed_24_bit_operand(args[1])
            return f"({left} * {right})"
        if function_name == "__umul24" and len(args) == 2:
            return f"(({args[0]} & 0x00ffffffu) * ({args[1]} & 0x00ffffffu))"
        if function_name == "__dp4a" and len(args) == 3:
            return f"(dot4I8Packed({args[0]}, {args[1]}) + {args[2]})"
        if function_name in {"__ffs", "__ffsll"} and len(args) == 1:
            return f"(findLSB({args[0]}) + 1)"
        if function_name in {"__brev", "__brevll"} and len(args) == 1:
            return f"bitfieldReverse({args[0]})"
        if function_name in {"__hadd", "__uhadd"} and len(args) == 2:
            return self.format_integer_average_floor(args[0], args[1])
        if function_name in {"__rhadd", "__urhadd"} and len(args) == 2:
            return self.format_integer_average_rounded(args[0], args[1])
        if function_name == "__sad" and len(args) == 3:
            return f"(abs({args[0]} - {args[1]}) + {args[2]})"
        if function_name == "__usad" and len(args) == 3:
            return (
                f"((({args[0]} > {args[1]}) ? ({args[0]} - {args[1]}) : "
                f"({args[1]} - {args[0]})) + {args[2]})"
            )
        if function_name == "__usad4" and len(args) in {2, 3}:
            bias = args[2] if len(args) == 3 else "0u"
            return self.format_cuda_usad4_intrinsic(args[0], args[1], bias)
        return None

    def format_cuda_usad4_intrinsic(self, left, right, bias):
        differences = [
            self.format_cuda_unsigned_byte_absdiff(left, right, shift)
            for shift in (0, 8, 16, 24)
        ]
        result = differences[0]
        for difference in differences[1:]:
            result = f"({result} + {difference})"

        if self.normalize_cuda_integer_literal(bias) == "0":
            return result
        return f"({result} + {bias})"

    def format_cuda_unsigned_byte_absdiff(self, left, right, shift):
        left_byte = self.format_cuda_packed_byte(left, shift)
        right_byte = self.format_cuda_packed_byte(right, shift)
        return (
            f"(({left_byte} > {right_byte}) ? ({left_byte} - {right_byte}) : "
            f"({right_byte} - {left_byte}))"
        )

    def format_cuda_packed_byte(self, value, shift):
        operand = self.format_bitwise_operand(value)
        if shift == 0:
            return f"({operand} & 0xffu)"
        return f"(({operand} >> {shift}) & 0xffu)"

    def normalize_cuda_integer_literal(self, value):
        text = str(value).strip().replace("'", "").lower()
        while text.endswith(("u", "l")):
            text = text[:-1]
        return text

    def format_bitwise_operand(self, value):
        text = str(value).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return text
        return f"({text})"

    def format_integer_average_floor(self, left, right):
        return f"(({left} & {right}) + (({left} ^ {right}) >> 1))"

    def format_integer_average_rounded(self, left, right):
        return f"(({left} | {right}) - (({left} ^ {right}) >> 1))"

    def format_cuda_float_intrinsic_call(self, function_name, args):
        if isinstance(function_name, str) and function_name.startswith("::"):
            function_name = function_name[2:]

        fast_math_intrinsics = {
            "__cosf": ("cos", 1),
            "__exp10f": ("pow", 1),
            "__frsqrt_rn": ("inversesqrt", 1),
            "__fsqrt_rd": ("sqrt", 1),
            "__fsqrt_rn": ("sqrt", 1),
            "__fsqrt_ru": ("sqrt", 1),
            "__fsqrt_rz": ("sqrt", 1),
            "__expf": ("exp", 1),
            "__log2f": ("log2", 1),
            "__logf": ("log", 1),
            "__powf": ("pow", 2),
            "__sinf": ("sin", 1),
            "__tanf": ("tan", 1),
            "__tanhf": ("tanh", 1),
        }
        mapped_intrinsic = fast_math_intrinsics.get(function_name)
        if mapped_intrinsic is not None:
            mapped_name, arity = mapped_intrinsic
            if len(args) == arity:
                if function_name == "__exp10f":
                    return f"{mapped_name}(10.0f, {args[0]})"
                return f"{mapped_name}({', '.join(args)})"

        if function_name == "__saturatef" and len(args) == 1:
            return f"clamp({args[0]}, 0.0f, 1.0f)"
        # fdivide and fdividef return a / b unless fast math is on (__clang_cuda_math.h); __fdividef is the fast form.
        if (
            function_name in {
                "fdivide",
                "fdividef",
                "__fdividef",
                "__fdiv_rd",
                "__fdiv_rn",
                "__fdiv_ru",
                "__fdiv_rz",
            }
            and len(args) == 2
        ):
            return f"({args[0]} / {args[1]})"
        if (
            function_name in {"__fadd_rd", "__fadd_rn", "__fadd_ru", "__fadd_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} + {args[1]})"
        if (
            function_name in {"__fsub_rd", "__fsub_rn", "__fsub_ru", "__fsub_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} - {args[1]})"
        if (
            function_name in {"__fmul_rd", "__fmul_rn", "__fmul_ru", "__fmul_rz"}
            and len(args) == 2
        ):
            return f"({args[0]} * {args[1]})"
        if (
            function_name in {"__frcp_rd", "__frcp_rn", "__frcp_ru", "__frcp_rz"}
            and len(args) == 1
        ):
            return f"(1.0f / {args[0]})"
        if (
            function_name in {
                "__fmaf_rd",
                "__fmaf_rn",
                "__fmaf_ru",
                "__fmaf_rz",
                "__fmaf_ieee_rd",
                "__fmaf_ieee_rn",
                "__fmaf_ieee_ru",
                "__fmaf_ieee_rz",
            }
            and len(args) == 3
        ):
            return f"fma({args[0]}, {args[1]}, {args[2]})"

        return None

    def format_cuda_signed_24_bit_operand(self, arg):
        return f"(({arg} << 8) >> 8)"

    def format_vector_component_access(self, expression, component):
        text = str(expression).strip()
        if text and all(char.isalnum() or char in "_." for char in text):
            return f"{text}.{component}"
        return f"({text}).{component}"
