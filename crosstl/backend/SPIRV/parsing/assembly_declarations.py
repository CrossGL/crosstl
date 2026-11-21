"""SPIR-V assembly module declarations in the SPIR-V parser: capabilities, debug names, annotations, entry
points and execution modes, type declarations, constants and module variables."""


class VulkanAssemblyDeclarationMixin:
    """Module declaration members of ``VulkanParser``: what ``parse_spirv_assembly_module`` records for each
    instruction before it builds the shader."""

    def spirv_assembly_record_annotation(
        self,
        opcode,
        operands,
        names,
        decorations,
        member_decorations,
        member_names,
        capabilities,
        entry_points,
        execution_modes,
        group_decorates,
        group_member_decorates,
    ):
        """Record a capability, debug name, decoration, decoration group, entry point or execution mode, and return
        whether the instruction was one."""
        if opcode == "OpCapability" and operands:
            capabilities.append(operands[0])
        elif opcode == "OpName" and len(operands) >= 2:
            names[operands[0]] = (
                self.spirv_identifier_name(operands[1], operands[0])
                if operands[1]
                else ""
            )
        elif opcode == "OpMemberName" and len(operands) >= 3:
            target, member = operands[0], operands[1]
            name = self.spirv_identifier_name(
                operands[2],
                f"{target}_{member}",
                prefix="member",
            )
            member_names.setdefault(target, {})[member] = name
        elif opcode == "OpDecorate" and len(operands) >= 2:
            target, decoration = operands[0], operands[1]
            decorations.setdefault(target, []).append((decoration, operands[2:]))
        elif opcode == "OpMemberDecorate" and len(operands) >= 3:
            target, member, decoration = operands[0], operands[1], operands[2]
            member_decorations.setdefault(target, []).append(
                (member, decoration, operands[3:])
            )
        elif opcode == "OpGroupDecorate" and len(operands) >= 2:
            group_decorates.append((operands[0], operands[1:]))
        elif opcode == "OpGroupMemberDecorate" and len(operands) >= 3:
            group_member_decorates.append((operands[0], operands[1:]))
        elif opcode == "OpEntryPoint" and len(operands) >= 3:
            entry_points.append(
                {
                    "execution_model": operands[0],
                    "id": operands[1],
                    "name": self.spirv_identifier_name(
                        operands[2], operands[1], prefix="entry_point"
                    ),
                    "interface_ids": operands[3:],
                }
            )
        elif opcode in {"OpExecutionMode", "OpExecutionModeId"} and len(operands) >= 2:
            execution_modes.setdefault(operands[0], []).append(
                {
                    "opcode": opcode,
                    "mode": operands[1],
                    "operands": operands[2:],
                }
            )
        else:
            return False
        return True

    def spirv_assembly_record_type(self, result_id, opcode, operands, types, constants):
        """Record a type declaration, and return whether the instruction was one."""
        if result_id and opcode == "OpTypeVoid":
            types[result_id] = {"kind": "scalar", "name": "void"}
        elif result_id and opcode == "OpTypeBool":
            types[result_id] = {"kind": "scalar", "name": "bool"}
        elif result_id and opcode == "OpTypeFloat" and operands:
            types[result_id] = {
                "kind": "scalar",
                "name": self.spirv_float_type_name(operands[0]),
            }
        elif result_id and opcode == "OpTypeInt" and len(operands) >= 2:
            types[result_id] = {
                "kind": "scalar",
                "name": self.spirv_int_type_name(operands[0], operands[1]),
            }
        elif result_id and opcode == "OpTypeVector" and len(operands) >= 2:
            component_type = self.spirv_type_name(operands[0], types)
            types[result_id] = {
                "kind": "vector",
                "name": self.spirv_vector_type_name(component_type, operands[1]),
                "component_type": operands[0],
                "component_count": operands[1],
            }
        elif result_id and opcode == "OpTypeMatrix" and len(operands) >= 2:
            column_type = types.get(operands[0], {})
            component_type = self.spirv_type_name(
                column_type.get("component_type"), types
            )
            types[result_id] = {
                "kind": "matrix",
                "name": self.spirv_matrix_type_name(
                    component_type,
                    column_type.get("component_count"),
                    operands[1],
                ),
                "column_type": operands[0],
                "column_count": operands[1],
            }
        elif (
            result_id
            and opcode in {
                "OpTypeCooperativeMatrixKHR",
                "OpTypeCooperativeMatrixNV",
            }
            and len(operands) >= 4
        ):
            types[result_id] = {
                "kind": "cooperative_matrix",
                "name": self.spirv_cooperative_matrix_type_name(
                    opcode, operands, types, constants
                ),
                "component_type": operands[0],
                "scope": operands[1],
                "rows": operands[2],
                "columns": operands[3],
                "use": operands[4] if len(operands) >= 5 else None,
            }
        elif result_id and opcode == "OpTypeArray" and len(operands) >= 2:
            types[result_id] = {
                "kind": "array",
                "element_type": operands[0],
                "length_id": operands[1],
            }
        elif result_id and opcode == "OpTypeRuntimeArray" and len(operands) >= 1:
            types[result_id] = {
                "kind": "runtime_array",
                "element_type": operands[0],
            }
        elif result_id and opcode == "OpTypeStruct":
            types[result_id] = {"kind": "struct", "member_types": operands}
        elif result_id and opcode == "OpTypeImage" and len(operands) >= 7:
            sampled_type = self.spirv_type_name(operands[0], types)
            types[result_id] = {
                "kind": "image",
                "name": self.spirv_image_type_name(
                    sampled_type,
                    operands[1],
                    operands[2],
                    operands[3],
                    operands[4],
                    operands[5],
                ),
                "sampled_type": operands[0],
                "dim": operands[1],
                "depth": operands[2],
                "arrayed": operands[3],
                "multisampled": operands[4],
                "sampled": operands[5],
                "format": operands[6],
                "access_qualifier": operands[7] if len(operands) >= 8 else None,
            }
        elif result_id and opcode == "OpTypeSampledImage" and operands:
            image_type = types.get(operands[0], {})
            types[result_id] = {
                "kind": "sampled_image",
                "name": self.spirv_sampled_image_type_name(image_type, types),
                "image_type": operands[0],
            }
        elif result_id and opcode == "OpTypeSampler":
            types[result_id] = {"kind": "sampler", "name": "sampler"}
        elif result_id and opcode == "OpTypeNamedBarrier":
            types[result_id] = {
                "kind": "named_barrier",
                "name": "spirvNamedBarrier",
            }
        elif result_id and opcode in {
            "OpTypeAccelerationStructureKHR",
            "OpTypeAccelerationStructureNV",
        }:
            types[result_id] = {
                "kind": "acceleration_structure",
                "name": "accelerationStructureEXT",
            }
        elif result_id and opcode in {"OpTypeRayQueryKHR", "OpTypeRayQueryNV"}:
            types[result_id] = {
                "kind": "ray_query",
                "name": "rayQueryEXT",
            }
        elif result_id and opcode == "OpTypePointer" and len(operands) >= 2:
            types[result_id] = {
                "kind": "pointer",
                "storage_class": operands[0],
                "type_id": operands[1],
            }
        elif result_id and opcode == "OpTypeFunction" and operands:
            types[result_id] = {
                "kind": "function",
                "return_type": operands[0],
                "parameter_types": operands[1:],
            }
        else:
            return False
        return True

    def spirv_assembly_record_value(
        self,
        result_id,
        opcode,
        operands,
        names,
        types,
        constants,
        constant_types,
        spec_constant_ids,
        extended_instruction_imports,
        variables,
    ):
        """Record a constant, string, undefined value, extended-instruction import or module variable."""
        if result_id and opcode in {"OpConstant", "OpSpecConstant"}:
            if len(operands) >= 2:
                constant_types[result_id] = operands[0]
                constants[result_id] = self.spirv_constant_literal_expression(
                    operands[0], operands[1], types
                )
                if opcode == "OpSpecConstant":
                    spec_constant_ids.append(result_id)
        elif result_id and opcode == "OpString" and operands:
            constants[result_id] = self.spirv_string_literal(operands[0])
        elif result_id and opcode in {
            "OpConstantFalse",
            "OpConstantTrue",
            "OpSpecConstantFalse",
            "OpSpecConstantTrue",
        }:
            if operands:
                constant_types[result_id] = operands[0]
                constants[result_id] = "true" if opcode.endswith("True") else "false"
                if opcode.startswith("OpSpecConstant"):
                    spec_constant_ids.append(result_id)
        elif result_id and opcode == "OpUndef" and operands:
            constant_types[result_id] = operands[0]
            constants[result_id] = self.spirv_assembly_undef_expression(
                operands[0], types
            )
        elif result_id and opcode == "OpConstantNull" and operands:
            constant_types[result_id] = operands[0]
            constants[result_id] = self.spirv_assembly_null_expression(
                operands[0], types
            )
        elif result_id and opcode in {
            "OpConstantComposite",
            "OpSpecConstantComposite",
        }:
            if operands:
                constant_types[result_id] = operands[0]
                constants[result_id] = self.spirv_constant_composite_expression(
                    operands[0], operands[1:], names, types, constants
                )
                if opcode == "OpSpecConstantComposite":
                    spec_constant_ids.append(result_id)
        elif result_id and opcode == "OpSpecConstantOp":
            if len(operands) >= 2:
                constant_types[result_id] = operands[0]
                constants[result_id] = self.spirv_spec_constant_op_expression(
                    operands[1], operands[2:], names, constants, types
                )
                spec_constant_ids.append(result_id)
        elif result_id and opcode == "OpExtInstImport" and operands:
            extended_instruction_imports[result_id] = operands[0]
        elif result_id and opcode == "OpVariable" and len(operands) >= 2:
            variables.append(
                {
                    "id": result_id,
                    "pointer_type_id": operands[0],
                    "storage_class": operands[1],
                    "initializer": operands[2] if len(operands) >= 3 else None,
                }
            )
