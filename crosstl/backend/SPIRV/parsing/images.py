"""Image sampling, sparse residency, gather, fetch, read, write, texel pointers, sized memory copies and image operand selection in the SPIR-V parser."""

from ..ray_tracing_import import SpirvInstructionUnsupportedError
from ..sampled_images import sampled_image_arguments
from ..VulkanAst import (
    FunctionCallNode,
    MemberAccessNode,
)


class VulkanImageMixin:
    """Image members of ``VulkanParser``."""

    def spirv_sample_coordinate_components(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        expression_type_ids,
        constant_types,
        types,
    ):
        """The coordinate components a sample consumes, when its coordinate operand has more; else ``None``.

        A SPIR-V sample coordinate holds ``(u[, v][, w][, array layer])`` as the image needs, then ``q`` for a
        projective sample; further components are ignored, and glslang passes, for instance, the whole ``vec3`` of
        ``texture(sampler2DShadow, P)`` with its reference. CrossGL coordinates hold exactly the consumed
        components.
        """
        sampled_type = types.get(expression_type_ids.get(image_operand), {})
        image_type = types.get(sampled_type.get("image_type"), {})
        axes = {"1D": 1, "2D": 2, "Rect": 2, "3D": 3, "Cube": 3}.get(
            image_type.get("dim")
        )
        if sampled_type.get("kind") != "sampled_image" or axes is None:
            return None
        needed = axes + (1 if str(image_type.get("arrayed")) == "1" else 0)
        if "Proj" in opcode:
            needed += 1
        coordinate_type = expression_type_ids.get(coordinate_operand)
        if coordinate_type is None:
            coordinate_type = constant_types.get(coordinate_operand)
        available = self.spirv_vector_component_count(coordinate_type, types)
        if available is None or available <= needed:
            return None
        return needed

    def spirv_assembly_image_sample_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
        coordinate_components=None,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        if coordinate_components is not None:
            coordinate = MemberAccessNode(coordinate, "xyzw"[:coordinate_components])
        dref = None
        if "Dref" in opcode and image_operands:
            dref = self.spirv_assembly_operand_expression(
                image_operands[0], expressions, names, decorations, constants
            )
            image_operands = image_operands[1:]
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        base_args = [*sampled_image_arguments(image), coordinate]
        if dref is not None:
            base_args.append(dref)
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        bias = self.spirv_assembly_image_bias_operand(parsed_operands)
        min_lod = self.spirv_assembly_image_min_lod_operand(parsed_operands)
        is_projected = "Proj" in opcode

        if min_lod is not None:
            return self.spirv_assembly_image_min_lod_sample_expression(
                opcode,
                base_args,
                parsed_operands,
                offset,
                bias,
                min_lod,
            )

        if "Grad" in parsed_operands and len(parsed_operands["Grad"]) >= 2:
            if dref is not None and is_projected and offset is not None:
                function_name = "textureCompareProjGradOffset"
            elif dref is not None and is_projected:
                function_name = "textureCompareProjGrad"
            elif dref is not None and offset is not None:
                function_name = "textureCompareGradOffset"
            elif dref is not None:
                function_name = "textureCompareGrad"
            elif is_projected and offset is not None:
                function_name = "textureProjGradOffset"
            elif is_projected:
                function_name = "textureProjGrad"
            elif offset is not None:
                function_name = "textureGradOffset"
            else:
                function_name = "textureGrad"
            args = [*base_args, *parsed_operands["Grad"][:2]]
            if offset is not None:
                args.append(offset)
            return FunctionCallNode(
                function_name,
                args,
            )

        if "Lod" in parsed_operands and parsed_operands["Lod"]:
            if dref is not None and is_projected and offset is not None:
                function_name = "textureCompareProjLodOffset"
            elif dref is not None and is_projected:
                function_name = "textureCompareProjLod"
            elif dref is not None and offset is not None:
                function_name = "textureCompareLodOffset"
            elif dref is not None:
                function_name = "textureCompareLod"
            elif is_projected and offset is not None:
                function_name = "textureProjLodOffset"
            elif is_projected:
                function_name = "textureProjLod"
            elif offset is not None:
                function_name = "textureLodOffset"
            else:
                function_name = "textureLod"
            args = [*base_args, parsed_operands["Lod"][0]]
            if offset is not None:
                args.append(offset)
            return FunctionCallNode(
                function_name,
                args,
            )

        if offset is not None:
            args = [*base_args, offset]
            if bias is not None:
                args.append(bias)
            if dref is not None and bias is None and is_projected:
                function_name = "textureCompareProjOffset"
            elif dref is not None and bias is None:
                function_name = "textureCompareOffset"
            else:
                function_name = "textureProjOffset" if is_projected else "textureOffset"
            return FunctionCallNode(function_name, args)

        if dref is not None and bias is None and is_projected:
            function_name = "textureCompareProj"
        elif dref is not None and bias is None:
            function_name = "textureCompare"
        else:
            function_name = "textureProj" if is_projected else "texture"
        args = list(base_args)
        if bias is not None:
            args.append(bias)
        return FunctionCallNode(function_name, args)

    def spirv_assembly_image_min_lod_sample_expression(
        self,
        opcode,
        base_args,
        parsed_operands,
        offset,
        bias,
        min_lod,
    ):
        """CrossGL ``texture*Clamp`` for a sample with the ``MinLod`` image operand (SPIR-V 3.14).

        ``base_args`` are the sampled image, the coordinate and, for a ``Dref`` sample, the reference.
        """
        is_compare = "Dref" in opcode
        gradient = parsed_operands.get("Grad") or []
        if "Proj" in opcode:
            raise SpirvInstructionUnsupportedError(
                opcode,
                "with a MinLod operand has no CrossGL form: CrossGL has no projective minimum-LOD clamped sample",
            )
        if parsed_operands.get("Lod") or (gradient and len(gradient) < 2):
            raise SpirvInstructionUnsupportedError(
                opcode,
                "combines MinLod with an explicit Lod, which SPIR-V allows only on implicit-LOD or Grad sampling",
            )
        if is_compare and bias is not None:
            raise SpirvInstructionUnsupportedError(
                opcode,
                "with Bias and MinLod has no CrossGL form: CrossGL depth compares take no bias",
            )
        name = "textureCompare" if is_compare else "texture"
        args = list(base_args)
        if gradient:
            name += "Grad"
            args.extend(gradient[:2])
        if offset is not None:
            name += "Offset"
            args.append(offset)
        args.append(min_lod)
        if bias is not None:
            args.append(bias)
        return FunctionCallNode(f"{name}Clamp", args)

    def spirv_assembly_image_sparse_sample_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
        coordinate_components=None,
    ):
        sample = self.spirv_assembly_image_sample_expression(
            opcode,
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
            coordinate_components=coordinate_components,
        )
        function_name = self.spirv_sparse_image_sample_function_name(
            opcode, sample.name
        )
        return FunctionCallNode(function_name, sample.args)

    def spirv_sparse_image_sample_function_name(self, opcode, sample_function_name):
        if sample_function_name.startswith("spirvTexture"):
            suffix = sample_function_name[len("spirvTexture") :]
        elif "Dref" in opcode and sample_function_name.startswith("textureCompare"):
            suffix = sample_function_name[len("textureCompare") :]
        elif sample_function_name.startswith("texture"):
            suffix = sample_function_name[len("texture") :]
        else:
            suffix = self.spirv_fallback_identifier(sample_function_name, "sample")

        dref = "Dref" if "Dref" in opcode else ""
        return f"spirvImageSparseSample{dref}{suffix}"

    def spirv_assembly_image_sparse_read_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        read = self.spirv_assembly_image_read_expression(
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_read_function_name(read.name)
        return FunctionCallNode(function_name, read.args)

    def spirv_sparse_image_read_function_name(self, read_function_name):
        if read_function_name == "imageLoad":
            return "spirvImageSparseRead"
        if read_function_name.startswith("spirvImageLoad"):
            suffix = read_function_name[len("spirvImageLoad") :]
            return f"spirvImageSparseRead{suffix}"
        suffix = self.spirv_fallback_identifier(read_function_name, "read")
        return f"spirvImageSparseRead{suffix}"

    def spirv_assembly_image_sparse_fetch_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        fetch = self.spirv_assembly_image_fetch_expression(
            image_operand,
            coordinate_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_fetch_function_name(fetch.name)
        return FunctionCallNode(function_name, fetch.args)

    def spirv_sparse_image_fetch_function_name(self, fetch_function_name):
        if fetch_function_name == "texelFetch":
            return "spirvImageSparseFetch"
        if fetch_function_name.startswith("texelFetch"):
            suffix = fetch_function_name[len("texelFetch") :]
            return f"spirvImageSparseFetch{suffix}"
        suffix = self.spirv_fallback_identifier(fetch_function_name, "fetch")
        return f"spirvImageSparseFetch{suffix}"

    def spirv_assembly_image_sparse_gather_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        gather_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        gather = self.spirv_assembly_image_gather_expression(
            {
                "OpImageSparseGather": "OpImageGather",
                "OpImageSparseDrefGather": "OpImageDrefGather",
            }[opcode],
            image_operand,
            coordinate_operand,
            gather_operand,
            image_operands,
            expressions,
            names,
            decorations,
            constants,
        )
        function_name = self.spirv_sparse_image_gather_function_name(
            opcode,
            gather.name,
        )
        return FunctionCallNode(function_name, gather.args)

    def spirv_sparse_image_gather_function_name(self, opcode, gather_function_name):
        if "Dref" in opcode and gather_function_name.startswith("textureGatherCompare"):
            suffix = gather_function_name[len("textureGatherCompare") :]
            return f"spirvImageSparseDrefGather{suffix}"
        if gather_function_name.startswith("textureGather"):
            suffix = gather_function_name[len("textureGather") :]
            return f"spirvImageSparseGather{suffix}"
        suffix = self.spirv_fallback_identifier(gather_function_name, "gather")
        return f"spirvImageSparseGather{suffix}"

    def spirv_assembly_image_gather_expression(
        self,
        opcode,
        image_operand,
        coordinate_operand,
        gather_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        gathered = self.spirv_assembly_operand_expression(
            gather_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        offsets = self.spirv_assembly_image_const_offsets_operand(parsed_operands)

        if opcode == "OpImageDrefGather":
            base_args = [*sampled_image_arguments(image), coordinate, gathered]
            if offsets is not None:
                return FunctionCallNode(
                    "textureGatherCompareOffsets", [*base_args, offsets]
                )
            if offset is not None:
                return FunctionCallNode(
                    "textureGatherCompareOffset", [*base_args, offset]
                )
            return FunctionCallNode("textureGatherCompare", base_args)

        base_args = [*sampled_image_arguments(image), coordinate]
        if offsets is not None:
            return FunctionCallNode(
                "textureGatherOffsets", [*base_args, offsets, gathered]
            )
        if offset is not None:
            return FunctionCallNode(
                "textureGatherOffset", [*base_args, offset, gathered]
            )
        return FunctionCallNode("textureGather", [*base_args, gathered])

    def spirv_assembly_image_fetch_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        if "Lod" in parsed_operands and parsed_operands["Lod"]:
            args.append(parsed_operands["Lod"][0])
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        if sample is not None:
            args.append(sample)
        offset = self.spirv_assembly_image_offset_operand(parsed_operands)
        if offset is not None:
            args.append(offset)
            return FunctionCallNode("texelFetchOffset", args)
        return FunctionCallNode("texelFetch", args)

    def spirv_assembly_image_read_expression(
        self,
        image_operand,
        coordinate_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        lod = self.spirv_assembly_image_lod_operand(parsed_operands)
        if lod is not None:
            args.append(lod)
            if sample is not None:
                args.append(sample)
            return FunctionCallNode("spirvImageLoadLod", args)
        if sample is not None:
            args.append(sample)
        return FunctionCallNode("imageLoad", args)

    def spirv_assembly_image_texel_pointer_expression(
        self,
        image_operand,
        coordinate_operand,
        sample_operand,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "spirvImageTexelPointer",
            [
                self.spirv_assembly_operand_expression(
                    image_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    coordinate_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
                self.spirv_assembly_operand_expression(
                    sample_operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                ),
            ],
        )

    def spirv_assembly_image_write_statement(
        self,
        image_operand,
        coordinate_operand,
        texel_operand,
        image_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        image = self.spirv_assembly_operand_expression(
            image_operand, expressions, names, decorations, constants
        )
        coordinate = self.spirv_assembly_operand_expression(
            coordinate_operand, expressions, names, decorations, constants
        )
        texel = self.spirv_assembly_operand_expression(
            texel_operand, expressions, names, decorations, constants
        )
        parsed_operands = self.spirv_assembly_image_operands(
            image_operands, expressions, names, decorations, constants
        )
        args = [image, coordinate]
        sample = self.spirv_assembly_image_sample_operand(parsed_operands)
        lod = self.spirv_assembly_image_lod_operand(parsed_operands)
        if lod is not None:
            args.append(lod)
            if sample is not None:
                args.append(sample)
            args.append(texel)
            return FunctionCallNode("spirvImageStoreLod", args)
        if sample is not None:
            args.append(sample)
        args.append(texel)
        return FunctionCallNode("imageStore", args)

    def spirv_assembly_copy_memory_sized_statement(
        self,
        target_operand,
        source_operand,
        size_operand,
        memory_operands,
        expressions,
        names,
        decorations,
        constants,
    ):
        return FunctionCallNode(
            "spirvCopyMemorySized",
            [
                self.spirv_assembly_operand_expression(
                    operand,
                    expressions,
                    names,
                    decorations,
                    constants,
                )
                for operand in (
                    target_operand,
                    source_operand,
                    size_operand,
                    *memory_operands,
                )
            ],
        )

    def spirv_assembly_image_sample_operand(self, parsed_operands):
        values = parsed_operands.get("Sample")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_lod_operand(self, parsed_operands):
        values = parsed_operands.get("Lod")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_offset_operand(self, parsed_operands):
        for operand_name in ("ConstOffset", "Offset"):
            values = parsed_operands.get(operand_name)
            if values:
                return values[0]
        return None

    def spirv_assembly_image_bias_operand(self, parsed_operands):
        values = parsed_operands.get("Bias")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_min_lod_operand(self, parsed_operands):
        values = parsed_operands.get("MinLod")
        if values:
            return values[0]
        return None

    def spirv_assembly_image_const_offsets_operand(self, parsed_operands):
        values = parsed_operands.get("ConstOffsets")
        if values:
            return values[0]
        return None
