"""Base types of ``Parser.parse_type`` that need more than one token: opaque sampler, texture and image types, and named types with their qualified path and generic arguments."""

from ..ast import (
    NamedType,
)

# Sampler, texture and image type keywords and the named type each one parses to.
SAMPLER_IMAGE_TYPE_NAMES = {
    "SAMPLER": "sampler",
    "SAMPLER1D": "sampler1D",
    "SAMPLER1DARRAY": "sampler1DArray",
    "SAMPLER2D": "sampler2D",
    "SAMPLER3D": "sampler3D",
    "SAMPLERCUBE": "samplerCube",
    "SAMPLER2DARRAY": "sampler2DArray",
    "SAMPLER2DSHADOW": "sampler2DShadow",
    "SAMPLER2DARRAYSHADOW": "sampler2DArrayShadow",
    "SAMPLERCUBESHADOW": "samplerCubeShadow",
    "SAMPLERCUBEARRAY": "samplerCubeArray",
    "SAMPLERCUBEARRAYSHADOW": "samplerCubeArrayShadow",
    "SAMPLER2DMS": "sampler2DMS",
    "SAMPLER2DMSARRAY": "sampler2DMSArray",
    "ISAMPLER1D": "isampler1D",
    "ISAMPLER1DARRAY": "isampler1DArray",
    "ISAMPLER2D": "isampler2D",
    "ISAMPLER3D": "isampler3D",
    "ISAMPLERCUBE": "isamplerCube",
    "ISAMPLERCUBEARRAY": "isamplerCubeArray",
    "ISAMPLER2DARRAY": "isampler2DArray",
    "ISAMPLER2DMS": "isampler2DMS",
    "ISAMPLER2DMSARRAY": "isampler2DMSArray",
    "USAMPLER1D": "usampler1D",
    "USAMPLER1DARRAY": "usampler1DArray",
    "USAMPLER2D": "usampler2D",
    "USAMPLER3D": "usampler3D",
    "USAMPLERCUBE": "usamplerCube",
    "USAMPLERCUBEARRAY": "usamplerCubeArray",
    "USAMPLER2DARRAY": "usampler2DArray",
    "USAMPLER2DMS": "usampler2DMS",
    "USAMPLER2DMSARRAY": "usampler2DMSArray",
    "IIMAGE1D": "iimage1D",
    "IIMAGE1DARRAY": "iimage1DArray",
    "IIMAGE2D": "iimage2D",
    "IIMAGE3D": "iimage3D",
    "IIMAGECUBE": "iimageCube",
    "IIMAGECUBEARRAY": "iimageCubeArray",
    "IIMAGE2DARRAY": "iimage2DArray",
    "IIMAGE2DMS": "iimage2DMS",
    "IIMAGE2DMSARRAY": "iimage2DMSArray",
    "UIMAGE1D": "uimage1D",
    "UIMAGE1DARRAY": "uimage1DArray",
    "UIMAGE2D": "uimage2D",
    "UIMAGE3D": "uimage3D",
    "UIMAGECUBE": "uimageCube",
    "UIMAGECUBEARRAY": "uimageCubeArray",
    "UIMAGE2DARRAY": "uimage2DArray",
    "UIMAGE2DMS": "uimage2DMS",
    "UIMAGE2DMSARRAY": "uimage2DMSArray",
    "IMAGE1D": "image1D",
    "IMAGE1DARRAY": "image1DArray",
    "IMAGE2D": "image2D",
    "IMAGE3D": "image3D",
    "IMAGECUBE": "imageCube",
    "IMAGECUBEARRAY": "imageCubeArray",
    "IMAGE2DARRAY": "image2DArray",
    "IMAGE2DMS": "image2DMS",
    "IMAGE2DMSARRAY": "image2DMSArray",
}


class ParserNamedTypeMixin:
    """Base-type members of ``Parser.parse_type``."""

    def parse_opaque_resource_type(
        self,
    ):
        """Parse a sampler, texture or image type keyword into its named type."""
        token_type = self.current_token[0]
        self.eat(token_type)
        base_type = NamedType(SAMPLER_IMAGE_TYPE_NAMES[token_type])
        return base_type

    def parse_named_base_type(
        self,
    ):
        """Parse a named type: its qualified path, generic arguments and the cooperative-matrix form."""
        name = self.current_token[1]
        self.eat("IDENTIFIER")
        while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
            separator = self.current_token[1]
            self.eat(self.current_token[0])
            segment = self.parse_binding_identifier()
            name = f"{name}{separator}{segment}"

        generic_args = []
        if self.current_token[0] == "LESS_THAN":
            generic_args = self.parse_generic_arguments()
        elif (
            self.current_token[0] == "LBRACKET"
            and self.square_brackets_form_type_arguments()
        ):
            generic_args = self.parse_square_generic_arguments()

        while self.current_token[0] == "LPAREN":
            name = f"{name}({self.parse_balanced_parenthesized_token_text()})"

        while self.current_token[0] in {"DOT", "DOUBLE_COLON"}:
            separator = self.current_token[1]
            self.eat(self.current_token[0])
            segment = self.parse_binding_identifier()

            segment_generic_args = []
            if self.current_token[0] == "LESS_THAN":
                segment_generic_args = self.parse_generic_arguments()
            elif self.current_token[0] == "LBRACKET" and (
                self.square_brackets_form_type_arguments()
                or (generic_args and self.square_brackets_form_member_type_arguments())
            ):
                segment_generic_args = self.parse_square_generic_arguments()

            if generic_args:
                name = f"{name}_{segment}"
                generic_args.extend(segment_generic_args)
            else:
                name = f"{name}{separator}{segment}"
                generic_args = segment_generic_args

        if name.rsplit("::", 1)[-1] == "CooperativeMatrix":
            base_type = self.cooperative_matrix_type(generic_args)
        else:
            base_type = NamedType(name, generic_args)
        return base_type
