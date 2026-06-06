from unittest import mock

from crosstl.formatter import (
    CodeFormatter,
    ShaderLanguage,
    format_file,
    format_shader_code,
)


class TestShaderLanguage:
    def test_enum_values(self):
        assert ShaderLanguage.HLSL.value == "hlsl"
        assert ShaderLanguage.GLSL.value == "glsl"
        assert ShaderLanguage.METAL.value == "metal"
        assert ShaderLanguage.SPIRV.value == "spirv"
        assert ShaderLanguage.UNKNOWN.value == "unknown"


class TestCodeFormatter:
    def test_init_with_no_tools(self):
        with mock.patch("shutil.which", return_value=None):
            formatter = CodeFormatter()
            assert formatter.has_clang is False
            assert formatter.has_spirv_tools is False

    def test_init_with_tools(self):
        with mock.patch("shutil.which", return_value="/usr/bin/fake-tool"):
            formatter = CodeFormatter()
            assert formatter.has_clang is True
            assert formatter.has_spirv_tools is True

    def test_detect_language(self):
        formatter = CodeFormatter()

        assert formatter.detect_language("shader.hlsl") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.hlsli") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.fx") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.fxh") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.glsl") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.vert") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.vsh") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.vertex") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.frag") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.fsh") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.fragment") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.csh") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.compute") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.gsh") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.geometry") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.mesh") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.task") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rgen") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rint") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rahit") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rchit") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rmiss") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.rcall") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.metal") == ShaderLanguage.METAL
        assert formatter.detect_language("shader.msl") == ShaderLanguage.METAL
        assert formatter.detect_language("shader.spvasm") == ShaderLanguage.SPIRV
        assert formatter.detect_language("shader.vulkan") == ShaderLanguage.SPIRV
        assert formatter.detect_language("shader.slangh") == ShaderLanguage.SLANG
        assert formatter.detect_language("shader.cuh") == ShaderLanguage.CUDA
        assert formatter.detect_language("shader.cuda") == ShaderLanguage.CUDA
        assert formatter.detect_language("shader.rust") == ShaderLanguage.RUST
        assert formatter.detect_language("shader.txt") == ShaderLanguage.UNKNOWN

    def test_format_code_language_detection(self):
        formatter = CodeFormatter()

        formatter._format_with_clang = mock.MagicMock(return_value="clang_formatted")
        formatter._format_spirv = mock.MagicMock(return_value="spirv_formatted")

        assert (
            formatter.format_code("code", file_path="shader.hlsl") == "clang_formatted"
        )
        assert (
            formatter.format_code("code", file_path="shader.glsl") == "clang_formatted"
        )
        assert (
            formatter.format_code("code", file_path="shader.metal") == "clang_formatted"
        )
        assert (
            formatter.format_code("code", file_path="shader.spvasm")
            == "spirv_formatted"
        )

        assert (
            formatter.format_code("code", language=ShaderLanguage.HLSL)
            == "clang_formatted"
        )
        assert (
            formatter.format_code("code", language=ShaderLanguage.SPIRV)
            == "spirv_formatted"
        )

        assert formatter.format_code("code", language="hlsl") == "clang_formatted"
        assert formatter.format_code("code", language="spirv") == "spirv_formatted"

    def test_format_with_clang(self):
        formatter = CodeFormatter()
        formatter.has_clang = False

        assert formatter._format_with_clang("code", ShaderLanguage.HLSL) == "code"

        formatter.has_clang = True
        formatter.clang_format_path = "clang-format"

        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile, mock.patch(
            "subprocess.run"
        ) as mock_run, mock.patch(
            "builtins.open", mock.mock_open(read_data="formatted_code")
        ), mock.patch(
            "os.unlink"
        ):

            mock_tempfile.return_value.__enter__.return_value.name = "temp.hlsl"
            mock_run.return_value.returncode = 0

            result = formatter._format_with_clang("code", ShaderLanguage.HLSL)
            assert result == "formatted_code"

            mock_run.return_value.returncode = 1
            result = formatter._format_with_clang("code", ShaderLanguage.HLSL)
            assert result == "code"

    def test_format_spirv(self):
        formatter = CodeFormatter()
        formatter.has_spirv_tools = False

        assert formatter._format_spirv("code") == "code"

        formatter._make_spirv_readable = mock.MagicMock(return_value="readable_code")

        formatter.has_spirv_tools = True
        formatter.spirv_as_path = "spirv-as"
        formatter.spirv_dis_path = "spirv-dis"

        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile, mock.patch(
            "subprocess.run"
        ) as mock_run, mock.patch("os.unlink"):

            mock_tempfile.return_value.__enter__.return_value.name = "temp.spvasm"

            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "formatted_spirv"

            result = formatter._format_spirv("code")
            assert result == "formatted_spirv"

            mock_run.side_effect = [
                mock.MagicMock(returncode=1),  # spirv-as fails
                mock.MagicMock(returncode=0),  # spirv-dis succeeds
            ]

            result = formatter._format_spirv("code")
            assert result == "readable_code"

    def test_make_spirv_readable(self):
        formatter = CodeFormatter()

        spirv_code = """OpCapability Shader
OpFunction
OpLabel
OpStore
OpFunctionEnd
"""

        expected = """OpCapability Shader
OpFunction
  OpLabel
    OpStore
OpFunctionEnd"""

        formatted = formatter._make_spirv_readable(spirv_code).strip()
        formatted_lines = [line.rstrip() for line in formatted.split("\n")]
        expected_lines = [line.rstrip() for line in expected.split("\n")]

        assert formatted_lines == expected_lines

    def test_validate_spirv(self):
        formatter = CodeFormatter()
        formatter.has_spirv_tools = False

        valid, msg = formatter.validate_spirv("code")
        assert valid is False
        assert "not available" in msg

        formatter.has_spirv_tools = True
        formatter.spirv_as_path = "spirv-as"
        formatter.spirv_val_path = "spirv-val"

        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile, mock.patch(
            "subprocess.run"
        ) as mock_run, mock.patch("os.unlink"):

            mock_tempfile.return_value.__enter__.return_value.name = "temp.spvasm"

            mock_run.return_value.returncode = 0

            valid, msg = formatter.validate_spirv("code")
            assert valid is True
            assert "Valid" in msg

            mock_run.side_effect = [
                mock.MagicMock(returncode=1, stderr="Assembly error"),  # spirv-as fails
                mock.MagicMock(returncode=0),  # spirv-val succeeds
            ]

            valid, msg = formatter.validate_spirv("code")
            assert valid is False
            assert "Assembly failed" in msg

            mock_run.side_effect = [
                mock.MagicMock(returncode=0),  # spirv-as succeeds
                mock.MagicMock(
                    returncode=1, stderr="Validation error"
                ),  # spirv-val fails
            ]

            valid, msg = formatter.validate_spirv("code")
            assert valid is False
            assert "Validation failed" in msg


def test_format_file():
    with mock.patch("builtins.open", mock.mock_open(read_data="code")), mock.patch(
        "crosstl.formatter.CodeFormatter.format_code", return_value="formatted_code"
    ):

        assert format_file("shader.hlsl") is True

    with mock.patch("builtins.open", side_effect=Exception("File error")):
        assert format_file("shader.hlsl") is False


def test_format_shader_code():
    with mock.patch("crosstl.formatter.CodeFormatter") as MockFormatter:
        mock_instance = MockFormatter.return_value
        mock_instance.format_code.return_value = "formatted_code"

        assert format_shader_code("code", "metal") == "formatted_code"
        assert format_shader_code("code", "msl") == "formatted_code"
        assert format_shader_code("code", "directx") == "formatted_code"
        assert format_shader_code("code", "dx") == "formatted_code"
        assert format_shader_code("code", "opengl") == "formatted_code"
        assert format_shader_code("code", "ogl") == "formatted_code"
        assert format_shader_code("code", "vulkan") == "formatted_code"
        assert format_shader_code("code", "rs") == "formatted_code"
        assert format_shader_code("code", "cu") == "formatted_code"
        assert format_shader_code("code", "slangh") == "formatted_code"

        format_shader_code("code", "metal", "output.metal")
        mock_instance.format_code.assert_called_with(
            "code", ShaderLanguage.METAL, "output.metal"
        )

        format_shader_code("code", "directx", "output.hlsl")
        mock_instance.format_code.assert_called_with(
            "code", ShaderLanguage.HLSL, "output.hlsl"
        )

        format_shader_code("code", "opengl", "output.glsl")
        mock_instance.format_code.assert_called_with(
            "code", ShaderLanguage.GLSL, "output.glsl"
        )

        format_shader_code("code", "vulkan", "output.spvasm")
        mock_instance.format_code.assert_called_with(
            "code", ShaderLanguage.SPIRV, "output.spvasm"
        )
