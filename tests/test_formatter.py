from unittest import mock
import pytest

from crosstl.formatter import (
    ShaderLanguage,
    CodeFormatter,
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
        # Test initialization when tools are not available
        with mock.patch("shutil.which", return_value=None):
            formatter = CodeFormatter()
            assert formatter.has_clang is False
            assert formatter.has_spirv_tools is False

    def test_init_with_tools(self):
        # Test initialization when tools are available
        with mock.patch("shutil.which", return_value="/usr/bin/fake-tool"):
            formatter = CodeFormatter()
            assert formatter.has_clang is True
            assert formatter.has_spirv_tools is True

    def test_detect_language(self):
        formatter = CodeFormatter()

        # Test different file extensions
        assert formatter.detect_language("shader.hlsl") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.fx") == ShaderLanguage.HLSL
        assert formatter.detect_language("shader.glsl") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.vert") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.frag") == ShaderLanguage.GLSL
        assert formatter.detect_language("shader.metal") == ShaderLanguage.METAL
        assert formatter.detect_language("shader.spirv") == ShaderLanguage.SPIRV
        assert formatter.detect_language("shader.vulkan") == ShaderLanguage.SPIRV
        assert formatter.detect_language("shader.txt") == ShaderLanguage.UNKNOWN

    def test_format_code_language_detection(self):
        formatter = CodeFormatter()

        # Mock format methods
        formatter._format_with_clang = mock.MagicMock(return_value="clang_formatted")
        formatter._format_spirv = mock.MagicMock(return_value="spirv_formatted")

        # Test language detection from file path
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
            formatter.format_code("code", file_path="shader.spirv") == "spirv_formatted"
        )

        # Test with explicit language
        assert (
            formatter.format_code("code", language=ShaderLanguage.HLSL)
            == "clang_formatted"
        )
        assert (
            formatter.format_code("code", language=ShaderLanguage.SPIRV)
            == "spirv_formatted"
        )

        # Test with string language
        assert formatter.format_code("code", language="hlsl") == "clang_formatted"
        assert formatter.format_code("code", language="spirv") == "spirv_formatted"

    def test_format_with_clang(self):
        formatter = CodeFormatter()
        formatter.has_clang = False

        # Test when clang-format is not available
        assert formatter._format_with_clang("code", ShaderLanguage.HLSL) == "code"

        # Mock subprocess.run for successful formatting
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

            # Test with failed formatting
            mock_run.return_value.returncode = 1
            result = formatter._format_with_clang("code", ShaderLanguage.HLSL)
            assert result == "code"

    def test_format_spirv(self):
        formatter = CodeFormatter()
        formatter.has_spirv_tools = False

        # Test when SPIRV-Tools are not available
        assert formatter._format_spirv("code") == "code"

        # Mock _make_spirv_readable for failed formatting
        formatter._make_spirv_readable = mock.MagicMock(return_value="readable_code")

        # Mock subprocess.run for successful formatting
        formatter.has_spirv_tools = True
        formatter.spirv_as_path = "spirv-as"
        formatter.spirv_dis_path = "spirv-dis"

        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile, mock.patch(
            "subprocess.run"
        ) as mock_run, mock.patch("os.unlink"):

            mock_tempfile.return_value.__enter__.return_value.name = "temp.spvasm"

            # Test successful formatting
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "formatted_spirv"

            result = formatter._format_spirv("code")
            assert result == "formatted_spirv"

            # Test failed assembly
            mock_run.side_effect = [
                mock.MagicMock(returncode=1),  # spirv-as fails
                mock.MagicMock(returncode=0),  # spirv-dis succeeds
            ]

            result = formatter._format_spirv("code")
            assert result == "readable_code"

    def test_make_spirv_readable(self):
        formatter = CodeFormatter()

        # Test indentation
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

        # Clean up whitespace before comparing
        formatted = formatter._make_spirv_readable(spirv_code).strip()
        formatted_lines = [line.rstrip() for line in formatted.split("\n")]
        expected_lines = [line.rstrip() for line in expected.split("\n")]

        assert formatted_lines == expected_lines

    def test_validate_spirv(self):
        formatter = CodeFormatter()
        formatter.has_spirv_tools = False

        # Test when SPIRV-Tools are not available
        valid, msg = formatter.validate_spirv("code")
        assert valid is False
        assert "not available" in msg

        # Mock subprocess.run for successful validation
        formatter.has_spirv_tools = True
        formatter.spirv_as_path = "spirv-as"
        formatter.spirv_val_path = "spirv-val"

        with mock.patch("tempfile.NamedTemporaryFile") as mock_tempfile, mock.patch(
            "subprocess.run"
        ) as mock_run, mock.patch("os.unlink"):

            mock_tempfile.return_value.__enter__.return_value.name = "temp.spvasm"

            # Test successful validation
            mock_run.return_value.returncode = 0

            valid, msg = formatter.validate_spirv("code")
            assert valid is True
            assert "Valid" in msg

            # Test failed assembly
            mock_run.side_effect = [
                mock.MagicMock(returncode=1, stderr="Assembly error"),  # spirv-as fails
                mock.MagicMock(returncode=0),  # spirv-val succeeds
            ]

            valid, msg = formatter.validate_spirv("code")
            assert valid is False
            assert "Assembly failed" in msg

            # Test failed validation
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
    # Test successful formatting
    with mock.patch("builtins.open", mock.mock_open(read_data="code")), mock.patch(
        "crosstl.formatter.CodeFormatter.format_code", return_value="formatted_code"
    ):

        assert format_file("shader.hlsl") is True

    # Test failed formatting
    with mock.patch("builtins.open", side_effect=Exception("File error")):
        assert format_file("shader.hlsl") is False


def test_format_shader_code():
    # Test backend mapping with direct calls to format_shader_code
    with mock.patch("crosstl.formatter.CodeFormatter") as MockFormatter:
        # Set up the mock to return formatted_code
        mock_instance = MockFormatter.return_value
        mock_instance.format_code.return_value = "formatted_code"

        # Test the function calls
        assert format_shader_code("code", "metal") == "formatted_code"
        assert format_shader_code("code", "directx") == "formatted_code"
        assert format_shader_code("code", "opengl") == "formatted_code"
        assert format_shader_code("code", "vulkan") == "formatted_code"

        # Check the language was mapped correctly
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

        format_shader_code("code", "vulkan", "output.spirv")
        mock_instance.format_code.assert_called_with(
            "code", ShaderLanguage.SPIRV, "output.spirv"
        )
