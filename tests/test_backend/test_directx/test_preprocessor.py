import unittest
from unittest.mock import patch, mock_open
from crosstl.backend.DirectX.DirectxPreprocessor import DirectxPreprocessor


class TestDirectxPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up common test variables."""
        self.preprocessor = DirectxPreprocessor()

    def test_include_directive(self):
        """Test proper handling of #include directives."""
        shader_code = (
            '#include "common.hlsl"\nfloat4 main() : SV_POSITION { return 0; }'
        )
        include_content = "float4 commonFunc() { return float4(1.0); }"

        with patch("builtins.open", mock_open(read_data=include_content)):
            result = self.preprocessor.preprocess(shader_code)
            self.assertIn("float4 commonFunc()", result)
            self.assertIn("float4 main()", result)

    def test_define_macro(self):
        """Test macro definition and substitution."""
        shader_code = "#define PI 3.14\nfloat piValue = PI;"
        result = self.preprocessor.preprocess(shader_code)
        self.assertIn("float piValue = 3.14;", result)

    def test_conditional_compilation(self):
        """Test conditional compilation blocks."""
        shader_code = """
        #define FEATURE 1
        #ifdef FEATURE
        float featureValue = 1.0;
        #else
        float featureValue = 0.0;
        #endif
        """
        result = self.preprocessor.preprocess(shader_code)
        self.assertIn("float featureValue = 1.0;", result)
        self.assertNotIn("float featureValue = 0.0;", result)

    def test_undefined_macro_error(self):
        """Test error handling for undefined macros."""
        shader_code = "float value = UNDEFINED_MACRO;"
        with self.assertRaises(ValueError):
            self.preprocessor.preprocess(shader_code)

    def test_nested_includes(self):
        """Test handling of nested include directives."""
        main_shader = '#include "file1.hlsl"'
        file1_content = '#include "file2.hlsl"\nfloat file1Value = 1.0;'
        file2_content = "float file2Value = 2.0;"

        with patch(
            "builtins.open",
            side_effect=[
                mock_open(read_data=file1_content).return_value,
                mock_open(read_data=file2_content).return_value,
            ],
        ):
            result = self.preprocessor.preprocess(main_shader)
            self.assertIn("float file1Value = 1.0;", result)
            self.assertIn("float file2Value = 2.0;", result)

    def test_missing_file_error(self):
        """Test handling of missing include files."""
        shader_code = '#include "nonexistent.hlsl"'
        with self.assertRaises(FileNotFoundError):
            self.preprocessor.preprocess(shader_code)

    def test_recursive_includes(self):
        """Test detection of recursive include loops."""
        main_shader = '#include "file1.hlsl"'
        file1_content = '#include "file2.hlsl"'
        file2_content = '#include "file1.hlsl"'

        with patch(
            "builtins.open",
            side_effect=[
                mock_open(read_data=file1_content).return_value,
                mock_open(read_data=file2_content).return_value,
            ],
        ):
            with self.assertRaises(RecursionError):
                self.preprocessor.preprocess(main_shader)

    def test_macro_expansion(self):
        """Test macro expansion within shader code."""
        shader_code = """
        #define MAX 10
        int value = MAX + 5;
        """
        result = self.preprocessor.preprocess(shader_code)
        self.assertIn("int value = 10 + 5;", result)

    def test_condition_stack(self):
        """Test the condition stack during #ifdef/#endif."""
        shader_code = """
        #ifdef FEATURE
        float featureValue = 1.0;
        #else
        float featureValue = 0.0;
        #endif
        """
        self.preprocessor.handle_define(
            "#define FEATURE 1"
        )  # Ensure the macro is defined
        result = self.preprocessor.preprocess(shader_code)
        self.assertIn("float featureValue = 1.0;", result)
        self.assertNotIn("float featureValue = 0.0;", result)

    def test_empty_shader(self):
        """Test an empty shader code."""
        shader_code = ""
        result = self.preprocessor.preprocess(shader_code)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
