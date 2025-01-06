import unittest
from DirectxPreprocessor import DirectxPreprocessor


class TestDirectxPreprocessor(unittest.TestCase):

    def test_preprocessor_with_defines_and_ifdef(self):
        shader_code = """
        #define PI 3.14159
        #define DEBUG 1

        #ifdef DEBUG
            float debugValue = 1.0;
        #else
            float debugValue = 0.0;
        #endif

        float computeCircleArea(float radius) {
            return PI * radius * radius;
        }
        """

        expected_output = """
            float debugValue = 1.0;

            float computeCircleArea(float radius) {
                return 3.14159 * radius * radius;
            }
        """

        preprocessor = DirectxPreprocessor()
        result = preprocessor.preprocess(shader_code)

        # Clean up the result by stripping leading/trailing whitespace for easier comparison
        result = result.strip()
        expected_output = expected_output.strip()

        self.assertEqual(result, expected_output)

    def test_preprocessor_with_no_debug(self):
        shader_code = """
        #define PI 3.14159
        #undef DEBUG

        #ifdef DEBUG
            float debugValue = 1.0;
        #else
            float debugValue = 0.0;
        #endif

        float computeCircleArea(float radius) {
            return PI * radius * radius;
        }
        """

        expected_output = """
            float debugValue = 0.0;

            float computeCircleArea(float radius) {
                return 3.14159 * radius * radius;
            }
        """

        preprocessor = DirectxPreprocessor()
        result = preprocessor.preprocess(shader_code)

        result = result.strip()
        expected_output = expected_output.strip()

        self.assertEqual(result, expected_output)

    def test_preprocessor_without_include(self):
        shader_code = """
        #define PI 3.14159

        float computeCircleArea(float radius) {
            return PI * radius * radius;
        }
        """

        expected_output = """
            float computeCircleArea(float radius) {
                return 3.14159 * radius * radius;
            }
        """

        preprocessor = DirectxPreprocessor()
        result = preprocessor.preprocess(shader_code)

        result = result.strip()
        expected_output = expected_output.strip()

        self.assertEqual(result, expected_output)

    def test_preprocessor_with_invalid_include(self):
        shader_code = """
        #include "common.h"
        #define PI 3.14159

        float computeCircleArea(float radius) {
            return PI * radius * radius;
        }
        """

        preprocessor = DirectxPreprocessor()

        with self.assertRaises(FileNotFoundError):
            preprocessor.preprocess(shader_code)
        result = self.preprocessor.preprocess(shader_code)
        self.assertEqual(result.strip(), expected_result.strip())


if __name__ == "__main__":
    unittest.main()
