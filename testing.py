import unittest
import os
import crosstl

class TestCodeGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("test", exist_ok=True)

    def test_cgl_to_glsl_translation(self):
        result = crosstl.translate("examples/PerlinNoise.cgl", backend='opengl')
        self.assertTrue(result, "GLSL translation failed or returned empty result")
        print("################### GLSL ###################")
        print(result)

    def test_cgl_to_metal_translation(self):
        result = crosstl.translate("examples/PerlinNoise.cgl", backend='metal')
        self.assertTrue(result, "Metal translation failed or returned empty result")
        print("################### Metal ###################")
        print(result)
    def test_cgl_to_hlsl_translation(self):
        result = crosstl.translate("examples/PerlinNoise.cgl", backend='directx')
        self.assertTrue(result, "HLSL translation failed or returned empty result")
        print("################### HLSL ###################")
        print(result)

    def test_glsl_to_crossgl_translation(self):
        result = crosstl.translate("examples/PerlinNoise.glsl", backend='cgl')
        self.assertTrue(result, "GLSL to CrossGL translation failed or returned empty result")
        print("################### GLSL TO CrossGL ###################")
        print(result)
    def test_hlsl_to_crossgl_translation(self):
        result = crosstl.translate("examples/PerlinNoise.hlsl", backend='cgl')
        self.assertTrue(result, "HLSL to CrossGL translation failed or returned empty result")
        print("################### HLSL TO CrossGL ###################")
        print(result)

    def test_metal_to_crossgl_translation(self):
        result = crosstl.translate("examples/PerlinNoise.metal", backend='cgl')
        self.assertTrue(result, "Metal to CrossGL translation failed or returned empty result")
        print("################### Metal TO CrossGL ###################")
        print(result)

    def test_translation_errors(self):
        with self.assertRaises(Exception):
            crosstl.translate("non_existent_file.cgl", backend='opengl')
        with self.assertRaises(Exception):
            crosstl.translate("examples/PerlinNoise.cgl", backend='invalid_backend')

if __name__ == "__main__":
    unittest.main()