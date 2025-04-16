import os
import sys
import pytest

# Add test_utils to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# For array testing
@pytest.fixture
def array_test_data():
    """Provide test data for array handling tests."""
    return {
        "hlsl": {
            "array_type_declarations": [
                "float values[4]", 
                "float weights[8]",
            ],
            "array_access": [
                "weights[2]",
                "material.values[0]",
                "material.colors[index]", 
                "particles[3].position"
            ],
        },
        "metal": {
            "array_type_declarations": [
                "float values[4]", 
                "float weights[8]",
            ],
            "array_access": [
                "weights[2]",
                "material.values[0]",
                "material.colors[index]", 
                "particles[3].position"
            ],
        },
        "glsl": {
            "array_type_declarations": [
                "float values[4]", 
                "float weights[8]",
            ],
            "array_access": [
                "weights[2]",
                "material.values[0]",
                "material.colors[index]", 
                "particles[3].position"
            ],
        }
    } 