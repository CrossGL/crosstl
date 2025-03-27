"""Test backend module."""

# This is just a marker file to make the directory a Python package

# Register all test directories with proper capitalization
import sys

# Map of lowercase to proper capitalization
backend_modules = {
    "opengl": "OpenGL",
    "directx": "DirectX",
    "slang": "Slang",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "mojo": "Mojo",
}

# Create package aliasing for case insensitivity
for lowercase, proper in backend_modules.items():
    # Create aliases for test_backend.test_lowercase -> test_backend.test_Proper
    module_name = f"tests.test_backend.test_{lowercase}"
    proper_name = f"tests.test_backend.test_{proper}"

    # Add the module to sys.modules if not already there
    if module_name not in sys.modules and proper_name in sys.modules:
        sys.modules[module_name] = sys.modules[proper_name]

    # Also check if we need to add the proper name
    if proper_name not in sys.modules and module_name in sys.modules:
        sys.modules[proper_name] = sys.modules[module_name]
