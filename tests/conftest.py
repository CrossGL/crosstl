"""
Pytest configuration for the CrossGL Translator tests
"""

import sys
import os
import pytest
import subprocess
import importlib.util
import importlib

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Map of lowercase to proper capitalization for backend modules
backend_modules = {
    "opengl": "OpenGL",
    "directx": "DirectX",
    "slang": "Slang",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "mojo": "Mojo",
}


# Import a module using a custom finder
def import_module_with_path(module_name, module_path):
    """Import a module with the specified name from the given path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Run setup script to create proper test directories
def pytest_sessionstart(session):
    """Run setup script to prepare test directories."""
    setup_script = os.path.join(os.path.dirname(__file__), "setup_tests.py")
    if os.path.exists(setup_script):
        try:
            subprocess.run([sys.executable, setup_script], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to set up test directories")

    # Set up aliases for lowercase directory imports
    test_dir = os.path.dirname(__file__)
    test_backend_dir = os.path.join(test_dir, "test_backend")
    
    for lowercase, proper in backend_modules.items():
        # For each backend module type (OpenGL, Slang, etc.)
        uppercase_dir = os.path.join(test_backend_dir, f"test_{proper}")
        lowercase_dir = os.path.join(test_backend_dir, f"test_{lowercase}")
        
        # If both directories exist, create bidirectional aliases
        if os.path.exists(uppercase_dir) and os.path.exists(lowercase_dir):
            # Create the module aliases
            lowercase_module = f"tests.test_backend.test_{lowercase}"
            proper_module = f"tests.test_backend.test_{proper}"
            
            # Try to import the proper module first
            try:
                proper_mod = importlib.import_module(proper_module)
                if proper_mod and lowercase_module not in sys.modules:
                    sys.modules[lowercase_module] = proper_mod
            except ImportError:
                pass
                
            # If we still don't have the lowercase module, try direct import
            if lowercase_module not in sys.modules:
                try:
                    lowercase_mod = importlib.import_module(lowercase_module)
                    if lowercase_mod and proper_module not in sys.modules:
                        sys.modules[proper_module] = lowercase_mod
                except ImportError:
                    pass
            
            # Set up aliases for test files
            for item in os.listdir(uppercase_dir):
                if item.endswith('.py') and item.startswith('test_'):
                    # Get module name without extension
                    modname = item[:-3]
                    
                    # Set up aliases for the test modules
                    upper_mod_name = f"{proper_module}.{modname}"
                    lower_mod_name = f"{lowercase_module}.{modname}"
                    
                    # Create bidirectional aliases
                    if upper_mod_name in sys.modules and lower_mod_name not in sys.modules:
                        sys.modules[lower_mod_name] = sys.modules[upper_mod_name]
                    elif lower_mod_name in sys.modules and upper_mod_name not in sys.modules:
                        sys.modules[upper_mod_name] = sys.modules[lower_mod_name]
