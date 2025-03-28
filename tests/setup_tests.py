#!/usr/bin/env python
"""
Setup script for CrossGL Translator tests.
Creates proper __init__.py files in test directories.
"""

import os
import sys
import shutil
import importlib
import subprocess


def setup_test_directories():
    """Set up test directories with proper __init__.py files"""
    test_dir = os.path.dirname(__file__)
    test_backend_dir = os.path.join(test_dir, "test_backend")

    # Make sure test_backend has __init__.py
    os.makedirs(test_backend_dir, exist_ok=True)
    backend_init_file = os.path.join(test_backend_dir, "__init__.py")
    
    # Write a basic __init__.py to the test directory
    with open(os.path.join(test_dir, "__init__.py"), "w") as f:
        f.write('"""Test module for CrossGL Translator."""\n')
    
    # Define test directories with proper capitalization and their lowercase aliases
    backend_dirs = {
        "test_OpenGL": "test_opengl",
        "test_DirectX": "test_directx",
        "test_Slang": "test_slang",
        "test_Metal": "test_metal",
        "test_Vulkan": "test_vulkan",
        "test_Mojo": "test_mojo",
    }

    # Create or update the backend __init__.py file with the import aliases
    with open(backend_init_file, "w") as f:
        f.write('"""Test backend module."""\n\n')
        f.write('# Register all test directories with proper capitalization\n')
        f.write('import sys\n\n')
        f.write('# Map of lowercase to proper capitalization\n')
        f.write('backend_modules = {\n')
        for uppercase_dir, lowercase_dir in backend_dirs.items():
            lowercase = lowercase_dir[5:]  # Remove "test_" prefix
            proper = uppercase_dir[5:]  # Remove "test_" prefix
            f.write(f'    "{lowercase}": "{proper}",\n')
        f.write('}\n\n')
        
        f.write('# Create package aliasing for case insensitivity\n')
        f.write('for lowercase, proper in backend_modules.items():\n')
        f.write('    # Create aliases for test_backend.test_lowercase -> test_backend.test_Proper\n')
        f.write('    module_name = f"tests.test_backend.test_{lowercase}"\n')
        f.write('    proper_name = f"tests.test_backend.test_{proper}"\n\n')
        f.write('    # Add the module to sys.modules if not already there\n')
        f.write('    if module_name not in sys.modules and proper_name in sys.modules:\n')
        f.write('        sys.modules[module_name] = sys.modules[proper_name]\n\n')
        f.write('    # Also check if we need to add the proper name\n')
        f.write('    if proper_name not in sys.modules and module_name in sys.modules:\n')
        f.write('        sys.modules[proper_name] = sys.modules[module_name]\n')

    # Update all uppercase directories with __init__.py files
    for uppercase_dir, lowercase_dir in backend_dirs.items():
        uppercase_path = os.path.join(test_backend_dir, uppercase_dir)
        # Skip if the directory doesn't exist
        if not os.path.exists(uppercase_path):
            continue
            
        # Create __init__.py in the uppercase directory if it doesn't exist
        init_file = os.path.join(uppercase_path, "__init__.py")
        if not os.path.exists(init_file):
            # Remove "test_" prefix and get backend name
            backend_name = uppercase_dir[5:]
            with open(init_file, "w") as f:
                f.write(f'"""Test module for {backend_name}."""\n')
        
        # Update the __init__.py in the lowercase directory as well
        lowercase_path = os.path.join(test_backend_dir, lowercase_dir)
        if os.path.exists(lowercase_path) and os.path.isdir(lowercase_path):
            lowercase_init = os.path.join(lowercase_path, "__init__.py")
            with open(lowercase_init, "w") as f:
                backend_name = lowercase_dir[5:]
                f.write(f'"""Test module for {backend_name} (lowercase alias)."""\n')
    
    # Create lowercase aliases for all backend tests
    create_aliases_script = os.path.join(test_dir, "create_lowercase_aliases.py")
    if os.path.exists(create_aliases_script):
        try:
            subprocess.run([sys.executable, create_aliases_script], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to create lowercase aliases")


if __name__ == "__main__":
    setup_test_directories()
    print("Test directories have been set up.")
