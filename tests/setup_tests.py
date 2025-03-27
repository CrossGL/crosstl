#!/usr/bin/env python
"""
Setup script for cross-platform testing.
Creates proper __init__.py files in test directories.
"""

import os


def setup_test_directories():
    """Set up test directories with proper __init__.py files"""
    test_backend_dir = os.path.join(os.path.dirname(__file__), "test_backend")

    # Make sure test_backend has __init__.py
    init_file = os.path.join(test_backend_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write('"""Test backend module."""\n')

    # Define mappings of capitalized directory names
    backend_dirs = [
        "test_OpenGL",
        "test_DirectX",
        "test_Slang",
        "test_Metal",
        "test_Vulkan",
        "test_Mojo",
    ]

    # Create or update __init__.py in each test directory
    for dir_name in backend_dirs:
        dir_path = os.path.join(test_backend_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f'"""Test module for {dir_name}."""\n')

        # Fix imports in test files
        fix_imports_in_directory(dir_path)


def fix_imports_in_directory(directory):
    """Fix imports in all Python files in the directory."""
    backend_modules = {
        "opengl": "OpenGL",
        "directx": "DirectX",
        "slang": "Slang",
        "metal": "Metal",
        "vulkan": "Vulkan",
        "mojo": "Mojo",
    }

    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename.startswith("test_"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for lowercase imports
                modified = False
                for lower, proper in backend_modules.items():
                    pattern = f"crosstl.backend.{lower}"
                    replacement = f"crosstl.backend.{proper}"
                    if pattern in content:
                        content = content.replace(pattern, replacement)
                        modified = True

                # Save changes if needed
                if modified:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Fixed imports in {filepath}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    setup_test_directories()
    print("Test directories have been set up.")
