#!/usr/bin/env python
"""
Setup script for CrossGL Translator tests.
Creates proper __init__.py files in test directories.
"""

import os


def setup_test_directories():
    """Set up test directories with proper __init__.py files"""
    test_backend_dir = os.path.join(os.path.dirname(__file__), "test_backend")

    # Make sure test_backend has __init__.py
    os.makedirs(test_backend_dir, exist_ok=True)
    init_file = os.path.join(test_backend_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write('"""Test backend module."""\n')

    # Define test directories with proper capitalization
    backend_dirs = [
        "test_OpenGL",
        "test_DirectX",
        "test_Slang",
        "test_Metal",
        "test_Vulkan",
        "test_Mojo",
    ]

    # Create or update directories and __init__.py files
    for dir_name in backend_dirs:
        # Ensure the directory exists
        dir_path = os.path.join(test_backend_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        # Create __init__.py in the directory
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            # Remove "test_" prefix and get backend name
            backend_name = dir_name[5:]
            with open(init_file, "w") as f:
                f.write(f'"""Test module for {backend_name}."""\n')


if __name__ == "__main__":
    setup_test_directories()
    print("Test directories have been set up.")
