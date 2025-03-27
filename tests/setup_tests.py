#!/usr/bin/env python
"""
Setup script for cross-platform testing.
Creates proper __init__.py files in test directories.
"""

import os
import sys
import shutil


def setup_test_directories():
    """Set up test directories with proper __init__.py files"""
    test_backend_dir = os.path.join(os.path.dirname(__file__), "test_backend")

    # Make sure test_backend has __init__.py
    os.makedirs(test_backend_dir, exist_ok=True)
    init_file = os.path.join(test_backend_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write('"""Test backend module."""\n')

    # Define mappings of capitalized directory names
    backend_dirs = {
        "OpenGL": ["test_OpenGL", "test_opengl"],
        "DirectX": ["test_DirectX", "test_directx"],
        "Slang": ["test_Slang", "test_slang"],
        "Metal": ["test_Metal", "test_metal"],
        "Vulkan": ["test_Vulkan", "test_vulkan"],
        "Mojo": ["test_Mojo", "test_mojo"],
    }

    # Create or update directories and __init__.py files
    for backend, dir_variants in backend_dirs.items():
        # Ensure the primary (capitalized) directory exists
        primary_dir_path = os.path.join(test_backend_dir, dir_variants[0])
        os.makedirs(primary_dir_path, exist_ok=True)

        # Create __init__.py in the primary directory
        init_file = os.path.join(primary_dir_path, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write(f'"""Test module for {backend}."""\n')

        # Fix imports in test files in primary directory
        fix_imports_in_directory(primary_dir_path)

        # Create lowercase directory variant if not the same as primary
        # and copy or symlink test files for backward compatibility
        lowercase_dir_path = os.path.join(test_backend_dir, dir_variants[1])
        if lowercase_dir_path != primary_dir_path:
            # Copy files if lowercase directory doesn't exist yet
            if not os.path.exists(lowercase_dir_path):
                try:
                    # Try symlink first (more efficient)
                    if (
                        sys.platform != "win32"
                    ):  # Symlinks work better on Unix-like systems
                        # Create relative symlink
                        os.symlink(
                            os.path.basename(primary_dir_path),
                            lowercase_dir_path,
                            target_is_directory=True,
                        )
                        print(
                            f"Created symlink from {lowercase_dir_path} -> {primary_dir_path}"
                        )
                    else:
                        # On Windows, do a regular copy
                        shutil.copytree(primary_dir_path, lowercase_dir_path)
                        print(f"Copied {primary_dir_path} to {lowercase_dir_path}")
                except (OSError, shutil.Error) as e:
                    print(f"Error creating directory {lowercase_dir_path}: {e}")

            # If lowercase directory exists but is not a symlink, ensure it has __init__.py
            elif not os.path.islink(lowercase_dir_path):
                init_file = os.path.join(lowercase_dir_path, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, "w") as f:
                        f.write(f'"""Test module for {backend} (lowercase)."""\n')

                # Fix imports in lowercase directory
                fix_imports_in_directory(lowercase_dir_path)


def fix_imports_in_directory(directory):
    """Fix imports in all Python files in the directory."""
    if not os.path.exists(directory):
        return

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
