#!/usr/bin/env python
"""
Script to create lowercase aliases for all backend test directories.
This helps maintain backward compatibility for projects that might
reference the lowercase directory names.
"""

import os

# Map of proper capitalization to lowercase
backend_modules = {
    "OpenGL": "opengl",
    "DirectX": "directx",
    "Slang": "slang",
    "Metal": "metal",
    "Vulkan": "vulkan",
    "Mojo": "mojo",
}


def create_lowercase_aliases():
    """Create lowercase directory aliases for all backend tests."""
    test_dir = os.path.dirname(__file__)

    # Create aliases directly under tests/
    for proper, lowercase in backend_modules.items():
        source_dir = os.path.join(test_dir, "test_backend", f"test_{proper}")
        target_dir = os.path.join(test_dir, f"test_{lowercase}")

        # Skip if source doesn't exist
        if not os.path.exists(source_dir):
            print(f"Skipping {source_dir} - directory doesn't exist")
            continue

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created directory: {target_dir}")

        # Create __init__.py file in the target directory
        init_file = os.path.join(target_dir, "__init__.py")
        with open(init_file, "w") as f:
            f.write(f'"""Test module for {lowercase} (alternate location)."""\n\n')
            f.write(f"# Import the original module to make its symbols available\n")
            f.write(f"from tests.test_backend.test_{proper} import *\n")

        # Create symlinks for all test files
        for item in os.listdir(source_dir):
            if item.endswith(".py") and item.startswith("test_"):
                source_file = os.path.join(source_dir, item)
                target_file = os.path.join(target_dir, item)

                # Use relative path for symlink
                rel_source = os.path.relpath(source_file, os.path.dirname(target_file))

                # Remove existing symlink if it exists
                if os.path.exists(target_file):
                    if os.path.islink(target_file):
                        os.unlink(target_file)
                    else:
                        continue  # Skip if it's a real file

                # Create symlink
                try:
                    os.symlink(rel_source, target_file)
                    print(f"Created symlink: {target_file} -> {rel_source}")
                except Exception as e:
                    print(f"Error creating symlink {target_file}: {e}")


if __name__ == "__main__":
    create_lowercase_aliases()
    print("Lowercase aliases have been created.")
