"""
Pytest configuration for the CrossGL Translator tests
"""

import sys
import os
import platform
import pytest

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session", autouse=True)
def check_backend_modules():
    """Verify that all backend modules exist with the correct case."""
    if platform.system() == "Linux":  # Only check on case-sensitive filesystems
        backend_dirs = [
            "crosstl/backend/DirectX",
            "crosstl/backend/Metal",
            "crosstl/backend/OpenGL",
            "crosstl/backend/Slang",
            "crosstl/backend/Vulkan",
            "crosstl/backend/Mojo",
        ]

        for dir_path in backend_dirs:
            full_path = os.path.join(project_root, dir_path)
            if not os.path.isdir(full_path):
                print(
                    f"Warning: {dir_path} directory does not exist or has incorrect case"
                )

                # Check if the directory exists with a different case
                parent_dir = os.path.dirname(full_path)
                if os.path.isdir(parent_dir):
                    expected_name = os.path.basename(dir_path)
                    for item in os.listdir(parent_dir):
                        if (
                            item.lower() == expected_name.lower()
                            and item != expected_name
                        ):
                            print(
                                f"  Found directory with different case: {os.path.join(parent_dir, item)}"
                            )
                            print(
                                f"  Please rename to match expected case: {expected_name}"
                            )
