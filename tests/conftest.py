"""
Pytest configuration for the CrossGL Translator tests
"""

import sys
import os
import platform
import pytest
import re
import importlib.util
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import our custom import hooks for case-insensitive module handling
from tests import import_hooks


# Run setup script to create proper test directories
def pytest_sessionstart(session):
    """Run setup script to prepare test directories."""
    setup_script = os.path.join(os.path.dirname(__file__), "setup_tests.py")
    if os.path.exists(setup_script):
        try:
            subprocess.run([sys.executable, setup_script], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to set up test directories")


# Hook to handle case-insensitive test path collection
def pytest_configure(config):
    """Configure pytest with import mappings for case sensitivity."""
    # Create an import hook for backend modules with proper capitalization
    sys.meta_path.append(CaseSensitiveImportFinder())


class CaseSensitiveImportFinder:
    """Custom import finder that redirects lowercase module names to proper capitalized versions."""

    def find_spec(self, fullname, path, target=None):
        # Handle only our specific modules
        if fullname.startswith("crosstl.backend."):
            parts = fullname.split(".")
            if len(parts) >= 3:
                # Map common lowercase module names to their capitalized versions
                backends = {
                    "opengl": "OpenGL",
                    "directx": "DirectX",
                    "slang": "Slang",
                    "metal": "Metal",
                    "vulkan": "Vulkan",
                    "mojo": "Mojo",
                }

                # Check if we need to correct the case
                if (
                    parts[2].lower() in backends
                    and parts[2] != backends[parts[2].lower()]
                ):
                    # Reconstruct the fullname with proper capitalization
                    parts[2] = backends[parts[2].lower()]
                    corrected_name = ".".join(parts)

                    # Redirect to the properly capitalized module
                    try:
                        return importlib.util.find_spec(corrected_name)
                    except (ImportError, AttributeError):
                        return None
        return None


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


# Define a mapping of lowercase to proper capitalization
BACKEND_NAMES = {
    "opengl": "OpenGL",
    "directx": "DirectX",
    "slang": "Slang",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "mojo": "Mojo",
}


# Add hook to handle test path collection with case-insensitive matching
def pytest_ignore_collect(collection_path, config):
    """Special hook to handle case-insensitive test directories."""
    # Convert to string for consistency
    path_str = str(collection_path)

    # Skip if not in test_backend area
    if "test_backend/test_" not in path_str.replace("\\", "/"):
        return False

    # Get the lowercase version of the path
    path_str.lower()

    # Extract the backend name from path
    match = re.search(
        r"test_backend/test_([^/\\]+)", path_str.replace("\\", "/"), re.IGNORECASE
    )
    if not match:
        return False

    backend_name = match.group(1).lower()

    # Check if this is a known backend
    if backend_name in BACKEND_NAMES:
        # If path doesn't exist but we're processing a known backend directory
        if not os.path.exists(path_str):
            # Check if the capitalized version exists
            cap_name = BACKEND_NAMES[backend_name]
            correct_dir = re.sub(
                rf"test_backend/test_{backend_name}",
                f"test_backend/test_{cap_name}",
                path_str.replace("\\", "/"),
                flags=re.IGNORECASE,
            ).replace("/", os.sep)

            # If the capitalized version exists, let pytest collect from there instead
            if os.path.exists(correct_dir):
                return (
                    True  # Ignore this path, we'll collect from the capitalized version
                )

    return False


# Add import hook for properly capitalized modules
def pytest_collect_file(parent, path):
    """Handle collection of test files with proper module imports."""
    # Path may be a different type depending on pytest version
    # Convert to Path object for consistency
    if not isinstance(path, Path):
        path = Path(str(path))

    path_str = str(path)
    if path.is_file() and path.name.startswith("test_") and path.suffix == ".py":
        # Create module mappings for test imports
        backend_modules = BACKEND_NAMES.copy()

        # Process the file content to fix imports
        if "test_backend/test_" in path_str.replace("\\", "/"):
            try:
                with open(path_str, "r", encoding="utf-8") as f:
                    file_content = f.read()

                # Replace lowercase module references with proper capitalization
                for lower, proper in backend_modules.items():
                    pattern = f"crosstl.backend.{lower}"
                    replacement = f"crosstl.backend.{proper}"
                    if pattern in file_content:
                        # Replace the module names in the file content
                        new_content = file_content.replace(pattern, replacement)
                        if new_content != file_content:
                            # Write the fixed imports back to the file
                            with open(path_str, "w", encoding="utf-8") as f:
                                f.write(new_content)
            except Exception as e:
                print(f"Warning: Failed to update imports in {path_str}: {e}")

    return None
