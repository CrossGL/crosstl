"""
Import hooks for handling case-insensitive module names.
"""

import sys
import importlib.abc
import importlib.machinery
import importlib.util


class CaseSensitiveBackendFinder(importlib.abc.MetaPathFinder):
    """
    A meta path finder that maps lowercase backend module names to their
    correctly capitalized versions.
    """

    def __init__(self):
        # Register mappings from lowercase to proper case
        self.backend_mappings = {
            "opengl": "OpenGL",
            "directx": "DirectX",
            "slang": "Slang",
            "metal": "Metal",
            "vulkan": "Vulkan",
            "mojo": "Mojo",
        }
        # Flag to prevent recursion
        self.in_find_spec = False

    def find_spec(self, fullname, path=None, target=None):
        """Find the module spec for the given name, with case correction."""
        # Prevent recursion
        if self.in_find_spec:
            return None

        self.in_find_spec = True
        try:
            # Only handle crosstl.backend.X modules
            if not fullname.startswith("crosstl.backend.") and not fullname.startswith(
                "tests.test_backend.test_"
            ):
                return None

            # Split the module name into parts
            parts = fullname.split(".")

            # Case 1: Handle crosstl.backend.Xxx imports
            if fullname.startswith("crosstl.backend.") and len(parts) >= 3:
                backend_name = parts[2].lower()

                # Check if this is a lowercase variant we need to redirect
                if (
                    backend_name in self.backend_mappings
                    and parts[2] != self.backend_mappings[backend_name]
                ):
                    # Create the properly capitalized name
                    parts[2] = self.backend_mappings[backend_name]
                    ".".join(parts)

                    # Let the next finder handle it
                    return None

            # Case 2: Handle tests.test_backend.test_xxx imports
            elif fullname.startswith("tests.test_backend.test_") and len(parts) >= 3:
                test_name = parts[2].lower()

                # Check if this is a "test_backend" submodule
                if test_name.startswith("test_"):
                    backend_name = test_name[5:].lower()  # Remove 'test_' prefix

                    # Check if we need to correct the case
                    if backend_name in self.backend_mappings:
                        # Create the properly capitalized name
                        parts[2] = f"test_{self.backend_mappings[backend_name]}"
                        ".".join(parts)

                        # Let the next finder handle it
                        return None

            # No match or correction needed
            return None
        finally:
            self.in_find_spec = False


def register_hooks():
    """Register the custom import hooks."""
    # Add our finder to the meta path
    sys.meta_path.insert(0, CaseSensitiveBackendFinder())

    print("Backend import hooks registered for case-insensitive module handling")


# Register hooks when the module is imported
register_hooks()
