"""CrossGL Translator Library"""

import sys

# Create case-insensitive aliases for backend modules
backend_modules = {
    "opengl": "OpenGL",
    "directx": "DirectX",
    "slang": "Slang",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "mojo": "Mojo",
}

# Import hook to alias modules
for lowercase, proper in backend_modules.items():
    # Create lowercase module aliases for each backend
    module_name = f"crosstl.backend.{lowercase}"
    proper_name = f"crosstl.backend.{proper}"

    try:
        # First try to import the proper-cased module
        __import__(proper_name)
        # If it succeeds, alias it to the lowercase name
        if proper_name in sys.modules and module_name not in sys.modules:
            sys.modules[module_name] = sys.modules[proper_name]
    except ImportError:
        # If that fails, try the lowercase version
        try:
            __import__(module_name)
            # If it succeeds, alias it to the proper name
            if module_name in sys.modules and proper_name not in sys.modules:
                sys.modules[proper_name] = sys.modules[module_name]
        except ImportError:
            pass  # Neither exists, skip

# Import the main API functions
from ._crosstl import translate
