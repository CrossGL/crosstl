# CrossGL Translator Tests

This directory contains tests for the CrossGL Translator project.

## Directory Structure

The tests are organized as follows:

```
tests/
├── test_backend/          # Tests for specific backends
│   ├── test_OpenGL/      # Tests for OpenGL backend (canonical location)
│   ├── test_Slang/       # Tests for Slang backend (canonical location)
│   ├── test_DirectX/     # Tests for DirectX backend
│   ├── test_Metal/       # Tests for Metal backend
│   ├── test_Vulkan/      # Tests for Vulkan backend
│   └── test_Mojo/        # Tests for Mojo backend
├── test_opengl/          # Alternate location for OpenGL tests (lowercase)
├── test_slang/           # Alternate location for Slang tests (lowercase)
└── test_translator/      # Tests for translator module
```

## Running Tests

You can run the tests using pytest:

```bash
# Run all tests
python -m pytest tests/

# Run tests for a specific backend
python -m pytest tests/test_backend/test_OpenGL/

# Run tests using lowercase directory names
python -m pytest tests/test_opengl/
```

## Directory Naming Convention

For backward compatibility, tests can be run using either the uppercase or lowercase directory names:

- Uppercase directories (e.g. `test_OpenGL`) are the canonical locations of the tests
- Lowercase directories (e.g. `test_opengl`) are alternative locations that point to the same tests

This allows flexibility in how the tests are referenced in scripts and documentation.
