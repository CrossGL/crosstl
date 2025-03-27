"""
Pytest configuration for the CrossGL Translator tests
"""

import sys
import os
import pytest
import subprocess

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


# Run setup script to create proper test directories
def pytest_sessionstart(session):
    """Run setup script to prepare test directories."""
    setup_script = os.path.join(os.path.dirname(__file__), "setup_tests.py")
    if os.path.exists(setup_script):
        try:
            subprocess.run([sys.executable, setup_script], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to set up test directories")
