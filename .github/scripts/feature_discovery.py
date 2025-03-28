#!/usr/bin/env python3
"""
Feature Discovery Script for CrossGL Translator

This script analyzes the supported features across different backends
and creates GitHub issues for missing implementations.
"""

import os
import re
import json
import sys
from pathlib import Path
from github import Github
from github.GithubException import GithubException

# Reference backends to compare against
REFERENCE_BACKENDS = ["metal", "directx", "opengl", "vulkan", "slang", "mojo"]

# Backend directories
BACKEND_DIRS = {
    "metal": "crosstl/backend/Metal",
    "directx": "crosstl/backend/DirectX",
    "opengl": "crosstl/backend/Opengl",
    "vulkan": "crosstl/backend/Vulkan",
    "slang": "crosstl/backend/slang",
    "mojo": "crosstl/backend/Mojo"
}

# Codegen files
CODEGEN_FILES = {
    "metal": "crosstl/translator/codegen/metal_codegen.py",
    "directx": "crosstl/translator/codegen/directx_codegen.py",
    "opengl": "crosstl/translator/codegen/opengl_codegen.py",
    "vulkan": "crosstl/translator/codegen/vulkan_codegen.py",
    "slang": "crosstl/translator/codegen/slang_codegen.py",
    "mojo": "crosstl/translator/codegen/mojo_codegen.py"
}

# Types of features to search for
FEATURE_TYPES = {
    "functions": r"def\s+(\w+)\s*\(",  # Match function definitions
    "operators": r"(visit_(?:Binary|Unary)Op).*?operator\s*==\s*['\"]([+\-*/%<>=!&|^~]+)['\"]",  # Match operators
    "types": r"(?:visit_TypeDecl|process_type)\(.*?['\"](\w+)['\"]"  # Match type declarations
}

def get_features_from_file(file_path, feature_type):
    """Extract features of a particular type from a file."""
    if not os.path.exists(file_path):
        return set()
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    pattern = FEATURE_TYPES[feature_type]
    if feature_type == "operators":
        # For operators, we need the second group from the regex
        return {match.group(2) for match in re.finditer(pattern, content, re.DOTALL)}
    else:
        # For functions and types, we need the first group
        return {match.group(1) for match in re.finditer(pattern, content, re.DOTALL)}

def get_features_for_backend(backend):
    """Get all implemented features for a backend."""
    features = {
        "functions": set(),
        "operators": set(),
        "types": set()
    }
    
    # Check the main codegen file
    codegen_file = CODEGEN_FILES.get(backend)
    if codegen_file and os.path.exists(codegen_file):
        for feature_type in FEATURE_TYPES:
            features[feature_type].update(get_features_from_file(codegen_file, feature_type))
    
    # Check backend directory
    backend_dir = BACKEND_DIRS.get(backend)
    if backend_dir and os.path.exists(backend_dir):
        for root, _, files in os.walk(backend_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    for feature_type in FEATURE_TYPES:
                        features[feature_type].update(get_features_from_file(file_path, feature_type))
    
    return features

def find_missing_features():
    """Find missing features across all backends."""
    all_features = {
        backend: get_features_for_backend(backend)
        for backend in REFERENCE_BACKENDS
    }
    
    # Aggregate all possible features across all backends
    all_possible_features = {
        feature_type: set().union(*(backend_features[feature_type] for backend_features in all_features.values()))
        for feature_type in FEATURE_TYPES
    }
    
    # Find missing features for each backend
    missing_features = {}
    for backend in REFERENCE_BACKENDS:
        missing_features[backend] = {
            feature_type: all_possible_features[feature_type] - all_features[backend][feature_type]
            for feature_type in FEATURE_TYPES
        }
    
    return missing_features

def get_feature_documentation(feature, feature_type, backend):
    """
    Attempt to get documentation for a feature from the implemented backends.
    This would search for docstrings or comments near the feature implementation.
    """
    # This is a simplified version - in a real implementation, you would
    # parse docstrings and comments from source files where the feature is implemented
    for ref_backend in REFERENCE_BACKENDS:
        if ref_backend == backend:
            continue
            
        # Check if this backend has implemented the feature
        features = get_features_for_backend(ref_backend)
        if feature in features[feature_type]:
            # Check codegen file first
            codegen_file = CODEGEN_FILES.get(ref_backend)
            if codegen_file and os.path.exists(codegen_file):
                with open(codegen_file, 'r') as f:
                    content = f.read()
                    
                # Look for function/method definition with feature name
                if feature_type == "functions":
                    method_pattern = r"def\s+(?:visit_)?(\w+)\s*\(.*?\).*?(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')|\#(.*?)$"
                    for match in re.finditer(method_pattern, content, re.DOTALL | re.MULTILINE):
                        if match.group(1) == feature or f"visit_{feature}" == match.group(1):
                            doc = match.group(2) or match.group(3)
                            if doc:
                                return f"Documentation from {ref_backend}:\n\n{doc.strip()}"
    
    return f"No documentation found for {feature} in any backend implementation."

def create_github_issues(missing_features):
    """Create GitHub issues for missing features."""
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("GITHUB_TOKEN environment variable is not set. Cannot create issues.")
        return
    
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'CrossGL/CrossGL-Translator')
    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    
    # Create or get the main TODO issue
    main_issue_title = "🚀 CrossGL Translator Feature Implementation TODO"
    main_issue_body = """
# CrossGL Translator Feature Implementation Tracking

This issue tracks missing features across different backends in the CrossGL Translator.
Each missing feature has its own issue linked below, organized by backend.

This issue is automatically updated daily by the Feature Discovery workflow.

## Missing Features by Backend
    """
    
    main_issue = None
    # Try to find existing main issue
    for issue in repo.get_issues(state='open'):
        if issue.title == main_issue_title:
            main_issue = issue
            break
    
    if not main_issue:
        main_issue = repo.create_issue(
            title=main_issue_title,
            body=main_issue_body,
            labels=["todo", "enhancement", "good first issue"]
        )
    
    # Process and create issues for each missing feature
    backend_sections = {}
    for backend in REFERENCE_BACKENDS:
        backend_sections[backend] = f"\n### {backend.capitalize()} Backend\n\n"
        
        for feature_type in FEATURE_TYPES:
            missing = missing_features[backend][feature_type]
            if missing:
                backend_sections[backend] += f"\n#### Missing {feature_type.capitalize()}\n\n"
                
                for feature in sorted(missing):
                    # Check if issue already exists
                    feature_issue_title = f"[{backend.upper()}] Implement {feature_type[:-1]} '{feature}'"
                    existing_issue = None
                    for issue in repo.get_issues(state='open'):
                        if issue.title == feature_issue_title:
                            existing_issue = issue
                            break
                    
                    if not existing_issue:
                        # Get documentation from other backends if available
                        docs = get_feature_documentation(feature, feature_type, backend)
                        
                        issue_body = f"""
# Missing {feature_type[:-1]} in {backend.capitalize()} Backend

The {feature} {feature_type[:-1]} is implemented in one or more other backends, but is missing in {backend}.

## Documentation

{docs}

## Implementation Task

Implement the {feature} {feature_type[:-1]} in the {backend} backend.

## Relevant Files

- {CODEGEN_FILES.get(backend)}
- {BACKEND_DIRS.get(backend)}
"""
                        
                        try:
                            new_issue = repo.create_issue(
                                title=feature_issue_title,
                                body=issue_body,
                                labels=[backend, feature_type[:-1], "enhancement", "good first issue"]
                            )
                            backend_sections[backend] += f"- [ ] [{feature}](#{new_issue.number})\n"
                        except GithubException as e:
                            print(f"Error creating issue for {feature}: {e}")
                            continue
                    else:
                        backend_sections[backend] += f"- [ ] [{feature}](#{existing_issue.number})\n"
    
    # Update main issue with new content
    updated_body = main_issue_body
    for backend, section in backend_sections.items():
        if "#### Missing" in section:  # Only add sections with actual missing features
            updated_body += section
    
    main_issue.edit(body=updated_body)
    print(f"Updated main issue: {main_issue.html_url}")

def main():
    """Main function to discover features and create issues."""
    print("Starting feature discovery...")
    
    missing_features = find_missing_features()
    
    # Print summary
    for backend in REFERENCE_BACKENDS:
        total_missing = sum(len(missing_features[backend][ft]) for ft in FEATURE_TYPES)
        print(f"{backend}: {total_missing} missing features")
        for feature_type in FEATURE_TYPES:
            print(f"  - {feature_type}: {len(missing_features[backend][feature_type])}")
    
    # Create issues
    create_github_issues(missing_features)
    
    print("Feature discovery completed successfully.")

if __name__ == "__main__":
    main() 