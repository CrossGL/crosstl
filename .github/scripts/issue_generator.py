#!/usr/bin/env python3
"""
Issue Generator for CrossGL Feature Discovery

This script takes the output from the docs_crawler.py analysis and:
1. Compares against the implemented features in CrossGL
2. Creates structured GitHub issues for missing features
3. Organizes issues into milestones by language
4. Creates parent issues with subtasks for each category
"""

import os
import re
import json
import glob
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime

from github import Github
from github.Issue import Issue
from github.Repository import Repository
from github.Milestone import Milestone
from github.GithubException import GithubException

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.github/logs/issue_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('issue_generator')

# Ensure directories exist
os.makedirs('.cache/analysis', exist_ok=True)
os.makedirs('.cache/issues', exist_ok=True)
os.makedirs('.github/logs', exist_ok=True)

# Category definitions for organizing issues
CATEGORIES = {
    'functions': {
        'title': 'Functions',
        'description': 'Missing function implementations in the CrossGL Translator',
        'labels': ['enhancement', 'function', 'good first issue'],
    },
    'operators': {
        'title': 'Operators',
        'description': 'Missing operator implementations in the CrossGL Translator',
        'labels': ['enhancement', 'operator', 'good first issue'],
    },
    'types': {
        'title': 'Types',
        'description': 'Missing type implementations in the CrossGL Translator',
        'labels': ['enhancement', 'type', 'good first issue'],
    }
}

# Language-specific settings
LANGUAGE_SETTINGS = {
    'metal': {
        'name': 'Metal Shading Language',
        'doc_url': 'https://developer.apple.com/documentation/metal/metal_shading_language',
        'priority': 1,
    },
    'directx': {
        'name': 'DirectX HLSL',
        'doc_url': 'https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl',
        'priority': 2,
    },
    'opengl': {
        'name': 'OpenGL GLSL',
        'doc_url': 'https://www.khronos.org/registry/OpenGL-Refpages/gl4/',
        'priority': 3,
    },
    'vulkan': {
        'name': 'Vulkan SPIR-V',
        'doc_url': 'https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html',
        'priority': 4,
    },
    'slang': {
        'name': 'Slang',
        'doc_url': 'https://shader-slang.com/slang/docs/reference/',
        'priority': 5,
    },
    'mojo': {
        'name': 'Mojo',
        'doc_url': 'https://docs.modular.com/mojo/stdlib',
        'priority': 6,
    }
}

# Implementation status
IMPLEMENTATION_STATUS = {
    'MISSING': 0,     # Not implemented at all
    'PARTIAL': 1,     # Partially implemented
    'COMPLETE': 2,    # Fully implemented
}

class CrossGLAnalyzer:
    """Analyzes the CrossGL codebase to detect implemented features"""
    
    def __init__(self):
        self.crosstl_dir = Path('crosstl')
        self.backends_dir = self.crosstl_dir / 'backend'
        self.translator_dir = self.crosstl_dir / 'translator'
        self.codegen_dir = self.translator_dir / 'codegen'
        
        self.implemented_features = {
            'functions': self._get_implemented_functions(),
            'operators': self._get_implemented_operators(),
            'types': self._get_implemented_types(),
        }
    
    def _get_implemented_functions(self) -> Dict[str, Set[str]]:
        """Get all implemented functions per backend"""
        implemented = {lang: set() for lang in LANGUAGE_SETTINGS.keys()}
        
        # Check codegen files for function implementations
        for lang in implemented.keys():
            backend_dir = self.backends_dir / lang.capitalize()
            codegen_file = self.codegen_dir / f"{lang}_codegen.py"
            
            if not backend_dir.exists() or not codegen_file.exists():
                continue
            
            # Extract functions from codegen files
            if codegen_file.exists():
                with open(codegen_file, 'r') as f:
                    content = f.read()
                    
                    # Look for function definitions and visit_* methods
                    for match in re.finditer(r'def\s+(?:visit_)?(\w+)', content):
                        function_name = match.group(1)
                        # Filter out common methods that aren't actual shader functions
                        if not function_name.startswith(('__', 'visit_', 'generate')):
                            implemented[lang].add(function_name)
            
            # Look through backend directory
            if backend_dir.exists():
                for py_file in backend_dir.glob('**/*.py'):
                    with open(py_file, 'r') as f:
                        content = f.read()
                        
                        # Look for function definitions specific to this language
                        for match in re.finditer(r'def\s+(\w+)', content):
                            function_name = match.group(1)
                            # Filter out common methods that aren't actual shader functions
                            if not function_name.startswith('__'):
                                implemented[lang].add(function_name)
        
        return implemented
    
    def _get_implemented_operators(self) -> Dict[str, Set[str]]:
        """Get all implemented operators per backend"""
        implemented = {lang: set() for lang in LANGUAGE_SETTINGS.keys()}
        
        # Extract operators from codegen files
        for lang in implemented.keys():
            codegen_file = self.codegen_dir / f"{lang}_codegen.py"
            
            if not codegen_file.exists():
                continue
            
            with open(codegen_file, 'r') as f:
                content = f.read()
                
                # Look for operator handling in visit_BinaryOp, visit_UnaryOp, etc.
                for match in re.finditer(r'(?:visit_(?:Binary|Unary)Op).*?operator\s*==\s*[\'"]([+\-*/%<>=!&|^~]+)[\'"]', content, re.DOTALL):
                    operator = match.group(1)
                    implemented[lang].add(operator)
        
        return implemented
    
    def _get_implemented_types(self) -> Dict[str, Set[str]]:
        """Get all implemented types per backend"""
        implemented = {lang: set() for lang in LANGUAGE_SETTINGS.keys()}
        
        # Extract types from codegen and backend files
        for lang in implemented.keys():
            codegen_file = self.codegen_dir / f"{lang}_codegen.py"
            
            if not codegen_file.exists():
                continue
            
            with open(codegen_file, 'r') as f:
                content = f.read()
                
                # Look for type processing
                for match in re.finditer(r'(?:visit_TypeDecl|process_type)\(.*?[\'"](\w+)[\'"]', content):
                    type_name = match.group(1)
                    implemented[lang].add(type_name)
        
        return implemented
    
    def get_implementation_status(self, feature_type: str, feature_name: str, language: str) -> int:
        """Check if a feature is implemented for a specific language"""
        if language not in self.implemented_features[feature_type]:
            return IMPLEMENTATION_STATUS['MISSING']
        
        if feature_name in self.implemented_features[feature_type][language]:
            return IMPLEMENTATION_STATUS['COMPLETE']
        
        # Check for partial implementation (e.g., function exists but with different signature)
        if feature_type == 'functions':
            # Check if a similar function exists
            similar_functions = {func for func in self.implemented_features[feature_type][language] 
                               if feature_name.lower() in func.lower()}
            if similar_functions:
                return IMPLEMENTATION_STATUS['PARTIAL']
        
        return IMPLEMENTATION_STATUS['MISSING']


class IssueGenerator:
    """Generates GitHub issues for missing features"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.gh = Github(github_token)
        self.repo = self.gh.get_repo(repo_name)
        self.analyzer = CrossGLAnalyzer()
        self.analysis_dir = Path('.cache/analysis')
        self.issues_cache_dir = Path('.cache/issues')
        self.issues_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for existing issues to avoid duplicate API calls
        self.existing_issues = self._load_existing_issues()
        self.existing_milestones = self._load_existing_milestones()
        
        # Track created issues for linking
        self.created_issues = {}
    
    def _load_existing_issues(self) -> Dict[str, Issue]:
        """Load existing issues from the repository to avoid duplicates"""
        logger.info("Loading existing issues...")
        
        # Try to load from cache first
        cache_file = self.issues_cache_dir / 'existing_issues.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is recent (less than 1 day old)
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                if (datetime.now() - cache_time).days < 1:
                    logger.info("Using cached issues data")
                    return {issue['title']: issue for issue in cache_data.get('issues', [])}
            except Exception as e:
                logger.warning(f"Error loading issues cache: {e}")
        
        # Load from GitHub API
        issues = {}
        for issue in self.repo.get_issues(state='open'):
            issues[issue.title] = {
                'title': issue.title,
                'number': issue.number,
                'body': issue.body,
                'labels': [label.name for label in issue.labels],
                'milestone': issue.milestone.number if issue.milestone else None,
            }
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'issues': list(issues.values())
            }, f, indent=2)
        
        logger.info(f"Loaded {len(issues)} existing issues")
        return issues
    
    def _load_existing_milestones(self) -> Dict[str, Milestone]:
        """Load existing milestones from the repository"""
        logger.info("Loading existing milestones...")
        milestones = {}
        
        for milestone in self.repo.get_milestones():
            milestones[milestone.title] = milestone
        
        logger.info(f"Loaded {len(milestones)} existing milestones")
        return milestones
    
    def _get_or_create_milestone(self, language: str) -> Milestone:
        """Get or create a milestone for a language"""
        milestone_title = f"{LANGUAGE_SETTINGS[language]['name']} Implementation"
        
        if milestone_title in self.existing_milestones:
            return self.existing_milestones[milestone_title]
        
        # Create milestone if it doesn't exist
        description = f"Implementation of {LANGUAGE_SETTINGS[language]['name']} features in CrossGL Translator"
        milestone = self.repo.create_milestone(
            title=milestone_title,
            state='open',
            description=description,
        )
        
        self.existing_milestones[milestone_title] = milestone
        logger.info(f"Created milestone: {milestone_title}")
        return milestone
    
    def _create_issue(self, title: str, body: str, labels: List[str] = None, milestone: Optional[Milestone] = None) -> Issue:
        """Create a GitHub issue with error handling and rate limit awareness"""
        if labels is None:
            labels = []
        
        # Check if issue already exists
        if title in self.existing_issues:
            issue_number = self.existing_issues[title]['number']
            logger.info(f"Issue already exists: {title} (#{issue_number})")
            return self.repo.get_issue(issue_number)
        
        # Create the issue with retry for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                issue = self.repo.create_issue(
                    title=title,
                    body=body,
                    labels=labels,
                    milestone=milestone
                )
                
                # Update cache
                self.existing_issues[title] = {
                    'title': title,
                    'number': issue.number,
                    'body': body,
                    'labels': labels,
                    'milestone': milestone.number if milestone else None,
                }
                
                logger.info(f"Created issue: {title} (#{issue.number})")
                return issue
            
            except GithubException as e:
                if e.status == 403 and attempt < max_retries - 1:
                    # Rate limit hit, wait and retry
                    logger.warning(f"Rate limit hit, waiting 60 seconds... ({attempt+1}/{max_retries})")
                    time.sleep(60)
                else:
                    logger.error(f"Error creating issue '{title}': {e}")
                    raise
        
        raise RuntimeError(f"Failed to create issue after {max_retries} attempts: {title}")
    
    def _format_feature_documentation(self, feature: Dict[str, Any], language: str) -> str:
        """Format feature documentation for inclusion in issue body"""
        feature_type = 'function' if 'signature' in feature else 'operator' if 'symbol' in feature else 'type'
        
        if feature_type == 'function':
            doc = f"### Function: `{feature['name']}`\n\n"
            
            if feature.get('signature'):
                doc += f"**Signature:**\n```{language}\n{feature['signature']}\n```\n\n"
            
            if feature.get('description'):
                doc += f"**Description:**\n{feature['description']}\n\n"
            
            if feature.get('parameters'):
                doc += "**Parameters:**\n"
                for param in feature['parameters']:
                    doc += f"- `{param['name']}`: {param.get('type', '')}\n"
                doc += "\n"
            
            if feature.get('return_type'):
                doc += f"**Returns:** {feature['return_type']}\n\n"
            
            if feature.get('url'):
                doc += f"**Documentation:** [Official {LANGUAGE_SETTINGS[language]['name']} Documentation]({feature['url']})\n\n"
        
        elif feature_type == 'operator':
            doc = f"### Operator: `{feature['symbol']}`\n\n"
            
            if feature.get('name') and feature['name'] != feature['symbol']:
                doc += f"**Name:** {feature['name']}\n\n"
            
            if feature.get('description'):
                doc += f"**Description:**\n{feature['description']}\n\n"
            
            if feature.get('example'):
                doc += f"**Example:**\n```{language}\n{feature['example']}\n```\n\n"
        
        else:  # type
            doc = f"### Type: `{feature['name']}`\n\n"
            
            if feature.get('description'):
                doc += f"**Description:**\n{feature['description']}\n\n"
            
            if feature.get('category'):
                doc += f"**Category:** {feature['category']}\n\n"
        
        return doc
    
    def _generate_task_description(self, feature: Dict[str, Any], language: str, feature_type: str) -> str:
        """Generate task description for implementation"""
        task_desc = f"""## Task: Implement {feature_type} in {LANGUAGE_SETTINGS[language]['name']}

### Description
Implement the {feature['name']} {feature_type} for the {LANGUAGE_SETTINGS[language]['name']} backend in CrossGL Translator.

### Implementation Details
You should implement this {feature_type} in the following files:

- `crosstl/translator/codegen/{language}_codegen.py` - Add support for this {feature_type}
- Add appropriate methods in the {language.capitalize()} backend directory

### Testing Requirements
- Create test cases in `tests/test_translator/test_codegen/test_{language}_codegen.py`
- Ensure tests verify correct translation of this {feature_type}
- Include both positive and negative test cases

### Documentation
{self._format_feature_documentation(feature, language)}
"""
        return task_desc
    
    def generate_parent_issues(self, language: str, analysis_data: Dict[str, Any]) -> Dict[str, Issue]:
        """Generate parent issues for each category of features"""
        milestone = self._get_or_create_milestone(language)
        parent_issues = {}
        
        for category, category_info in CATEGORIES.items():
            if category not in analysis_data:
                continue
            
            missing_count = sum(1 for feature in analysis_data[category] 
                              if self.analyzer.get_implementation_status(category, feature['name'], language) == IMPLEMENTATION_STATUS['MISSING'])
            
            if missing_count == 0:
                logger.info(f"No missing {category} for {language}, skipping parent issue")
                continue
            
            title = f"[{language.upper()}] Implement Missing {category_info['title']}"
            body = f"""# {LANGUAGE_SETTINGS[language]['name']} {category_info['title']} Implementation

This issue tracks the implementation of missing {category_info['title'].lower()} in the {LANGUAGE_SETTINGS[language]['name']} backend of CrossGL Translator.

## Description
{category_info['description']} for the {LANGUAGE_SETTINGS[language]['name']} backend.

## Progress
There are {missing_count} missing {category} to implement.

## Tasks
The following sub-issues will be created for individual {category}:

"""
            
            # Create parent issue
            labels = category_info['labels'].copy()
            labels.append(language)
            parent_issue = self._create_issue(title, body, labels, milestone)
            parent_issues[category] = parent_issue
        
        return parent_issues
    
    def generate_feature_issues(self, language: str, parent_issues: Dict[str, Issue]) -> None:
        """Generate individual feature issues linked to parent issues"""
        milestone = self._get_or_create_milestone(language)
        
        # Load analysis data
        analysis_file = self.analysis_dir / f"{language}.json"
        if not analysis_file.exists():
            logger.warning(f"Analysis data not found for {language}")
            return
        
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        # Create feature issues
        for category, category_info in CATEGORIES.items():
            if category not in analysis_data:
                continue
            
            # Skip if no parent issue (meaning no missing features)
            if category not in parent_issues:
                continue
            
            parent_issue = parent_issues[category]
            issue_references = []
            
            for feature in analysis_data[category]:
                feature_name = feature['name']
                
                # Check if already implemented
                status = self.analyzer.get_implementation_status(category, feature_name, language)
                if status != IMPLEMENTATION_STATUS['MISSING']:
                    continue
                
                # Create issue for missing feature
                title = f"[{language.upper()}] Implement {category[:-1]} '{feature_name}'"
                labels = category_info['labels'].copy()
                labels.append(language)
                
                # Generate task description
                body = self._generate_task_description(feature, language, category[:-1])
                
                # Add reference to parent issue
                body += f"\n\nParent issue: #{parent_issue.number}"
                
                try:
                    issue = self._create_issue(title, body, labels, milestone)
                    issue_references.append(f"- [ ] [{feature_name}](#{issue.number})")
                except Exception as e:
                    logger.error(f"Error creating issue for {feature_name}: {e}")
                    issue_references.append(f"- [ ] {feature_name} (issue creation failed)")
            
            # Update parent issue with references to child issues
            if issue_references:
                updated_body = parent_issue.body + "\n" + "\n".join(issue_references)
                parent_issue.edit(body=updated_body)
                logger.info(f"Updated parent issue #{parent_issue.number} with {len(issue_references)} child issues")
    
    def generate_all_issues(self) -> None:
        """Generate all issues for all languages"""
        logger.info("Starting issue generation for all languages...")
        
        # Process each language
        for language in LANGUAGE_SETTINGS.keys():
            analysis_file = self.analysis_dir / f"{language}.json"
            if not analysis_file.exists():
                logger.warning(f"Skipping {language}: No analysis data found")
                continue
            
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            logger.info(f"Processing {language}...")
            
            try:
                # Create parent issues first
                parent_issues = self.generate_parent_issues(language, analysis_data)
                
                # Then create child issues linked to parents
                self.generate_feature_issues(language, parent_issues)
                
                logger.info(f"Completed issue generation for {language}")
            except Exception as e:
                logger.error(f"Error generating issues for {language}: {e}")
        
        logger.info("Issue generation complete")


def main():
    """Main entry point"""
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.error("GITHUB_TOKEN environment variable is not set")
        return 1
    
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'CrossGL/CrossGL-Translator')
    
    try:
        issue_generator = IssueGenerator(github_token, repo_name)
        issue_generator.generate_all_issues()
        return 0
    except Exception as e:
        logger.error(f"Error in issue generation: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 