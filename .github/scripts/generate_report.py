#!/usr/bin/env python3
"""
Report Generator for CrossGL Translator Implementation Status

This script generates comprehensive reports on the implementation status
of different shader language features in the CrossGL Translator.
"""

import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(".github/logs/report_generator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("report_generator")

# Ensure directories exist
os.makedirs(".cache/analysis", exist_ok=True)
os.makedirs(".cache/reports", exist_ok=True)
os.makedirs(".github/logs", exist_ok=True)

# Implementation status
IMPLEMENTATION_STATUS = {
    "MISSING": 0,  # Not implemented at all
    "PARTIAL": 1,  # Partially implemented
    "COMPLETE": 2,  # Fully implemented
}

# Category types
FEATURE_CATEGORIES = ["functions", "operators", "types"]

# Language-specific settings (same as in issue_generator.py)
LANGUAGE_SETTINGS = {
    "metal": {
        "name": "Metal Shading Language",
        "doc_url": (
            "https://developer.apple.com/documentation/metal/metal_shading_language"
        ),
        "priority": 1,
    },
    "directx": {
        "name": "DirectX HLSL",
        "doc_url": "https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl",
        "priority": 2,
    },
    "opengl": {
        "name": "OpenGL GLSL",
        "doc_url": "https://www.khronos.org/registry/OpenGL-Refpages/gl4/",
        "priority": 3,
    },
    "vulkan": {
        "name": "Vulkan SPIR-V",
        "doc_url": (
            "https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/vkspec.html"
        ),
        "priority": 4,
    },
    "slang": {
        "name": "Slang",
        "doc_url": "https://shader-slang.com/slang/docs/reference/",
        "priority": 5,
    },
    "mojo": {
        "name": "Mojo",
        "doc_url": "https://docs.modular.com/mojo/stdlib",
        "priority": 6,
    },
}


class CrossGLAnalyzer:
    """Analyzes the CrossGL codebase to detect implemented features"""

    def __init__(self):
        self.crosstl_dir = Path("crosstl")
        self.backends_dir = self.crosstl_dir / "backend"
        self.translator_dir = self.crosstl_dir / "translator"
        self.codegen_dir = self.translator_dir / "codegen"

        self.implemented_features = {
            "functions": self._get_implemented_functions(),
            "operators": self._get_implemented_operators(),
            "types": self._get_implemented_types(),
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
                with open(codegen_file, "r") as f:
                    content = f.read()

                    # Look for function definitions and visit_* methods
                    for match in re.finditer(r"def\s+(?:visit_)?(\w+)", content):
                        function_name = match.group(1)
                        # Filter out common methods that aren't actual shader functions
                        if not function_name.startswith(("__", "visit_", "generate")):
                            implemented[lang].add(function_name)

            # Look through backend directory
            if backend_dir.exists():
                for py_file in backend_dir.glob("**/*.py"):
                    with open(py_file, "r") as f:
                        content = f.read()

                        # Look for function definitions specific to this language
                        for match in re.finditer(r"def\s+(\w+)", content):
                            function_name = match.group(1)
                            # Filter out common methods that aren't actual shader functions
                            if not function_name.startswith("__"):
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

            with open(codegen_file, "r") as f:
                content = f.read()

                # Look for operator handling in visit_BinaryOp, visit_UnaryOp, etc.
                for match in re.finditer(
                    r'(?:visit_(?:Binary|Unary)Op).*?operator\s*==\s*[\'"]([+\-*/%<>=!&|^~]+)[\'"]',
                    content,
                    re.DOTALL,
                ):
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

            with open(codegen_file, "r") as f:
                content = f.read()

                # Look for type processing
                for match in re.finditer(
                    r'(?:visit_TypeDecl|process_type)\(.*?[\'"](\w+)[\'"]', content
                ):
                    type_name = match.group(1)
                    implemented[lang].add(type_name)

        return implemented

    def get_implementation_status(
        self, feature_type: str, feature_name: str, language: str
    ) -> int:
        """Check if a feature is implemented for a specific language"""
        if language not in self.implemented_features[feature_type]:
            return IMPLEMENTATION_STATUS["MISSING"]

        if feature_name in self.implemented_features[feature_type][language]:
            return IMPLEMENTATION_STATUS["COMPLETE"]

        # Check for partial implementation (e.g., function exists but with different signature)
        if feature_type == "functions":
            # Check if a similar function exists
            similar_functions = {
                func
                for func in self.implemented_features[feature_type][language]
                if feature_name.lower() in func.lower()
            }
            if similar_functions:
                return IMPLEMENTATION_STATUS["PARTIAL"]

        return IMPLEMENTATION_STATUS["MISSING"]


class ReportGenerator:
    """Generates implementation status reports"""

    def __init__(self):
        self.analyzer = CrossGLAnalyzer()
        self.analysis_dir = Path(".cache/analysis")
        self.reports_dir = Path(".cache/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load analysis data
        self.analysis_data = self._load_analysis_data()

    def _load_analysis_data(self) -> Dict[str, Dict]:
        """Load all analysis data files"""
        data = {}

        for analysis_file in self.analysis_dir.glob("*.json"):
            language = analysis_file.stem
            with open(analysis_file, "r") as f:
                data[language] = json.load(f)

        return data

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary of implementation status across all languages"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "languages": {},
            "overall_progress": 0.0,
        }

        total_features = 0
        total_implemented = 0

        for language, language_data in self.analysis_data.items():
            language_summary = {
                "name": LANGUAGE_SETTINGS[language]["name"],
                "categories": {},
                "progress": 0.0,
                "total_features": 0,
                "implemented_features": 0,
                "partial_features": 0,
                "missing_features": 0,
            }

            language_total = 0
            language_implemented = 0
            language_partial = 0

            for category in FEATURE_CATEGORIES:
                if category not in language_data:
                    continue

                features = language_data[category]
                category_total = len(features)
                category_implemented = 0
                category_partial = 0
                category_missing = 0

                for feature in features:
                    status = self.analyzer.get_implementation_status(
                        category, feature["name"], language
                    )
                    if status == IMPLEMENTATION_STATUS["COMPLETE"]:
                        category_implemented += 1
                    elif status == IMPLEMENTATION_STATUS["PARTIAL"]:
                        category_partial += 1
                    else:
                        category_missing += 1

                # Calculate progress (complete = 1.0, partial = 0.5)
                category_progress = (
                    (category_implemented + 0.5 * category_partial) / category_total
                    if category_total > 0
                    else 1.0
                )

                language_summary["categories"][category] = {
                    "total": category_total,
                    "implemented": category_implemented,
                    "partial": category_partial,
                    "missing": category_missing,
                    "progress": category_progress,
                }

                language_total += category_total
                language_implemented += category_implemented
                language_partial += category_partial

            # Calculate language progress
            language_summary["total_features"] = language_total
            language_summary["implemented_features"] = language_implemented
            language_summary["partial_features"] = language_partial
            language_summary["missing_features"] = (
                language_total - language_implemented - language_partial
            )
            language_summary["progress"] = (
                (language_implemented + 0.5 * language_partial) / language_total
                if language_total > 0
                else 1.0
            )

            summary["languages"][language] = language_summary

            total_features += language_total
            total_implemented += language_implemented + 0.5 * language_partial

        # Calculate overall progress
        summary["overall_progress"] = (
            total_implemented / total_features if total_features > 0 else 1.0
        )
        summary["total_features"] = total_features
        summary["total_implemented"] = total_implemented

        return summary

    def generate_detailed_report(self, language: str) -> Dict[str, Any]:
        """Generate a detailed report for a specific language"""
        if language not in self.analysis_data:
            logger.warning(f"No analysis data found for {language}")
            return {}

        language_data = self.analysis_data[language]

        detailed_report = {
            "language": language,
            "name": LANGUAGE_SETTINGS[language]["name"],
            "timestamp": datetime.now().isoformat(),
            "categories": {},
        }

        for category in FEATURE_CATEGORIES:
            if category not in language_data:
                continue

            features = language_data[category]
            feature_details = []

            for feature in features:
                status = self.analyzer.get_implementation_status(
                    category, feature["name"], language
                )
                status_text = (
                    "Implemented"
                    if status == IMPLEMENTATION_STATUS["COMPLETE"]
                    else (
                        "Partial"
                        if status == IMPLEMENTATION_STATUS["PARTIAL"]
                        else "Missing"
                    )
                )

                feature_details.append(
                    {
                        "name": feature["name"],
                        "status": status_text,
                        "status_code": status,
                        "description": feature.get("description", ""),
                        "signature": feature.get("signature", ""),
                        "url": feature.get("url", ""),
                    }
                )

            # Sort by status (missing first, then partial, then implemented)
            feature_details.sort(key=lambda f: f["status_code"])

            detailed_report["categories"][category] = feature_details

        return detailed_report

    def generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Generate an HTML report from the summary data"""
        template_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CrossGL Translator Implementation Status</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #444; }
                .progress-bar { background-color: #f1f1f1; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
                .progress-bar-fill { height: 100%; background-color: #4CAF50; float: left; }
                .progress-bar-partial { height: 100%; background-color: #FFC107; float: left; }
                .summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .summary-table th, .summary-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .summary-table th { background-color: #f2f2f2; }
                .summary-table tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { width: 100%; height: 400px; margin: 20px 0; }
                .category-section { margin-top: 30px; }
                .timestamp { color: #888; font-size: 0.8em; margin-top: 40px; }
            </style>
        </head>
        <body>
            <h1>CrossGL Translator Implementation Status</h1>
            <p>Overall implementation progress: {{ "%.1f"|format(summary.overall_progress * 100) }}%</p>
            
            <div class="progress-bar">
                <div class="progress-bar-fill" style="width: {{ summary.overall_progress * 100 }}%;"></div>
            </div>
            
            <h2>Summary by Language</h2>
            <table class="summary-table">
                <tr>
                    <th>Language</th>
                    <th>Progress</th>
                    <th>Total Features</th>
                    <th>Implemented</th>
                    <th>Partial</th>
                    <th>Missing</th>
                </tr>
                {% for lang_code, lang_data in summary.languages.items() %}
                <tr>
                    <td><a href="details_{{ lang_code }}.html">{{ lang_data.name }}</a></td>
                    <td>
                        <div class="progress-bar" style="margin: 0;">
                            <div class="progress-bar-fill" style="width: {{ lang_data.implemented_features / lang_data.total_features * 100 if lang_data.total_features > 0 else 0 }}%;"></div>
                            <div class="progress-bar-partial" style="width: {{ lang_data.partial_features / lang_data.total_features * 100 if lang_data.total_features > 0 else 0 }}%;"></div>
                        </div>
                        {{ "%.1f"|format(lang_data.progress * 100) }}%
                    </td>
                    <td>{{ lang_data.total_features }}</td>
                    <td>{{ lang_data.implemented_features }}</td>
                    <td>{{ lang_data.partial_features }}</td>
                    <td>{{ lang_data.missing_features }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Implementation by Category</h2>
            {% for lang_code, lang_data in summary.languages.items() %}
            <div class="category-section">
                <h3>{{ lang_data.name }}</h3>
                <table class="summary-table">
                    <tr>
                        <th>Category</th>
                        <th>Progress</th>
                        <th>Total</th>
                        <th>Implemented</th>
                        <th>Partial</th>
                        <th>Missing</th>
                    </tr>
                    {% for cat_name, cat_data in lang_data.categories.items() %}
                    <tr>
                        <td>{{ cat_name.title() }}</td>
                        <td>
                            <div class="progress-bar" style="margin: 0;">
                                <div class="progress-bar-fill" style="width: {{ cat_data.implemented / cat_data.total * 100 if cat_data.total > 0 else 0 }}%;"></div>
                                <div class="progress-bar-partial" style="width: {{ cat_data.partial / cat_data.total * 100 if cat_data.total > 0 else 0 }}%;"></div>
                            </div>
                            {{ "%.1f"|format(cat_data.progress * 100) }}%
                        </td>
                        <td>{{ cat_data.total }}</td>
                        <td>{{ cat_data.implemented }}</td>
                        <td>{{ cat_data.partial }}</td>
                        <td>{{ cat_data.missing }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endfor %}
            
            <p class="timestamp">Report generated on {{ summary.timestamp }}</p>
        </body>
        </html>
        """

        template = Template(template_html)
        return template.render(summary=summary)

    def generate_detailed_html_report(
        self, language: str, detailed_report: Dict[str, Any]
    ) -> str:
        """Generate a detailed HTML report for a specific language"""
        template_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ report.name }} Implementation Details</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #444; }
                .nav-links { margin: 20px 0; }
                .feature-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                .feature-table th, .feature-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .feature-table th { background-color: #f2f2f2; }
                .feature-table tr:nth-child(even) { background-color: #f9f9f9; }
                .status-implemented { color: #4CAF50; }
                .status-partial { color: #FF9800; }
                .status-missing { color: #F44336; }
                .timestamp { color: #888; font-size: 0.8em; margin-top: 40px; }
                .signature { font-family: monospace; background-color: #f8f8f8; padding: 8px; border-radius: 4px; white-space: pre-wrap; }
                .description { max-width: 400px; }
            </style>
        </head>
        <body>
            <div class="nav-links">
                <a href="index.html">← Back to Summary</a>
            </div>
            
            <h1>{{ report.name }} Implementation Details</h1>
            
            {% for category, features in report.categories.items() %}
            <h2>{{ category.title() }}</h2>
            <table class="feature-table">
                <tr>
                    <th>Feature</th>
                    <th>Status</th>
                    <th>Signature</th>
                    <th>Description</th>
                </tr>
                {% for feature in features %}
                <tr>
                    <td>
                        {% if feature.url %}
                        <a href="{{ feature.url }}" target="_blank">{{ feature.name }}</a>
                        {% else %}
                        {{ feature.name }}
                        {% endif %}
                    </td>
                    <td class="status-{{ feature.status.lower() }}">{{ feature.status }}</td>
                    <td>
                        {% if feature.signature %}
                        <div class="signature">{{ feature.signature }}</div>
                        {% endif %}
                    </td>
                    <td class="description">{{ feature.description }}</td>
                </tr>
                {% endfor %}
            </table>
            {% endfor %}
            
            <p class="timestamp">Report generated on {{ report.timestamp }}</p>
        </body>
        </html>
        """

        template = Template(template_html)
        return template.render(report=detailed_report)

    def generate_visualizations(self, summary: Dict[str, Any]):
        """Generate visualization charts for the reports"""
        reports_dir = self.reports_dir

        # 1. Overall progress bar chart by language
        languages = []
        implemented = []
        partial = []
        missing = []

        for lang_code, lang_data in summary["languages"].items():
            languages.append(LANGUAGE_SETTINGS[lang_code]["name"])
            implemented.append(lang_data["implemented_features"])
            partial.append(lang_data["partial_features"])
            missing.append(lang_data["missing_features"])

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        bar_width = 0.8
        ind = np.arange(len(languages))

        p1 = ax.bar(ind, implemented, bar_width, color="#4CAF50", label="Implemented")
        p2 = ax.bar(
            ind,
            partial,
            bar_width,
            bottom=implemented,
            color="#FFC107",
            label="Partial",
        )
        p3 = ax.bar(
            ind,
            missing,
            bar_width,
            bottom=np.array(implemented) + np.array(partial),
            color="#F44336",
            label="Missing",
        )

        ax.set_title("Implementation Status by Language")
        ax.set_ylabel("Number of Features")
        ax.set_xticks(ind)
        ax.set_xticklabels(languages, rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()
        plt.savefig(reports_dir / "implementation_by_language.png", dpi=300)
        plt.close()

        # 2. Progress by category (radar chart)
        categories = FEATURE_CATEGORIES

        # Prepare data
        fig = plt.figure(figsize=(10, 8))

        # Number of variables
        N = len(categories)

        # Compute angles for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Initialize the radar plot
        ax = plt.subplot(111, polar=True)

        # Draw progress for each language
        for lang_code, lang_data in summary["languages"].items():
            values = []
            for category in categories:
                if category in lang_data["categories"]:
                    values.append(lang_data["categories"][category]["progress"])
                else:
                    values.append(0)

            # Close the loop
            values += values[:1]

            # Plot values
            ax.plot(
                angles,
                values,
                linewidth=2,
                linestyle="solid",
                label=LANGUAGE_SETTINGS[lang_code]["name"],
            )
            ax.fill(angles, values, alpha=0.1)

        # Set ticks and labels
        plt.xticks(angles[:-1], categories)

        # Add legend
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

        plt.savefig(reports_dir / "category_radar_chart.png", dpi=300)
        plt.close()

    def generate_all_reports(self):
        """Generate all reports and visualizations"""
        logger.info("Generating implementation status reports...")

        # Generate summary report
        summary = self.generate_summary_report()

        # Generate HTML summary report
        summary_html = self.generate_html_report(summary)
        with open(self.reports_dir / "index.html", "w") as f:
            f.write(summary_html)

        # Generate detailed reports for each language
        for language in self.analysis_data.keys():
            detailed_report = self.generate_detailed_report(language)
            detailed_html = self.generate_detailed_html_report(
                language, detailed_report
            )

            with open(self.reports_dir / f"details_{language}.html", "w") as f:
                f.write(detailed_html)

        # Generate visualizations
        self.generate_visualizations(summary)

        # Save summary data as JSON for further processing
        with open(self.reports_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Reports generated successfully. Available at {self.reports_dir}")


def main():
    """Main entry point"""
    try:
        report_generator = ReportGenerator()
        report_generator.generate_all_reports()
        return 0
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
