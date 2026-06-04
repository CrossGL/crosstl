"""Repo-scale CrossTL project porting APIs."""

from .pipeline import (
    ProjectConfig,
    ProjectDiagnostic,
    ProjectPortabilityReport,
    ProjectScan,
    ProjectTranslationUnit,
    inspect_project_report,
    load_project_config,
    scan_project,
    translate_project,
    validate_project_report,
)

__all__ = [
    "ProjectConfig",
    "ProjectDiagnostic",
    "ProjectPortabilityReport",
    "ProjectScan",
    "ProjectTranslationUnit",
    "inspect_project_report",
    "load_project_config",
    "scan_project",
    "translate_project",
    "validate_project_report",
]
