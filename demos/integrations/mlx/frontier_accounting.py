"""Check the MLX project-porting frontier accounting after ``run_porting`` (CI script).

The ``mlx-metal-porting`` job runs this with ``python -m demos.integrations.mlx.frontier_accounting``
from the repository root once the porting reports exist. It reads ``MLX_COMMIT`` and ``RUNNER_OS``
from the environment and exits non-zero when a summary, frontier or toolchain count is wrong.
Running it is its only use: the checks execute at import time, like the workflow step it replaced.
"""

import json
import os
from pathlib import Path

from demos.integrations.mlx.run_porting import (
    GEMV_DIRECTX_COMPILER_ENTRY_POINTS,
    GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES,
    GEMV_DIRECTX_EXPECTED_ENTRY_POINTS,
    GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE,
    GEMV_EXPECTED_ENTRY_POINT_COUNT,
    GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES,
    GEMV_OPENGL_INDEX_RANGE_ASSERTIONS,
    GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES,
    GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT,
    GEMV_REPORT_WORKGROUP_SIZE_RULE,
    GEMV_SUBGROUP_WIDTH,
    GEMV_WORKGROUP_SIZE_RULE,
    MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE,
    MLX_DIRECTX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS,
    MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
    MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT,
    MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS,
    MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES,
    MLX_DIRECTX_VULKAN_FRONTIER_SOURCES,
    MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE,
    MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE,
    MLX_GEMV_SHA256,
    MLX_GEMV_SOURCE,
    MLX_GEMV_SOURCE_SIZE_BYTES,
    MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE,
    MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY,
    MLX_LAYER_NORM_DISPATCH_VARIANTS,
    MLX_LAYER_NORM_SHA256,
    MLX_LAYER_NORM_SOURCE,
    MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY,
    MLX_LOGSUMEXP_DISPATCH_VARIANTS,
    MLX_LOGSUMEXP_SHA256,
    MLX_LOGSUMEXP_SOURCE,
    MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES,
    MLX_OPENGL_FRONTIER_SOURCES,
    MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS,
    MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
    MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
    MLX_OPENGL_INDEX_RANGE_ASSERTIONS,
    MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES,
    MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES,
    MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY,
    MLX_RMS_NORM_DISPATCH_VARIANTS,
    MLX_RMS_NORM_SHA256,
    MLX_RMS_NORM_SOURCE,
    MLX_VULKAN_FRONTIER_SOURCES,
)

fence_source = "mlx/backend/metal/kernels/fence.metal"
fence_issue = "https://github.com/CrossGL/crosstl/issues/1537"
directx_frontier_sources = set(MLX_DIRECTX_TOOLCHAIN_FRONTIER_SOURCES)
directx_entry_point_counts = dict(MLX_DIRECTX_TOOLCHAIN_ENTRY_POINT_COUNTS)
directx_artifact_count = MLX_DIRECTX_TOOLCHAIN_ARTIFACT_COUNT
directx_entry_point_count = sum(directx_entry_point_counts.values())
directx_blocked_sources = set(MLX_DIRECTX_DYNAMIC_WORKGROUP_FRONTIER_SOURCES)
directx_blocked_entry_point_counts = dict(
    MLX_DIRECTX_DYNAMIC_WORKGROUP_ENTRY_POINT_COUNTS
)
directx_blocked_entry_point_count = sum(directx_blocked_entry_point_counts.values())
vulkan_frontier_sources = set(MLX_DIRECTX_VULKAN_FRONTIER_SOURCES)
# Refused sources are checked by run_porting against their tracked diagnostics.
vulkan_translated_sources = set(MLX_VULKAN_FRONTIER_SOURCES)
gemv_workgroup_size_rule = list(GEMV_WORKGROUP_SIZE_RULE)
gemv_report_workgroup_size_rule = list(GEMV_REPORT_WORKGROUP_SIZE_RULE)
gemv_resolved_workgroup_sizes = list(GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES)
opengl_frontier_sources = set(MLX_OPENGL_FRONTIER_SOURCES)
opengl_frontier_count = len(opengl_frontier_sources)
opengl_toolchain_frontier_sources = set(MLX_OPENGL_TOOLCHAIN_FRONTIER_SOURCES)
opengl_toolchain_frontier_count = len(opengl_toolchain_frontier_sources)
opengl_blocked_sources = set(MLX_OPENGL_DYNAMIC_WORKGROUP_FRONTIER_SOURCES)
required_opengl_promotions = {
    "mlx/backend/metal/kernels/binary_two.metal",
    "mlx/backend/metal/kernels/ternary.metal",
}
if not required_opengl_promotions <= opengl_toolchain_frontier_sources:
    raise SystemExit("required OpenGL frontier promotions are missing")
if opengl_frontier_count != 8:
    raise SystemExit("expected 8 OpenGL attempted frontier sources")
if opengl_toolchain_frontier_count != 3:
    raise SystemExit("expected 3 OpenGL toolchain frontier sources")
payload = json.loads(
    Path("mlx-upstream/.crosstl-mlx-porting/summary.json").read_text(encoding="utf-8")
)
expected_repository = {
    "name": "ml-explore/mlx",
    "url": "https://github.com/ml-explore/mlx",
    "commit": os.environ["MLX_COMMIT"],
}
if (
    payload.get("schema_version") != 1
    or payload.get("repository") != expected_repository
    or payload.get("status") != "passed"
):
    raise SystemExit("MLX summary identity or status is incorrect")
scope = payload["scope"]
if set(scope["nonFenceFrontierSources"]) != vulkan_frontier_sources:
    raise SystemExit("expected the exact 11-source non-fence MLX frontier")
if scope["blockedFrontierSources"] != [fence_source]:
    raise SystemExit("fence.metal must be the sole blocked frontier source")
if scope["blockedFrontierIssues"] != [fence_issue]:
    raise SystemExit("blocked frontier must track issue #1537")
if (
    set(scope["directxTranslatedFrontierSources"]) != directx_frontier_sources
    or scope["directxTranslatedFrontierArtifactCount"] != directx_artifact_count
    or set(scope["directxWorkgroupBlockedFrontierSources"]) != directx_blocked_sources
    or set(scope["vulkanTranslatedFrontierSources"]) != vulkan_translated_sources
    or set(scope["openglTranslatedFrontierSources"])
    != set(MLX_OPENGL_TRANSLATED_FRONTIER_SOURCES)
    or set(scope["openglWorkgroupBlockedFrontierSources"]) != opengl_blocked_sources
    or scope["hostDispatchImportResolvedIssue"]
    != MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
):
    raise SystemExit("target-split frontier scope accounting is incomplete")
check_records = payload.get("checks")
if not isinstance(check_records, list) or not all(
    isinstance(check, dict) and isinstance(check.get("name"), str)
    for check in check_records
):
    raise SystemExit("MLX summary checks are malformed")
checks = {check["name"]: check for check in check_records}
if len(checks) != len(check_records):
    raise SystemExit("MLX summary check names must be unique")
checkout = checks.get("mlx-checkout")
if (
    not isinstance(checkout, dict)
    or checkout.get("status") != "passed"
    or checkout.get("repository") != expected_repository["url"]
    or checkout.get("commit") != expected_repository["commit"]
):
    raise SystemExit("MLX checkout proof does not match the pinned revision")
fence = checks["atomic-fence-contract"]
if fence["failedArtifactCount"] != 3 or fence["emittedArtifactCount"] != 0:
    raise SystemExit("fence contract accounting must be 3 failed, 0 emitted")
reference_accessor = checks["reference-accessor-lvalue-identity"]
expected_directx_required = os.environ["RUNNER_OS"] == "Windows"
expected_opengl_required = os.environ["RUNNER_OS"] == "Linux"
if (
    scope["referenceAccessorProofIncluded"] is not True
    or scope["referenceAccessorTargets"] != ["directx", "opengl"]
    or (
        scope["referenceAccessorDirectxToolchainRequired"]
        is not expected_directx_required
    )
    or (
        scope["referenceAccessorOpenglToolchainRequired"]
        is not expected_opengl_required
    )
    or reference_accessor["proofStatus"] != "verified-original-storage-write"
    or reference_accessor["constReadProofStatus"]
    != "verified-original-storage-const-read"
    or reference_accessor["nestedConstAliasProofStatus"]
    != "verified-original-nested-storage-const-alias-read"
    or reference_accessor["artifactCount"] != 2
    or reference_accessor["projectDiagnosticCount"] != 0
    or reference_accessor["runtimeParityClaimed"] is not False
    or reference_accessor["upstreamMlxRuntimeExecuted"] is not False
):
    raise SystemExit("reference accessor proof accounting is incomplete")
expected_native_target = {
    "Linux": "opengl",
    "Windows": "directx",
}.get(os.environ["RUNNER_OS"])
for target in ("directx", "opengl"):
    proof = reference_accessor["targetProofs"][target]
    evidence = proof["writeEvidence"]
    const_read = proof["constReadEvidence"]
    nested_const_alias = proof["nestedConstAliasEvidence"]
    if (
        evidence["status"] != "verified-original-storage-write"
        or evidence["storageMember"] != "val_frags"
        or evidence["readBackFromWrittenLvalue"] is not True
        or evidence["valueReturningHelperUsedForWrite"] is not False
    ):
        raise SystemExit(f"reference accessor {target} storage evidence is incomplete")
    if (
        const_read["status"] != "verified-original-storage-const-read"
        or const_read["storageMember"] != "val_frags"
        or const_read["implicitReceiver"] != "self"
        or const_read["passedDirectlyToHelper"] is not True
        or const_read["accessorCallEliminated"] is not True
        or const_read["kernelPathInvoked"] is not True
    ):
        raise SystemExit(
            f"reference accessor {target} const-read evidence is incomplete"
        )
    if (
        nested_const_alias["status"]
        != "verified-original-nested-storage-const-alias-read"
        or nested_const_alias["storageMember"] != "val_frags"
        or nested_const_alias["storagePath"] != "self.nestedTile.val_frags"
        or nested_const_alias["outerReceiver"] != "self"
        or nested_const_alias["tileMember"] != "nestedTile"
        or nested_const_alias["fragmentType"] != "float2"
        or nested_const_alias["aliasName"] != "accum"
        or nested_const_alias["aliasEliminated"] is not True
        or nested_const_alias["accessorCallEliminated"] is not True
        or nested_const_alias["indexedAliasRead"] != "accum[k]"
        or nested_const_alias["readFromOriginalStorage"] is not True
        or nested_const_alias["kernelPathInvoked"] is not True
    ):
        raise SystemExit(
            f"reference accessor {target} nested const-alias evidence is incomplete"
        )
    expected_status = (
        "validated" if target == expected_native_target else "not-required"
    )
    expected_required = expected_status == "validated"
    native_validation = proof["nativeValidation"]
    expected_compiler = {
        "directx": "dxc",
        "opengl": "glslangValidator",
    }[target]
    if (
        native_validation["status"] != expected_status
        or native_validation["required"] is not expected_required
        or native_validation["nativeCompiler"] != expected_compiler
        or (target == "opengl" and native_validation["spirvValidator"] != "spirv-val")
    ):
        raise SystemExit(
            f"reference accessor {target} native validation must be "
            f"{expected_status} on {os.environ['RUNNER_OS']}"
        )
directx = checks["directx-frontier"]
if (
    directx["status"] != "passed-with-bounded-dispatch-and-pending-contracts"
    or set(directx["sources"]) != vulkan_frontier_sources
    or directx["artifactCount"] != directx_artifact_count + len(directx_blocked_sources)
    or set(directx["translatedSources"]) != directx_frontier_sources
    or directx["translatedArtifactCount"] != directx_artifact_count
    or set(directx["workgroupBlockedSources"]) != directx_blocked_sources
    or directx["workgroupBlockedArtifactCount"] != len(directx_blocked_sources)
    or set(directx["directxToolchainSources"]) != directx_frontier_sources
    or directx["directxToolchainArtifactCount"] != directx_artifact_count
    or directx["directxToolchainExpectedEntryPointCounts"] != directx_entry_point_counts
    or directx["directxToolchainExpectedEntryPointCount"] != directx_entry_point_count
    or directx["trackedIssues"] != []
    or directx["resolvedIssues"] != [MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE]
):
    raise SystemExit("DirectX frontier accounting is incomplete")
if set(directx["dynamicWorkgroupDispatchEvidence"]) != directx_blocked_sources:
    raise SystemExit("DirectX workgroup blocker source join is incomplete")
if directx["bfloat16LoweringEvidence"] != MLX_DIRECTX_BFLOAT16_LOWERING_EVIDENCE:
    raise SystemExit("DirectX bfloat16 lowering evidence is incomplete")
for source in directx_blocked_sources:
    expected_evidence = MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source]
    observed_evidence = directx["dynamicWorkgroupDispatchEvidence"][source]
    if (
        any(
            observed_evidence.get(key) != value
            for key, value in expected_evidence.items()
        )
        or observed_evidence["entryPointCount"]
        != directx_blocked_entry_point_counts[source]
        or observed_evidence["diagnosticCode"] != MLX_DYNAMIC_WORKGROUP_DIAGNOSTIC_CODE
        or observed_evidence["artifactStatus"] != "failed"
        or observed_evidence["artifactEmitted"] is not False
        or observed_evidence["sourceEntryPointIdentityStatus"]
        != "matched-materialized-host-names"
    ):
        raise SystemExit(f"DirectX workgroup blocker evidence changed for {source}")
if directx_blocked_entry_point_count != 76:
    raise SystemExit("expected 76 fail-closed DirectX compute entries")
layer_norm = directx["layerNormDispatchEvidence"]
expected_layer_norm_status = (
    "translated-dxc-validated" if os.environ["RUNNER_OS"] == "Windows" else "translated"
)
expected_layer_norm_validated = (
    len(MLX_LAYER_NORM_DISPATCH_VARIANTS) if os.environ["RUNNER_OS"] == "Windows" else 0
)
if (
    layer_norm["status"] != expected_layer_norm_status
    or layer_norm["source"] != MLX_LAYER_NORM_SOURCE
    or layer_norm["sourceSha256"] != MLX_LAYER_NORM_SHA256
    or layer_norm["target"] != "directx"
    or layer_norm["dispatchContract"]["contentIdentity"]
    != MLX_LAYER_NORM_DISPATCH_CONTENT_IDENTITY
    or layer_norm["dispatchContract"]["variantCount"]
    != len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
    or layer_norm["dispatchContract"]["resolvedIssue"]
    != MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    or layer_norm["artifactCount"] != len(MLX_LAYER_NORM_DISPATCH_VARIANTS)
    or set(layer_norm["variants"]) != set(MLX_LAYER_NORM_DISPATCH_VARIANTS)
    or layer_norm["dxcValidatedArtifactCount"] != expected_layer_norm_validated
    or layer_norm["runtimeExecutionAttempted"] is not False
    or layer_norm["numericalParityClaimed"] is not False
):
    raise SystemExit("LayerNorm dispatch frontier evidence is incomplete")
logsumexp = directx["logsumexpDispatchEvidence"]
expected_logsumexp_status = (
    "translated-dxc-validated" if os.environ["RUNNER_OS"] == "Windows" else "translated"
)
expected_logsumexp_validated = (
    len(MLX_LOGSUMEXP_DISPATCH_VARIANTS) if os.environ["RUNNER_OS"] == "Windows" else 0
)
if (
    logsumexp["status"] != expected_logsumexp_status
    or logsumexp["source"] != MLX_LOGSUMEXP_SOURCE
    or logsumexp["sourceSha256"] != MLX_LOGSUMEXP_SHA256
    or logsumexp["target"] != "directx"
    or logsumexp["testSources"]
    != [
        "python/tests/test_ops.py::test_logsumexp",
        "python/tests/test_autograd.py::test_logsumexp_grad",
    ]
    or logsumexp["dispatchContract"]["contentIdentity"]
    != MLX_LOGSUMEXP_DISPATCH_CONTENT_IDENTITY
    or logsumexp["dispatchContract"]["variantCount"]
    != len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    or logsumexp["dispatchContract"]["resolvedIssue"]
    != MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    or logsumexp["artifactCount"] != len(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    or set(logsumexp["variants"]) != set(MLX_LOGSUMEXP_DISPATCH_VARIANTS)
    or logsumexp["dxcValidatedArtifactCount"] != expected_logsumexp_validated
    or logsumexp["runtimeExecutionAttempted"] is not False
    or logsumexp["numericalParityClaimed"] is not False
):
    raise SystemExit("LogSumExp dispatch frontier evidence is incomplete")
rms_norm = directx["rmsNormDispatchEvidence"]
expected_rms_norm_status = (
    "translated-dxc-validated" if os.environ["RUNNER_OS"] == "Windows" else "translated"
)
expected_rms_norm_validated = (
    len(MLX_RMS_NORM_DISPATCH_VARIANTS) if os.environ["RUNNER_OS"] == "Windows" else 0
)
if (
    rms_norm["status"] != expected_rms_norm_status
    or rms_norm["source"] != MLX_RMS_NORM_SOURCE
    or rms_norm["sourceSha256"] != MLX_RMS_NORM_SHA256
    or rms_norm["target"] != "directx"
    or rms_norm["testSources"]
    != [
        "python/tests/test_fast.py::test_rms_norm",
        "python/tests/test_fast.py::test_rms_norm_grad",
    ]
    or rms_norm["dispatchContract"]["contentIdentity"]
    != MLX_RMS_NORM_DISPATCH_CONTENT_IDENTITY
    or rms_norm["dispatchContract"]["variantCount"]
    != len(MLX_RMS_NORM_DISPATCH_VARIANTS)
    or rms_norm["dispatchContract"]["resolvedIssue"]
    != MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE
    or rms_norm["artifactCount"] != len(MLX_RMS_NORM_DISPATCH_VARIANTS)
    or set(rms_norm["variants"]) != set(MLX_RMS_NORM_DISPATCH_VARIANTS)
    or rms_norm["dxcValidatedArtifactCount"] != expected_rms_norm_validated
    or rms_norm["runtimeExecutionAttempted"] is not False
    or rms_norm["numericalParityClaimed"] is not False
):
    raise SystemExit("RMSNorm dispatch frontier evidence is incomplete")
if os.environ["RUNNER_OS"] == "Windows":
    if (
        directx["directxToolchainRequired"] is not True
        or directx["directxValidationStatus"] != "validated"
        or directx["directxToolchainValidatedArtifactCount"] != directx_artifact_count
        or set(directx["directxToolchainValidatedSources"]) != directx_frontier_sources
        or directx["directxToolchainValidatedEntryPointCounts"]
        != directx_entry_point_counts
        or directx["directxToolchainValidatedEntryPointCount"]
        != directx_entry_point_count
        or directx["toolchainRuns"] != directx_entry_point_count
    ):
        raise SystemExit(
            "DirectX frontier toolchain must validate every configured "
            "source artifact and compute entry"
        )
vulkan = checks["vulkan-frontier"]
if (
    vulkan["status"] != "passed"
    or set(vulkan["sources"]) != vulkan_translated_sources
    or vulkan["unitCount"] != len(vulkan_translated_sources)
    or vulkan["artifactCount"] != len(vulkan_translated_sources)
    or vulkan["translatedArtifactCount"] != len(vulkan_translated_sources)
    or vulkan["target"] != "vulkan"
):
    raise SystemExit("Vulkan frontier accounting is incomplete")
if os.environ["RUNNER_OS"] == "Linux" and (
    vulkan["vulkanToolchainRequired"] is not True
    or vulkan["vulkanValidationStatus"] != "validated"
    or vulkan["toolchainRuns"] < len(vulkan_translated_sources)
):
    raise SystemExit("Vulkan frontier toolchain validation is incomplete")
expected_directx_gemv = os.environ["RUNNER_OS"] == "Windows"
if scope["directxGemvCompilerFrontierRequired"] is not expected_directx_gemv:
    raise SystemExit("DirectX GEMV compiler-frontier scope is incorrect")
if expected_directx_gemv:
    gemv_directx = checks["gemv-directx-compiler-frontier"]
    library_run = gemv_directx["libraryCompilerRun"]
    if (
        gemv_directx["status"] != "passed"
        or gemv_directx["source"] != MLX_GEMV_SOURCE
        or gemv_directx["sourceHash"] != MLX_GEMV_SHA256
        or gemv_directx["sourceSizeBytes"] != MLX_GEMV_SOURCE_SIZE_BYTES
        or gemv_directx["artifactStatus"] != "translated"
        or gemv_directx["artifactPackaging"] != "single-aggregate-artifact"
        or gemv_directx["projectDiagnosticCount"] != 0
        or gemv_directx["templateSpecializationCount"] != 226
        or gemv_directx["hostNamedMaterializationCount"] != 224
        or gemv_directx["unsupportedSpecializationCount"] != 0
        or gemv_directx["workgroupSizeRule"] != gemv_workgroup_size_rule
        or gemv_directx["reportWorkgroupSizeRule"] != gemv_report_workgroup_size_rule
        or gemv_directx["reportWorkgroupSizeRuleCount"] != 1
        or gemv_directx["reportExecutionEntryCount"] != 224
        or gemv_directx["executionIdentityJoinCount"] != 224
        or gemv_directx["generatedTargetEntryIdentityCount"] != 224
        or gemv_directx["generatedNumthreadsContractCount"] != 224
        or gemv_directx["materializationResidueCount"] != 0
        or gemv_directx["bareValueDiscardCount"] != 0
        or gemv_directx["computeEntryPointCount"] != 224
        or gemv_directx["artifactProvenance"] != {
            "intermediate": "crossgl",
            "pipeline": "single-file-translate",
        }
        or gemv_directx["artifactHash"]["algorithm"] != "sha256"
        or gemv_directx["artifactSizeBytes"] <= 0
    ):
        raise SystemExit("DirectX GEMV project-artifact evidence is incomplete")
    resolved_size_counts = {
        tuple(item["workgroupSize"]): item["entryCount"]
        for item in gemv_directx["resolvedWorkgroupSizeCounts"]
    }
    if (
        len(gemv_directx["resolvedWorkgroupSizeCounts"])
        != len(gemv_resolved_workgroup_sizes)
        or gemv_directx["resolvedWorkgroupSizes"]
        != [list(size) for size in gemv_resolved_workgroup_sizes]
        or set(resolved_size_counts) != set(gemv_resolved_workgroup_sizes)
        or sum(resolved_size_counts.values()) != 224
        or any(count <= 0 for count in resolved_size_counts.values())
    ):
        raise SystemExit("DirectX GEMV resolved workgroup-size evidence is incomplete")
    if (
        gemv_directx["entryProfile"] != "cs_6_2"
        or gemv_directx["compilerArguments"] != ["-enable-16bit-types"]
        or gemv_directx["minimumShaderModel"] != "6.2"
        or gemv_directx["entryProfileCompilerEntryPoints"]
        != list(GEMV_DIRECTX_COMPILER_ENTRY_POINTS)
        or gemv_directx["entryProfileCompiledEntryPointCount"] != 3
        or gemv_directx["entryProfileDiagnosticCount"] != 0
        or gemv_directx["entryProfileUnusedValueWarningCount"] != 0
        or any(
            run["profile"] != "cs_6_2"
            or run["compilerArguments"] != ["-enable-16bit-types"]
            or run["minimumShaderModel"] != "6.2"
            for run in gemv_directx["entryProfileCompilerRuns"]
        )
        or gemv_directx["libraryProfile"] != "lib_6_6"
        or gemv_directx["libraryExports"] != list(GEMV_DIRECTX_EXPECTED_ENTRY_POINTS)
        or gemv_directx["libraryExportCount"] != 224
        or library_run["profile"] != "lib_6_6"
        or library_run["compilerArguments"] != ["-enable-16bit-types"]
        or library_run["minimumShaderModel"] != "6.2"
        or library_run["status"] != "compiled"
        or library_run["outputSizeBytes"] <= 0
        or library_run["unusedValueWarningCount"] != 0
        or gemv_directx["libraryUnusedValueWarningCount"] != 0
        or gemv_directx["compilerCoveredEntryPointCount"] != 224
        or gemv_directx["uncompiledEntryPointCount"] != 0
        or gemv_directx["compilerCoverageStatus"]
        != "all-exported-entry-points-compiled"
    ):
        raise SystemExit("DirectX GEMV compiler coverage is incomplete")
    library_warnings = gemv_directx["libraryAllowedWarnings"]
    expected_warning_counts = {
        f"[numthreads({size[0]}, {size[1]}, {size[2]})]": resolved_size_counts[size]
        for size in gemv_resolved_workgroup_sizes
    }
    observed_warning_counts = {}
    for warning in library_warnings:
        source_expression = warning["sourceExpression"]
        if (
            warning["classification"] != "library-profile-numthreads-ignored"
            or warning["severity"] != "warning"
            or warning["message"] != GEMV_DIRECTX_LIBRARY_NUMTHREADS_WARNING_MESSAGE
            or source_expression in observed_warning_counts
        ):
            raise SystemExit("DirectX GEMV compiler warning classification changed")
        observed_warning_counts[source_expression] = warning["count"]
    if observed_warning_counts != expected_warning_counts or library_run[
        "allowedWarningCounts"
    ] != {"libraryNumthreads": 224}:
        raise SystemExit("DirectX GEMV compiler warning evidence changed")
    if (
        gemv_directx["wholeArtifactSemanticValidityClaimed"] is not False
        or gemv_directx["libraryExecutionSemanticsEstablished"] is not False
        or gemv_directx["observedNumthreadsDirectives"] != list(expected_warning_counts)
        or gemv_directx["numthreadsDirectiveCount"] != 224
        or gemv_directx["numthreadsContractEstablished"] is not True
        or gemv_directx["exactWorkgroupSizeEstablished"] is not True
        or gemv_directx["requiredWaveSize"] != 32
        or gemv_directx["requiredWaveSizeEstablished"] is not False
        or gemv_directx["executionContractBlockedBy"]
        != list(GEMV_DIRECTX_EXECUTION_TRACKED_ISSUES)
        or gemv_directx["runtimeExecutionAttempted"] is not False
        or gemv_directx["runtimeIntegrationIncluded"] is not False
        or gemv_directx["numericalParityClaimed"] is not False
        or gemv_directx["runtimeParityClaimed"] is not False
    ):
        raise SystemExit("DirectX GEMV execution non-claims changed")
elif "gemv-directx-compiler-frontier" in checks:
    raise SystemExit("DirectX GEMV compiler frontier ran outside Windows")
expected_opengl_gemv = os.environ["RUNNER_OS"] == "Linux"
if scope["openglGemvToolchainRequired"] is not expected_opengl_gemv:
    raise SystemExit("OpenGL GEMV toolchain scope is incorrect")
if expected_opengl_gemv:
    gemv_opengl = checks["gemv-opengl-toolchain"]
    if (
        gemv_opengl["status"] != "passed"
        or gemv_opengl["source"] != MLX_GEMV_SOURCE
        or gemv_opengl["sourceHash"] != MLX_GEMV_SHA256
        or gemv_opengl["artifactStatus"] != "translated"
        or gemv_opengl["artifactCount"] != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["emittedTargetFileCount"] != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["reportExecutionEntryCount"] != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["sourceMapCount"] != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["sourceRemapCount"] != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["nativeValidationAttempted"] is not True
        or gemv_opengl["nativeValidationStatus"] != "validated"
        or gemv_opengl["toolchainValidatedArtifactCount"]
        != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or len(gemv_opengl["toolchainValidationRuns"])
        != GEMV_EXPECTED_ENTRY_POINT_COUNT
        or gemv_opengl["workgroupSizeRule"] != gemv_workgroup_size_rule
        or gemv_opengl["reportWorkgroupSizeRule"] != gemv_report_workgroup_size_rule
        or gemv_opengl["reportWorkgroupSizeRuleCount"] != 1
        or gemv_opengl["workgroupSizeRuleConfigured"] is not True
        or gemv_opengl["resolvedWorkgroupSizes"]
        != [list(size) for size in GEMV_EXPECTED_RESOLVED_WORKGROUP_SIZES]
        or gemv_opengl["subgroupWidth"] != GEMV_SUBGROUP_WIDTH
        or gemv_opengl["subgroupWidthRuleConfigured"] is not True
        or gemv_opengl["subgroupWidthEnforcement"]
        != GEMV_OPENGL_SUBGROUP_WIDTH_ENFORCEMENT
        or gemv_opengl["indexRangeAssertionEvidence"]["assertionCount"]
        != len(GEMV_OPENGL_INDEX_RANGE_ASSERTIONS)
        or gemv_opengl["indexRangeAssertionEvidence"]["runtimeEnforced"] is not False
        or gemv_opengl["translationBlockedBy"] != []
        or gemv_opengl["executionContractBlockedBy"] != []
        or gemv_opengl["trackedIssues"] != list(GEMV_OPENGL_PORTABILITY_TRACKED_ISSUES)
        or gemv_opengl["runtimeExecutionAttempted"] is not False
        or gemv_opengl["runtimeIntegrationIncluded"] is not False
        or gemv_opengl["runnableArtifactClaimed"] is not False
        or gemv_opengl["compilerValidatedArtifactClaimed"] is not True
        or gemv_opengl["numericalParityClaimed"] is not False
        or gemv_opengl["runtimeParityClaimed"] is not False
    ):
        raise SystemExit("OpenGL GEMV toolchain evidence is incomplete")
elif "gemv-opengl-toolchain" in checks:
    raise SystemExit("OpenGL GEMV toolchain ran outside Linux")
opengl = checks["opengl-frontier"]
if (
    opengl["status"] != "passed-with-expected-workgroup-blockers"
    or set(opengl["sources"]) != opengl_frontier_sources
    or opengl["sourceCount"] != opengl_frontier_count
    or opengl["artifactCount"] != opengl_frontier_count
    or set(opengl["translatedSources"]) != opengl_toolchain_frontier_sources
    or opengl["translatedArtifactCount"] != opengl_toolchain_frontier_count
    or set(opengl["workgroupBlockedSources"]) != opengl_blocked_sources
    or opengl["workgroupBlockedArtifactCount"] != len(opengl_blocked_sources)
    or opengl["projectDiagnosticCount"] != 0
    or opengl["expectedProjectErrorCount"] != len(opengl_blocked_sources)
    or set(opengl["toolchainSources"]) != opengl_toolchain_frontier_sources
    or opengl["trackedIssues"] != []
    or opengl["resolvedIssues"] != [MLX_HOST_DISPATCH_IMPORT_RESOLVED_ISSUE]
    or opengl["runtimeIntegrationIncluded"] is not False
):
    raise SystemExit("OpenGL target-split frontier accounting is incomplete")
opengl_index_evidence = opengl["indexRangeAssertionEvidence"]
if (
    opengl_index_evidence["assertionCount"] != len(MLX_OPENGL_INDEX_RANGE_ASSERTIONS)
    or opengl_index_evidence["inclusiveBounds"] != {
        "minimum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MINIMUM,
        "maximum": MLX_OPENGL_INDEX_RANGE_ASSERTION_MAXIMUM,
    }
    or opengl_index_evidence["expressionsBySource"] != {
        source: list(expressions)
        for source, expressions in MLX_OPENGL_INDEX_RANGE_ASSERTION_EXPRESSIONS.items()
    }
    or opengl_index_evidence["contractKind"]
    != "explicit-host-runtime-portability-preconditions"
    or opengl_index_evidence["inferred"] is not False
    or opengl_index_evidence["runtimeEnforced"] is not False
):
    raise SystemExit("OpenGL index-range portability evidence is incomplete")
opengl_dispatch_evidence = opengl["dynamicWorkgroupDispatchEvidence"]
if set(opengl_dispatch_evidence) != opengl_blocked_sources:
    raise SystemExit("OpenGL workgroup blocker source join is incomplete")
for source in opengl_blocked_sources:
    expected_evidence = MLX_DYNAMIC_WORKGROUP_DISPATCH_EVIDENCE[source]
    observed_evidence = opengl_dispatch_evidence[source]
    if any(
        observed_evidence.get(key) != value for key, value in expected_evidence.items()
    ) or (
        observed_evidence["sourceEntryPointIdentityStatus"]
        != "matched-materialized-host-names"
    ):
        raise SystemExit(f"OpenGL workgroup blocker evidence changed for {source}")
if os.environ["RUNNER_OS"] == "Linux":
    validated_sources = set(opengl["toolchainValidatedSources"])
    validation_outputs = set(opengl["nativeValidationOutputs"])
    if (
        opengl["toolchainRequired"] is not True
        or opengl["nativeValidationStatus"] != "validated"
        or opengl["toolchainValidatedArtifactCount"] != opengl_toolchain_frontier_count
        or validated_sources != opengl_toolchain_frontier_sources
        or validation_outputs != opengl_toolchain_frontier_sources
    ):
        raise SystemExit("OpenGL frontier toolchain must validate every emitted source")
