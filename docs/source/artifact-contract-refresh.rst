Refreshing Artifact Contracts
=============================

Exact generated-source identities are useful regression checks, but legitimate
translator changes can make those identities stale. The refresh tool regenerates
an existing entry contract through the public project API and requires a fresh
native compilation before proposing replacement hashes and byte counts.

The tool accepts the entry-contract format used by external corpus proofs:
``source``, ``sourceSha256``, ``target``, and an ``entries`` array with
``entryPoint``, ``sha256`` and ``sizeBytes`` fields. An optional ``commit`` pins
the repository revision and requires a clean tracked checkout. The project
configuration supplies include paths, defines and frontend limits. Targets need
host-interface reflection support.

For example, from the translator checkout with an existing pinned source
repository and project configuration:

.. code-block:: console

   python -m tools.refresh_artifact_contract \
     --root /path/to/source-repository \
     --config /path/to/source-repository/crosstl.toml \
     --contract /path/to/artifact-contract.json \
     --work-dir .contract-refresh/metal \
     --compiler-command '["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", "{artifact}", "-o", "{output}"]' \
     --jobs 4

Compiler commands are explicit JSON argument arrays, never shell expressions.
``{artifact}`` and ``{output}`` are required standalone arguments;
``{entry_point}`` is available for compilers that require the generated entry
name. Use the compiler profile and flags required by the existing proof.
Successful process exit alone is insufficient: compilation must create a new,
nonempty output, and the source artifact must retain its verified identity.

Outputs And Review
------------------

The work directory must remain below the source repository and cannot contain
the input contract, configuration or pinned source. It contains the
portability report, translation checkpoint, generated artifacts, per-entry
compiler evidence, and ``audit.json``. A complete successful run also writes
``candidate.json``. The input contract is never modified.

Only generated hashes, byte counts and their aggregate size summaries change.
Entry coverage, source pins, materialization digests and resource ABI contracts
remain unchanged and are checked where present. Aggregate checks cover artifact,
specialization and reflected-resource counts, including resource types and
per-shape specialization/resource totals. Corpus-specific annotations and
historical proof metadata are copied, not independently re-established; their
existing regression tests and evidence review remain required.
Uniform recorded workgroup sizes are applied to every selected entry. Partial
audits selected with repeated ``--entry`` options retain their evidence but exit
nonzero and do not produce an incomplete candidate.

Metal template contracts use entry-level launch rules; other contracts use the
concrete project workgroup size. In both cases the report must retain the
requested size. A target that cannot provide that metadata fails the audit.

``--resume`` uses the project's existing validation for interrupted translation
checkpoints. Completed checkpoints are deliberately rejected; omit ``--resume``
to regenerate a completed run. Native compilers still run again and cannot reuse
an old output as successful evidence.
``--timeout-seconds`` bounds individual translation jobs and compiler calls.
Warnings, translation failures, interface drift, compiler failures and incomplete
coverage prevent a candidate from being written.
Changes to the source, configuration or input contract during an audit also
invalidate the result. Compiler timeouts retain captured output for diagnosis.

Review the candidate diff before replacing a baseline. Update dependent contract
file hashes and size summaries together, then run the existing native corpus
tests and numerical gates. A refreshed artifact identity establishes neither
numerical correctness nor a passing upstream test suite; those remain separate
requirements.

The project-porting workflow requires a small native refresh on macOS (Metal),
Windows (DXC) and Linux (GLSL). Missing compilers fail the required check. Each
job retains its generated source, compiler output, audit, candidate and test
report. This checks the refresh workflow, not numerical execution or complete
corpus coverage.
