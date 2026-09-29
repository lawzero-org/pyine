"""Utilities for local exportation of PyINE traces for external evaluation datasets."""

# pyarrow does not publish complete type information:
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false

from __future__ import annotations

import collections
import dataclasses
import datetime
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import pathlib  # noqa: TC003 - pydantic resolves this annotation at runtime
import platform
import re
import shutil
import time
import typing

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic

import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.utils.splits
import pyine.utils.code.execution
import pyine.utils.code.output_compare
import pyine.utils.concurrency
import pyine.utils.filesystem
import pyine.utils.portability
import pyine.utils.reprod

__all__ = [
    "EVAL_EXPORT_SCHEMA_VERSION",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_VALIDATION_FRACTION",
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_REEXECUTE_MAX_STEP_COUNT",
    "DEFAULT_RECHECK_TIMEOUT_SECONDS",
    "EvalExportConfig",
    "ExportManifest",
    "ValueRendering",
    "ResolvedOutcome",
    "CertificationResult",
    "assign_export_split",
    "certify_trace",
    "export_eval_traces",
    "get_eval_export_schema",
    "render_value",
    "resolve_trace_outcome",
    "select_problem_identifiers",
    "validate_trace_record",
]
"""Public API exposed by the evaluation-export module."""

logger = logging.getLogger(__name__)
"""Module logger for export progress and integrity checks."""

EVAL_EXPORT_SCHEMA_VERSION = "1.0.0"
"""Semantic version of the flat evaluation-export schema."""
OUTCOME_POLICY_VERSION = "strict-channel-v1"
"""Versioned policy used to select one exported outcome channel."""
COMPARISON_POLICY_VERSION = "pyine-soft-v1"
"""Versioned policy used for semantic outcome comparisons."""
DEFAULT_TRAIN_FRACTION = 0.50
"""Default fraction of eligible problems assigned to derived train."""
DEFAULT_VALIDATION_FRACTION = 0.25
"""Default fraction of eligible problems assigned to derived validation."""
DEFAULT_TEST_FRACTION = 0.25
"""Default fraction of eligible problems assigned to derived test."""
DEFAULT_REEXECUTE_MAX_STEP_COUNT = 20_000
"""Recommended opt-in stored valid-step ceiling for certification by re-execution."""
DEFAULT_RECHECK_TIMEOUT_SECONDS = 60.0
"""Default timeout in seconds for one certification re-execution."""
EXPORT_SPLIT_NAMES = ("train", "validation", "test")
"""Ordered names of the derived export partitions."""
COUNTED_FLAG_NAMES = (
    "problem_is_banned",
    "has_return_value",
    "has_stdout",
    "has_stderr",
    "has_exception",
    "expected_matches_outcome",
)
"""Boolean row flags summarized as true-value counts in the manifest."""
ExportSplitName = typing.Literal["train", "validation", "test"]
"""Type alias for a derived export partition name."""
OutcomeKind = typing.Literal["return_value", "stdout", "exception"]
"""Type alias for the observable channel selected as a trace's final outcome."""
RecheckMethod = typing.Literal["reexecuted", "record_integrity_checked"]
"""Type alias for the certification method applied to an exported row."""
RecheckOutcome = typing.Literal["pass", "fail"]
"""Type alias for the result of certifying an exported row."""
OutcomeSource = typing.Literal["reexecuted_live", "stored_record"]
"""Type alias for the source of the authoritative exported outcome text."""
InvocationKind = typing.Literal["callable", "stdin"]
"""Type alias for the broad execution interface represented by a trace."""
SHARD_PART_PATTERN = re.compile(r"(?P<part>\d{6})of(?P<total>\d{6})")
"""Filename pattern used by PyINE's numbered trace shards."""
SHARD_SPECIFIC_WRITER_CONFIG_FIELDS = frozenset({"target_problem_ids"})
"""Writer settings allowed to differ between otherwise compatible source shards."""


class EvalExportConfig(pydantic.BaseModel):
    """Configuration for a local evaluation export."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown settings."""

    source_lmdb_paths: list[pathlib.Path] = pydantic.Field(min_length=1)
    """Native PyINE trace LMDB shards to export."""
    split_file_path: pathlib.Path
    """Original PyINE problem-level split result used to enforce the train-only embargo."""
    output_dir: pathlib.Path
    """Directory in which the local Parquet artifact is written."""
    source_dataset_name: str = "TACO"
    """Name of the source problem dataset represented by the trace shards."""
    seed: int = 0
    """Seed included in deterministic problem capping and derived splitting."""
    train_fraction: float = pydantic.Field(default=DEFAULT_TRAIN_FRACTION, ge=0.0, le=1.0)
    """Fraction of selected problems assigned to derived train."""
    validation_fraction: float = pydantic.Field(default=DEFAULT_VALIDATION_FRACTION, ge=0.0, le=1.0)
    """Fraction of selected problems assigned to derived validation."""
    test_fraction: float = pydantic.Field(default=DEFAULT_TEST_FRACTION, ge=0.0, le=1.0)
    """Fraction of selected problems assigned to derived test."""
    max_problem_count: pydantic.PositiveInt | None = None
    """Optional deterministic whole-problem cap. None exports every allowed problem."""
    reexecute_max_step_count: pydantic.NonNegativeInt | None = None
    """Optional stored valid-step ceiling that opts eligible rows into re-execution."""
    reexecute_all: bool = False
    """Whether every structurally valid row should be freshly re-executed."""
    recheck_timeout_seconds: pydantic.PositiveFloat = DEFAULT_RECHECK_TIMEOUT_SECONDS
    """Generous timeout for each certification re-execution."""
    execution_seed_override: int | None = None
    """Optional seed override for re-execution; by default the stored execution seed is reused."""
    certification_workers: pydantic.PositiveInt = 4
    """Number of concurrent certification workers."""
    parquet_batch_size: pydantic.PositiveInt = 256
    """Number of records written per Parquet row group."""
    allow_partial_source: bool = False
    """Whether an incomplete set of conventionally numbered source shards is allowed."""

    @pydantic.model_validator(mode="after")
    def _validate_configuration(self) -> EvalExportConfig:
        """Validate partition fractions and mutually exclusive re-execution settings."""
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"export split fractions must sum to 1.0, got {total:.12f}")
        if self.reexecute_all and self.reexecute_max_step_count is not None:
            raise ValueError("reexecute_all and reexecute_max_step_count are mutually exclusive")
        return self


class ExportFileInfo(pydantic.BaseModel):
    """Integrity metadata for one artifact file."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    relative_path: str
    """Artifact path relative to the export root directory."""
    size_bytes: int
    """Artifact file size in bytes."""
    sha256: str
    """SHA-256 digest of the artifact file contents."""


class SourceShardInfo(pydantic.BaseModel):
    """Provenance metadata for one source LMDB shard."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    path: str
    """Absolute path of the native source LMDB shard."""
    trace_count: int
    """Number of trace records indexed in the source shard."""
    dataset_hash: str
    """Content hash of the source shard's LMDB data file."""
    parent_dataset: dict[str, pydantic.JsonValue]
    """Source coding-dataset provenance embedded by the trace writer."""
    writer_config: dict[str, pydantic.JsonValue]
    """Trace-writer configuration embedded in this shard."""


class SourceFileInfo(pydantic.BaseModel):
    """Integrity metadata for a source-side input file."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    path: str
    """Absolute path of the source-side input file."""
    size_bytes: int
    """Source file size in bytes."""
    sha256: str
    """SHA-256 digest of the source file contents."""


class SourceSplitInfo(pydantic.BaseModel):
    """Auditable identity and configuration of the original problem split."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    file: SourceFileInfo
    """Path and integrity metadata for the serialized split file."""
    identifier_count: int
    """Number of source problems represented by the split."""
    assignment_counts: dict[str, int]
    """Problem count assigned to each original PyINE partition."""
    config: dict[str, pydantic.JsonValue]
    """Original problem-splitting configuration."""
    creation_metadata: dict[str, pydantic.JsonValue]
    """Reproducibility metadata recorded when the split was created."""


class DerivedSplitSpec(pydantic.BaseModel):
    """The deterministic derived-split method recorded in the manifest."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    method: str
    """Versioned name of the deterministic assignment method."""
    key: str
    """Description of the problem-level value hashed for assignment."""
    seed: int
    """Seed incorporated into deterministic split hashing."""
    fractions: dict[str, float]
    """Requested assignment fraction for each derived partition."""
    max_problem_count: int | None
    """Optional whole-problem cap applied before derived assignment."""


class EnvironmentFingerprint(pydantic.BaseModel):
    """Runtime information needed to identify the export environment."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown fields."""

    python_version: str
    """Python interpreter version used for the export."""
    platform: str
    """Operating-system and architecture description for the export runtime."""
    uv_lock_sha256: str | None
    """SHA-256 digest of the dependency lockfile when available."""
    installed_packages_sha256: str
    """SHA-256 digest of the frozen installed-package inventory."""


class ExportManifest(pydantic.BaseModel):
    """Self-contained manifest for a completed local evaluation export."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    """Pydantic configuration that freezes instances and rejects unknown manifest fields."""

    schema_version: str
    """Semantic version of the exported row and manifest contract."""
    created_at: str
    """UTC timestamp recorded after artifact generation."""
    pyine_version: str
    """Installed PyINE package version used to build the artifact."""
    pyine_commit: str
    """Git revision of the PyINE working tree used for the export."""
    pyine_worktree_clean: bool
    """Whether the PyINE worktree was clean, including untracked files."""
    source_dataset_name: str
    """Name of the source coding-problem dataset."""
    source_dataset_hash: str
    """Source dataset hash recorded by the original split result."""
    source_shards: list[SourceShardInfo]
    """Ordered provenance records for the native trace LMDB shards."""
    source_split: SourceSplitInfo
    """Identity, configuration, and integrity metadata for the original PyINE split."""
    export_config: dict[str, pydantic.JsonValue]
    """Complete JSON-serialized configuration used for the export."""
    derived_split: DerivedSplitSpec
    """Method and parameters used for derived problem-level assignment."""
    environment: EnvironmentFingerprint
    """Portable fingerprint of the environment that performed certification and export."""
    outcome_policy_version: str
    """Version of the channel-selection semantics used for exported outcomes."""
    comparison_policy_version: str
    """Version of the semantic comparison policy used by the exporter."""
    comparison_options: dict[str, pydantic.JsonValue]
    """Complete soft-comparison settings used for flags and certification."""
    schema_columns: list[str]
    """Ordered names of all columns in the shared Parquet schema."""
    partition_row_counts: dict[str, int]
    """Exported row count for each derived partition."""
    partition_problem_counts: dict[str, int]
    """Distinct full-problem count for each derived partition."""
    certification_counts: dict[str, dict[str, int]]
    """Row counts grouped by certification method and outcome."""
    flag_counts: dict[str, int]
    """True-value counts for the summarized Boolean row flags."""
    files: dict[str, ExportFileInfo]
    """Integrity metadata for Parquet partitions and self-contained contract files."""
    certification_log: ExportFileInfo
    """Integrity metadata for the structured certification JSONL log."""


@dataclasses.dataclass(frozen=True)
class ValueRendering:
    """Canonical JSON and display forms of a Python value."""

    json_value: str | None
    """Canonical JSON text, or null when lossless JSON encoding is unavailable."""
    text: str
    """Deterministic display text available for every value."""
    canonical: bool
    """Whether the value has a lossless canonical JSON representation."""


@dataclasses.dataclass(frozen=True)
class ResolvedOutcome:
    """The observable program outcome selected for an exported trace."""

    kind: OutcomeKind
    """Observable channel selected as the final outcome."""
    value: typing.Any
    """Actual value observed through the selected outcome channel."""
    expected_matches: bool
    """Whether the source expectation softly matches the selected actual value."""
    comparison_reason: str
    """Diagnostic reason returned by the output comparison."""


@dataclasses.dataclass(frozen=True)
class CertificationResult:
    """Certification method and result for one exported trace."""

    identifier: str
    """Exact internal identifier of the certified trace."""
    method: RecheckMethod
    """Method used to certify the trace record."""
    outcome: RecheckOutcome
    """Pass or fail result of certification."""
    reason: str
    """Human-readable explanation of the certification result."""
    stored_outcome: ResolvedOutcome
    """Outcome reconstructed from the stored trace record."""
    rerun_outcome: ResolvedOutcome | None
    """Freshly observed outcome when re-execution completed."""
    outcome_source: OutcomeSource
    """Source selected for the authoritative exported outcome."""
    exact_match: bool | None
    """Whether stored and rerun outcome kinds and representations matched exactly."""
    semantic_match: bool | None
    """Whether stored and rerun outcomes matched with PyINE's soft comparator."""
    execution_seed: int | None
    """Seed used for fresh re-execution, if any."""
    duration_seconds: float
    """Elapsed wall time for integrity checking and any requested re-execution."""
    error_phase: str | None
    """Processing phase responsible for a failure, when applicable."""

    @property
    def selected_outcome(self) -> ResolvedOutcome:
        """Return the live passing outcome when available, otherwise the stored outcome."""
        if self.outcome_source == "reexecuted_live":
            if self.rerun_outcome is None:
                raise RuntimeError("live outcome source requires a rerun outcome")
            return self.rerun_outcome
        return self.stored_outcome


@dataclasses.dataclass(frozen=True)
class TraceReference:
    """Lightweight pointer to one selected trace before its full record is loaded."""

    identifier: str
    """Exact internal trace identifier used for ordering and integrity checks."""
    reader_idx: int
    """Index of the source reader containing the trace."""
    trace_idx: int
    """External trace position within the selected source reader."""
    problem_identifier: str
    """Full internal problem identifier used for derived splitting."""
    export_split: ExportSplitName
    """Derived partition assigned to the trace's complete problem family."""


def _is_canonical_json_value(value: typing.Any) -> bool:
    """Return whether a value is losslessly representable by the export's JSON subset."""
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_canonical_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _is_canonical_json_value(item) for key, item in value.items())
    return False


def render_value(value: typing.Any) -> ValueRendering:
    """Render a value without silently normalizing non-JSON Python types.

    Args:
        value: Value to render.

    Returns:
        Canonical compact JSON plus identical text when lossless, otherwise a display-only
        representation and a null JSON field.
    """
    if _is_canonical_json_value(value):
        json_value = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return ValueRendering(json_value=json_value, text=json_value, canonical=True)
    return ValueRendering(
        json_value=None,
        text=_render_display_value(value),
        canonical=False,
    )


def _render_display_value(value: typing.Any) -> str:
    """Render a potentially non-canonical value into deterministic display-only text."""
    if type(value) is tuple:
        rendered_items = [_render_display_value(item) for item in value]
        suffix = "," if len(rendered_items) == 1 else ""
        return f"({', '.join(rendered_items)}{suffix})"
    if type(value) is list:
        return f"[{', '.join(_render_display_value(item) for item in value)}]"
    if type(value) in (set, frozenset):
        rendered_items = sorted(_render_display_value(item) for item in value)
        if type(value) is frozenset:
            return f"frozenset({{{', '.join(rendered_items)}}})"
        return f"{{{', '.join(rendered_items)}}}" if rendered_items else "set()"
    if type(value) is dict:
        rendered_items = sorted(
            (
                _render_display_value(key),
                _render_display_value(item),
            )
            for key, item in value.items()
        )
        return "{" + ", ".join(f"{key}: {item}" for key, item in rendered_items) + "}"
    return pyine.utils.portability.get_portable_representation(value)


def _compare_values(
    first: typing.Any,
    second: typing.Any,
) -> pyine.utils.code.output_compare.CompareResult:
    """Compare two outcome values with PyINE's established soft equality settings."""
    return pyine.utils.code.output_compare.compare(
        first,
        second,
        options=pyine.utils.code.output_compare.get_default_comparison_config(),
    )


def _compare_stdout(
    stdout: str,
    other: typing.Any,
) -> pyine.utils.code.output_compare.CompareResult:
    """Compare stdout softly, also accepting a list of strings joined with newlines."""
    comparison = _compare_values(stdout, other)
    if not comparison and isinstance(other, list) and all(isinstance(item, str) for item in other):
        merged_comparison = _compare_values(stdout, "\n".join(typing.cast("list[str]", other)))
        if merged_comparison:
            return merged_comparison
    return comparison


def resolve_trace_outcome(
    trace_result: pyine.utils.code.execution.TraceResult,
) -> ResolvedOutcome:
    """Select the authoritative outcome using ``strict-channel-v1``.

    Args:
        trace_result: Stored or freshly executed trace containing its entrypoint,
            return value, stdout, exception, and source-test expectation.

    Returns:
        The selected channel/value plus its soft comparison with the source expectation.
        Exceptions take precedence, except that a script's SystemExit with nonempty
        stdout uses that stdout. Otherwise callables use their return value and scripts
        use stdout. The expectation never determines which channel wins, and this
        function does not validate or re-execute the trace.
    """
    expected_output = trace_result.expected_output
    if trace_result.exception is not None:
        exception_value = str(trace_result.exception)
        if trace_result.entrypoint_name is not None or trace_result.exception.type != SystemExit.__name__:
            comparison = _compare_values(exception_value, str(expected_output))
            return ResolvedOutcome("exception", exception_value, bool(comparison), comparison.reason)
        if not trace_result.stdout:
            comparison = _compare_values(exception_value, str(expected_output))
            return ResolvedOutcome("exception", exception_value, bool(comparison), comparison.reason)
    if trace_result.entrypoint_name is not None:
        comparison = _compare_values(trace_result.return_value, expected_output)
        return ResolvedOutcome("return_value", trace_result.return_value, bool(comparison), comparison.reason)
    stdout_comparison = _compare_stdout(trace_result.stdout, expected_output)
    if stdout_comparison:
        return ResolvedOutcome("stdout", trace_result.stdout, True, stdout_comparison.reason)
    return ResolvedOutcome(
        "stdout",
        trace_result.stdout,
        False,
        f"unexpected stdout output: {stdout_comparison.reason}",
    )


def compare_candidate(
    outcome: ResolvedOutcome,
    candidate: typing.Any,
) -> bool:
    """Grade a candidate against an already resolved authoritative outcome.

    Args:
        outcome: Actual execution outcome selected by ``resolve_trace_outcome`` or
            certification. Its source-test expectation does not determine this label.
        candidate: Proposed Python value or display representation. Comparison uses
            ``pyine-soft-v1`` tolerances and container rules. For stdout outcomes, a
            list of strings is also compared after joining its elements with newlines.

    Returns:
        Whether the candidate matches the resolved value. This does not assert an
        outcome channel or exact Python type and does not execute the program.
    """
    if outcome.kind == "stdout":
        return bool(_compare_stdout(outcome.value, candidate))
    return bool(_compare_values(outcome.value, candidate))


def _hash_for_assignment(
    domain: str,
    seed: int,
    problem_identifier: str,
) -> int:
    """Return a domain-separated SHA-256 integer for deterministic problem assignment."""
    payload = f"{domain}\0{seed}\0{problem_identifier}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def assign_export_split(
    problem_identifier: str,
    config: EvalExportConfig,
) -> ExportSplitName:
    """Assign a whole problem family to a deterministic derived partition.

    Args:
        problem_identifier: Full original problem identity, shared by all of its
            solutions, tests, and augmentations. Do not pass a trace identifier.
        config: Validated export settings supplying the seed and train/validation/test
            fractions. Other selection and certification settings do not affect assignment.

    Returns:
        ``train``, ``validation``, or ``test`` from a domain-separated seeded hash.
        Assignment is independent of traversal order and available problem count;
        fractions specify probabilities rather than exact partition sizes.
    """
    unit_interval = _hash_for_assignment("split", config.seed, problem_identifier) / (1 << 256)
    if unit_interval < config.train_fraction:
        return "train"
    if unit_interval < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def select_problem_identifiers(
    problem_identifiers: typing.Iterable[str],
    config: EvalExportConfig,
) -> set[str]:
    """Apply an order-independent cap to already eligible whole-problem identities.

    Args:
        problem_identifiers: Iterable of eligible full problem identifiers. Duplicates
            are removed; this helper does not check original splits or source eligibility.
        config: Settings supplying ``seed`` and optional ``max_problem_count``.

    Returns:
        All unique identities when uncapped or below the cap; otherwise the capped set
        selected by a deterministic seeded hash rank. No trace-level subsampling occurs.
    """
    unique_identifiers = set(problem_identifiers)
    if config.max_problem_count is None or len(unique_identifiers) <= config.max_problem_count:
        return unique_identifiers
    ordered = sorted(
        unique_identifiers,
        key=lambda identifier: (_hash_for_assignment("cap", config.seed, identifier), identifier),
    )
    return set(ordered[: config.max_problem_count])


def _values_record_equal(
    first: typing.Any,
    second: typing.Any,
) -> bool:
    """Compare stored record values without applying semantic output tolerances."""
    first_rendering = render_value(first)
    second_rendering = render_value(second)
    if first_rendering.canonical and second_rendering.canonical:
        return first_rendering.json_value == second_rendering.json_value
    return first_rendering.text == second_rendering.text


def validate_trace_record(
    trace_result: pyine.utils.code.execution.TraceResult,
    trace_metadata: pyine.data.traces.dataset_utils.TraceMetadata,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
) -> CertificationResult:
    """Check consistency between a stored trace, its index metadata, and its problem.

    Args:
        trace_result: Native trace record to check, without modifying or executing it.
        trace_metadata: Reader metadata for that same record, including code, inputs,
            outputs, exception, streams, identity, and valid-step count.
        problem: Parent coding problem used to check problem identity and entrypoint.

    Returns:
        A ``record_integrity_checked`` certification result with pass/fail status,
        mismatch reasons, and the resolved stored outcome. Expected record mismatches
        are returned as failures rather than raised. Passing checks do not independently
        validate intermediate assertions, prove source-test correctness, or apply bans.
    """
    start_time = time.perf_counter()
    failures: list[str] = []
    if trace_result.identifier is None:
        failures.append("trace identifier is null")
        identifier = trace_metadata.identifier
    else:
        identifier = trace_result.identifier
    if identifier != trace_metadata.identifier:
        failures.append("trace and metadata identifiers differ")
    try:
        trace_identifier = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(identifier)
    except ValueError as error:
        failures.append(f"identifier is invalid: {error}")
    else:
        if trace_identifier.get_parent_identifier().get_parent_identifier() != problem.problem_id:
            failures.append("trace and problem identifiers differ")
    if trace_result.code_string != trace_metadata.code_string:
        failures.append("code strings differ")
    if trace_result.entrypoint_name != problem.entrypoint_name:
        failures.append("trace and problem entrypoints differ")
    for field_name in ("inputs", "expected_output", "return_value"):
        if not _values_record_equal(getattr(trace_result, field_name), getattr(trace_metadata, field_name)):
            failures.append(f"{field_name} values differ")
    if trace_result.exception != trace_metadata.exception:
        failures.append("exceptions differ")
    if trace_result.stdout != trace_metadata.stdout:
        failures.append("stdout values differ")
    if trace_result.stderr != trace_metadata.stderr:
        failures.append("stderr values differ")
    if trace_result.valid_step_count != trace_metadata.step_count:
        failures.append("valid step counts differ")
    stored_outcome = resolve_trace_outcome(trace_result)
    return CertificationResult(
        identifier=identifier,
        method="record_integrity_checked",
        outcome="fail" if failures else "pass",
        reason="; ".join(failures) if failures else "stored trace record invariants passed",
        stored_outcome=stored_outcome,
        rerun_outcome=None,
        outcome_source="stored_record",
        exact_match=None,
        semantic_match=None,
        execution_seed=None,
        duration_seconds=time.perf_counter() - start_time,
        error_phase="record_integrity" if failures else None,
    )


def _get_stored_execution_seed(
    trace_result: pyine.utils.code.execution.TraceResult,
) -> int | None:
    """Parse the execution seed recorded in a stored trace's metadata."""
    raw_seed = trace_result.metadata.get("seed")
    if raw_seed in (None, "None"):
        return None
    if isinstance(raw_seed, int):
        return raw_seed
    return int(str(raw_seed))


def certify_trace(
    trace_result: pyine.utils.code.execution.TraceResult,
    trace_metadata: pyine.data.traces.dataset_utils.TraceMetadata,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: EvalExportConfig,
) -> CertificationResult:
    """Check record integrity and optionally perform outcome-only safe re-execution.

    Args:
        trace_result: Native execution record whose code/input outcome is being certified.
        trace_metadata: Reader metadata for the same record, checked before any rerun.
        problem: Parent problem used for identity and entrypoint validation.
        config: Certification settings: opt-in step threshold or all-traces selection,
            per-run timeout, and optional execution-seed override. Without opt-in,
            certification checks stored-record integrity only.

    Returns:
        A certification result recording its method, status, reason, selected evidence
        source, and optional exact/semantic rerun comparisons. Integrity failures prevent
        execution. A passing same-channel soft rerun match selects the live outcome;
        rerun errors or mismatches produce failed results retaining stored evidence.

    Notes:
        Reruns use the safe execution wrapper with event capture disabled, so they cannot
        certify stored intermediate events. Ordinary rerun exceptions become diagnostics;
        process-control interruptions propagate to the caller.
    """
    start_time = time.perf_counter()
    record_result = validate_trace_record(trace_result, trace_metadata, problem)
    if record_result.outcome == "fail" or not _should_reexecute(trace_result, config):
        return record_result
    identifier = typing.cast("str", trace_result.identifier)
    seed = (
        config.execution_seed_override
        if config.execution_seed_override is not None
        else _get_stored_execution_seed(trace_result)
    )
    try:
        rerun_result = pyine.utils.code.execution.execute_and_trace_code(
            code_string=trace_result.code_string,
            inputs=trace_result.inputs,
            expected_output=trace_result.expected_output,
            identifier=identifier,
            entrypoint_name=trace_result.entrypoint_name,
            trace_only_inside_code_string=True,
            max_valid_events=None,
            max_events_per_line=None,
            max_var_repr_length=None,
            timeout_seconds=config.recheck_timeout_seconds,
            seed=seed,
            request_metadata="eval export certification",
            capture_trace_events=False,
            precomputed_complexity_metrics=trace_result.complexity_metrics,
            use_safe_execution=True,
        )
    except pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS:
        raise
    except Exception as error:
        return CertificationResult(
            identifier=identifier,
            method="reexecuted",
            outcome="fail",
            reason=f"re-execution raised {type(error).__name__}: {error}",
            stored_outcome=record_result.stored_outcome,
            rerun_outcome=None,
            outcome_source="stored_record",
            exact_match=None,
            semantic_match=None,
            execution_seed=seed,
            duration_seconds=time.perf_counter() - start_time,
            error_phase="reexecution",
        )
    stored_outcome = record_result.stored_outcome
    rerun_outcome = resolve_trace_outcome(rerun_result)
    comparison = _compare_values(stored_outcome.value, rerun_outcome.value)
    kind_matches = stored_outcome.kind == rerun_outcome.kind
    exact_match = kind_matches and _values_record_equal(stored_outcome.value, rerun_outcome.value)
    semantic_match = kind_matches and bool(comparison)
    reason = (
        "stored final outcome reproduced semantically"
        if semantic_match
        else (
            f"stored outcome {stored_outcome.kind} did not match rerun outcome {rerun_outcome.kind}: "
            f"{comparison.reason}"
        )
    )
    return CertificationResult(
        identifier=identifier,
        method="reexecuted",
        outcome="pass" if semantic_match else "fail",
        reason=reason,
        stored_outcome=stored_outcome,
        rerun_outcome=rerun_outcome,
        outcome_source="reexecuted_live" if semantic_match else "stored_record",
        exact_match=exact_match,
        semantic_match=semantic_match,
        execution_seed=seed,
        duration_seconds=time.perf_counter() - start_time,
        error_phase=None if semantic_match else "comparison",
    )


def _should_reexecute(
    trace_result: pyine.utils.code.execution.TraceResult,
    config: EvalExportConfig,
) -> bool:
    """Return whether the configuration explicitly requests fresh execution for a trace."""
    if config.reexecute_all:
        return True
    if config.reexecute_max_step_count is None:
        return False
    return trace_result.valid_step_count <= config.reexecute_max_step_count


def _get_recheck_method(
    trace_result: pyine.utils.code.execution.TraceResult,
    config: EvalExportConfig,
) -> RecheckMethod:
    """Return the method that should own success or worker-failure accounting for a trace."""
    return "reexecuted" if _should_reexecute(trace_result, config) else "record_integrity_checked"


def get_eval_export_schema() -> pa.Schema:
    """Return the fixed facts-v1 schema shared by all derived partitions.

    Returns:
        An Arrow schema with explicit field types and nullability for source identities,
        task/fact values, outcome evidence, flags, and certification results. Empty
        partitions use the same schema; v2 uses ``get_statement_export_schema`` instead.
    """
    return pa.schema(
        [
            pa.field("identifier", pa.string(), nullable=False),
            pa.field("dataset_name", pa.string(), nullable=False),
            pa.field("source_subset", pa.string(), nullable=False),
            pa.field("source_dataset_names", pa.list_(pa.string()), nullable=False),
            pa.field("pyine_subset", pa.string(), nullable=False),
            pa.field("export_split", pa.string(), nullable=False),
            pa.field("problem_idx", pa.int64(), nullable=False),
            pa.field("solution_idx", pa.int64(), nullable=False),
            pa.field("test_idx", pa.int64(), nullable=False),
            pa.field("is_augmented", pa.bool_(), nullable=False),
            pa.field("augment_category", pa.string()),
            pa.field("augment_idx", pa.int64()),
            pa.field("code_string", pa.string(), nullable=False),
            pa.field("entrypoint_name", pa.string()),
            pa.field("invocation_kind", pa.string(), nullable=False),
            pa.field("invocation_text", pa.string(), nullable=False),
            pa.field("invocation_source", pa.string(), nullable=False),
            pa.field("inputs_json", pa.string()),
            pa.field("inputs_text", pa.string(), nullable=False),
            pa.field("problem_statement", pa.string(), nullable=False),
            pa.field("problem_tags", pa.list_(pa.string()), nullable=False),
            pa.field("tags", pa.list_(pa.string()), nullable=False),
            pa.field("expected_output_json", pa.string()),
            pa.field("expected_output_text", pa.string(), nullable=False),
            pa.field("outcome_kind", pa.string(), nullable=False),
            pa.field("outcome_json", pa.string()),
            pa.field("outcome_text", pa.string(), nullable=False),
            pa.field("outcome_source", pa.string(), nullable=False),
            pa.field("return_value_json", pa.string()),
            pa.field("stdout", pa.string(), nullable=False),
            pa.field("stderr", pa.string(), nullable=False),
            pa.field("exception_type", pa.string()),
            pa.field("exception_message", pa.string()),
            pa.field("problem_is_banned", pa.bool_(), nullable=False),
            pa.field("has_return_value", pa.bool_(), nullable=False),
            pa.field("has_stdout", pa.bool_(), nullable=False),
            pa.field("has_stderr", pa.bool_(), nullable=False),
            pa.field("has_exception", pa.bool_(), nullable=False),
            pa.field("expected_matches_outcome", pa.bool_(), nullable=False),
            pa.field("recheck_method", pa.string(), nullable=False),
            pa.field("recheck_outcome", pa.string(), nullable=False),
            pa.field("recheck_exact_match", pa.bool_()),
            pa.field("recheck_semantic_match", pa.bool_()),
            pa.field("valid_step_count", pa.int64(), nullable=False),
            pa.field("total_step_count", pa.int64(), nullable=False),
        ]
    )


def _get_source_dataset_names(
    problem: pyine.data.traces.dataset_utils.CodingProblem,
) -> list[str]:
    """Extract sorted upstream dataset names from a problem's source tags."""
    return sorted({tag.removeprefix("source:") for tag in problem.problem_tags if tag.startswith("source:")})


def _get_invocation_fields(
    trace_result: pyine.utils.code.execution.TraceResult,
    inputs: ValueRendering,
) -> tuple[InvocationKind, str, str]:
    """Render stored invocation facts without claiming an exact Python calling convention."""
    if trace_result.entrypoint_name is not None:
        invocation_text = f"entrypoint {trace_result.entrypoint_name}; stored input payload: {inputs.text}"
        return "callable", invocation_text, "stored_record"
    return "stdin", f"stored stdin payload: {inputs.text}", "stored_record"


def _project_trace_row(
    trace_result: pyine.utils.code.execution.TraceResult,
    trace_metadata: pyine.data.traces.dataset_utils.TraceMetadata,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    export_split: ExportSplitName,
    certification: CertificationResult,
) -> dict[str, typing.Any]:
    """Project a native trace and its context into one flat export row."""
    if trace_result.identifier is None:
        raise ValueError("cannot export a trace with a null identifier")
    trace_identifier = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_result.identifier)
    inputs = render_value(trace_result.inputs)
    expected_output = render_value(trace_result.expected_output)
    return_value = render_value(trace_result.return_value)
    invocation_kind, invocation_text, invocation_source = _get_invocation_fields(trace_result, inputs)
    outcome = certification.selected_outcome
    outcome_rendering = render_value(outcome.value)
    exception = trace_result.exception
    return {
        "identifier": trace_result.identifier,
        "dataset_name": trace_identifier.dataset,
        "source_subset": trace_identifier.subset,
        "source_dataset_names": _get_source_dataset_names(problem),
        "pyine_subset": "train",
        "export_split": export_split,
        "problem_idx": trace_identifier.problem_idx,
        "solution_idx": trace_identifier.solution_idx,
        "test_idx": trace_identifier.test_idx,
        "is_augmented": trace_identifier.is_augmented,
        "augment_category": trace_identifier.augment_category,
        "augment_idx": trace_identifier.augment_idx,
        "code_string": trace_result.code_string,
        "entrypoint_name": trace_result.entrypoint_name,
        "invocation_kind": invocation_kind,
        "invocation_text": invocation_text,
        "invocation_source": invocation_source,
        "inputs_json": inputs.json_value,
        "inputs_text": inputs.text,
        "problem_statement": problem.problem_statement,
        "problem_tags": sorted(problem.problem_tags),
        "tags": sorted(trace_metadata.tags),
        "expected_output_json": expected_output.json_value,
        "expected_output_text": expected_output.text,
        "outcome_kind": outcome.kind,
        "outcome_json": outcome_rendering.json_value,
        "outcome_text": outcome_rendering.text,
        "outcome_source": certification.outcome_source,
        "return_value_json": return_value.json_value,
        "stdout": trace_result.stdout,
        "stderr": trace_result.stderr,
        "exception_type": exception.type if exception is not None else None,
        "exception_message": exception.message if exception is not None else None,
        "problem_is_banned": problem.is_banned,
        "has_return_value": trace_result.return_value is not None,
        "has_stdout": bool(trace_result.stdout),
        "has_stderr": bool(trace_result.stderr),
        "has_exception": exception is not None,
        "expected_matches_outcome": outcome.expected_matches,
        "recheck_method": certification.method,
        "recheck_outcome": certification.outcome,
        "recheck_exact_match": certification.exact_match,
        "recheck_semantic_match": certification.semantic_match,
        "valid_step_count": trace_result.valid_step_count,
        "total_step_count": trace_result.total_step_count,
    }


def _get_file_info(
    path: pathlib.Path,
    output_dir: pathlib.Path,
) -> ExportFileInfo:
    """Build relative-path and integrity metadata for one artifact file."""
    return ExportFileInfo(
        relative_path=str(path.relative_to(output_dir)),
        size_bytes=path.stat().st_size,
        sha256=pyine.utils.reprod.compute_hash(path, algorithm="sha256"),
    )


def _prepare_output_paths(
    config: EvalExportConfig,
) -> tuple[pathlib.Path, dict[str, pathlib.Path]]:
    """Create a fresh sibling staging directory for a non-overwriting export."""
    staging_dir = _validate_output_paths_available(config)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    paths = {split_name: staging_dir / f"{split_name}.parquet" for split_name in EXPORT_SPLIT_NAMES}
    paths["manifest"] = staging_dir / "export_manifest.json"
    paths["specification"] = staging_dir / "EVAL_EXPORT_SPEC.md"
    paths["schema"] = staging_dir / "export_schema.json"
    paths["packages"] = staging_dir / "installed_packages.json"
    paths["certification"] = staging_dir / "certification" / "certification.jsonl"
    paths["certification"].parent.mkdir(parents=True, exist_ok=True)
    return staging_dir, paths


def _validate_output_paths_available(config: EvalExportConfig) -> pathlib.Path:
    """Reject completed or interrupted output paths without changing the filesystem."""
    if config.output_dir.exists():
        raise FileExistsError(f"export output directory already exists: {config.output_dir}")
    staging_dir = config.output_dir.with_name(f"{config.output_dir.name}.incomplete")
    if staging_dir.exists():
        raise FileExistsError(f"incomplete export directory already exists: {staging_dir}")
    return staging_dir


def build_trace_references(
    readers: list[pyine.data.traces.dataset_reader.DatasetProtocol],
    split_result: pyine.data.utils.splits.SplitResult,
    config: EvalExportConfig,
) -> dict[ExportSplitName, list[TraceReference]]:
    """Index train-scoped source traces and assign their problems to derived partitions."""
    candidate_problem_identifiers: list[str] = []
    candidate_metadata: list[tuple[int, int, pyine.data.traces.dataset_utils.TraceMetadata]] = []
    all_identifiers: set[str] = set()
    for reader_idx, reader in enumerate(readers):
        for trace_idx, trace_metadata in enumerate(reader.trace_metadata):
            if trace_metadata.identifier in all_identifiers:
                raise ValueError(f"duplicate trace identifier across source shards: {trace_metadata.identifier}")
            all_identifiers.add(trace_metadata.identifier)
            problem_identifier = str(trace_metadata.problem_id)
            if problem_identifier not in split_result.subset_assignments:
                raise ValueError(f"source trace problem is absent from the supplied split: {problem_identifier}")
            if split_result.subset_assignments[problem_identifier] != "train":
                continue
            candidate_problem_identifiers.append(problem_identifier)
            candidate_metadata.append((reader_idx, trace_idx, trace_metadata))
    selected_problems = select_problem_identifiers(candidate_problem_identifiers, config)
    references: dict[ExportSplitName, list[TraceReference]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for reader_idx, trace_idx, trace_metadata in candidate_metadata:
        problem_identifier = str(trace_metadata.problem_id)
        if problem_identifier not in selected_problems:
            continue
        export_split = assign_export_split(problem_identifier, config)
        references[export_split].append(
            TraceReference(
                identifier=trace_metadata.identifier,
                reader_idx=reader_idx,
                trace_idx=trace_idx,
                problem_identifier=problem_identifier,
                export_split=export_split,
            )
        )
    for split_references in references.values():
        split_references.sort(key=lambda reference: reference.identifier)
    return references


def _write_json_line(
    file_descriptor: typing.TextIO,
    payload: dict[str, typing.Any],
) -> None:
    """Write one canonical compact JSON object to a JSONL stream."""
    file_descriptor.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    file_descriptor.write("\n")


def _get_outcome_digest(outcome: ResolvedOutcome) -> str:
    """Hash an outcome's selected channel and deterministic display representation."""
    payload = f"{outcome.kind}\0{render_value(outcome.value).text}".encode()
    return hashlib.sha256(payload).hexdigest()


def get_certification_log_payload(certification: CertificationResult) -> dict[str, typing.Any]:
    """Project a certification result into a compact JSON-safe audit record."""
    return {
        "identifier": certification.identifier,
        "method": certification.method,
        "outcome": certification.outcome,
        "reason": certification.reason,
        "outcome_source": certification.outcome_source,
        "stored_outcome_kind": certification.stored_outcome.kind,
        "stored_outcome_sha256": _get_outcome_digest(certification.stored_outcome),
        "rerun_outcome_kind": certification.rerun_outcome.kind if certification.rerun_outcome is not None else None,
        "rerun_outcome_sha256": (
            _get_outcome_digest(certification.rerun_outcome) if certification.rerun_outcome is not None else None
        ),
        "exact_match": certification.exact_match,
        "semantic_match": certification.semantic_match,
        "execution_seed": certification.execution_seed,
        "duration_seconds": certification.duration_seconds,
        "error_phase": certification.error_phase,
    }


def _certify_loaded_trace(
    trace_result: pyine.utils.code.execution.TraceResult,
    trace_metadata: pyine.data.traces.dataset_utils.TraceMetadata,
    problem: pyine.data.traces.dataset_utils.CodingProblem,
    config: EvalExportConfig,
) -> CertificationResult:
    """Adapt trace certification to the zero-argument parallel-work wrapper."""
    return certify_trace(trace_result, trace_metadata, problem, config)


def _write_partition(
    split_name: ExportSplitName,
    references: list[TraceReference],
    readers: list[pyine.data.traces.dataset_reader.DatasetProtocol],
    path: pathlib.Path,
    certification_log: typing.TextIO,
    config: EvalExportConfig,
    certification_counts: dict[str, collections.Counter[str]],
    flag_counts: collections.Counter[str],
) -> None:
    """Certify traces with bounded heavy-record concurrency and write one partition."""
    schema = get_eval_export_schema()
    with pq.ParquetWriter(
        path,
        schema=schema,
        compression="zstd",
        use_dictionary=True,
        version="2.6",
    ) as parquet_writer:
        pending_rows: list[dict[str, typing.Any]] = []
        processed_count = 0
        for batch_start in range(0, len(references), config.certification_workers):
            batch_references = references[batch_start : batch_start + config.certification_workers]
            loaded_records: list[
                tuple[
                    pyine.utils.code.execution.TraceResult,
                    pyine.data.traces.dataset_utils.TraceMetadata,
                    pyine.data.traces.dataset_utils.CodingProblem,
                ]
            ] = []
            for reference in batch_references:
                reader = readers[reference.reader_idx]
                trace_result = reader[reference.trace_idx]
                trace_metadata = reader.trace_metadata[reference.trace_idx]
                problem = reader.get_problem_data(reference.trace_idx)
                if trace_result.identifier != reference.identifier:
                    raise ValueError(
                        f"source trace identifier changed after indexing: {reference.identifier} != "
                        f"{trace_result.identifier}"
                    )
                if str(problem.problem_id) != reference.problem_identifier:
                    raise ValueError(
                        f"source problem identifier changed after indexing: {reference.problem_identifier} != "
                        f"{problem.problem_id}"
                    )
                loaded_records.append((trace_result, trace_metadata, problem))
            callables = [
                functools.partial(
                    _certify_loaded_trace,
                    trace_result,
                    trace_metadata,
                    problem,
                    config,
                )
                for trace_result, trace_metadata, problem in loaded_records
            ]
            results, errors = pyine.utils.concurrency.run_in_parallel(
                callables,
                use_processes=False,
                max_workers=config.certification_workers,
            )
            for reference, loaded_record, result, error in zip(
                batch_references,
                loaded_records,
                results,
                errors,
                strict=True,
            ):
                trace_result, trace_metadata, problem = loaded_record
                if error is not None:
                    if isinstance(error, pyine.utils.code.execution.DONT_CATCH_EXCEPTIONS):
                        raise error
                    certification = CertificationResult(
                        identifier=reference.identifier,
                        method=_get_recheck_method(trace_result, config),
                        outcome="fail",
                        reason=f"certification worker raised {type(error).__name__}: {error}",
                        stored_outcome=resolve_trace_outcome(trace_result),
                        rerun_outcome=None,
                        outcome_source="stored_record",
                        exact_match=None,
                        semantic_match=None,
                        execution_seed=None,
                        duration_seconds=0.0,
                        error_phase="worker",
                    )
                else:
                    if not isinstance(result, CertificationResult):
                        raise TypeError(f"unexpected certification result for {reference.identifier}: {result!r}")
                    certification = result
                certification_counts[certification.method][certification.outcome] += 1
                _write_json_line(
                    certification_log,
                    get_certification_log_payload(certification),
                )
                row = _project_trace_row(
                    trace_result,
                    trace_metadata,
                    problem,
                    split_name,
                    certification,
                )
                for flag_name in COUNTED_FLAG_NAMES:
                    if row[flag_name]:
                        flag_counts[flag_name] += 1
                pending_rows.append(row)
            while len(pending_rows) >= config.parquet_batch_size:
                rows_to_write = pending_rows[: config.parquet_batch_size]
                parquet_writer.write_table(pa.Table.from_pylist(rows_to_write, schema=schema))
                del pending_rows[: config.parquet_batch_size]
            processed_count += len(batch_references)
            if (
                batch_start == 0
                or processed_count == len(references)
                or processed_count % (100 * config.parquet_batch_size) == 0
            ):
                logger.info(f"exported {processed_count}/{len(references)} rows to derived {split_name}")
        if pending_rows:
            parquet_writer.write_table(pa.Table.from_pylist(pending_rows, schema=schema))


def get_source_shard_info(
    path: pathlib.Path,
    reader: pyine.data.traces.dataset_reader.DatasetProtocol,
) -> SourceShardInfo:
    """Verify a source shard's current hash and return its provenance metadata."""
    logger.info(f"verifying source shard hash: {path}")
    dataset_hash = reader.hash
    if reader.trace_metadata:
        recorded_hash = reader.trace_metadata[0].parent_dataset_hash
        if any(metadata.parent_dataset_hash != recorded_hash for metadata in reader.trace_metadata):
            raise ValueError(f"inconsistent parent dataset hashes in source shard: {path}")
        if recorded_hash != dataset_hash:
            raise ValueError(f"source shard changed after trace metadata was prepared: {path}")
    metadata = reader.metadata
    parent_dataset = metadata.get("parent_dataset")
    writer_config = metadata.get("writer_config")
    if not isinstance(parent_dataset, dict):
        raise ValueError(f"source shard is missing parent_dataset metadata: {path}")
    if not isinstance(writer_config, dict):
        raise ValueError(f"source shard is missing writer_config metadata: {path}")
    return SourceShardInfo(
        path=str(path.resolve()),
        trace_count=len(reader),
        dataset_hash=dataset_hash,
        parent_dataset=typing.cast("dict[str, pydantic.JsonValue]", parent_dataset),
        writer_config=typing.cast("dict[str, pydantic.JsonValue]", writer_config),
    )


def validate_source_shards(
    source_shards: list[SourceShardInfo],
    config: EvalExportConfig,
) -> None:
    """Validate source-dataset identity, writer compatibility, and numbered-shard coverage."""
    parent_dataset_hashes: set[str] = set()
    normalized_writer_configs: set[str] = set()
    for shard in source_shards:
        parent_name = shard.parent_dataset.get("dataset_name")
        if parent_name != config.source_dataset_name:
            raise ValueError(
                f"source shard parent dataset {parent_name!r} does not match configured source "
                f"{config.source_dataset_name!r}: {shard.path}"
            )
        parent_dataset_hash = shard.parent_dataset.get("dataset_hash")
        if not isinstance(parent_dataset_hash, str):
            raise ValueError(f"source shard has no recorded parent dataset hash: {shard.path}")
        parent_dataset_hashes.add(parent_dataset_hash)
        normalized_config = {
            key: value for key, value in shard.writer_config.items() if key not in SHARD_SPECIFIC_WRITER_CONFIG_FIELDS
        }
        normalized_writer_configs.add(
            json.dumps(normalized_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    if len(parent_dataset_hashes) != 1:
        raise ValueError("source shards do not share one recorded parent dataset hash")
    if len(normalized_writer_configs) != 1:
        raise ValueError("source shards were produced with incompatible writer configurations")
    _validate_numbered_shard_coverage(source_shards, config.allow_partial_source)


def _validate_numbered_shard_coverage(
    source_shards: list[SourceShardInfo],
    allow_partial_source: bool,
) -> None:
    """Require complete ordinal coverage when source shard names use PyINE's numbered convention."""
    matches = [SHARD_PART_PATTERN.search(pathlib.Path(shard.path).name) for shard in source_shards]
    if not any(matches) or allow_partial_source:
        return
    if not all(matches):
        raise ValueError("source paths mix numbered and unnumbered shards; use allow_partial_source if intentional")
    typed_matches = typing.cast("list[re.Match[str]]", matches)
    total_counts = {int(match.group("total")) for match in typed_matches}
    if len(total_counts) != 1:
        raise ValueError("numbered source shards disagree on their total shard count")
    total_count = total_counts.pop()
    found_parts = [int(match.group("part")) for match in typed_matches]
    expected_parts = set(range(1, total_count + 1))
    if len(found_parts) != len(set(found_parts)) or set(found_parts) != expected_parts:
        missing_parts = sorted(expected_parts - set(found_parts))
        raise ValueError(
            f"incomplete numbered source shard set: expected {total_count}, found {len(found_parts)}, "
            f"missing {missing_parts}"
        )


def validate_source_problem_hashes(
    readers: list[pyine.data.traces.dataset_reader.DatasetProtocol],
    split_result: pyine.data.utils.splits.SplitResult,
) -> None:
    """Tie every source problem identifier and content hash to the supplied split revision."""
    split_hashes = dict(zip(split_result.identifiers, split_result.source_data_hashes, strict=True))
    validated_problems: set[tuple[int, str]] = set()
    for reader_idx, reader in enumerate(readers):
        for trace_idx, trace_metadata in enumerate(reader.trace_metadata):
            problem_identifier = str(trace_metadata.problem_id)
            expected_hash = split_hashes.get(problem_identifier)
            if expected_hash is None:
                raise ValueError(f"source trace problem is absent from the supplied split: {problem_identifier}")
            validation_key = (reader_idx, problem_identifier)
            if validation_key in validated_problems:
                continue
            problem = reader.get_problem_data(trace_idx)
            if problem.source_data_hash != expected_hash:
                raise ValueError(
                    f"source problem hash does not match supplied split for {problem_identifier}: "
                    f"{problem.source_data_hash} != {expected_hash}"
                )
            validated_problems.add(validation_key)


def _write_json_file(
    path: pathlib.Path,
    payload: typing.Any,
) -> None:
    """Write one indented, deterministically keyed JSON artifact."""
    with open(path, "w", encoding="utf-8") as file_descriptor:
        json.dump(payload, file_descriptor, ensure_ascii=False, sort_keys=True, indent=2)
        file_descriptor.write("\n")


def _write_support_files(
    output_paths: dict[str, pathlib.Path],
) -> str:
    """Copy the frozen contract and write schema and installed-package inventories."""
    specification_source = pathlib.Path(__file__).with_name("EVAL_EXPORT_SPEC_V1.md")
    if not specification_source.is_file():
        raise FileNotFoundError(f"missing packaged evaluation export specification: {specification_source}")
    shutil.copyfile(specification_source, output_paths["specification"])
    row_schema = get_eval_export_schema()
    schema_payload = {
        "schema_version": EVAL_EXPORT_SCHEMA_VERSION,
        "row_fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in row_schema
        ],
        "manifest": ExportManifest.model_json_schema(),
    }
    _write_json_file(output_paths["schema"], schema_payload)
    installed_packages = sorted(
        {
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        },
        key=str.casefold,
    )
    _write_json_file(output_paths["packages"], installed_packages)
    return pyine.utils.reprod.compute_hash(output_paths["packages"], algorithm="sha256")


def _get_environment_fingerprint(installed_packages_sha256: str) -> EnvironmentFingerprint:
    """Capture the portable runtime fields used to identify the export environment."""
    project_root = pyine.utils.filesystem.get_project_root_path()
    uv_lock_path = project_root / "uv.lock"
    uv_lock_hash = pyine.utils.reprod.compute_hash(uv_lock_path, algorithm="sha256") if uv_lock_path.is_file() else None
    return EnvironmentFingerprint(
        python_version=platform.python_version(),
        platform=platform.platform(),
        uv_lock_sha256=uv_lock_hash,
        installed_packages_sha256=installed_packages_sha256,
    )


def _validate_staged_artifact(
    staging_dir: pathlib.Path,
    manifest: ExportManifest,
) -> None:
    """Recheck staged artifact files, Parquet schemas, and row counts before promotion."""
    for file_info in [*manifest.files.values(), manifest.certification_log]:
        path = staging_dir / file_info.relative_path
        if path.stat().st_size != file_info.size_bytes:
            raise ValueError(f"staged artifact file size changed before promotion: {path}")
        if pyine.utils.reprod.compute_hash(path, algorithm="sha256") != file_info.sha256:
            raise ValueError(f"staged artifact digest changed before promotion: {path}")
    expected_schema = get_eval_export_schema()
    for split_name in EXPORT_SPLIT_NAMES:
        parquet_file = pq.ParquetFile(staging_dir / manifest.files[split_name].relative_path)
        if parquet_file.schema_arrow != expected_schema:
            raise ValueError(f"staged Parquet schema mismatch for derived {split_name}")
        if parquet_file.metadata.num_rows != manifest.partition_row_counts[split_name]:
            raise ValueError(f"staged Parquet row-count mismatch for derived {split_name}")


def export_eval_traces(config: EvalExportConfig) -> ExportManifest:
    """Export train-scoped PyINE traces to deterministic local Parquet partitions.

    Args:
        config: Validated source LMDB paths, original split, fresh output directory,
            whole-problem selection/split settings, and certification/write controls.
            Only original-training problem families participate. Re-execution is opt-in.

    Returns:
        The completed facts-v1 manifest, also saved in the output directory, containing
        partition row/problem counts, source provenance, certification summaries, and
        artifact hashes. V1 retains banned and failed-certification rows with diagnostic
        flags; consumers must apply their intended filtering policy.

    Raises:
        FileExistsError: The output or its sibling incomplete directory already exists.
        ValueError: Sources or split hashes are inconsistent, no eligible original-training
            traces exist, or final artifact validation fails.

    Notes:
        Readers may prepare source metadata caches. Output files, certification logs,
        and support files are written to a sibling incomplete directory and promoted
        only after validation. Failed exports can leave that incomplete directory.
    """
    _validate_output_paths_available(config)
    split_result = pyine.data.utils.splits.SplitResult.from_file(config.split_file_path)
    if split_result.source_dataset_name != config.source_dataset_name:
        raise ValueError(
            f"split source dataset {split_result.source_dataset_name!r} does not match "
            f"configured source {config.source_dataset_name!r}"
        )
    readers: list[pyine.data.traces.dataset_reader.DatasetProtocol] = [
        pyine.data.traces.dataset_reader.DatasetReader(path) for path in config.source_lmdb_paths
    ]
    for reader in readers:
        if reader.parent_dataset_name != config.source_dataset_name:
            raise ValueError(
                f"trace source dataset {reader.parent_dataset_name!r} does not match "
                f"configured source {config.source_dataset_name!r}"
            )
    source_shards = [
        get_source_shard_info(path, reader) for path, reader in zip(config.source_lmdb_paths, readers, strict=True)
    ]
    validate_source_shards(source_shards, config)
    validate_source_problem_hashes(readers, split_result)
    references = build_trace_references(readers, split_result, config)
    if not any(references.values()):
        raise ValueError("no source traces belong to problems assigned to PyINE's original train partition")
    staging_dir, output_paths = _prepare_output_paths(config)
    installed_packages_sha256 = _write_support_files(output_paths)
    certification_counts: dict[str, collections.Counter[str]] = {
        "reexecuted": collections.Counter(),
        "record_integrity_checked": collections.Counter(),
    }
    flag_counts: collections.Counter[str] = collections.Counter()
    with open(output_paths["certification"], "w", encoding="utf-8") as certification_log:
        for split_name in typing.cast("tuple[ExportSplitName, ...]", EXPORT_SPLIT_NAMES):
            logger.info(f"exporting {len(references[split_name])} rows to derived {split_name}")
            _write_partition(
                split_name,
                references[split_name],
                readers,
                output_paths[split_name],
                certification_log,
                config,
                certification_counts,
                flag_counts,
            )
    partition_row_counts = {split_name: len(references[split_name]) for split_name in EXPORT_SPLIT_NAMES}
    partition_problem_counts = {
        split_name: len({reference.problem_identifier for reference in references[split_name]})
        for split_name in EXPORT_SPLIT_NAMES
    }
    artifact_files = {
        split_name: _get_file_info(output_paths[split_name], staging_dir) for split_name in EXPORT_SPLIT_NAMES
    }
    artifact_files.update(
        {
            file_name: _get_file_info(output_paths[file_name], staging_dir)
            for file_name in ("specification", "schema", "packages")
        }
    )
    certification_log_info = _get_file_info(output_paths["certification"], staging_dir)
    split_file_info = SourceFileInfo(
        path=str(config.split_file_path.resolve()),
        size_bytes=config.split_file_path.stat().st_size,
        sha256=pyine.utils.reprod.compute_hash(config.split_file_path, algorithm="sha256"),
    )
    assignment_counts = collections.Counter(split_result.subset_assignments.values())
    comparison_options = pyine.utils.code.output_compare.get_default_comparison_config()
    manifest = ExportManifest(
        schema_version=EVAL_EXPORT_SCHEMA_VERSION,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        pyine_version=pyine.utils.reprod.get_framework_version(),
        pyine_commit=pyine.utils.reprod.get_git_revision_hash(),
        pyine_worktree_clean=pyine.utils.reprod.is_git_repo_clean(include_untracked=True),
        source_dataset_name=config.source_dataset_name,
        source_dataset_hash=split_result.source_dataset_hash,
        source_shards=source_shards,
        source_split=SourceSplitInfo(
            file=split_file_info,
            identifier_count=len(split_result.identifiers),
            assignment_counts=dict(sorted(assignment_counts.items())),
            config=typing.cast("dict[str, pydantic.JsonValue]", split_result.config.model_dump(mode="json")),
            creation_metadata=split_result.creation_metadata,
        ),
        export_config=typing.cast("dict[str, pydantic.JsonValue]", config.model_dump(mode="json")),
        derived_split=DerivedSplitSpec(
            method="sha256-threshold-v1",
            key="full PyINE problem identifier",
            seed=config.seed,
            fractions={
                "train": config.train_fraction,
                "validation": config.validation_fraction,
                "test": config.test_fraction,
            },
            max_problem_count=config.max_problem_count,
        ),
        environment=_get_environment_fingerprint(installed_packages_sha256),
        outcome_policy_version=OUTCOME_POLICY_VERSION,
        comparison_policy_version=COMPARISON_POLICY_VERSION,
        comparison_options=typing.cast(
            "dict[str, pydantic.JsonValue]",
            comparison_options.model_dump(mode="json"),
        ),
        schema_columns=get_eval_export_schema().names,
        partition_row_counts=partition_row_counts,
        partition_problem_counts=partition_problem_counts,
        certification_counts={
            method: {outcome: counts.get(outcome, 0) for outcome in ("pass", "fail")}
            for method, counts in certification_counts.items()
        },
        flag_counts={flag_name: flag_counts.get(flag_name, 0) for flag_name in COUNTED_FLAG_NAMES},
        files=artifact_files,
        certification_log=certification_log_info,
    )
    with open(output_paths["manifest"], "w", encoding="utf-8") as manifest_file:
        manifest_file.write(manifest.model_dump_json(indent=2))
        manifest_file.write("\n")
    _validate_staged_artifact(staging_dir, manifest)
    staging_dir.rename(config.output_dir)
    logger.info(
        f"completed evaluation export with partition problem counts {partition_problem_counts} at: {config.output_dir}"
    )
    return manifest
