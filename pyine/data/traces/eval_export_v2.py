"""Deterministic statement exports, author artifacts, and source-free public projection."""

# pyarrow lacks complete annotations.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownParameterType=false

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import importlib.metadata
import itertools
import json
import pathlib
import random
import re
import shutil
import time
import types
import typing

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import tokenizers
import yaml

import pyine.data.traces.dataset_reader as trace_reader
import pyine.data.traces.dataset_utils as trace_utils
import pyine.data.traces.eval_export as legacy
import pyine.data.traces.export_consumer as consumer
import pyine.data.traces.statement_config as contract
import pyine.data.traces.statement_events as events
import pyine.data.utils.splits
import pyine.utils.code.execution as execution
import pyine.utils.code.input_mock
import pyine.utils.code.output_compare
import pyine.utils.concurrency
import pyine.utils.reprod

SCHEMA_VERSION = "2.0.0"
STATEMENT_TARGET_VERSION = "reference-execution-v1"
SPLITS = ("train", "validation", "test")
EXECUTION_POLICY_KEYS = (
    "seed",
    "blacklisted_modules",
    "blacklisted_objects",
    "trace_only_inside_code_string",
    "capture_trace_events",
)
GROUP_SHARED_FIELDS = (
    "problem_id",
    "context_id",
    "export_split",
    "code_string",
    "entrypoint_name",
    "invocation_kind",
    "invocation_text",
    "inputs_text",
    "inputs_json",
    "observed_bindings",
    "supplied_stdin",
    "problem_statement",
    "reasoning_events",
    "category_labels",
)


def load_recipe(path: pathlib.Path | None) -> contract.Recipe:
    """Load and validate a statement-export recipe without fetching remote resources.

    Args:
        path: YAML recipe file, or None for ``contract.Recipe`` defaults. Relative
            tokenizer paths are resolved against the recipe file's parent directory.

    Returns:
        A validated recipe with defaults filled in. Token budgets require a readable
        local tokenizer file whose bytes match the configured SHA-256 digest.

    Raises:
        OSError: The recipe or a required tokenizer resource cannot be read.
        yaml.YAMLError: The recipe is not valid YAML.
        ValueError: The YAML root is not a mapping.
        pydantic.ValidationError: Fields, constraints, or tokenizer settings are invalid.
    """
    if path is None:
        return contract.Recipe()
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("recipe must be a YAML mapping")
    payload = typing.cast("dict[str, typing.Any]", payload)
    budget = payload.get("budget")
    if isinstance(budget, dict) and budget.get("tokenizer_file") is not None:
        budget["tokenizer_file"] = path.parent / budget["tokenizer_file"]
    return contract.Recipe.model_validate(payload)


def _arrow_type(annotation: typing.Any) -> pa.DataType:
    """Map a pydantic field annotation onto its Arrow type; optional types map to their inner type."""
    origin, arguments = typing.get_origin(annotation), typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        return _arrow_type(next(argument for argument in arguments if argument is not type(None)))
    if origin is typing.Literal:
        return _arrow_type(type(arguments[0]))
    if origin is list:
        return pa.list_(_arrow_type(arguments[0]))
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return pa.struct(list(_model_schema(annotation)))
    return {str: pa.string(), int: pa.int64(), bool: pa.bool_()}[annotation]


def _model_schema(model: type[pydantic.BaseModel]) -> pa.Schema:
    """Derive an Arrow schema from a pydantic model, with optional fields marked nullable."""
    return pa.schema(
        [
            pa.field(name, _arrow_type(field.annotation), nullable=type(None) in typing.get_args(field.annotation))
            for name, field in model.model_fields.items()
        ]
    )


def get_statement_export_schema(author: bool = False) -> pa.Schema:
    """Return the typed, nested Arrow schema for a v2 artifact.

    Args:
        author: Include the private ``author`` provenance struct when True. False
            returns the public row schema, including nullable targets and categories.

    Returns:
        An Arrow schema derived from ``contract.AuthorExample`` or ``contract.Example``.
        Visibility settings change field values, not this schema or its nullability.
    """
    return _model_schema(contract.AuthorExample if author else contract.Example)


def _value(value: typing.Any) -> tuple[str, str | None]:
    """Return a value's displayed text and its canonical JSON, or None when it has no JSON form."""
    rendered = legacy.render_value(value)
    if rendered.canonical:
        return repr(value), json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return rendered.text, None


def _invocation(trace: execution.TraceResult) -> dict[str, typing.Any]:
    """Build a trace's displayed task fields: code, entrypoint, input, observed bindings, and stdin."""
    input_text, input_json = _value(trace.inputs)
    arguments = events.observed_entrypoint_bindings(trace)  # incomplete bindings fall back to the stored payload
    bindings = [{"name": name, "value": value} for name, value in (arguments or {}).items()]
    supplied_stdin = None
    if trace.entrypoint_name is None:
        supplied_stdin = pyine.utils.code.input_mock.MockInput(str(trace.inputs)).read()
    return {
        "code_string": trace.code_string,
        "entrypoint_name": trace.entrypoint_name,
        "invocation_kind": "callable" if trace.entrypoint_name is not None else "stdin",
        "invocation_text": f"entrypoint {trace.entrypoint_name}; observed bindings (not call syntax)"
        if bindings
        else (
            f"entrypoint {trace.entrypoint_name}; stored input payload"
            if trace.entrypoint_name is not None
            else "script with supplied stdin"
        ),
        "inputs_text": input_text,
        "inputs_json": input_json,
        "observed_bindings": bindings,
        "supplied_stdin": supplied_stdin,
    }


def _is_harness_failure(trace: execution.TraceResult) -> bool:
    """Check whether a recorded exception arose entirely outside the displayed program.

    Invocation failures, such as inputs that cannot be mapped onto the entrypoint or a
    missing entrypoint, never enter a program frame. Program, library, module-initialization,
    stdin exhaustion, and ``exit()`` exceptions do, even when raised in library or mock code.
    """
    exception = trace.exception
    program_frame = f'File "{execution.EXEC_TRACE_FILE_NAME}"'
    return exception is not None and program_frame not in (exception.traceback or "")


@dataclasses.dataclass(frozen=True)
class _Context:
    """One certified execution, with its task, outcome, and the keys that pair it with donors."""

    reference: legacy.TraceReference
    outcome: legacy.ResolvedOutcome
    task: dict[str, typing.Any]
    input_key: str
    supplied_input_key: str
    pool_key: str
    outcome_evidence: str
    native_event_count: int
    original_identifier: str | None
    categories: tuple[str, ...]
    prompt_parent_id: str | None

    @property
    def reference_input_keys(self) -> frozenset[str]:
        """Match cross-code references by normalized invocation or exact supplied payload."""
        return frozenset((self.input_key, self.supplied_input_key))


def _input_keys(task: dict[str, typing.Any]) -> tuple[str, str]:
    """Hash a task's normalized invocation and its exact supplied payload."""
    payload = (
        task["supplied_stdin"] if task["invocation_kind"] == "stdin" else [task["inputs_json"], task["inputs_text"]]
    )
    normalized = payload if task["invocation_kind"] == "stdin" else (task["observed_bindings"] or payload)
    return (
        contract.opaque_id("normalized_input", task["entrypoint_name"], normalized),
        contract.opaque_id("supplied_input", task["entrypoint_name"], task["invocation_kind"], payload),
    )


def _pool_key(
    trace: execution.TraceResult,
    reference: legacy.TraceReference,
) -> str:
    """Hash the solution lineage, exact code, interface, execution policy, and split shared by donors."""
    solution = trace_utils.TraceIdentifier.from_string(reference.identifier).get_parent_identifier()
    policy = {key: trace.metadata.get(key) for key in EXECUTION_POLICY_KEYS}
    return contract.opaque_id(
        "pool", str(solution), trace.code_string, trace.entrypoint_name, policy, reference.export_split
    )


@dataclasses.dataclass
class _Candidate:
    """One distinct candidate outcome of a query group, with every role and source that produced it."""

    context: _Context
    roles: list[str]
    sources: list[str]
    required: bool


def _problem_id(context: _Context) -> str:
    """Return the public opaque identity of a context's source problem."""
    return contract.opaque_id("problem", context.reference.problem_identifier)


def _same_outcome(
    first: legacy.ResolvedOutcome,
    second: legacy.ResolvedOutcome,
) -> bool:
    """Check whether two outcomes match under the comparison policy, in either direction."""
    return legacy.compare_candidate(first, second.value) or legacy.compare_candidate(second, first.value)


def _categories(identifier: trace_utils.TraceIdentifier) -> tuple[str, ...]:
    """Derive a trace's ``code.*`` category labels from its identifier."""
    names = ["code.augmented"] if identifier.is_augmented else ["code.original"]
    for flag in ("bugged", "hinted", "misleading", "obfuscated"):
        if getattr(identifier, f"is_{flag}"):
            names.append(f"code.{flag}")
    if any("stub" in category for category in identifier.split_augment_categories):
        names.append("code.stubbed")
    return tuple(names)


def _load_sources(
    config: legacy.EvalExportConfig,
) -> tuple[list[trace_reader.DatasetProtocol], dict[str, typing.Any], list[legacy.TraceReference]]:
    """Open and validate the source shards; return readers, provenance, and references in split order."""
    split = pyine.data.utils.splits.SplitResult.from_file(config.split_file_path)
    if split.source_dataset_name != config.source_dataset_name:
        raise ValueError("split source dataset does not match configured source")
    readers: list[trace_reader.DatasetProtocol] = [
        trace_reader.DatasetReader(path) for path in sorted(config.source_lmdb_paths)
    ]
    shards = [
        legacy.get_source_shard_info(path, reader)
        for path, reader in zip(sorted(config.source_lmdb_paths), readers, strict=True)
    ]
    legacy.validate_source_shards(shards, config)
    legacy.validate_source_problem_hashes(readers, split)
    references = legacy.build_trace_references(readers, split, config)
    source = {
        "shards": [shard.model_dump(mode="json") for shard in shards],
        "split_sha256": pyine.utils.reprod.compute_hash(config.split_file_path, algorithm="sha256"),
        "config": config.model_dump(mode="json"),
        "pyine_commit": pyine.utils.reprod.get_git_revision_hash(),
        "pyine_version": pyine.utils.reprod.get_framework_version(),
    }
    return readers, source, [reference for split_name in SPLITS for reference in references[split_name]]


def _certified_records(
    readers: list[trace_reader.DatasetProtocol],
    references: list[legacy.TraceReference],
    config: legacy.EvalExportConfig,
    recipe: contract.Recipe,
    counts: collections.Counter[str],
    skip: typing.Callable[[str, str], None],
) -> typing.Iterator[
    tuple[legacy.TraceReference, execution.TraceResult, trace_utils.CodingProblem, legacy.CertificationResult]
]:
    """Load eligible traces in batches and certify their outcomes in parallel."""
    for start in range(0, len(references), config.certification_workers):
        loaded = []
        for reference in references[start : start + config.certification_workers]:
            reader = readers[reference.reader_idx]
            problem = reader.get_problem_data(reference.trace_idx)
            counts["source.contexts"] += 1
            if problem.is_banned and not recipe.eligibility.include_banned_problems:
                skip(reference.identifier, "banned_problem")
                continue
            trace = reader[reference.trace_idx]
            if trace.identifier != reference.identifier or str(problem.problem_id) != reference.problem_identifier:
                raise ValueError("source identity changed after indexing")
            loaded.append((reference, trace, problem, reader.get_trace_metadata(reference.trace_idx)))
        if not loaded:
            continue
        results, errors = pyine.utils.concurrency.run_in_parallel(
            [
                functools.partial(legacy.certify_trace, trace, metadata, problem, config)
                for _, trace, problem, metadata in loaded
            ],
            use_processes=False,
            max_workers=config.certification_workers,
        )
        for (reference, trace, problem, _), result, error in zip(loaded, results, errors, strict=True):
            if error is not None:
                raise error
            if not isinstance(result, legacy.CertificationResult):
                raise TypeError("invalid certification worker result")
            yield reference, trace, problem, result


def _index_contexts(
    readers: list[trace_reader.DatasetProtocol],
    references: list[legacy.TraceReference],
    config: legacy.EvalExportConfig,
    recipe: contract.Recipe,
    counts: collections.Counter[str],
    skip: typing.Callable[[str, str], None],
    certification_log: typing.TextIO,
) -> dict[str, _Context]:
    """Certify every source trace and keep the eligible ones as contexts, keyed by identifier."""
    contexts: dict[str, _Context] = {}
    for reference, trace, problem, certification in _certified_records(
        readers, references, config, recipe, counts, skip
    ):
        reader = readers[reference.reader_idx]
        certification_log.write(contract.canonical_json(legacy.get_certification_log_payload(certification)) + "\n")
        counts[f"certification.{certification.method}.{certification.outcome}"] += 1
        if certification.outcome != "pass":
            skip(reference.identifier, "failed_certification")
            continue
        if _is_harness_failure(trace):
            skip(reference.identifier, "harness_invocation_failure")
            continue
        outcome_text, outcome_json = _value(certification.selected_outcome.value)
        visible_outcome = json.loads(outcome_json) if outcome_json is not None else outcome_text
        if not legacy.compare_candidate(certification.selected_outcome, visible_outcome):
            skip(reference.identifier, "outcome_display_changes_truth")
            continue
        identifier = trace_utils.TraceIdentifier.from_string(reference.identifier)
        task = _invocation(trace)
        task["problem_statement"] = problem.problem_statement if recipe.statements.include_problem_statement else None
        task["source_expected"] = _value(trace.expected_output)
        input_key, supplied_input_key = _input_keys(task)
        # prompt record UIDs start with the prompted trace identifier (see result_db._build_record_uid_prefix)
        parent_match = re.match(
            r"^([^/]+/[^/]+/p\d+/s\d+/t\d+(?:/a:[^:]+:\d+)?)_",
            str(trace.metadata.get("request_metadata", "")),
        )
        contexts[reference.identifier] = _Context(
            reference,
            certification.selected_outcome,
            task,
            input_key,
            supplied_input_key,
            _pool_key(trace, reference),
            certification.outcome_source,
            trace.valid_step_count,
            reader.augment_key_to_parent_trace_key.get(reference.identifier),
            _categories(identifier),
            parent_match.group(1) if parent_match is not None else None,
        )
        counts["eligible.outcome_contexts"] += 1
        counts.update(f"source.{category}" for category in contexts[reference.identifier].categories)
    return contexts


def _resolve_originals(
    contexts: dict[str, _Context],
    counts: collections.Counter[str],
) -> dict[str, _Context]:
    """Point each bugged context at its unbugged original on the same input, preferring prompt lineage.

    The declared unaugmented parent is used only when no prompt lineage is stored. Lineage that
    cannot be followed to an eligible non-bugged execution on the same input clears the original,
    so the bugged recipient is excluded rather than paired with another program's execution.
    """
    by_code_input: dict[tuple[str, str, str], list[_Context]] = collections.defaultdict(list)
    for context in contexts.values():
        solution = str(trace_utils.TraceIdentifier.from_string(context.reference.identifier).get_parent_identifier())
        for input_key in context.reference_input_keys:
            by_code_input[(solution, context.task["code_string"], input_key)].append(context)
    resolved = dict(contexts)
    for identifier, context in contexts.items():
        if "code.bugged" not in context.categories:
            continue
        if context.prompt_parent_id is None:
            has_parent = context.original_identifier is not None
            counts["original_reference.augmentless_parent" if has_parent else "original_reference.unavailable"] += 1
            continue
        ancestor = contexts.get(context.prompt_parent_id)
        visited = {identifier}
        while ancestor is not None and "code.bugged" in ancestor.categories:
            if ancestor.reference.identifier in visited:
                raise ValueError("cyclic stored augmentation lineage")
            visited.add(ancestor.reference.identifier)
            ancestor = contexts.get(ancestor.prompt_parent_id or "")
        candidates: list[_Context] = []
        if ancestor is not None:
            solution = str(trace_utils.TraceIdentifier.from_string(identifier).get_parent_identifier())
            candidates = [
                candidate
                for input_key in context.reference_input_keys
                for candidate in by_code_input[(solution, ancestor.task["code_string"], input_key)]
                if "code.bugged" not in candidate.categories
            ]
        selected = min(candidates, key=lambda candidate: candidate.reference.identifier) if candidates else None
        resolved[identifier] = dataclasses.replace(
            context, original_identifier=selected.reference.identifier if selected is not None else None
        )
        counts["original_reference.prompt_lineage" if selected is not None else "original_reference.unresolved"] += 1
    return resolved


def _candidate_bundle(
    recipient: _Context,
    donor: _Context | None,
    original: _Context | None,
    alternatives: list[_Context],
    extra_cap: int,
) -> list[_Candidate]:
    """Collect a group's distinct candidates: recipient, donor, original, then capped extra negatives."""
    bundle: list[_Candidate] = []

    def append(
        context: _Context,
        role: str,
        required: bool,
    ) -> bool:
        """Add a candidate, or merge its role into one with the same outcome; return whether it was new."""
        text, _ = _value(context.outcome.value)
        for existing in bundle:
            if text == _value(existing.context.outcome.value)[0] or _same_outcome(
                existing.context.outcome, context.outcome
            ):
                existing.roles.append(role)
                existing.sources.append(context.reference.identifier)
                existing.required = existing.required or required
                return False
        bundle.append(_Candidate(context, [role], [context.reference.identifier], required))
        return True

    append(recipient, "recipient", True)
    if donor is not None:
        append(donor, "donor", True)
    if original is not None:
        append(original, "original", True)
    extra_count = 0
    for alternative in alternatives:
        if extra_count >= extra_cap:
            break
        extra_count += append(alternative, "additional_observed_negative", False)
    return bundle


class _TextBudget:
    """Measure rendered reference text in characters or tokens against the recipe budget."""

    def __init__(
        self,
        config: contract.Budget,
    ) -> None:
        """Load the tokenizer when the budget counts tokens, with truncation and padding disabled."""
        self.config = config
        self.tokenizer = tokenizers.Tokenizer.from_file(str(config.tokenizer_file)) if config.unit == "tokens" else None
        if self.tokenizer is not None:
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()

    def cost(
        self,
        example: dict[str, typing.Any],
    ) -> int:
        """Return the rendered example's size in the budget unit."""
        text = consumer.render_example(example)
        return (
            len(self.tokenizer.encode(text, add_special_tokens=False).ids) if self.tokenizer is not None else len(text)
        )

    def fits(
        self,
        example: dict[str, typing.Any],
    ) -> bool:
        """Check whether the rendered example fits within the budget maximum."""
        return self.config.maximum is None or self.cost(example) <= self.config.maximum


def _display_index_characters(count: int) -> int:
    """Count the digits of the displayed step indices 0 to ``count - 1``."""
    total = min(count, 10)
    lower, width = 10, 2
    while count > lower:
        total += (min(count, lower * 10) - lower) * width
        lower *= 10
        width += 1
    return total


def _budget_group(
    rows: list[dict[str, typing.Any]],
    candidates: list[_Candidate],
    view: list[contract.Event] | None,
    reference: list[contract.Event],
    budget: _TextBudget,
    seed: str,
    max_matches: int,
) -> list[dict[str, typing.Any]]:
    """Fit a query group's rows to the budget and grade the reasoning events they keep.

    Optional rows that cannot fit are dropped, while an oversized required row raises
    ``IneligibleTraceError``. Events are then omitted deepest first until every row fits.
    """
    retained_rows: list[dict[str, typing.Any]] = []
    for row, candidate in zip(rows, candidates, strict=True):
        row["reasoning_events"] = None if view is None else []
        if not budget.fits(row):
            if candidate.required:
                raise events.IneligibleTraceError("required_query_budget")
        else:
            retained_rows.append(row)
    if view is None:
        kept_events = None
    else:
        tiers: dict[int, list[int]] = collections.defaultdict(list)
        for index, event in enumerate(view):
            tiers[event.source_depth].append(index)
        rng = random.Random(seed)
        removal_order: list[int] = []
        for depth in sorted(tiers, reverse=True):
            rng.shuffle(tiers[depth])
            removal_order.extend(tiers[depth])
        removed: set[int] = set()
        if budget.config.maximum is not None and budget.config.unit == "characters":
            # event content is fixed; only display-index widths and separators change on omission.
            event_sizes = [len(consumer.render_event(event.model_dump(), 0)) - 1 for event in view]
            content_size, count = sum(event_sizes), len(view)
            mandatory_size = max(budget.cost(row) for row in retained_rows)
            for index in removal_order:
                size = mandatory_size + content_size + _display_index_characters(count) + max(count - 1, 0)
                if size <= budget.config.maximum:
                    break
                removed.add(index)
                content_size -= event_sizes[index]
                count -= 1
        elif budget.config.maximum is not None:
            removal_iter = iter(removal_order)
            while True:
                serialized = [event.model_dump() for index, event in enumerate(view) if index not in removed]
                for row in retained_rows:
                    row["reasoning_events"] = serialized
                if all(budget.fits(row) for row in retained_rows):
                    break
                removed.add(next(removal_iter))
        kept_events = [event for index, event in enumerate(view) if index not in removed]
        kept_events = events.grade_events(kept_events, reference, max_matches)
    for row in retained_rows:
        row["reasoning_events"] = [event.model_dump() for event in kept_events] if kept_events is not None else None
        row["budget_metadata"] = contract.BudgetMetadata(
            unit=budget.config.unit,
            cost=budget.cost(row),
            maximum=budget.config.maximum,
            original_event_count=len(view or []),
            retained_event_count=len(kept_events or []),
        ).model_dump()
        if not budget.fits(row):
            raise ValueError("final reference text exceeds budget")
    return retained_rows


def _write_json(
    path: pathlib.Path,
    payload: typing.Any,
) -> None:
    """Write a payload as sorted, indented JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _file_info(path: pathlib.Path) -> dict[str, typing.Any]:
    """Return a file's SHA-256 hash and size, as recorded in manifests."""
    return {"sha256": pyine.utils.reprod.compute_hash(path, algorithm="sha256"), "size_bytes": path.stat().st_size}


def _stage(path: pathlib.Path) -> pathlib.Path:
    """Create the ``.incomplete`` staging directory for an output, failing if either already exists."""
    stage = path.with_name(f"{path.name}.incomplete")
    if path.exists() or stage.exists():
        raise FileExistsError(f"output or incomplete directory already exists: {path}")
    stage.mkdir(parents=True)
    return stage


def _check_roots(
    author_dir: pathlib.Path,
    public_dir: pathlib.Path,
    source_paths: typing.Iterable[pathlib.Path] = (),
) -> None:
    """Fail if the author, public, and source roots overlap or the outputs already exist."""
    roots = [author_dir.resolve(), public_dir.resolve()]
    outputs = [*roots, *(root.with_name(f"{root.name}.incomplete") for root in roots)]
    sources = [path.resolve() for path in source_paths]
    for index, output in enumerate(outputs):
        for other in [*outputs[index + 1 :], *sources]:
            if output == other or output.is_relative_to(other) or other.is_relative_to(output):
                raise ValueError("author, public, and source roots must not overlap")
    for output in roots:
        if output.exists() or output.with_name(f"{output.name}.incomplete").exists():
            raise FileExistsError(f"output or incomplete directory already exists: {output}")


def _rank_alternatives(
    recipient: _Context,
    pool: list[_Context],
    seed: int,
) -> list[list[_Context]]:
    """Rank distinct negative outcomes while retaining their ordered source executions."""
    eligible = [
        context
        for context in pool
        if context.input_key != recipient.input_key
        and not legacy.compare_candidate(recipient.outcome, context.outcome.value)
    ]
    eligible.sort(
        key=lambda context: (
            contract.opaque_id("outcome_rank", seed, recipient.reference.identifier, _value(context.outcome.value)),
            context.reference.identifier,
        )
    )
    distinct: list[list[_Context]] = []
    for context in eligible:
        for sources in distinct:
            if (
                _same_outcome(context.outcome, sources[0].outcome)
                or _value(context.outcome.value)[0] == _value(sources[0].outcome.value)[0]
            ):
                sources.append(context)
                break
        else:
            distinct.append([context])
    return distinct


def _first_donor_with_events(
    candidates: typing.Iterable[_Context],
    search_limit: int,
    extracted: typing.Callable[[str], events.ExtractedEvents] | None,
    counts: collections.Counter[str],
) -> _Context | None:
    """Pick the first ranked candidate within the limit; check its events only when they are spliced."""
    for donor in itertools.islice(candidates, search_limit):
        if extracted is not None:
            try:
                extracted(donor.reference.identifier)
            except events.IneligibleTraceError:
                counts["donors.ineligible_reasoning_attempts"] += 1
                continue
        return donor
    return None


def _effective_bug_donors(
    recipient: _Context,
    bugged_variants: list[_Context],
    seed: int,
) -> list[_Context]:
    """Order bugged variants by seed, keeping those whose outcome on the recipient's input differs."""
    ordered = sorted(
        bugged_variants, key=lambda context: contract.opaque_id("bug_donor", seed, context.reference.identifier)
    )
    return [
        donor
        for donor in ordered
        if not donor.reference_input_keys.isdisjoint(recipient.reference_input_keys)
        and not legacy.compare_candidate(recipient.outcome, donor.outcome.value)
    ]


DONOR_CATEGORIES = {
    "alternate_input": "donor.alternate_input",
    "buggy_reasoning": "donor.bugged",
    "original_reasoning": "donor.original",
}


def _build_group(
    recipient: _Context,
    donor: _Context | None,
    original: _Context | None,
    candidates: list[_Candidate],
    view: list[contract.Event] | None,
    reference: list[contract.Event],
    pairing: str,
    pairing_kind: contract.PairingKind,
    variant: str,
    view_idx: int,
    recipe: contract.Recipe,
    budget: _TextBudget,
    counts: collections.Counter[str],
    displayed_labels: dict[bytes, bool],
) -> list[dict[str, typing.Any]]:
    """Build one query group's rows, fit them to the budget, and assign their IDs and categories.

    Raises ``IneligibleTraceError`` when the group cannot be emitted; otherwise updates ``counts``
    and ``displayed_labels``.
    """
    construction = contract.opaque_id("construction", pairing, variant, view_idx)
    context_id = contract.opaque_id(
        "context", recipient.reference.identifier, recipient.task["code_string"], recipient.input_key
    )
    expected = recipient.task["source_expected"]
    task = {key: value for key, value in recipient.task.items() if key != "source_expected"}
    rows: list[dict[str, typing.Any]] = []
    for candidate in candidates:
        candidate_text, candidate_json = _value(candidate.context.outcome.value)
        visible_candidate = json.loads(candidate_json) if candidate_json is not None else candidate_text
        label = legacy.compare_candidate(recipient.outcome, candidate.context.outcome.value)
        if legacy.compare_candidate(recipient.outcome, visible_candidate) != label:
            raise events.IneligibleTraceError("candidate_display_changes_truth")
        outcome_text, outcome_json = _value(recipient.outcome.value)
        rows.append(
            {
                **task,
                "row_id": "pending",
                "export_split": recipient.reference.export_split,
                "problem_id": _problem_id(recipient),
                "context_id": context_id,
                "query_group_id": "pending",
                "candidate_output_text": candidate_text,
                "candidate_output_json": candidate_json,
                "label": label,
                "reasoning_events": [],
                "category_labels": [],
                "author": contract.AuthorInfo(
                    source_identifier=recipient.reference.identifier,
                    problem_identifier=recipient.reference.problem_identifier,
                    donor_identifier=donor.reference.identifier if donor else None,
                    original_identifier=original.reference.identifier if original else None,
                    candidate_source_identifiers=candidate.sources,
                    candidate_roles=candidate.roles,
                    pairing_id=pairing,
                    pairing_kind=pairing_kind,
                    requested_variant=variant,
                    outcome_kind=recipient.outcome.kind,
                    outcome_text=outcome_text,
                    outcome_json=outcome_json,
                    source_test_expected_output_text=expected[0],
                    source_test_expected_output_json=expected[1],
                    source_test_passed=recipient.outcome.expected_matches,
                    outcome_evidence=recipient.outcome_evidence,
                    statement_evidence="stored_native_events" if view is not None else "disabled",
                    native_event_count=recipient.native_event_count,
                ).model_dump(),
            }
        )
    rows = _budget_group(
        rows,
        candidates,
        view,
        reference,
        budget,
        contract.opaque_id("prune", construction),
        recipe.max_oracle_matches,
    )
    if sum(row["label"] is True for row in rows) != 1:
        raise ValueError("a query group must contain exactly one distinct positive")
    task_labels = [(_displayed_task_key(row), row["label"]) for row in rows]
    if any(displayed_labels.get(key, label) is not label for key, label in task_labels):
        raise events.IneligibleTraceError("contradicts_displayed_label")
    retained_events = rows[0]["reasoning_events"]
    surviving_origins = {event["origin_token"] for event in retained_events or []}
    recipient_origin = reference[0].origin_token if reference else None
    donor_survived = variant != "faithful" and bool(surviving_origins - {recipient_origin})
    if retained_events is None:
        final_view = "reasoning.disabled"
    elif not retained_events:
        final_view = "reasoning.empty"
    else:
        final_view = f"reasoning.{variant}" if donor_survived else "reasoning.faithful"
    categories = [*recipient.categories, final_view]
    if donor_survived:
        categories.append(DONOR_CATEGORIES[pairing_kind])
    elif variant != "faithful":
        counts[f"splices.{variant}.no_surviving_donor"] += 1
    if len(retained_events or []) < len(view or []):
        categories.append("reasoning.budget_truncated")
    group_id = contract.opaque_id(
        "group",
        construction,
        task,
        SCHEMA_VERSION,
        STATEMENT_TARGET_VERSION,
        consumer.CONTRACT,
        recipe.budget.model_dump(mode="json", exclude={"tokenizer_file"}),
        None if retained_events is None else [event["event_id"] for event in retained_events],
        [row["candidate_output_text"] for row in rows],
    )
    for row in rows:
        row["query_group_id"] = group_id
        row["row_id"] = contract.opaque_id("row", group_id, row["candidate_output_text"], row["candidate_output_json"])
        row["category_labels"] = sorted(categories)
        contract.AuthorExample.model_validate(row)
    rows.sort(key=lambda row: row["row_id"])
    counts["emitted.groups"] += 1
    counts["emitted.rows"] += len(rows)
    counts["labels.true"] += 1
    counts["labels.false"] += len(rows) - 1
    counts[f"groups.size.{len(rows)}"] += 1
    counts["queries.extra_retained"] += sum(
        "additional_observed_negative" in row["author"]["candidate_roles"]
        and not any(role in row["author"]["candidate_roles"] for role in ("recipient", "donor", "original"))
        for row in rows
    )
    counts["queries.extra_requested"] += recipe.queries.max_additional_observed_negatives
    counts["queries.extra_shortage"] += recipe.queries.max_additional_observed_negatives - sum(
        not candidate.required for candidate in candidates
    )
    counts["queries.extra_budget_excluded"] += len(candidates) - len(rows)
    counts.update(f"groups.{category}" for category in categories)
    for event in retained_events or []:
        counts[f"statements.{str(event['targets']['statement_correct']).lower()}"] += 1
    counts.update(_baseline_counts(rows, final_view))
    displayed_labels.update(task_labels)
    return rows


def _baseline_counts(
    rows: list[dict[str, typing.Any]],
    final_view: str,
) -> collections.Counter[str]:
    """Score the displayed-return shortcut on one query group, keyed by split and final view."""
    counts: collections.Counter[str] = collections.Counter()
    prefix = f"baseline.{rows[0]['export_split']}"
    shown_return = _last_shown_return(rows[0])
    if shown_return is not None and any(row["candidate_output_text"] == shown_return for row in rows):
        counts[f"{prefix}.{final_view}.applicable_rows"] += len(rows)
        counts[f"{prefix}.{final_view}.errors"] += sum(
            (row["candidate_output_text"] == shown_return) != row["label"] for row in rows
        )
    else:
        counts[f"{prefix}.unavailable_rows"] += len(rows)
    return counts


def _displayed_task_key(row: dict[str, typing.Any]) -> bytes:
    """Digest the model-visible task and candidate, which alone determine the output label."""
    return hashlib.sha256(consumer.render_example({**row, "reasoning_events": None}).encode()).digest()


def _last_shown_return(row: dict[str, typing.Any]) -> str | None:
    """Use only displayed return assertions; never treat stream fragments as complete."""
    entrypoint = row["entrypoint_name"]
    if entrypoint is None:
        return None
    name = entrypoint.rsplit(".", 1)[-1]
    for event in reversed(row["reasoning_events"] or []):
        if event["kind"] == "return" and event["code_object"].split(":", 1)[0] == name:
            return event["return_value"]
    return None


def export_statements(
    config: legacy.EvalExportConfig,
    author_output_dir: pathlib.Path,
    recipe: contract.Recipe | None = None,
) -> dict[str, typing.Any]:
    """Construct v2 examples from native traces and publish author and public artifacts.

    Source validation, whole-problem splits, coordinated candidates, reasoning edits,
    budgeting, and statement grading precede artifact promotion. Each construction
    pairing with an enabled splice and a donor also receives one faithful comparison
    view sharing its candidates; a recipient without such a pairing receives a single
    standalone faithful view built from its alternate-input candidates. Stored execution
    is sufficient by default; only explicit certification settings re-execute programs.
    Source readers may prepare metadata caches beside their LMDB directories.

    Args:
        config: Source LMDB paths, original split artifact, public output directory,
            deterministic selection/split settings, and certification/write controls.
        author_output_dir: New private artifact root. Must not overlap the public root
            or native sources; neither it nor its sibling ``.incomplete`` may exist.
        recipe: Construction, budget, visibility, coverage, and mixture settings. None uses
            ``contract.Recipe`` defaults. Tokenizer paths in an explicit recipe must
            already resolve to readable local resources; use ``load_recipe`` for YAML.

    Returns:
        The public manifest saved as ``export_manifest.json``, including per-split row
        counts, permitted summaries, contract versions, and component file hashes.
        Full source provenance and construction diagnostics stay in the author root.

    Raises:
        FileExistsError: An output root or its sibling incomplete directory exists.
        ValueError: Roots overlap, sources or completed rows fail validation, or a
            configured coverage minimum is unknown or unmet.

    Notes:
        Each artifact is validated in a sibling incomplete directory before renaming.
        A failure can leave incomplete data for inspection. Once the author artifact
        is promoted, a public-projection failure does not remove it; retry projection
        into a fresh destination with ``project_statements``.
    """
    recipe = recipe or contract.Recipe()
    _check_roots(author_output_dir, config.output_dir, config.source_lmdb_paths)
    started = time.perf_counter()
    readers, source, references = _load_sources(config)
    stage = _stage(author_output_dir)
    counts: collections.Counter[str] = collections.Counter(
        dict.fromkeys(
            (
                "emitted.groups",
                "emitted.rows",
                "emitted.contexts",
                "labels.true",
                "labels.false",
                "queries.extra_requested",
                "queries.extra_retained",
                "queries.extra_shortage",
                "queries.extra_budget_excluded",
                "requested.whole_replacement",
                "requested.partial_suffix",
                "statements.true",
                "statements.false",
            ),
            0,
        )
    )
    seen_ids: set[str] = set()
    emitted_contexts: set[str] = set()
    displayed_labels: dict[bytes, bool] = {}  # identical model-visible tasks must never disagree
    budget = _TextBudget(recipe.budget)
    with (
        (stage / "skipped_examples.jsonl").open("w") as skip_file,
        (stage / "certification.jsonl").open("w") as cert_file,
    ):

        def skip(
            identifier: str,
            reason: str,
        ) -> None:
            """Count a skipped source trace and log its reason."""
            counts[f"skipped.{reason}"] += 1
            skip_file.write(contract.canonical_json({"source_identifier": identifier, "reason": reason}) + "\n")

        contexts = _resolve_originals(
            _index_contexts(readers, references, config, recipe, counts, skip, cert_file), counts
        )
        pools: dict[str, list[_Context]] = collections.defaultdict(list)
        buggy_donors: dict[str, list[_Context]] = collections.defaultdict(list)
        for context in contexts.values():
            pools[context.pool_key].append(context)
            if "code.bugged" in context.categories and context.original_identifier is not None:
                buggy_donors[context.original_identifier].append(context)

        ineligible_reasons: dict[str, str] = {}  # lru_cache does not memoize raised exceptions

        @functools.lru_cache(maxsize=recipe.event_cache_size)
        def load_events(identifier: str) -> events.ExtractedEvents:
            """Extract a trace's scoped events, cached by identifier."""
            pointer = contexts[identifier].reference
            trace = readers[pointer.reader_idx][pointer.trace_idx]
            return events.extract_events(trace, recipe.statements.scope)

        def extracted(identifier: str) -> events.ExtractedEvents:
            """Return a trace's events, re-raising a known ineligibility without extracting again."""
            if identifier in ineligible_reasons:
                raise events.IneligibleTraceError(ineligible_reasons[identifier])
            try:
                return load_events(identifier)
            except events.IneligibleTraceError as error:
                ineligible_reasons[identifier] = str(error)
                raise

        def recipient_groups(recipient: _Context) -> typing.Iterator[list[dict[str, typing.Any]]]:
            """Yield the emitted query groups of one displayed recipient, recording skips."""
            is_bugged = "code.bugged" in recipient.categories
            original = contexts.get(recipient.original_identifier or "") if is_bugged else None
            if is_bugged and (
                original is None
                or original.reference_input_keys.isdisjoint(recipient.reference_input_keys)
                or "code.bugged" in original.categories
            ):
                skip(recipient.reference.identifier, "missing_same_input_original")
                return
            if original is not None and legacy.compare_candidate(recipient.outcome, original.outcome.value):
                counts["bugs.ineffective_outcome_contrast"] += 1
            reasoning = recipe.statements.include_reasoning
            try:
                reference = extracted(recipient.reference.identifier).events if reasoning else []
            except events.IneligibleTraceError as error:
                skip(recipient.reference.identifier, str(error))
                return
            alternative_sources = _rank_alternatives(recipient, pools[recipient.pool_key], config.seed)
            alternatives = [sources[0] for sources in alternative_sources]
            pairings: list[tuple[_Context | None, contract.PairingKind, contract.Splice]] = [
                (
                    _first_donor_with_events(
                        itertools.chain.from_iterable(alternative_sources),
                        recipe.donor_search_limit,
                        extracted if reasoning and recipe.variants.alternate_input_suffix.enabled else None,
                        counts,
                    ),
                    "alternate_input",
                    recipe.variants.alternate_input_suffix,
                )
            ]
            if recipe.variants.buggy_reasoning_suffix.enabled and not is_bugged:
                bug_candidates = _effective_bug_donors(
                    recipient, buggy_donors[recipient.reference.identifier], config.seed
                )
                bug_donor = _first_donor_with_events(
                    bug_candidates, recipe.donor_search_limit, extracted if reasoning else None, counts
                )
                if not bug_candidates:
                    skip(recipient.reference.identifier, "no_effective_buggy_donor")
                elif bug_donor is None:
                    skip(recipient.reference.identifier, "no_eligible_buggy_donor")
                else:
                    pairings.append((bug_donor, "buggy_reasoning", recipe.variants.buggy_reasoning_suffix))
            if recipe.variants.original_reasoning_suffix.enabled and original is not None:
                if legacy.compare_candidate(recipient.outcome, original.outcome.value):
                    skip(recipient.reference.identifier, "ineffective_original_donor")
                elif _first_donor_with_events([original], 1, extracted if reasoning else None, counts) is None:
                    skip(recipient.reference.identifier, "no_eligible_original_donor")
                else:
                    pairings.append((original, "original_reasoning", recipe.variants.original_reasoning_suffix))
            spliced = [donor is not None and splice.enabled and reasoning for donor, _, splice in pairings]
            for pairing_idx, (donor, donor_kind, splice) in enumerate(pairings):
                pairing = contract.opaque_id(
                    "pairing",
                    config.seed,
                    recipient.reference.identifier,
                    donor.reference.identifier if donor else None,
                    donor_kind,
                )
                views: list[tuple[str, int, list[contract.Event] | None]] = []
                # each spliced pairing gets a faithful comparison; without any, one standalone faithful view
                if (
                    recipe.variants.faithful
                    and (not is_bugged or recipe.variants.bugged_own_reasoning)
                    and (spliced[pairing_idx] or (pairing_idx == 0 and not any(spliced)))
                ):
                    views.append(("faithful", 0, reference if reasoning else None))
                if donor is not None and splice.enabled and reasoning:
                    for view_idx in range(splice.views_per_pairing):
                        whole = (
                            random.Random(contract.opaque_id("whole", pairing, view_idx)).random()
                            < splice.whole_replacement_probability
                        )
                        variant = "whole_replacement" if whole else "partial_suffix"
                        counts[f"requested.{variant}"] += 1
                        try:
                            view = events.splice_events(
                                reference,
                                extracted(donor.reference.identifier).events,
                                recipient.task["code_string"],
                                donor.task["code_string"],
                                whole,
                                contract.opaque_id("cut", pairing, view_idx),
                                tuple(splice.cut_kinds),
                            )
                        except events.IneligibleTraceError as error:
                            skip(recipient.reference.identifier, str(error))
                            continue
                        views.append((variant, view_idx, view))
                elif splice.enabled and reasoning:
                    skip(recipient.reference.identifier, "no_negative_reasoning_donor")
                bundle = _candidate_bundle(
                    recipient, donor, original, alternatives, recipe.queries.max_additional_observed_negatives
                )
                for variant, view_idx, view in views:
                    try:
                        rows = _build_group(
                            recipient,
                            donor,
                            original,
                            bundle,
                            view,
                            reference,
                            pairing,
                            donor_kind,
                            variant,
                            view_idx,
                            recipe,
                            budget,
                            counts,
                            displayed_labels,
                        )
                    except events.IneligibleTraceError as error:
                        skip(recipient.reference.identifier, str(error))
                        continue
                    yield rows

        author_schema = get_statement_export_schema(author=True)
        pending: list[dict[str, typing.Any]] = []
        with pq.ParquetWriter(stage / "examples.parquet", author_schema, compression="zstd") as writer:
            # problems stay contiguous for consumers, and each donor pool for the small event cache
            for recipient in sorted(
                contexts.values(),
                key=lambda context: (_problem_id(context), context.pool_key, context.reference.identifier),
            ):
                if not recipe.code_selection.selects(recipient.categories):
                    continue
                for rows in recipient_groups(recipient):
                    for row in rows:
                        if row["row_id"] in seen_ids:
                            raise ValueError("duplicate row identity")
                        seen_ids.add(row["row_id"])
                    emitted_contexts.add(recipient.reference.identifier)
                    pending.extend(rows)
                    if len(pending) >= config.parquet_batch_size:
                        writer.write_table(pa.Table.from_pylist(pending, schema=author_schema))
                        pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=author_schema))
    counts["emitted.contexts"] = len(emitted_contexts)
    for name, minimum in recipe.coverage.minimum_counts.items():
        if name not in counts:
            raise ValueError(f"coverage counter was never recorded (unknown name or zero count): {name}")
        if counts[name] < minimum:
            raise ValueError(f"coverage minimum unmet: {name} = {counts[name]} < {minimum}")
    _write_json(stage / "coverage.json", dict(counts))
    resolved_recipe = recipe.model_dump(mode="json")
    if recipe.budget.tokenizer_file is not None:
        shutil.copyfile(recipe.budget.tokenizer_file, stage / "tokenizer.json")
        resolved_recipe["budget"]["tokenizer_file"] = "tokenizer.json"
    _write_json(stage / "resolved_recipe.json", resolved_recipe)
    _write_json(
        stage / "installed_packages.json",
        dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            )
        ),
    )
    shutil.copyfile(pathlib.Path(consumer.__file__), stage / "pyine_consumer.py")
    shutil.copyfile(pathlib.Path(__file__).with_name("EVAL_EXPORT_SPEC_V2.md"), stage / "EVAL_EXPORT_SPEC.md")
    shutil.copyfile(pathlib.Path(__file__).with_name("EVAL_EXPORT_GUIDE.md"), stage / "EVAL_EXPORT_GUIDE.md")
    _write_json(
        stage / "export_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "public": contract.Example.model_json_schema(),
            "author": contract.AuthorExample.model_json_schema(),
        },
    )
    author_manifest = {
        "schema_version": SCHEMA_VERSION,
        "renderer": consumer.CONTRACT,
        "statement_target_version": STATEMENT_TARGET_VERSION,
        "outcome_policy_version": legacy.OUTCOME_POLICY_VERSION,
        "comparison_policy_version": legacy.COMPARISON_POLICY_VERSION,
        "budget": {
            **resolved_recipe["budget"],
            "tokenizer_backend": "tokenizers" if budget.tokenizer is not None else None,
            "tokenizer_backend_version": importlib.metadata.version("tokenizers")
            if budget.tokenizer is not None
            else None,
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
        },
        "source": source,
        "counts": dict(counts),
        "duration_seconds": time.perf_counter() - started,
        "files": {path.name: _file_info(path) for path in sorted(stage.iterdir()) if path.is_file()},
    }
    _write_json(stage / "author_manifest.json", author_manifest)
    _validate_author(stage)
    stage.rename(author_output_dir)
    return project_statements(author_output_dir, config.output_dir, recipe.visibility, config.parquet_batch_size)


def _check_files(
    root: pathlib.Path,
    manifest: dict[str, typing.Any],
) -> None:
    """Check that every manifest file is a direct, non-symlink child with the recorded hash and size."""
    for name, expected in manifest["files"].items():
        if pathlib.Path(name).name != name:
            raise ValueError("artifact file names must be direct children")
        path = root / name
        if path.is_symlink() or not path.is_file() or _file_info(path) != expected:
            raise ValueError(f"artifact hash or size mismatch: {name}")


def _validate_author_row(
    row: dict[str, typing.Any],
    text_budget: _TextBudget,
) -> None:
    """Replay one author row's label, presented positions, and exact reference-text cost."""
    contract.AuthorExample.model_validate(row)
    author = row["author"]
    outcome = legacy.ResolvedOutcome(
        kind=author["outcome_kind"],
        value=json.loads(author["outcome_json"]) if author["outcome_json"] is not None else author["outcome_text"],
        expected_matches=author["source_test_passed"],
        comparison_reason="author evidence replay",
    )
    candidate = (
        json.loads(row["candidate_output_json"])
        if row["candidate_output_json"] is not None
        else row["candidate_output_text"]
    )
    if row["label"] is not legacy.compare_candidate(outcome, candidate):
        raise ValueError("candidate truth differs from author outcome evidence")
    reasoning = row["reasoning_events"]
    if reasoning is not None:
        if [event["view_index"] for event in reasoning] != list(range(len(reasoning))):
            raise ValueError("invalid presented event positions")
        if any(event["targets"]["statement_correct"] is None for event in reasoning):
            raise ValueError("author statement targets must be present")
    budget = row["budget_metadata"]
    if budget["unit"] != text_budget.config.unit or budget["maximum"] != text_budget.config.maximum:
        raise ValueError("row budget differs from resolved recipe")
    if budget["maximum"] is not None and budget["cost"] > budget["maximum"]:
        raise ValueError("recorded text exceeds budget")
    if text_budget.cost(row) != budget["cost"]:
        raise ValueError("reference text cost mismatch")
    if budget["retained_event_count"] != len(reasoning or []):
        raise ValueError("retained event count mismatch")


def _validate_author_group(rows: list[dict[str, typing.Any]]) -> None:
    """Check that a query group shares one task/view, one positive, and every required role."""
    if len({contract.canonical_json({key: row[key] for key in GROUP_SHARED_FIELDS}) for row in rows}) != 1:
        raise ValueError("query group reasoning/task fields differ")
    if sum(row["label"] is True for row in rows) != 1:
        raise ValueError("group must contain one positive")
    texts = [row["candidate_output_text"] for row in rows]
    if len(set(texts)) != len(texts):
        raise ValueError("duplicate displayed candidate within query group")
    author = rows[0]["author"]
    required_roles = {"recipient"}
    if author["donor_identifier"] is not None:
        required_roles.add("donor")
    if author["original_identifier"] is not None:
        required_roles.add("original")
    if not required_roles <= {role for row in rows for role in row["author"]["candidate_roles"]}:
        raise ValueError("group is missing a required query role")


def _validate_author(root: pathlib.Path) -> dict[str, typing.Any]:
    """Validate an author artifact's contracts, files, schema, rows, and groups; return its manifest."""
    manifest = json.loads((root / "author_manifest.json").read_text())
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["renderer"] != consumer.CONTRACT
        or manifest["statement_target_version"] != STATEMENT_TARGET_VERSION
        or manifest["outcome_policy_version"] != legacy.OUTCOME_POLICY_VERSION
        or manifest["comparison_policy_version"] != legacy.COMPARISON_POLICY_VERSION
    ):
        raise ValueError("incompatible author schema/renderer/target contract")
    _check_files(root, manifest)
    budget_payload = json.loads((root / "resolved_recipe.json").read_text())["budget"]
    if budget_payload["unit"] == "tokens":
        if manifest["budget"]["tokenizer_backend_version"] != importlib.metadata.version("tokenizers"):
            raise ValueError("tokenizer backend version differs from the author artifact")
        if budget_payload["tokenizer_file"] != "tokenizer.json":
            raise ValueError("author tokenizer must be an artifact-local resource")
        budget_payload["tokenizer_file"] = root / "tokenizer.json"
    text_budget = _TextBudget(contract.Budget.model_validate(budget_payload))
    parquet = pq.ParquetFile(root / "examples.parquet")
    if parquet.schema_arrow != get_statement_export_schema(author=True):
        raise ValueError("author Arrow schema mismatch")
    row_ids: set[str] = set()
    group_ids: set[str] = set()
    displayed_labels: dict[bytes, bool] = {}
    finished: dict[str, set[str]] = {"problem_id": set(), "context_id": set()}
    current: dict[str, str | None] = {"problem_id": None, "context_id": None}
    rows = consumer.iter_examples(root / "examples.parquet")
    for group_id, group_rows in itertools.groupby(rows, key=lambda row: row["query_group_id"]):
        if group_id in group_ids:
            raise ValueError("noncontiguous or repeated author query group")
        group_ids.add(group_id)
        group = list(group_rows)
        for key in ("problem_id", "context_id"):
            if group[0][key] != current[key]:
                if group[0][key] in finished[key]:
                    raise ValueError(f"noncontiguous author {key}")
                finished[key].add(group[0][key])
                current[key] = group[0][key]
        for row in group:
            if row["row_id"] in row_ids:
                raise ValueError("duplicate row ID")
            row_ids.add(row["row_id"])
            _validate_author_row(row, text_budget)
            if displayed_labels.setdefault(_displayed_task_key(row), row["label"]) is not row["label"]:
                raise ValueError("identical displayed task and candidate carry different labels")
        _validate_author_group(group)
    if len(row_ids) != manifest["counts"].get("emitted.rows", 0) or len(group_ids) != manifest["counts"].get(
        "emitted.groups", 0
    ):
        raise ValueError("author counts do not match stored rows")
    return typing.cast("dict[str, typing.Any]", manifest)


def _final_view(row: dict[str, typing.Any]) -> str:
    """Return the realized reasoning view of an author row from its categories."""
    return next(
        label
        for label in row["category_labels"]
        if label.startswith("reasoning.") and label != "reasoning.budget_truncated"
    )


def _mixture_family(row: dict[str, typing.Any]) -> str:
    """Assign an author row's query group to its ``contract.Mixture`` family."""
    if "code.bugged" in row["category_labels"]:
        return "bugged_code"
    return "buggy_reasoning" if row["author"]["pairing_kind"] == "buggy_reasoning" else "clean_code"


def _mixture_selection(
    path: pathlib.Path,
    mixture: contract.Mixture,
    batch_size: int,
) -> tuple[set[str], dict[str, collections.Counter[str]]]:
    """Select whole pairings whose query groups realize the mixture shares in every split.

    Within each split and family, pairings are ranked by a deterministic draw and kept whenever
    that brings the family's group count closer to its target. The family with the least supply
    relative to its share is kept entirely. A family with a positive share but no groups in a
    nonempty split cannot be realized and fails projection.

    Returns:
        The kept pairing IDs, and the available query groups per split and family.
    """
    pairing_sizes: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    supply: dict[str, collections.Counter[str]] = {split: collections.Counter() for split in SPLITS}
    columns = ["export_split", "query_group_id", "category_labels", "author.pairing_id", "author.pairing_kind"]
    previous_group = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            if row["query_group_id"] == previous_group:
                continue
            previous_group = row["query_group_id"]
            family = _mixture_family(row)
            supply[row["export_split"]][family] += 1
            pairing_sizes[row["export_split"], family][row["author"]["pairing_id"]] += 1
    shares = mixture.model_dump()
    kept: set[str] = set()
    for split, available in supply.items():
        missing = [family for family, share in shares.items() if share > 0 and not available[family]]
        if available and missing:
            raise ValueError(f"mixture families {missing} have no query groups in the {split} split")
        if not available:
            continue
        total = min(available[family] / share for family, share in shares.items() if share > 0)
        for family, share in shares.items():
            target, kept_groups = share * total, 0
            for pairing_id, size in sorted(
                pairing_sizes[split, family].items(), key=lambda item: contract.opaque_id("mixture", item[0])
            ):
                if abs(kept_groups + size - target) < abs(kept_groups - target):
                    kept.add(pairing_id)
                    kept_groups += size
    return kept, supply


def _project_group(
    author_group: list[dict[str, typing.Any]],
    visibility: contract.Visibility,
    split: str,
    counts: dict[str, collections.Counter[str]],
    problems: dict[str, set[str]],
    contexts: dict[str, set[str]],
    groups: dict[str, set[str]],
) -> list[dict[str, typing.Any]]:
    """Project one author query group onto public fields and visibility, updating public counts."""
    rows: list[dict[str, typing.Any]] = []
    for author_row in author_group:
        row = {field: author_row[field] for field in contract.Example.model_fields}
        if not getattr(visibility.categories, split):
            row["category_labels"] = None
        if not getattr(visibility.output_targets, split):
            row["label"] = None
        if not getattr(visibility.statement_targets, split) and row["reasoning_events"] is not None:
            for event in row["reasoning_events"]:
                event["targets"] = {"statement_correct": None}
        if consumer.render_example(author_row) != consumer.render_example(row):
            raise ValueError("public projection changed displayed task inputs")
        contract.Example.model_validate(row)
        problems[split].add(row["problem_id"])
        contexts[split].add(row["context_id"])
        if row["query_group_id"] not in groups[split]:
            counts[split].update(row["category_labels"] or [])
        groups[split].add(row["query_group_id"])
        counts[split]["rows"] += 1
        counts[split]["target.withheld" if row["label"] is None else f"label.{str(row['label']).lower()}"] += 1
        rows.append(row)
    return rows


def project_statements(
    author_dir: pathlib.Path,
    public_dir: pathlib.Path,
    visibility: contract.Visibility | None = None,
    batch_size: int = 256,
    mixture: contract.Mixture | None | typing.Literal["recorded"] = "recorded",
) -> dict[str, typing.Any]:
    """Validate a completed author artifact and project its public fields and visibility.

    Projection uses the stored rows and evidence without native sources, program
    execution, donor selection, or new labels. It requires compatible contract versions,
    an installed renderer that reproduces every recorded text cost, and the recorded
    tokenizers backend version when used. The public artifact ships the author's
    hashed ``pyine_consumer.py``, not the installed copy.

    Args:
        author_dir: Completed author artifact containing its manifest and all hashed
            components, including an artifact-local tokenizer for token budgets.
        public_dir: New destination, separate from the author root. Neither this path
            nor its sibling ``.incomplete`` directory may already exist.
        visibility: Per-split category and target visibility. None reuses the settings
            in the author's resolved recipe; construction and task content stay fixed.
        batch_size: Positive number of rows per Parquet read/write batch.
        mixture: Target family shares for every public split, None to keep every group, or
            ``"recorded"`` to reuse the author's resolved recipe. Whole pairings (spliced
            views with their faithful twin) are selected by a deterministic ranking.

    Returns:
        The public manifest written to ``public_dir/export_manifest.json``. All three
        split files are written, including empty splits, with public counts and hashes.

    Raises:
        FileExistsError: The public destination or its incomplete directory exists.
        ValueError: Roots overlap, the batch size is invalid, artifact integrity, schema,
            renderer, tokenizer, grouping, labels, or budgets fail validation, or a mixture
            family with a positive share has no groups in a nonempty split.
        OSError: Required artifact files cannot be read or output files cannot be written.

    Notes:
        The author artifact is not modified. Public files are promoted by renaming a
        validated incomplete directory; a failed projection may leave that directory.
    """
    author_dir, public_dir = author_dir.resolve(), public_dir.resolve()
    for output in (public_dir, public_dir.with_name(f"{public_dir.name}.incomplete")):
        if output == author_dir or output.is_relative_to(author_dir) or author_dir.is_relative_to(output):
            raise ValueError("author and public roots must not overlap")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = _validate_author(author_dir)
    resolved_recipe = json.loads((author_dir / "resolved_recipe.json").read_text())
    visibility = visibility or contract.Visibility.model_validate(resolved_recipe["visibility"])
    if mixture == "recorded":
        mixture = None if resolved_recipe["mixture"] is None else contract.Mixture(**resolved_recipe["mixture"])
    kept_pairings, supply = (None, None)
    if mixture is not None:
        kept_pairings, supply = _mixture_selection(author_dir / "examples.parquet", mixture, batch_size)
    stage = _stage(public_dir)
    schema = get_statement_export_schema()
    counts: dict[str, collections.Counter[str]] = {split: collections.Counter() for split in SPLITS}
    problems: dict[str, set[str]] = {split: set() for split in SPLITS}
    contexts: dict[str, set[str]] = {split: set() for split in SPLITS}
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    kept_groups: dict[str, collections.Counter[str]] = {split: collections.Counter() for split in SPLITS}
    shortcut_counts: collections.Counter[str] = collections.Counter()
    for split in SPLITS:
        pending: list[dict[str, typing.Any]] = []
        with pq.ParquetWriter(stage / f"{split}.parquet", schema, compression="zstd") as writer:
            for author_group in consumer.iter_groups(author_dir / "examples.parquet", batch_size):
                if author_group[0]["export_split"] != split:
                    continue
                if kept_pairings is not None and author_group[0]["author"]["pairing_id"] not in kept_pairings:
                    continue
                kept_groups[split][_mixture_family(author_group[0])] += 1
                shortcut_counts.update(_baseline_counts(author_group, _final_view(author_group[0])))
                pending.extend(_project_group(author_group, visibility, split, counts, problems, contexts, groups))
                if len(pending) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
        counts[split]["problems"] = len(problems[split])
        counts[split]["contexts"] = len(contexts[split])
        counts[split]["groups"] = len(groups[split])
        if pq.ParquetFile(stage / f"{split}.parquet").metadata.num_rows != counts[split]["rows"]:
            raise ValueError("projected row counts differ")
    for name in ("pyine_consumer.py", "EVAL_EXPORT_SPEC.md", "EVAL_EXPORT_GUIDE.md"):
        shutil.copyfile(author_dir / name, stage / name)
    if manifest["budget"]["unit"] == "tokens":
        shutil.copyfile(author_dir / "tokenizer.json", stage / "tokenizer.json")
    _write_json(
        stage / "export_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "row": contract.Example.model_json_schema(),
            "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
            "model_input_fields": [
                "code_string",
                "entrypoint_name",
                "invocation_kind",
                "invocation_text",
                "inputs_text",
                "inputs_json",
                "observed_bindings",
                "supplied_stdin",
                "problem_statement",
                "candidate_output_text",
                "candidate_output_json",
                "reasoning_events (assertions only)",
            ],
            "event_assertions": [
                "kind",
                "code_object",
                "line",
                "arguments",
                "deltas",
                "return_value",
                "stdout_since_prev_capture",
                "stderr_since_prev_capture",
                "exception",
            ],
        },
    )
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "renderer": manifest["renderer"],
        "statement_target_version": STATEMENT_TARGET_VERSION,
        "outcome_policy_version": legacy.OUTCOME_POLICY_VERSION,
        "comparison_policy_version": legacy.COMPARISON_POLICY_VERSION,
        "comparison_options": pyine.utils.code.output_compare.get_default_comparison_config().model_dump(),
        "value_display_version": "value-display-v1",
        "budget": manifest["budget"],
        "visibility": visibility.model_dump(),
        "partition_row_counts": {split: counts[split]["rows"] for split in SPLITS},
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "shortcut_baseline": {
            "method": "last-visible-entrypoint-return-exact-display-v1",
            "counts": {
                name: value
                for name, value in sorted(shortcut_counts.items())
                if getattr(visibility.categories, name.split(".")[1])
                and getattr(visibility.output_targets, name.split(".")[1])
            },
        },
        "mixture": None
        if mixture is None or supply is None
        else {
            "shares": mixture.model_dump(),
            "groups": {
                split: {
                    family: {"available": supply[split][family], "kept": kept_groups[split][family]}
                    for family in mixture.model_dump()
                }
                for split in SPLITS
                if getattr(visibility.categories, split)
            },
        },
        "files": {path.name: _file_info(path) for path in sorted(stage.iterdir()) if path.is_file()},
    }
    _write_json(stage / "export_manifest.json", public_manifest)
    _check_files(stage, public_manifest)
    stage.rename(public_dir)
    return public_manifest


def report_source(
    config: legacy.EvalExportConfig,
    recipe: contract.Recipe | None = None,
    inspect_events: bool = False,
) -> dict[str, typing.Any]:
    """Inspect source eligibility without exporting examples or executing programs.

    Args:
        config: Source, split, problem-selection, and shard-validation settings.
            The output directory and certification re-execution controls are not used.
        recipe: Recipe supplying banned-problem eligibility and native event scope.
            None uses defaults. This preflight does not apply final recipient mixtures,
            donor search limits, query construction, visibility, or text budgets.
        inspect_events: Also load stored traces to check record integrity, event
            extraction eligibility, and shared-cut availability across eligible
            alternate-input negative pairs, using that family's ``cut_kinds`` and the
            same cut rules as ``statement_events.splice_events``. False reports
            metadata-level counts only.

    Returns:
        A JSON-compatible producer report containing source provenance, inspection
        mode, counters, the assessed cut scope and kinds, and an explicit ``not_measured`` list.
        Counts describe the inspected source pool, not expected post-budget emissions.

    Raises:
        ValueError: Source shards, split identities, or problem-content hashes disagree.
        OSError: Required source or split files cannot be read.

    Notes:
        Reader initialization may write prepared metadata caches beside sources. Even
        with event inspection, buggy-pair cut coverage and task difficulty are unmeasured.
    """
    recipe = recipe or contract.Recipe()
    cut_kinds = recipe.variants.alternate_input_suffix.cut_kinds
    readers, source, references = _load_sources(config)
    counts: collections.Counter[str] = collections.Counter()
    cut_pools: dict[str, list[tuple[str, legacy.ResolvedOutcome, set[events.CutKey], set[events.CutKey]]]] = (
        collections.defaultdict(list)
    )
    for reference in references:
        reader = readers[reference.reader_idx]
        problem = reader.get_problem_data(reference.trace_idx)
        counts["selected.contexts"] += 1
        if problem.is_banned and not recipe.eligibility.include_banned_problems:
            counts["excluded.banned_problem"] += 1
            continue
        identifier = trace_utils.TraceIdentifier.from_string(reference.identifier)
        counts.update(_categories(identifier))
        if inspect_events:
            trace = reader[reference.trace_idx]
            integrity = legacy.validate_trace_record(trace, reader.get_trace_metadata(reference.trace_idx), problem)
            if integrity.outcome != "pass":
                counts["excluded.failed_integrity"] += 1
                continue
            if _is_harness_failure(trace):
                counts["excluded.harness_invocation_failure"] += 1
                continue
            try:
                extracted = events.extract_events(trace, recipe.statements.scope)
            except events.IneligibleTraceError as error:
                counts[f"excluded.{error}"] += 1
            else:
                counts["events.eligible_contexts"] += 1
                counts["events.in_scope"] += len(extracted.events)
                input_key, _ = _input_keys(_invocation(trace))
                code = trace.code_string  # alternate-input donors share the recipient's exact code
                recipient_cuts = set(events.cut_locations(extracted.events, code, code, cut_kinds, as_recipient=True))
                donor_cuts = set(events.cut_locations(extracted.events, code, code, cut_kinds, as_recipient=False))
                cut_pools[_pool_key(trace, reference)].append(
                    (input_key, integrity.selected_outcome, recipient_cuts, donor_cuts)
                )
    if inspect_events:
        for name in ("negative_pairs", "shared_cut_pairs", "no_shared_cut_pairs"):
            counts[f"cuts.alternate_input.{name}"] = 0
        for pool in cut_pools.values():
            for recipient_input, outcome, recipient_cuts, _ in pool:
                for donor_input, donor_outcome, _, donor_cuts in pool:
                    if recipient_input == donor_input or legacy.compare_candidate(outcome, donor_outcome.value):
                        continue
                    counts["cuts.alternate_input.negative_pairs"] += 1
                    kind = "shared_cut_pairs" if recipient_cuts & donor_cuts else "no_shared_cut_pairs"
                    counts[f"cuts.alternate_input.{kind}"] += 1
    return {
        "source": source,
        "inspection": "stored_event_scope" if inspect_events else "metadata_only",
        "counts": dict(counts),
        "cut_scope": "all stored-integrity-passing, event-eligible alternate-input pairs before donor limits/selection"
        if inspect_events
        else None,
        "cut_kinds": cut_kinds if inspect_events else None,
        "not_measured": ["realized_queries", "buggy_shared_cut_coverage", "post_budget_counts", "task_difficulty"]
        + ([] if inspect_events else ["event_scope_eligibility", "alternate_input_shared_cut_coverage"]),
    }
