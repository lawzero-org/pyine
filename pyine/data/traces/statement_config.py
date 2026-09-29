"""Versioned configuration and structured records for statement exports."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib  # noqa: TC003 - pydantic resolves this type at runtime
import typing

import pydantic

import pyine.data.traces.export_consumer as consumer

CodeCategory = typing.Literal[
    "code.original",
    "code.augmented",
    "code.bugged",
    "code.hinted",
    "code.misleading",
    "code.obfuscated",
    "code.stubbed",
]
"""Displayed-code categories derived from stored augmentation identities."""

PairingKind = typing.Literal["alternate_input", "buggy_reasoning", "original_reasoning"]
"""Donor family of a construction pairing: same code on another input, bugged variant, or original."""


class Record(pydantic.BaseModel):
    """Reject unknown fields throughout the v2 contract."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
    """Pydantic configuration that freezes instances and rejects unknown fields."""


class Eligibility(Record):
    """Source eligibility switches."""

    include_banned_problems: bool = False
    """Whether banned problems may supply recipients and donors."""


class CodeSelection(Record):
    """Producer selection of stored code variants."""

    mode: typing.Literal["all_available", "original_only", "bugged_only"] = "all_available"
    """Keep ``all_available`` code, unaugmented ``original_only`` code, or ``bugged_only`` code."""
    include_categories: list[CodeCategory] = pydantic.Field(default_factory=lambda: [])
    """Keep recipients having any listed category; empty keeps all."""
    exclude_categories: list[CodeCategory] = pydantic.Field(default_factory=lambda: [])
    """Drop recipients having any listed category."""

    def selects(
        self,
        categories: typing.Iterable[str],
    ) -> bool:
        """Check whether a recipient with these ``code.*`` categories is displayed.

        Args:
            categories: Displayed-code categories of one recipient execution.

        Returns:
            Whether the mode and include/exclude filters all keep the recipient. Donor
            and reference eligibility do not depend on this selection.
        """
        present = set(categories)
        if self.mode == "original_only" and "code.original" not in present:
            return False
        if self.mode == "bugged_only" and "code.bugged" not in present:
            return False
        if self.include_categories and present.isdisjoint(self.include_categories):
            return False
        return present.isdisjoint(self.exclude_categories)


class Queries(Record):
    """Coordinated observed-outcome queries."""

    construction: typing.Literal["coordinated_observed_outcomes"] = "coordinated_observed_outcomes"
    """Candidate construction method; candidates are always observed outcomes."""
    max_additional_observed_negatives: pydantic.NonNegativeInt = 1
    """Cap on extra distinct negatives from the same code on other inputs."""


class Statements(Record):
    """Displayed task and assertion selection."""

    include_reasoning: bool = True
    """Whether rows carry reasoning events; False disables every view."""
    include_problem_statement: bool = False
    """Whether rows carry the problem text."""
    scope: typing.Literal["task_execution", "all_in_program"] = "task_execution"
    """Native events kept: the task's own execution, or also a callable's module initialization."""


class Splice(Record):
    """One bounded suffix-splice recipe."""

    enabled: bool = True
    """Whether to build spliced views for this donor family."""
    views_per_pairing: pydantic.PositiveInt = 1
    """Spliced views per recipient/donor pairing."""
    whole_replacement_probability: float = pydantic.Field(default=0.1, ge=0, le=1)
    """Chance that a view is the donor's entire execution."""
    cut_kinds: list[typing.Literal["line", "call", "return"]] = pydantic.Field(
        default_factory=lambda: ["line", "call", "return"], min_length=1
    )
    """Event kinds usable as partial-splice cuts; call and return cuts also need a matching call site."""


class Variants(Record):
    """Explicit view families; no automatic cross product."""

    faithful: bool = True
    """Whether to emit faithful views: one twin per spliced pairing, or one standalone view."""
    alternate_input_suffix: Splice = pydantic.Field(default_factory=Splice)
    """Splice the same code's execution on another input."""
    buggy_reasoning_suffix: Splice = pydantic.Field(default_factory=lambda: Splice(enabled=False))
    """For non-bugged code, splice a bugged variant's execution on the same input."""
    original_reasoning_suffix: Splice = pydantic.Field(default_factory=Splice)
    """For bugged code, splice its original's execution on the same input (the intended reasoning)."""
    bugged_own_reasoning: bool = True
    """Whether to allow faithful views of bugged code."""


class Budget(Record):
    """Reference-text budget and a pinned, local tokenizer resource."""

    unit: typing.Literal["characters", "tokens"] = "characters"
    """Unicode code points (``characters``) or full-string tokenization (``tokens``), excluding
    added special tokens, truncation, and padding."""
    maximum: pydantic.NonNegativeInt | None = 16000
    """Cap on the complete reference-rendered example, including mandatory task text. None is
    unlimited; zero is a valid cap."""
    tokenizer_file: pathlib.Path | None = None
    """Local tokenizers JSON, required only for token budgets. ``load_recipe`` resolves relative
    paths beside its YAML recipe."""
    tokenizer_sha256: str | None = None
    """Required digest of the tokenizer file in token mode. Construction validates the file
    immediately; no resource is downloaded."""

    @pydantic.model_validator(mode="after")
    def validate_tokenizer(self) -> Budget:
        """Require an explicit local tokenizers JSON and digest for token mode."""
        if self.unit == "tokens":
            if self.tokenizer_file is None or self.tokenizer_sha256 is None:
                raise ValueError("token budgets require tokenizer_file and tokenizer_sha256")
            if hashlib.sha256(self.tokenizer_file.read_bytes()).hexdigest() != self.tokenizer_sha256:
                raise ValueError("tokenizer_sha256 does not match tokenizer_file")
        elif self.tokenizer_file is not None or self.tokenizer_sha256 is not None:
            raise ValueError("tokenizer settings require token budgets")
        return self


class SplitVisibility(Record):
    """Visibility settings for each derived split."""

    train: bool = True
    """Whether the setting's values are published in the train split."""
    validation: bool = True
    """Whether the setting's values are published in the validation split."""
    test: bool = True
    """Whether the setting's values are published in the test split."""


class Visibility(Record):
    """Public projection settings; never rendered as task inputs."""

    categories: SplitVisibility = pydantic.Field(
        default_factory=lambda: SplitVisibility(train=False, test=False),
    )
    """Where ``category_labels`` are published; hidden in train and test by default."""
    output_targets: SplitVisibility = pydantic.Field(default_factory=SplitVisibility)
    """Where candidate ``label`` values are published."""
    statement_targets: SplitVisibility = pydantic.Field(default_factory=SplitVisibility)
    """Where per-event ``statement_correct`` targets are published."""


class Mixture(Record):
    """Target shares of query groups per construction family, enforced in each public split."""

    clean_code: float = pydantic.Field(default=0.0, ge=0, le=1)
    """Share of non-bugged code with faithful and alternate-input views."""
    buggy_reasoning: float = pydantic.Field(default=0.0, ge=0, le=1)
    """Share of non-bugged code paired with a bugged variant's reasoning."""
    bugged_code: float = pydantic.Field(default=0.0, ge=0, le=1)
    """Share of bugged code with its own, alternate-input, or original reasoning."""

    @pydantic.model_validator(mode="after")
    def validate_total(self) -> Mixture:
        """Require shares that sum to one."""
        if not math.isclose(self.clean_code + self.buggy_reasoning + self.bugged_code, 1.0):
            raise ValueError("mixture shares must sum to 1")
        return self


class Coverage(Record):
    """Optional minima over named coverage counters, disabled by default."""

    minimum_counts: dict[str, pydantic.NonNegativeInt] = pydantic.Field(default_factory=dict)
    """Minimum value of each named author counter; unmet or unavailable counters prevent promotion."""


class Recipe(Record):
    """Validated construction and release settings for statement exports."""

    recipe_version: typing.Literal[1] = 1
    """Configuration contract version; currently 1."""
    eligibility: Eligibility = pydantic.Field(default_factory=Eligibility)
    """Source restrictions shared by recipients and donors."""
    code_selection: CodeSelection = pydantic.Field(default_factory=CodeSelection)
    """Filters for displayed recipient code; donor and reference eligibility stay independent."""
    queries: Queries = pydantic.Field(default_factory=Queries)
    """Coordinated candidate construction and the distinct extra-negative cap."""
    statements: Statements = pydantic.Field(default_factory=Statements)
    """Reasoning inclusion, optional problem text, and native event scope."""
    variants: Variants = pydantic.Field(default_factory=Variants)
    """Faithful and explicitly requested suffix-splice view families."""
    budget: Budget = pydantic.Field(default_factory=Budget)
    """Complete reference-text length limit and optional pinned tokenizer."""
    visibility: Visibility = pydantic.Field(default_factory=Visibility)
    """Public category and target visibility, configured per derived split."""
    coverage: Coverage = pydantic.Field(default_factory=Coverage)
    """Optional named author-counter minima; empty by default."""
    mixture: Mixture | None = None
    """Optional target shares of query groups per family, applied to each split of the public
    projection; the author artifact keeps every group."""
    max_oracle_matches: pydantic.PositiveInt = 1_000_000
    """Work limit for ambiguous statement alignment, passed to ``grade_events``. Exceeded limits
    exclude unverifiable views."""
    donor_search_limit: pydantic.PositiveInt = 32
    """Maximum ranked executions checked for each reasoning donor, separately for alternate-input
    sources (including retries among sources with the same outcome) and for effective bugged
    variants of the recipient."""
    event_cache_size: pydantic.PositiveInt = 4
    """Maximum number of extracted native trajectories kept in the exporter's in-memory LRU cache."""


class Binding(Record):
    """An observed native representation, never evaluated by consumers."""

    name: str
    """Bound variable or parameter name."""
    value: str
    """Native representation of the bound value."""


class Delta(Record):
    """A namespace-specific change relative to native observations."""

    namespace: typing.Literal["local", "global"]
    """Namespace that changed."""
    operation: typing.Literal["add", "set", "delete"]
    """Kind of change to the name."""
    name: str
    """Changed variable name."""
    value: str | None
    """Native representation of the new value; None for deletions."""


class EventTargets(Record):
    """View-dependent supervision."""

    statement_correct: bool | None = None
    """Whether the event is part of the recipient's actual execution; None when withheld."""


class Event(Record):
    """One immutable native assertion with auxiliary occurrence metadata."""

    event_id: str
    """Opaque identity of the native event within its source execution."""
    origin_token: str
    """Opaque identity of the execution that recorded the event."""
    native_index: int
    """Position of the event in its source execution's native trace."""
    view_index: int
    """Position of the event in the presented reasoning, from 0."""
    source_depth: int
    """Depth of the event's frame in its source execution's program stack, from 0."""
    activation_id: str
    """Opaque identity of the frame activation that recorded the event."""
    kind: typing.Literal["call", "line", "return", "exception"]
    """Native event kind."""
    code_object: str
    """Displayed code-object label, ``name:first_line`` or ``<module>``."""
    line: int
    """Source line reported by the event, numbered as in the displayed program."""
    arguments: list[Binding]
    """Arguments observed at a ``call`` event."""
    deltas: list[Delta]
    """Local and global changes since the previous native observation."""
    return_value: str | None
    """Native representation of the value returned at a ``return`` event."""
    stdout_since_prev_capture: str | None
    """Standard output captured since the previous capture."""
    stderr_since_prev_capture: str | None
    """Standard error captured since the previous capture."""
    exception: str | None
    """Summary of the exception recorded with the event."""
    targets: EventTargets = pydantic.Field(default_factory=EventTargets)
    """Statement supervision for this event in this view."""


class BudgetMetadata(Record):
    """Exact finalized reference cost, separate from model inputs."""

    unit: str
    """Budget unit, ``characters`` or ``tokens``."""
    cost: int
    """Size of the complete reference-rendered row in that unit."""
    maximum: int | None
    """Budget cap; None is unlimited."""
    renderer_version: str = consumer.RENDERER_VERSION
    """Renderer version that measured the cost."""
    original_event_count: int
    """Events in the view before budget omission."""
    retained_event_count: int
    """Events kept after budget omission."""


class Example(Record):
    """Public v2 Parquet row contract."""

    row_id: str
    """Opaque identity of the row."""
    export_split: typing.Literal["train", "validation", "test"]
    """Derived split holding the row's problem."""
    problem_id: str
    """Opaque identity of the source problem; no execution is used by more than one problem."""
    context_id: str
    """Opaque identity of the displayed program and input, shared by every view of one recipient."""
    query_group_id: str
    """Opaque identity of the prompt shared by the group's candidate rows."""
    code_string: str
    """Displayed program source."""
    entrypoint_name: str | None
    """Function called by callable tasks; None for scripts."""
    invocation_kind: typing.Literal["callable", "stdin"]
    """Whether the task calls an entrypoint or runs a script on stdin."""
    invocation_text: str
    """Displayed description of how the program is invoked."""
    inputs_text: str
    """Displayed input payload."""
    inputs_json: str | None
    """The input payload as JSON, or None when it has no JSON form."""
    observed_bindings: list[Binding]
    """Entrypoint arguments observed at its call; empty when unavailable."""
    supplied_stdin: str | None
    """Text given to a script on stdin; None for callables."""
    problem_statement: str | None
    """Problem text, or None unless the recipe includes it."""
    candidate_output_text: str
    """Displayed candidate outcome."""
    candidate_output_json: str | None
    """The candidate as JSON, or None when it has no JSON form."""
    label: bool | None
    """Whether the candidate is the program's actual outcome; None when withheld."""
    reasoning_events: list[Event] | None
    """Presented reasoning events in view order; None when reasoning is disabled."""
    category_labels: list[str] | None
    """Sorted ``code.*``, ``reasoning.*``, and ``donor.*`` labels; None when hidden."""
    budget_metadata: BudgetMetadata
    """Reference-text cost of the row."""


class AuthorInfo(Record):
    """Private execution evidence and construction provenance."""

    source_identifier: str
    """Stored trace identifier of the recipient."""
    problem_identifier: str
    """Stored identifier of the recipient's problem."""
    donor_identifier: str | None
    """Trace identifier of the pairing's donor, if any."""
    original_identifier: str | None
    """Trace identifier of a bugged recipient's original, if any."""
    candidate_source_identifiers: list[str]
    """Traces whose outcome the row's candidate represents."""
    candidate_roles: list[str]
    """Roles of those traces: ``recipient``, ``donor``, ``original``, or ``additional_observed_negative``."""
    pairing_id: str
    """Identity shared by a spliced view and its faithful twin, or of a standalone faithful view."""
    pairing_kind: PairingKind
    """Donor family of the pairing."""
    requested_variant: str
    """View requested before budgeting: ``faithful``, ``partial_suffix``, or ``whole_replacement``."""
    outcome_kind: str
    """Observable channel of the recipient's actual outcome."""
    outcome_text: str
    """Displayed text of the recipient's actual outcome."""
    outcome_json: str | None
    """The actual outcome as JSON, or None when it has no JSON form."""
    source_test_expected_output_text: str
    """Displayed text of the source test's expected output."""
    source_test_expected_output_json: str | None
    """The expected output as JSON, or None when it has no JSON form."""
    source_test_passed: bool
    """Whether the actual outcome softly matches the source test's expected output."""
    outcome_evidence: str
    """How the outcome was certified: from the stored record or a live re-execution."""
    statement_evidence: str
    """Source of the statement targets: ``stored_native_events`` or ``disabled``."""
    native_event_count: int
    """Valid native steps in the recipient's stored trace."""


class AuthorExample(Example):
    """An author row projects to Example exclusively through its field allowlist."""

    author: AuthorInfo
    """Private provenance, removed by public projection."""


def canonical_json(value: typing.Any) -> str:
    """Serialize construction identities deterministically.

    Args:
        value: JSON-serializable value; mapping keys must support sorted serialization.

    Returns:
        Compact JSON with sorted mapping keys, preserved sequence order, and unescaped
        Unicode characters. This is a serialization, not a hash.

    Raises:
        TypeError: A value cannot be serialized or its mapping keys cannot be sorted.
        ValueError: A value contains NaN, infinity, or a circular reference.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def opaque_id(
    domain: str,
    *parts: typing.Any,
) -> str:
    """Hash a domain-separated canonical identity without visible source tags.

    Args:
        domain: Stable namespace distinguishing identity roles, such as row or event.
        *parts: Ordered JSON-serializable identity components, serialized with the
            domain by ``canonical_json``. Ordering and JSON representations affect the digest.

    Returns:
        A deterministic SHA-256 hexadecimal string. The digest is unkeyed: opacity
        hides direct tags but does not prevent inference from known candidate inputs.
    """
    return hashlib.sha256(canonical_json([domain, *parts]).encode()).hexdigest()
