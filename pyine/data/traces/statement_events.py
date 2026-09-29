"""Native event extraction, suffix composition, and independent statement supervision."""

from __future__ import annotations

import ast
import bisect
import collections
import dataclasses
import random
import typing

import pyine.data.traces.export_consumer as consumer
import pyine.data.traces.statement_config as contract
import pyine.utils.code.execution as execution


class IneligibleTraceError(ValueError):
    """Stored evidence cannot support the requested reasoning contract."""


@dataclasses.dataclass
class _Frame:
    """Replay state of one active frame: code object, activation, locals, and whether it is in task scope."""

    object_id: str
    activation_id: str
    locals: dict[str, str]
    task_scope: bool


@dataclasses.dataclass(frozen=True)
class ExtractedEvents:
    """Scoped assertions derived before reasoning edits."""

    events: list[contract.Event]
    """In-scope events in native order, with immutable native deltas and occurrence metadata.
    Statement targets have not yet been graded."""
    native_event_count: int
    """Number of non-null in-program events examined before scope selection; excludes external
    events and skipped placeholders."""


def _can_suspend(scope: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda | ast.GeneratorExp) -> bool:
    """Check whether a code object's own body can suspend and later resume its frame."""
    if isinstance(scope, (ast.AsyncFunctionDef, ast.GeneratorExp)):
        return True
    pending: list[ast.AST] = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            return True
        children = list(ast.iter_child_nodes(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            # nested bodies are separate code objects; decorators, defaults, and bases still run here
            body = node.body if isinstance(node.body, list) else [node.body]
            children = [child for child in children if all(child is not item for item in body)]
        pending.extend(children)
    return False


class _CodeObject(typing.NamedTuple):
    """A function, class, lambda, or generator expression, with its line span and whether it can suspend."""

    name: str
    first_line: int
    last_line: int
    suspended: bool

    @property
    def identity(self) -> str:
        """Displayed label; line spans stay internal so donor code layouts are not revealed."""
        return f"{self.name}:{self.first_line}"


def _code_objects(code: str) -> list[_CodeObject]:
    """List the source's code objects, rejecting invalid syntax and dynamic ``exec`` or ``eval``."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise IneligibleTraceError("invalid_source_syntax") from error
    objects: list[_CodeObject] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval"}:
            raise IneligibleTraceError("dynamic_execution_namespace")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            objects.append(_CodeObject(node.name, first_line, node.end_lineno or node.lineno, _can_suspend(node)))
        elif isinstance(node, (ast.Lambda, ast.GeneratorExp)):  # python 3.12 inlines list/set/dict comprehensions
            name = "<lambda>" if isinstance(node, ast.Lambda) else "<genexpr>"
            objects.append(_CodeObject(name, node.lineno, node.end_lineno or node.lineno, _can_suspend(node)))
    return objects


def _can_run_code_while_unwinding(code: str) -> bool:
    """Check whether program code could execute while an exception unwinds its frames."""
    return any(
        isinstance(node, (ast.Try, ast.TryStar, ast.With, ast.AsyncWith))
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"__del__", "__exit__", "__aexit__"}
        )
        for node in ast.walk(ast.parse(code))
    )


def _object_identity(
    key: execution.TraceKey,
    objects: list[_CodeObject],
) -> str:
    """Resolve a trace key to its displayed code-object label, rejecting ambiguous or resumable ones."""
    if key.object == "<module>":
        return "<module>"
    candidates = [item for item in objects if item.name == key.object and item.first_line <= key.line <= item.last_line]
    if len(candidates) != 1:
        raise IneligibleTraceError("ambiguous_code_object")
    if candidates[0].suspended:
        raise IneligibleTraceError("unsupported_resumption")
    return candidates[0].identity


def _object_spans(code: str) -> dict[str, tuple[int, int] | None]:
    """Map each code-object label to its line span, or None when several objects share the label."""
    spans: dict[str, tuple[int, int] | None] = {"<module>": (1, len(consumer.source_lines(code)))}
    for item in _code_objects(code):
        spans[item.identity] = (
            None if item.identity in spans else (item.first_line, item.last_line)
        )  # shared labels never cut
    return spans


def find_entrypoint_call(trace: execution.TraceResult) -> execution.TraceEvent | None:
    """Locate the native CALL event that starts the task's entrypoint activation.

    Args:
        trace: Stored native execution. Its ``entrypoint_step_idx`` is the next native
            index before invocation, so the CALL may occur exactly at that index.

    Returns:
        The first in-program CALL at or after that index whose code object has the
        entrypoint's unqualified name and whose stack holds no other in-program frame.
        None for scripts or when no such CALL was recorded. Its arguments are observed
        bindings, including receivers and defaults, not the original call syntax.
    """
    if trace.entrypoint_name is None or trace.entrypoint_step_idx is None:
        return None
    name = trace.entrypoint_name.rsplit(".", 1)[-1]
    for step in trace.traced_steps:
        if (
            step is not None
            and step.trace_step_idx >= trace.entrypoint_step_idx
            and step.event_type == execution.TraceEventType.CALL
            and step.trace_key.file == execution.EXEC_TRACE_FILE_NAME
            and step.trace_key.object == name
            and sum(key.file == execution.EXEC_TRACE_FILE_NAME for key in step.stack_trace) == 1
        ):
            return step
    return None


def _declared_parameters(
    code: str,
    key: execution.TraceKey,
) -> set[str] | None:
    """Collect the parameter names of the code object entered at a CALL key, if exactly one matches."""
    matches: list[set[str]] = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first_line = min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)])
            name = node.name
        elif isinstance(node, ast.Lambda):
            first_line, name = node.lineno, "<lambda>"
        else:
            continue
        if name == key.object and first_line == key.line:
            arguments = node.args
            declared = [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ]
            matches.append({argument.arg for argument in declared if argument is not None})
    return matches[0] if len(matches) == 1 else None


def observed_entrypoint_bindings(trace: execution.TraceResult) -> dict[str, str] | None:
    """Return the entrypoint CALL's observed bindings when they cover every declared parameter.

    Args:
        trace: Stored native execution. The source is inspected syntactically; nothing
            is executed to recover a signature.

    Returns:
        Native argument representations in observed order, or None for scripts, unrecorded
        entrypoint calls, and calls whose captured arguments omit a declared parameter (the
        tracer drops ``__``-prefixed names). Callers then display the stored input payload.
    """
    call = find_entrypoint_call(trace)
    if call is None:
        return None
    arguments = call.arguments or {}
    parameters = _declared_parameters(trace.code_string, call.trace_key)
    return dict(arguments) if parameters is not None and parameters <= arguments.keys() else None


def _deltas(
    before: dict[str, str],
    after: dict[str, str],
    namespace: typing.Literal["local", "global"],
) -> list[contract.Delta]:
    """Diff two namespace snapshots into add, set, and delete deltas, sorted by name."""
    changes: list[contract.Delta] = []
    for name in sorted(before.keys() | after.keys()):
        if name not in after:
            operation, value = "delete", None
        elif name not in before:
            operation, value = "add", after[name]
        elif before[name] != after[name]:
            operation, value = "set", after[name]
        else:
            continue
        changes.append(contract.Delta(namespace=namespace, operation=operation, name=name, value=value))
    return changes


def extract_events(
    trace: execution.TraceResult,
    scope: typing.Literal["task_execution", "all_in_program"] = "task_execution",
) -> ExtractedEvents:
    """Derive immutable deltas on full native trajectories, then select task scope.

    Args:
        trace: Complete stored native execution of the displayed code and supplied
            input. Variable values and stream fragments must retain their native
            representations; this function does not execute code or recover objects.
        scope: ``task_execution`` selects the validated callable entrypoint activation
            and nested calls, or all in-program events for scripts. ``all_in_program``
            also includes callable module initialization.

    Returns:
        Newly constructed events and the pre-scope in-program event count. Code objects
        are identified as ``name:first_line`` (``<module>`` for module code), where the
        first line includes decorators. Local/global baselines are derived before scoping;
        omission does not cause unchanged values to be emitted again. The source trace
        is not modified and returned statement targets remain unset. A ``SystemExit``
        raised in untraced code (such as the site ``exit()`` builtin) leaves no native
        unwind events; such a trace is accepted with its frames left open only when its
        source cannot run program code while unwinding (no ``try``/``with`` statements
        and no ``__del__``/``__exit__``/``__aexit__`` methods).

    Raises:
        IneligibleTraceError: Scope, activation, namespace, source-location, or
            completeness evidence is missing or unsupported. The error text is a
            quality/eligibility reason, not evidence that a statement is false.
    """
    objects = _code_objects(trace.code_string)
    entry_call = find_entrypoint_call(trace)
    if trace.entrypoint_name is not None and entry_call is None:
        raise IneligibleTraceError("unresolved_entrypoint_scope")
    origin = contract.opaque_id("origin", trace.identifier, trace.code_string, repr(trace.inputs))
    globals_before: dict[str, str] = {}
    frames: list[_Frame] = []
    result: list[contract.Event] = []
    native_count = 0
    prior_index = -1
    for step in trace.traced_steps:
        if step is None or step.trace_key.file != execution.EXEC_TRACE_FILE_NAME:
            continue
        if step.trace_step_idx <= prior_index:
            raise IneligibleTraceError("nonmonotonic_native_indices")
        prior_index = step.trace_step_idx
        native_count += 1
        stack = [key for key in reversed(step.stack_trace) if key.file == execution.EXEC_TRACE_FILE_NAME]
        if not stack or stack[-1] != step.trace_key:
            raise IneligibleTraceError("missing_or_malformed_stack")
        identities = [_object_identity(key, objects) for key in stack]
        kind = step.event_type.value
        if kind == "call":
            if identities[:-1] != [frame.object_id for frame in frames]:
                raise IneligibleTraceError("ambiguous_activation_call")
            entry = entry_call is not None and step.trace_step_idx == entry_call.trace_step_idx
            task_scope = entry_call is None or entry or any(frame.task_scope for frame in frames)
            frames.append(
                _Frame(
                    identities[-1],
                    contract.opaque_id("activation", origin, step.trace_step_idx),
                    dict(step.arguments or {}),
                    task_scope,
                )
            )
        elif identities != [frame.object_id for frame in frames]:
            raise IneligibleTraceError("ambiguous_activation_stack")
        frame = frames[-1]
        changes = _deltas(globals_before, step.global_variables, "global")
        if identities[-1] != "<module>":
            changes.extend(_deltas(frame.locals, step.local_variables, "local"))
        globals_before = dict(step.global_variables)
        frame.locals = dict(step.local_variables)
        if scope == "all_in_program" or frame.task_scope:
            result.append(
                contract.Event(
                    event_id=contract.opaque_id("event", origin, step.trace_step_idx),
                    origin_token=origin,
                    native_index=step.trace_step_idx,
                    view_index=len(result),
                    source_depth=len(stack) - 1,
                    activation_id=frame.activation_id,
                    kind=kind,
                    code_object=identities[-1],
                    line=step.trace_key.line,
                    arguments=[
                        contract.Binding(name=name, value=value) for name, value in (step.arguments or {}).items()
                    ],
                    deltas=changes,
                    return_value=step.return_value,
                    stdout_since_prev_capture=step.stdout,
                    stderr_since_prev_capture=step.stderr,
                    exception=str(step.exception) if step.exception is not None else None,
                )
            )
        if kind == "return":
            frames.pop()
    exited = trace.exception is not None and trace.exception.type == SystemExit.__name__
    if frames and not (exited and not _can_run_code_while_unwinding(trace.code_string)):
        raise IneligibleTraceError("incomplete_activation")
    if not result:
        raise IneligibleTraceError("no_in_scope_events")
    return ExtractedEvents(result, native_count)


def _call_sites(events: list[contract.Event]) -> list[tuple[str, int]]:
    """Locate each event's call site: the latest preceding step of its calling frame, if any."""
    latest_by_depth: dict[int, tuple[str, int]] = {}
    sites: list[tuple[str, int]] = []
    for event in events:
        sites.append(latest_by_depth.get(event.source_depth - 1, ("", 0)))
        latest_by_depth[event.source_depth] = (event.code_object, event.line)
    return sites


CutKey = tuple[str, str, int, str, int]
"""Partial-splice cut identity: event kind, code object, line, and call-site code object and line."""


def cut_locations(
    events: list[contract.Event],
    recipient_code: str,
    donor_code: str,
    cut_kinds: typing.Iterable[str],
    as_recipient: bool,
) -> dict[CutKey, list[int]]:
    """Index the events of one trace that may serve as partial-splice cuts between two programs.

    Args:
        events: Scoped, unmodified events of either the recipient or the donor, in native order.
        recipient_code: Complete source text of the recipient program.
        donor_code: Complete source text of the donor program.
        cut_kinds: Event kinds that may serve as cuts, among ``line``, ``call``, and ``return``.
        as_recipient: Whether the events are the recipient's, whose first event is never a cut
            because a partial splice keeps a nonempty recipient prefix.

    Returns:
        Event positions keyed by kind, code object, line, and call site. A location qualifies
        when its code object spans the same source lines in both programs and its line text is
        identical. Call and return cuts also require a call site (the calling frame's latest code
        object and line) meeting the same conditions; line cuts key on an empty call site. A
        recipient and a donor can be spliced at any key present in both of their indexes.
    """
    recipient_lines, donor_lines = consumer.source_lines(recipient_code), consumer.source_lines(donor_code)
    recipient_spans, donor_spans = _object_spans(recipient_code), _object_spans(donor_code)
    compatible_objects = {
        identity for identity, span in recipient_spans.items() if span is not None and donor_spans.get(identity) == span
    }

    def shared_location(
        code_object: str,
        line: int,
    ) -> bool:
        """Check whether a line lies in a compatible code object and reads the same in both programs."""
        return (
            code_object in compatible_objects
            and 1 <= line <= min(len(recipient_lines), len(donor_lines))
            and recipient_lines[line - 1] == donor_lines[line - 1]
        )

    kinds = set(cut_kinds)
    locations: dict[CutKey, list[int]] = collections.defaultdict(list)
    for index, (event, site) in enumerate(zip(events, _call_sites(events), strict=True)):
        if event.kind not in kinds or (as_recipient and index == 0):
            continue
        if not shared_location(event.code_object, event.line):
            continue
        if event.kind == "line":
            site = ("", 0)
        elif site != ("", 0) and not shared_location(*site):
            continue
        locations[(event.kind, event.code_object, event.line, site[0], site[1])].append(index)
    return dict(locations)


def splice_events(
    recipient: list[contract.Event],
    donor: list[contract.Event],
    recipient_code: str,
    donor_code: str,
    whole: bool,
    seed: str,
    cut_kinds: tuple[str, ...] = ("line", "call", "return"),
) -> list[contract.Event]:
    """Compose a donor suffix with a recipient prefix without changing assertions.

    Args:
        recipient: Recipient's scoped, unmodified events in native order.
        donor: Eligible donor's scoped, unmodified events in native order. Callers
            must establish donor lineage, input, outcome, and execution-policy eligibility.
        recipient_code: Complete source text underlying the recipient events.
        donor_code: Complete source text underlying the donor events.
        whole: Return the complete donor view when True. Otherwise, choose cuts at a key
            shared by both traces' ``cut_locations`` and return
            ``recipient[:recipient_cut] + donor[donor_cut:]``.
        seed: Deterministic seed for partial-cut selection: uniform over shared keys,
            then uniform over each trace's occurrences of the selected key.
        cut_kinds: Event kinds that may serve as cuts, among ``line``, ``call``, and ``return``.

    Returns:
        A new list referencing the original event objects. Native deltas and indices
        stay unchanged. Presented indices and targets must be recomputed by calling
        ``grade_events`` after any budget pruning.

    Raises:
        IneligibleTraceError: The donor is empty or no eligible partial cut exists.
            A partial splice requires a nonempty recipient prefix and donor suffix;
            failure never falls back to whole replacement.
    """
    if not donor:
        raise IneligibleTraceError("empty_donor")
    if whole:
        return list(donor)
    recipient_locations = cut_locations(recipient, recipient_code, donor_code, cut_kinds, as_recipient=True)
    donor_locations = cut_locations(donor, recipient_code, donor_code, cut_kinds, as_recipient=False)
    shared = sorted(recipient_locations.keys() & donor_locations.keys())
    if not shared:
        raise IneligibleTraceError("no_shared_cut")
    rng = random.Random(seed)
    key = rng.choice(shared)
    recipient_cut, donor_cut = rng.choice(recipient_locations[key]), rng.choice(donor_locations[key])
    return recipient[:recipient_cut] + donor[donor_cut:]


def event_signature(event: contract.Event) -> str:
    """Serialize exactly the event assertions used by the reference renderer and oracle.

    Args:
        event: Event whose kind, location, bindings, deltas, return, streams, and
            exception payload are to be compared. Native value strings remain opaque.

    Returns:
        Canonical JSON of the asserted fields, excluding identities, indices, depth,
        activation provenance, and targets. Equal signatures mean equal displayed
        assertions; the string is not a digest or a claim about their correctness.
    """
    fields = (
        "kind",
        "code_object",
        "line",
        "arguments",
        "deltas",
        "return_value",
        "stdout_since_prev_capture",
        "stderr_since_prev_capture",
        "exception",
    )
    return contract.canonical_json(event.model_dump(include=set(fields)))


def grade_events(
    view: list[contract.Event],
    reference: list[contract.Event],
    max_matches: int = 1_000_000,
) -> list[contract.Event]:
    """Find a maximum ordered matching with the specified lexicographic tie-break.

    Sparse matching uses a Fenwick maximum tree over reference positions. Its work is
    bounded by the number of equal-signature occurrence pairs, not a dense trace product.

    Args:
        view: Final presented events after composition and pruning, in display order.
            Existing targets and view indices are ignored when comparing assertions.
        reference: Complete unmodified scoped execution of the displayed recipient
            code/input. Each reference occurrence can support at most one view event.
        max_matches: Positive limit on equal-signature occurrence pairs for ambiguous
            alignments. A view that already matches in full uses a linear greedy fast
            path and is accepted even when its possible pair count exceeds this limit.

    Returns:
        Event copies in the original view order, with consecutive zero-based view
        indices and Boolean statement targets. The maximum ordered matching is chosen
        by the lexicographically smallest list of ``(view_index, reference_index)``
        pairs. Matched events are true and unmatched events false, independently of
        donor identity or candidate output. Empty views return an empty list.

    Raises:
        IneligibleTraceError: An ambiguous alignment exceeds ``max_matches``. Callers
            must treat the view as unverifiable rather than assign false targets.
    """
    positions: dict[str, list[int]] = collections.defaultdict(list)
    for index, event in enumerate(reference):
        positions[event_signature(event)].append(index)
    candidates = [positions.get(event_signature(event), []) for event in view]
    previous = -1
    for matches in candidates:
        occurrence = bisect.bisect_right(matches, previous)
        if occurrence == len(matches):
            break
        previous = matches[occurrence]
    else:
        return [
            event.model_copy(update={"view_index": index, "targets": contract.EventTargets(statement_correct=True)})
            for index, event in enumerate(view)
        ]
    if sum(map(len, candidates)) > max_matches:
        raise IneligibleTraceError("oracle_work_limit")
    tree = [0] * (len(reference) + 2)
    lengths: list[list[int]] = [[] for _ in view]
    for view_idx in range(len(view) - 1, -1, -1):
        for reference_idx in candidates[view_idx]:
            tree_idx, best = len(reference) - reference_idx - 1, 0
            while tree_idx > 0:
                best = max(best, tree[tree_idx])
                tree_idx -= tree_idx & -tree_idx
            lengths[view_idx].append(best + 1)
        for reference_idx, length in zip(candidates[view_idx], lengths[view_idx], strict=True):
            tree_idx = len(reference) - reference_idx
            while tree_idx < len(tree):
                tree[tree_idx] = max(tree[tree_idx], length)
                tree_idx += tree_idx & -tree_idx
    remaining = max(tree, default=0)
    matched: set[int] = set()
    last_reference = -1
    for view_idx, reference_indices in enumerate(candidates):
        for match_idx in range(bisect.bisect_right(reference_indices, last_reference), len(reference_indices)):
            if lengths[view_idx][match_idx] >= remaining and remaining > 0:
                matched.add(view_idx)
                last_reference = reference_indices[match_idx]
                remaining -= 1
                break
    return [
        event.model_copy(
            update={"view_index": index, "targets": contract.EventTargets(statement_correct=index in matched)}
        )
        for index, event in enumerate(view)
    ]
