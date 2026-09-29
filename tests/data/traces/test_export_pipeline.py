import collections
import dataclasses
import hashlib
import io
import itertools
import json
import pathlib
import subprocess
import sys
import typing

import click.testing
import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import pytest
import tokenizers
import yaml

import pyine.apps.traces.eval_exporter as cli
import pyine.data.traces.dataset_reader as readers
import pyine.data.traces.dataset_utils as source_types
import pyine.data.traces.dataset_writer as writer
import pyine.data.traces.eval_export as legacy
import pyine.data.traces.eval_export_v2 as exporter
import pyine.data.traces.export_consumer as consumer
import pyine.data.traces.statement_config as contract
import pyine.data.traces.statement_events as events
import pyine.data.utils.lmdb_io as lmdb_io
import pyine.data.utils.splits as splits
import pyine.prompts.result_db as prompt_db
import pyine.utils.code.execution as execution
import pyine.utils.reprod
import tests.data.traces.test_dataset_writer_checks as writer_fixtures
import tests.data.traces.test_eval_export as export_fixtures

PROGRAM = """counter = 0
def helper(value):
    global counter
    counter += 1
    scratch = value
    del scratch
    return value * 2
def solution(value):
    total = 0
    for idx in range(value):
        total += helper(idx)
    return total
"""
BUGGED = PROGRAM.replace("return total", "return total + 1")
HINTED = PROGRAM.replace("counter = 0", "counter = 0  # the loop sums twice every integer below value")
RECIPES = pathlib.Path(exporter.__file__).with_name("recipes")


def _config(
    source: tuple[pathlib.Path, pathlib.Path],
    output: pathlib.Path,
    **updates: typing.Any,
) -> legacy.EvalExportConfig:
    return legacy.EvalExportConfig(
        source_lmdb_paths=[source[0]], split_file_path=source[1], output_dir=output, **updates
    )


@pytest.fixture(scope="module")
def native_source(tmp_path_factory: pytest.TempPathFactory) -> tuple[pathlib.Path, pathlib.Path]:
    root = tmp_path_factory.mktemp("statement-source")
    source_dir = root / "input"
    source_dir.mkdir()
    (source_dir / "fixture.txt").write_text("known-answer source generation")
    source_path, split_path = root / "source.lmdb", root / "split.bin"
    selection_config = _config((source_path, split_path), root / "unused")
    selected: dict[str, int] = {}
    for problem_idx in range(100):
        split = legacy.assign_export_split(f"TACO/train/p{problem_idx:06d}", selection_config)
        selected.setdefault(split, problem_idx)
        if len(selected) == 3:
            break
    database_path = root / "prompts.sqlite"
    database = prompt_db.PromptResultDB(database_path)
    problems = []
    for problem_idx in selected.values():
        identifier = source_types.CodingProblemIdentifier("TACO", "train", problem_idx)
        solution_id = source_types.SolutionIdentifier("TACO", "train", problem_idx, 0)
        problem = source_types.CodingProblem(
            source_dataset_name="TACO",
            source_data_path=str(source_dir),
            source_data_hash=f"fixture-{problem_idx}",
            problem_id=identifier,
            problem_statement="Sum twice every integer below the input.",
            problem_tags=["source:fixture"],
            test_inout_pairs=[(value, value * (value - 1)) for value in (10, 12, 14)],
            entrypoint_name="solution",
            potential_solution_ids=[solution_id],
            parsing_errors=None,
            is_banned=False,
        )
        solution = source_types.Solution(
            parent_id=identifier,
            solution_id=solution_id,
            code=PROGRAM,
            analysis_errors=None,
            analysis_results=writer_fixtures._make_analysis_response(input_type="callable", output_type="callable"),
            is_banned=False,
        )
        problems.append((problem, [solution]))
        for name, code in [("issues/iterators", BUGGED), ("hints/docs", HINTED)]:
            database.store(
                identifier=str(solution_id),
                group="fixture",
                prompt_name=name,
                prompt_version="fixture",
                prompt="stored code fixture",
                result=code,
            )
    config = writer.TraceDatasetWriterConfig(
        source_dataset_name="TACO",
        prompt_result_db_path=str(database_path),
        fetch_augmentations={"issues/iterators": 1, "hints/docs": 1},
        writer_serialization_config=lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.JSON_ZSTD),
        failed_test_log_dir=root / "writer-failures",
        execution_timeout_seconds=30,
        pick_random_tests_per_solution=False,
        generate_obfuscated_solutions=False,
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(source_types, "CodingProblemIterator", lambda **kwargs: problems)
        writer.write_dataset(source_dir, source_path, config)
    splits.SplitResult(
        source_dataset_name="TACO",
        source_dataset_hash="fixture-source",
        identifiers=[str(problem.problem_id) for problem, _ in problems],
        source_data_hashes=[problem.source_data_hash for problem, _ in problems],
        tag_lists=[["subset:train"] for _ in problems],
        subset_assignments={str(problem.problem_id): "train" for problem, _ in problems},
        creation_metadata={"fixture": True},
        config=splits.SplitConfig(
            subset_names=["train", "valid", "test"],
            subset_assign_prob_map={"train": 0.8, "valid": 0.1, "test": 0.1},
        ),
    ).to_file(split_path)
    reader = readers.DatasetReader(source_path)
    assert len(reader) == 27
    assert {source_types.TraceIdentifier.from_string(key).augment_category for key in reader.trace_keys} == {
        None,
        "issues_iterators",
        "hints_docs",
    }
    return source_path, split_path


def _trace(
    code: str,
    inputs: typing.Any = 3,
    entrypoint: str | None = "solution",
) -> execution.TraceResult:
    return execution.execute_and_trace_code(
        code_string=code,
        inputs=inputs,
        expected_output=None,
        entrypoint_name=entrypoint,
        identifier="TACO/train/p000000/s0000/t0000",
        use_safe_execution=False,
        trace_only_inside_code_string=True,
        seed=0,
    )


class TestNativeEvents:
    def test_native_deltas_scope_and_global_baselines(self) -> None:
        trace = _trace(PROGRAM, 10)
        extracted = events.extract_events(trace)
        assert len(extracted.events) == 85
        assert extracted.native_event_count == 90
        assert extracted.events[0].kind == "call"
        assert {event.code_object for event in extracted.events} == {"helper:2", "solution:8"}
        assert exporter._invocation(trace)["observed_bindings"] == [{"name": "value", "value": "10"}]
        assert extracted.events[-1].return_value == "90"
        global_updates = [
            delta.value
            for event in extracted.events
            for delta in event.deltas
            if delta.namespace == "global" and delta.name == "counter"
        ]
        assert global_updates == [str(value) for value in range(1, 11)]
        deletions = [delta for event in extracted.events for delta in event.deltas if delta.operation == "delete"]
        assert len(deletions) == 10 and all(delta.name == "scratch" for delta in deletions)
        before_increment = [event for event in extracted.events if event.line == 4]
        assert all(not any(delta.name == "counter" for delta in event.deltas) for event in before_increment)
        assert all(
            event.targets.statement_correct for event in events.grade_events(extracted.events[::3], extracted.events)
        )
        full = events.extract_events(trace, "all_in_program")
        assert len(full.events) > len(extracted.events)
        assert [
            delta.namespace for event in full.events if event.code_object == "<module>" for delta in event.deltas
        ] == ["global"] * sum(len(event.deltas) for event in full.events if event.code_object == "<module>")

    def test_recursion_and_caught_exception(self) -> None:
        trace = _trace(
            "def solution(value):\n    if value == 0:\n        try:\n            raise ValueError('fixture')\n"
            "        except ValueError:\n            return 0\n    return 1 + solution(value - 1)\n"
        )
        extracted = events.extract_events(trace)
        assert max(event.source_depth for event in extracted.events) == 3
        assert [event.exception for event in extracted.events if event.exception is not None] == ["ValueError(fixture)"]
        assert extracted.events[-1].return_value == "3"

    def test_unsupported_resumption_is_not_false_supervision(self) -> None:
        trace = _trace("def values():\n    yield 1\ndef solution(value):\n    return list(values())\n")
        with pytest.raises(events.IneligibleTraceError, match="unsupported_resumption"):
            events.extract_events(trace)
        script = _trace("generator = (lambda: (yield 1))()\nlist(generator)\n", "", None)
        with pytest.raises(events.IneligibleTraceError, match="unsupported_resumption"):
            events.extract_events(script)
        expression = _trace("def solution(value):\n    return sum(item for item in range(value))\n")
        with pytest.raises(events.IneligibleTraceError, match="unsupported_resumption"):
            events.extract_events(expression)

    @pytest.mark.parametrize(
        "code",
        [
            "def solution(value):\n    def unused():\n        yield value\n    return value\n",
            "class Helper:\n    def items(self):\n        yield 1\ndef solution(value):\n    return value\n",
        ],
    )
    def test_uncalled_nested_generators_do_not_exclude_callers(
        self,
        code: str,
    ) -> None:
        assert events.extract_events(_trace(code)).events[-1].return_value == "3"

    @pytest.mark.parametrize("exit_call", ["exit()", "sys.exit()"])
    def test_exit_terminated_scripts_keep_their_recorded_events(
        self,
        exit_call: str,
    ) -> None:
        code = f"import sys\nvalue = int(input())\nif value < 0:\n    print(-1)\n    {exit_call}\nprint(value * 2)\n"
        extracted = events.extract_events(_trace(code, "-5", None)).events
        assert [event.line for event in extracted if event.kind == "line"] == [1, 2, 3, 4, 5]
        assert [event.stdout_since_prev_capture for event in extracted if event.stdout_since_prev_capture] == ["-1\n"]

    def test_exit_through_unwinding_code_stays_ineligible(self) -> None:
        # the untraced exit() hides the executed finally block, so accepting this trace would drop code
        trace = _trace("try:\n    exit()\nfinally:\n    print('cleanup')\n", "", None)
        assert trace.stdout == "cleanup\n"
        with pytest.raises(events.IneligibleTraceError, match="incomplete_activation"):
            events.extract_events(trace)

    def test_non_json_numeric_inputs_do_not_abort_extraction(self) -> None:
        trace = _trace("def solution(value):\n    return value\n", float("nan"))
        assert events.extract_events(trace).events[-1].return_value == "nan"
        assert all(exporter._input_keys(exporter._invocation(trace)))

    @pytest.mark.parametrize(
        "payload, supplied",
        [
            ("", ""),
            ("\n", "\n"),
            ("a\r\nb", "a\nb\n"),
            ("a\r", "a\n"),
            ("a\u2028", "a\n\n"),
            ("a\v", "a\n\n"),
            (17, "17\n"),
        ],
    )
    def test_stdin_matches_native_text_and_buffer_reads(
        self,
        payload: typing.Any,
        supplied: str,
    ) -> None:
        for expression in ("sys.stdin.read()", "sys.stdin.buffer.read().decode()"):
            trace = _trace(f"import sys\nprint(repr({expression}))\n", payload, None)
            invocation = exporter._invocation(trace)
            assert invocation["supplied_stdin"] == supplied
            assert trace.stdout == f"{supplied!r}\n"

    def test_scope_missing_entrypoint_does_not_guess(self) -> None:
        trace = _trace(PROGRAM).model_copy(update={"entrypoint_step_idx": None})
        with pytest.raises(events.IneligibleTraceError, match="unresolved_entrypoint"):
            events.extract_events(trace)
        invocation = exporter._invocation(trace)
        assert invocation["observed_bindings"] == []
        assert invocation["invocation_text"] == "entrypoint solution; stored input payload"
        row = {**invocation, "problem_statement": None, "reasoning_events": None, "candidate_output_text": "6"}
        assert "\n\nStored input payload:\n3\n\n" in consumer.render_example(row)

    @pytest.mark.parametrize(
        "code, inputs, expected",
        [
            ("def solution(value, offset=1):\n    return value + offset\n", 3, {"value": "3", "offset": "1"}),
            (
                "def solution(*values, **options):\n    return len(values)\n",
                [3, 4],
                {"values": "([3, 4],)", "options": "{}"},
            ),
            (
                "def trace(function):\n    return function\n@trace\ndef solution(value):\n    return value\n",
                3,
                {"value": "3"},
            ),
            ("def solution(value, __offset):\n    return value + __offset\n", {"value": 3, "__offset": 10}, None),
        ],
    )
    def test_observed_bindings_must_cover_every_declared_parameter(
        self,
        code: str,
        inputs: typing.Any,
        expected: dict[str, str] | None,
    ) -> None:
        assert events.observed_entrypoint_bindings(_trace(code, inputs)) == expected

    def test_lambda_identity_and_ambiguous_same_line(self) -> None:
        trace = _trace("def solution(value):\n    return list(map(lambda item: item + 1, [value]))\n")
        extracted = events.extract_events(trace)
        assert any(event.code_object.startswith("<lambda>:") for event in extracted.events)
        assert extracted.events[-1].return_value == "[4]"
        ambiguous = _trace("def solution(value):\n    return (lambda item: item + 1)((lambda item: item * 2)(value))\n")
        with pytest.raises(events.IneligibleTraceError, match="ambiguous_code_object"):
            events.extract_events(ambiguous)

    def test_partial_splice_cannot_fall_back_to_whole(self) -> None:
        first_code = "def solution(value):\n    return value\n"
        second_code = "def solution(value):\n    return value + 1\n"
        first = events.extract_events(_trace(first_code)).events
        second = events.extract_events(_trace(second_code)).events
        with pytest.raises(events.IneligibleTraceError, match="no_shared_cut"):
            events.splice_events(first, second, first_code, second_code, False, "fixture")
        assert events.splice_events(first, second, first_code, second_code, True, "fixture") == second

    def test_partial_splice_meets_at_one_shared_line_occurrence(self) -> None:
        recipient = events.extract_events(_trace(PROGRAM, 3)).events
        donor = events.extract_events(_trace(PROGRAM, 4)).events
        cuts = set()
        donor_facts_true = False
        for seed in range(20):
            view = events.splice_events(recipient, donor, PROGRAM, PROGRAM, False, f"cut-{seed}", ("line",))
            recipient_cut = next(idx for idx, event in enumerate(view) if event.origin_token == donor[0].origin_token)
            donor_cut = donor.index(view[recipient_cut])
            cuts.add((recipient_cut, donor_cut))
            assert 0 < recipient_cut < len(recipient)
            assert view == recipient[:recipient_cut] + donor[donor_cut:]
            assert view[recipient_cut].kind == recipient[recipient_cut].kind == "line"
            cut_location = (view[recipient_cut].code_object, view[recipient_cut].line)
            assert cut_location == (recipient[recipient_cut].code_object, recipient[recipient_cut].line)
            graded = events.grade_events(view, recipient)
            assert [event.view_index for event in graded] == list(range(len(view)))
            assert [event.native_index for event in graded] == [event.native_index for event in view]
            donor_targets = [event.targets.statement_correct for event in graded[recipient_cut:]]
            assert donor_targets[-1] is False  # the donor returns 12 rather than 6
            donor_facts_true = donor_facts_true or any(donor_targets)
        assert donor_facts_true
        assert len(cuts) > 1 and any(first != second for first, second in cuts)

    def test_partial_splice_requires_matching_code_object_layout(self) -> None:
        inserted = PROGRAM.replace("    return total\n", "    total += 0\n    return total\n")
        recipient = events.extract_events(_trace(PROGRAM, 3)).events
        donor = events.extract_events(_trace(inserted, 4)).events
        assert {event.code_object for event in donor} == {"helper:2", "solution:8"}
        for seed in range(20):
            view = events.splice_events(recipient, donor, PROGRAM, inserted, False, f"cut-{seed}")
            recipient_cut = next(idx for idx, event in enumerate(view) if event.origin_token == donor[0].origin_token)
            assert view[recipient_cut].code_object == "helper:2"

    def test_call_and_return_cuts_require_the_same_call_site(self) -> None:
        recipient = events.extract_events(_trace(PROGRAM, 3)).events
        donor = events.extract_events(_trace(PROGRAM, 4)).events
        for kind in ("call", "return"):
            for seed in range(10):
                view = events.splice_events(recipient, donor, PROGRAM, PROGRAM, False, f"cut-{seed}", (kind,))
                donor_token = donor[0].origin_token
                recipient_cut = next(idx for idx, event in enumerate(view) if event.origin_token == donor_token)
                donor_cut = donor.index(view[recipient_cut])
                assert view == recipient[:recipient_cut] + donor[donor_cut:]
                assert recipient[recipient_cut].kind == donor[donor_cut].kind == kind
                assert recipient[recipient_cut].code_object == donor[donor_cut].code_object
        # helper is now called from a line whose text differs, so its calls and returns cannot be cuts
        rewritten = PROGRAM.replace("total += helper(idx)", "total = total + helper(idx)")
        rewritten_donor = events.extract_events(_trace(rewritten, 4)).events
        with pytest.raises(events.IneligibleTraceError, match="no_shared_cut"):
            events.splice_events(recipient, rewritten_donor, PROGRAM, rewritten, False, "cut", ("call",))
        view = events.splice_events(recipient, rewritten_donor, PROGRAM, rewritten, False, "cut", ("return",))
        assert view == recipient[:-1] + rewritten_donor[-1:]  # only the entrypoint's final return is shared

    def test_cut_locations_key_call_sites_and_keep_a_recipient_prefix(self) -> None:
        trace_events = events.extract_events(_trace(PROGRAM, 2)).events
        as_donor = events.cut_locations(trace_events, PROGRAM, PROGRAM, ("call", "return"), as_recipient=False)
        as_recipient = events.cut_locations(trace_events, PROGRAM, PROGRAM, ("call", "return"), as_recipient=True)
        entry_call = ("call", "solution:8", 8, "", 0)
        assert as_donor[entry_call] == [0]
        assert as_recipient == {key: positions for key, positions in as_donor.items() if key != entry_call}
        assert {key[0] for key in as_donor} == {"call", "return"}
        assert all(key[3:] == ("solution:8", 11) for key in as_donor if key[1] == "helper:2")  # called in the loop
        lines = events.cut_locations(trace_events, PROGRAM, PROGRAM, ("line",), as_recipient=False)
        assert lines and all(key[0] == "line" and key[3:] == ("", 0) for key in lines)


class TestReferenceRendering:
    def _render(
        self,
        code: str,
        inputs: typing.Any,
        entrypoint: str | None,
        candidate: str,
    ) -> str:
        trace = _trace(code, inputs, entrypoint)
        extracted = events.extract_events(trace).events
        graded = events.grade_events(extracted, extracted)
        row = {
            **exporter._invocation(trace),
            "problem_statement": None,
            "reasoning_events": [event.model_dump() for event in graded],
            "candidate_output_text": candidate,
        }
        return consumer.render_example(row)

    def test_callable_known_answer(self) -> None:
        assert self._render("def solution(value):\n    return value + 1\n", 3, "solution", "4") == (
            "Program:\n1: def solution(value):\n2:     return value + 1\n\n"
            "Supplied invocation:\nentrypoint solution; observed bindings (not call syntax)\n\n"
            "Observed entrypoint bindings:\nvalue = 3\n\n"
            "Proposed execution reasoning (could be incomplete or imperfect):\n"
            "Step 0, line 1: call to solution:1; value = 3\n"
            "Step 1, line 2: solution:1\n"
            "Step 2, line 2: return from solution:1; returns 4\n\n"
            "Is the program's actual outcome 4?"
        )

    def test_script_known_answer(self) -> None:
        # python 3.12 reports line 0 for the module CALL event
        assert self._render("text = input()\nprint(text * 2)\n", "ab", None, "'abab\\n'") == (
            "Program:\n1: text = input()\n2: print(text * 2)\n\n"
            "Supplied invocation:\nscript with supplied stdin\n\n"
            'Supplied stdin stream:\n"ab\\n"\n\n'
            "Proposed execution reasoning (could be incomplete or imperfect):\n"
            "Step 0, line 0: call to <module>\n"
            "Step 1, line 1: <module>\n"
            "Step 2, line 2: <module>; global add text = 'ab'\n"
            'Step 3, line 2: return from <module>; returns None; stdout "abab\\n"\n\n'
            "Is the program's actual outcome 'abab\\n'?"
        )

    def test_documented_examples_match_the_renderer(self) -> None:
        docs = pathlib.Path(exporter.__file__).parent
        callable_text = self._render("def solution(value):\n    return value + 1\n", 3, "solution", "4")
        assert callable_text in (docs / "EVAL_EXPORT_GUIDE.md").read_text()
        script_text = self._render("print(input() * 2)\n", "ab", None, "'abab\\n'")
        assert script_text in (docs / "EVAL_EXPORT_SPEC_V2.md").read_text()


class TestStatementOracle:
    @pytest.fixture()
    def alphabet(self) -> list[contract.Event]:
        base = events.extract_events(_trace("def solution(value):\n    return value\n")).events[0]
        return [base.model_copy(update={"event_id": str(index), "line": index}) for index in range(3)]

    def test_sparse_matching_agrees_with_exhaustive_tie_break(
        self,
        alphabet: list[contract.Event],
    ) -> None:
        for reference_indices in itertools.product(range(2), repeat=3):
            for view_indices in itertools.product(range(2), repeat=4):
                reference = [alphabet[index] for index in reference_indices]
                view = [alphabet[index] for index in view_indices]
                alignments = []
                for length in range(4):
                    for left in itertools.combinations(range(4), length):
                        for right in itertools.combinations(range(3), length):
                            if all(
                                view_indices[view_idx] == reference_indices[ref_idx]
                                for view_idx, ref_idx in zip(left, right, strict=True)
                            ):
                                alignments.append(tuple(zip(left, right, strict=True)))
                expected = min(alignments, key=lambda pairs: (-len(pairs), pairs))
                assert [event.targets.statement_correct for event in events.grade_events(view, reference)] == [
                    index in {pair[0] for pair in expected} for index in range(4)
                ]

    def test_subsampling_can_change_statement_targets(
        self,
        alphabet: list[contract.Event],
    ) -> None:
        first, second = alphabet[:2]
        assert [
            event.targets.statement_correct for event in events.grade_events([second, first, second], [first, second])
        ] == [False, True, True]
        assert [event.targets.statement_correct for event in events.grade_events([second, first], [first, second])] == [
            True,
            False,
        ]
        with pytest.raises(events.IneligibleTraceError, match="oracle_work_limit"):
            events.grade_events([first] * 10, [first] * 5, max_matches=10)

    def test_long_repeated_faithful_view_avoids_quadratic_work(
        self,
        alphabet: list[contract.Event],
    ) -> None:
        graded = events.grade_events([alphabet[0]] * 1000, [alphabet[0]] * 2000, max_matches=1)
        assert all(event.targets.statement_correct for event in graded)


class TestExportPipeline:
    def test_determinism_workers_batches_and_counter_units(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        recipe = contract.Recipe(
            budget=contract.Budget(maximum=1800),
            queries=contract.Queries(max_additional_observed_negatives=5),
        )
        artifacts = []
        for index, (workers, batch) in enumerate([(1, 1), (3, 13)]):
            author = tmp_path / f"author-{index}"
            exporter.export_statements(
                _config(
                    native_source, tmp_path / f"public-{index}", certification_workers=workers, parquet_batch_size=batch
                ),
                author,
                recipe,
            )
            artifacts.append(list(consumer.iter_examples(author / "examples.parquet")))
            counts = json.loads((author / "coverage.json").read_text())
            assert counts["queries.extra_requested"] == counts["emitted.groups"] * 5
            assert counts["queries.extra_requested"] == sum(
                counts[key]
                for key in ("queries.extra_retained", "queries.extra_shortage", "queries.extra_budget_excluded")
            )
            assert counts["queries.extra_shortage"] > 0
        assert artifacts[0] == artifacts[1]

    @pytest.mark.parametrize("whole_probability", [0.0, 1.0])
    def test_real_storage_cli_groups_bugs_and_portable_consumer(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
        whole_probability: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(pyine.utils.reprod, "entrypoint_setup", lambda: None)
        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text(
            yaml.safe_dump(
                {
                    "budget": {"maximum": None},
                    "variants": {
                        "alternate_input_suffix": {"whole_replacement_probability": whole_probability},
                        "buggy_reasoning_suffix": {"enabled": True, "whole_replacement_probability": whole_probability},
                    },
                }
            )
        )
        output, author = tmp_path / "public", tmp_path / "author"
        result = click.testing.CliRunner().invoke(
            cli.main,
            [
                "--export-mode",
                "statements-v2",
                "--lmdb-path",
                str(native_source[0]),
                "--split-file",
                str(native_source[1]),
                "--recipe-file",
                str(recipe_path),
                "--output-dir",
                str(output),
                "--author-output-dir",
                str(author),
                "--parquet-batch-size",
                "7",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        all_rows = []
        manifest = json.loads((output / "export_manifest.json").read_text())
        for split in exporter.SPLITS:
            rows = pq.read_table(output / f"{split}.parquet").to_pylist()
            assert rows
            all_rows.extend(rows)
            assert all((row["category_labels"] is not None) == (split == "validation") for row in rows)
            assert all("author" not in row for row in rows)
            split_groups = list(consumer.iter_groups(output / f"{split}.parquet"))
            assert len({rows[0]["query_group_id"] for rows in split_groups}) == len(split_groups)  # adjacency
            assert all(len({row["query_group_id"] for row in rows}) == 1 for rows in split_groups)
            context_runs = [key for key, _ in itertools.groupby(rows[0]["context_id"] for rows in split_groups)]
            assert len(context_runs) == len(set(context_runs))
            split_problems = list(consumer.iter_problems(output / f"{split}.parquet"))
            assert len({groups[0][0]["problem_id"] for groups in split_problems}) == len(split_problems)
            assert [rows for groups in split_problems for rows in groups] == split_groups
            assert manifest["counts"][split]["problems"] == len(split_problems)
        problem_splits = {(row["problem_id"], row["export_split"]) for row in all_rows}
        assert len(problem_splits) == len({problem_id for problem_id, _ in problem_splits})
        consumer.verify_export(output)
        consumer.verify_export(author)
        specification = pathlib.Path(exporter.__file__).with_name("EVAL_EXPORT_SPEC_V2.md").read_bytes()
        for root in (output, author):
            assert (root / "EVAL_EXPORT_SPEC.md").read_bytes() == specification
            assert not (root / "EVAL_EXPORT_SPEC_V2.md").exists()
        groups = collections.defaultdict(list)
        for row in all_rows:
            groups[row["query_group_id"]].append(row)
        bugged_group_sizes = set()
        for rows in groups.values():
            assert sum(row["label"] for row in rows) == 1
            assert all(row["reasoning_events"] == rows[0]["reasoning_events"] for row in rows)
            expected = int(rows[0]["observed_bindings"][0]["value"])
            expected = expected * (expected - 1) + (rows[0]["code_string"] == BUGGED)
            assert [json.loads(row["candidate_output_json"]) for row in rows if row["label"]] == [expected]
            if rows[0]["code_string"] == BUGGED:
                assert any(
                    json.loads(row["candidate_output_json"]) == expected - 1 and not row["label"] for row in rows
                )
                bugged_group_sizes.add(len(rows))
        # intended-reasoning groups merge the donor with the original, so they have one candidate fewer
        assert bugged_group_sizes == {3, 4}
        variant = "whole_replacement" if whole_probability else "partial_suffix"
        assert any(f"reasoning.{variant}" in (row["category_labels"] or []) for row in all_rows)
        # guide example: one candidate-independent mask per group, with stale statement targets withheld
        group = next(rows for rows in groups.values() if len(rows[0]["reasoning_events"]) > 8)
        for row in group:
            subsampled = {**row, "reasoning_events": [dict(event) for event in row["reasoning_events"][::4]]}
            for event in subsampled["reasoning_events"]:
                event["targets"] = {"statement_correct": None}
            text = consumer.render_example(subsampled)
            shown_steps = [line.split(",", 1)[0] for line in text.split("\n") if line.startswith("Step ")]
            assert shown_steps == [f"Step {index}" for index in range(len(subsampled["reasoning_events"]))]
            assert len(text) < row["budget_metadata"]["cost"] and subsampled["label"] == row["label"]
        assert (output / "pyine_consumer.py").read_bytes() == pathlib.Path(consumer.__file__).read_bytes()
        script = (
            "import json, pyine_consumer; row=json.load(open('example.json')); "
            "print(pyine_consumer.render_example(row))"
        )
        (output / "example.json").write_text(json.dumps(all_rows[0]))
        isolated = subprocess.run(
            [sys.executable, "-S", "-c", script], cwd=output, check=True, text=True, capture_output=True
        )
        assert isolated.stdout.rstrip("\n") == consumer.render_example(all_rows[0])
        assert all_rows[0]["row_id"] not in isolated.stdout
        public_manifest = json.loads((output / "export_manifest.json").read_text())
        author_counts = json.loads((author / "author_manifest.json").read_text())["counts"]
        # public shortcut counts are recomputed from projected rows and must match construction
        assert public_manifest["shortcut_baseline"]["counts"] == {
            name: count for name, count in sorted(author_counts.items()) if name.startswith("baseline.validation.")
        }
        assert public_manifest["mixture"] is None
        projected = exporter.project_statements(
            author,
            tmp_path / "hidden",
            contract.Visibility(
                output_targets=contract.SplitVisibility(train=False, validation=False, test=False),
                statement_targets=contract.SplitVisibility(train=False, validation=False, test=False),
            ),
        )
        assert (
            projected["partition_row_counts"]
            == json.loads((output / "export_manifest.json").read_text())["partition_row_counts"]
        )
        assert all(
            row["label"] is None
            and all(event["targets"]["statement_correct"] is None for event in row["reasoning_events"])
            for row in consumer.iter_examples(tmp_path / "hidden" / "train.parquet")
        )

    def test_mixed_recipe_adds_intended_reasoning_and_enforces_shares(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        recipe = exporter.load_recipe(RECIPES / "statements-mixed.yaml")
        author, public = tmp_path / "author", tmp_path / "public"
        exporter.export_statements(_config(native_source, public), author, recipe)
        author_groups = list(consumer.iter_groups(author / "examples.parquet"))
        intended = [
            rows
            for rows in author_groups
            if rows[0]["author"]["pairing_kind"] == "original_reasoning"
            and "donor.original" in rows[0]["category_labels"]
        ]
        assert intended and all("code.bugged" in rows[0]["category_labels"] for rows in intended)
        for rows in intended:
            (donor_row,) = (row for row in rows if "donor" in row["author"]["candidate_roles"])
            assert "original" in donor_row["author"]["candidate_roles"] and donor_row["label"] is False
        assert not any(
            rows[0]["author"]["pairing_kind"] == "alternate_input" and "donor.bugged" in rows[0]["category_labels"]
            for rows in author_groups
        )
        pairing_groups: dict[str, set[str]] = collections.defaultdict(set)
        for rows in author_groups:
            pairing_groups[rows[0]["author"]["pairing_id"]].add(rows[0]["query_group_id"])
        public_group_ids = {
            rows[0]["query_group_id"]
            for split in exporter.SPLITS
            for rows in consumer.iter_groups(public / f"{split}.parquet")
        }
        assert all(ids <= public_group_ids or ids.isdisjoint(public_group_ids) for ids in pairing_groups.values())
        manifest = json.loads((public / "export_manifest.json").read_text())
        assert manifest["mixture"]["shares"] == recipe.mixture.model_dump()
        # only validation categories are visible by default, so only its mixture counts are published
        (validation_groups,) = manifest["mixture"]["groups"].values()
        assert list(manifest["mixture"]["groups"]) == ["validation"]
        assert all(family["kept"] <= family["available"] for family in validation_groups.values())
        assert any(family["kept"] == family["available"] > 0 for family in validation_groups.values())
        unmixed = exporter.project_statements(author, tmp_path / "unmixed", mixture=None)
        assert sum(unmixed["partition_row_counts"].values()) == sum(len(rows) for rows in author_groups)
        assert unmixed["mixture"] is None

    def test_mixture_requires_every_positive_share_family(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        author = tmp_path / "author"
        exporter.export_statements(_config(native_source, tmp_path / "public"), author)
        with pytest.raises(ValueError, match="buggy_reasoning"):
            exporter.project_statements(
                author, tmp_path / "mixed", mixture=contract.Mixture(clean_code=0.5, buggy_reasoning=0.5)
            )

    def test_author_validation_requires_contiguous_problems(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        author = tmp_path / "author"
        exporter.export_statements(_config(native_source, tmp_path / "public"), author)
        examples = author / "examples.parquet"
        groups = list(consumer.iter_groups(examples))
        assert groups[0][0]["problem_id"] == groups[1][0]["problem_id"] != groups[-1][0]["problem_id"]
        reordered = [row for rows in groups[1:] + groups[:1] for row in rows]
        pq.write_table(pa.Table.from_pylist(reordered, schema=exporter.get_statement_export_schema(True)), examples)
        manifest_path = author / "author_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["examples.parquet"] = exporter._file_info(examples)
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="noncontiguous author problem_id"):
            exporter.project_statements(author, tmp_path / "reprojected")

    def test_shipped_buggy_reasoning_recipe_pairs_one_faithful_comparison(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        recipe = exporter.load_recipe(RECIPES / "statements-buggy-reasoning.yaml")
        author = tmp_path / "author"
        exporter.export_statements(_config(native_source, tmp_path / "public"), author, recipe)
        groups = collections.defaultdict(list)
        for row in consumer.iter_examples(author / "examples.parquet"):
            groups[row["query_group_id"]].append(row)
        views = collections.defaultdict(list)
        for rows in groups.values():
            author_info = rows[0]["author"]
            candidates = {row["candidate_output_text"] for row in rows}
            views[author_info["source_identifier"]].append((author_info["requested_variant"], candidates))
        assert views and all(rows[0]["code_string"] != BUGGED for rows in groups.values())
        spliced_recipients = 0
        for recipient_views in views.values():
            faithful = [candidates for variant, candidates in recipient_views if variant == "faithful"]
            spliced = [candidates for variant, candidates in recipient_views if variant != "faithful"]
            assert len(faithful) == 1
            assert all(candidates == faithful[0] for candidates in spliced)
            spliced_recipients += bool(spliced)
        assert spliced_recipients == 9

    def test_v1_export_from_the_same_native_source(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        manifest = legacy.export_eval_traces(_config(native_source, tmp_path / "v1"))
        assert sum(manifest.partition_row_counts.values()) == 27
        assert pq.read_schema(tmp_path / "v1" / "train.parquet").equals(legacy.get_eval_export_schema())
        assert not (tmp_path / "v1" / "pyine_consumer.py").exists()
        assert not (tmp_path / "v1" / "author_manifest.json").exists()

    def test_budgets_disabled_reasoning_and_default_shortages(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        for name, recipe in [
            ("disabled", contract.Recipe(statements=contract.Statements(include_reasoning=False))),
            ("tight", contract.Recipe(budget=contract.Budget(maximum=1800))),
            ("zero", contract.Recipe(budget=contract.Budget(maximum=0))),
        ]:
            output, author = tmp_path / name, tmp_path / f"{name}-author"
            exporter.export_statements(_config(native_source, output), author, recipe)
            rows = list(consumer.iter_examples(output / "train.parquet"))
            if name == "zero":
                assert not rows
            else:
                assert rows
                for row in rows:
                    assert len(consumer.render_example(row)) <= recipe.budget.maximum
                    if name == "disabled":
                        assert row["reasoning_events"] is None
                if name == "tight":
                    assert any(
                        row["budget_metadata"]["retained_event_count"] < row["budget_metadata"]["original_event_count"]
                        for row in rows
                    )

    def test_source_report_and_opt_in_minimum(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        config = _config(native_source, tmp_path / "public")
        report = exporter.report_source(config, inspect_events=True)
        assert report["counts"]["code.bugged"] == 9
        assert report["counts"]["events.eligible_contexts"] == 27
        assert report["counts"]["cuts.alternate_input.negative_pairs"] == 54
        assert report["counts"]["cuts.alternate_input.shared_cut_pairs"] == 54
        assert report["cut_kinds"] == ["line", "call", "return"]
        assert "post_budget_counts" in report["not_measured"]
        recipe = contract.Recipe(coverage=contract.Coverage(minimum_counts={"emitted.rows": 100000}))
        with pytest.raises(ValueError, match="coverage minimum unmet"):
            exporter.export_statements(config, tmp_path / "author", recipe)
        assert not config.output_dir.exists()
        assert not (tmp_path / "author").exists()

    def test_v1_rejects_v2_switches(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        result = click.testing.CliRunner().invoke(
            cli.main, ["--author-output-dir", str(tmp_path / "author"), "--output-dir", str(tmp_path / "public")]
        )
        assert result.exit_code == 2
        assert "v2-only" in result.output

    def test_source_report_rejects_output_directories(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        for option in ("--output-dir", "--author-output-dir"):
            result = click.testing.CliRunner().invoke(
                cli.main, ["--export-mode", "statements-v2", "--source-report", option, str(tmp_path / "unused")]
            )
            assert result.exit_code == 2
            assert "source reports write no artifacts" in result.output

    def test_portable_token_budget_and_hash_validation(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> None:
        unknown_word = "[UNK]"
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({unknown_word: 0}, unk_token=unknown_word))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        resource = tmp_path / "original-tokenizer.json"
        tokenizer.save(str(resource))
        recipe = contract.Recipe(
            budget=contract.Budget(
                unit="tokens",
                maximum=400,
                tokenizer_file=resource,
                tokenizer_sha256=hashlib.sha256(resource.read_bytes()).hexdigest(),
            )
        )
        author, public = tmp_path / "author", tmp_path / "public"
        exporter.export_statements(_config(native_source, public), author, recipe)
        rows = list(consumer.iter_examples(public / "train.parquet"))
        assert rows
        for row in rows:
            assert row["budget_metadata"]["cost"] == len(
                tokenizer.encode(consumer.render_example(row), add_special_tokens=False).ids
            )
            assert row["budget_metadata"]["cost"] <= 400
        resource.unlink()
        exporter.project_statements(author, tmp_path / "moved-public")
        assert (author / "tokenizer.json").read_bytes() == (public / "tokenizer.json").read_bytes()
        with (author / "pyine_consumer.py").open("a") as stream:
            stream.write("\n# changed\n")
        with pytest.raises(ValueError, match="hash or size mismatch"):
            exporter.project_statements(author, tmp_path / "invalid-public")
        assert not (tmp_path / "invalid-public").exists()

    @pytest.fixture()
    def hinted_lineage(
        self,
        native_source: tuple[pathlib.Path, pathlib.Path],
        tmp_path: pathlib.Path,
    ) -> tuple[dict[str, exporter._Context], exporter._Context, exporter._Context]:
        """Return eligible contexts, a bugged context prompted from hinted code, and that prompted parent."""
        config = _config(native_source, tmp_path / "unused")
        source_readers, _, references = exporter._load_sources(config)
        contexts = exporter._index_contexts(
            source_readers,
            references,
            config,
            contract.Recipe(),
            collections.Counter(),
            lambda identifier, reason: pytest.fail(f"unexpected exclusion: {identifier}: {reason}"),
            io.StringIO(),
        )
        bug = next(context for context in contexts.values() if "code.bugged" in context.categories)
        parent = next(
            context
            for context in contexts.values()
            if "code.hinted" in context.categories
            and context.reference.problem_identifier == bug.reference.problem_identifier
            and context.input_key != bug.input_key
        )
        bug = dataclasses.replace(bug, prompt_parent_id=parent.reference.identifier)
        contexts[bug.reference.identifier] = bug
        assert bug.original_identifier is not None  # the unaugmented parent a fallback would pick
        return contexts, bug, parent

    def test_nearest_nonbugged_prompt_ancestor_on_same_input(
        self,
        hinted_lineage: tuple[dict[str, exporter._Context], exporter._Context, exporter._Context],
    ) -> None:
        contexts, bug, _ = hinted_lineage
        counts: collections.Counter[str] = collections.Counter()
        resolved = exporter._resolve_originals(contexts, counts)
        original = resolved[resolved[bug.reference.identifier].original_identifier]
        assert original.input_key == bug.input_key
        assert original.task["code_string"] == HINTED
        assert counts["original_reference.prompt_lineage"] == 1

    def test_ancestor_without_same_input_execution_excludes_the_bug(
        self,
        hinted_lineage: tuple[dict[str, exporter._Context], exporter._Context, exporter._Context],
    ) -> None:
        contexts, bug, _ = hinted_lineage
        contexts = {
            identifier: context
            for identifier, context in contexts.items()
            if not (context.task["code_string"] == HINTED and context.input_key == bug.input_key)
        }
        counts: collections.Counter[str] = collections.Counter()
        assert exporter._resolve_originals(contexts, counts)[bug.reference.identifier].original_identifier is None
        assert counts["original_reference.unresolved"] == 1

    def test_ineligible_prompted_parent_excludes_the_bug(
        self,
        hinted_lineage: tuple[dict[str, exporter._Context], exporter._Context, exporter._Context],
    ) -> None:
        contexts, bug, parent = hinted_lineage
        del contexts[parent.reference.identifier]
        counts: collections.Counter[str] = collections.Counter()
        assert exporter._resolve_originals(contexts, counts)[bug.reference.identifier].original_identifier is None
        assert counts["original_reference.unresolved"] == 1

    def test_output_stage_roots_must_not_overlap(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        with pytest.raises(ValueError, match="must not overlap"):
            exporter._check_roots(tmp_path / "author", tmp_path / "author.incomplete")

    def test_projection_rejects_ignored_construction_settings(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        author = tmp_path / "author"
        author.mkdir()
        result = click.testing.CliRunner().invoke(
            cli.main,
            [
                "--export-mode",
                "statements-v2",
                "--project-from-author",
                str(author),
                "--output-dir",
                str(tmp_path / "public"),
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 2
        assert "cannot change source, selection, or certification settings" in result.output


class TestSourcePairing:
    @pytest.fixture()
    def write_source(
        self,
        tmp_path: pathlib.Path,
    ) -> typing.Callable[..., tuple[pathlib.Path, pathlib.Path]]:
        def write(
            traces: list[execution.TraceResult],
            banned: bool = False,
        ) -> tuple[pathlib.Path, pathlib.Path]:
            _, problem = export_fixtures._make_trace_context(traces[0])
            problem = problem.model_copy(
                update={
                    "is_banned": banned,
                    "source_data_path": str(tmp_path),
                    "test_inout_pairs": [
                        (trace.inputs, trace.expected_output)
                        for trace in traces
                        if not source_types.TraceIdentifier.from_string(trace.identifier).is_augmented
                    ],
                }
            )
            source_path, split_path = tmp_path / "source.lmdb", tmp_path / "split.bin"
            with lmdb_io.LMDBWriter(
                source_path,
                serialization_config=lmdb_io.SerializationConfig(method=lmdb_io.SerializationMethod.JSON_ZSTD),
            ) as source_writer:
                source_writer.write_metadata(
                    {
                        "parent_dataset": {"dataset_name": "TACO", "dataset_hash": "fixture"},
                        "writer_config": {"source_dataset_name": "TACO"},
                    }
                )
                source_writer.put(f"{problem.problem_id}{source_types.PROBLEM_DATA_SUFFIX}", problem.model_dump())
                for trace in traces:
                    source_writer.put(trace.identifier, trace.model_dump())
            splits.SplitResult(
                source_dataset_name="TACO",
                source_dataset_hash="fixture",
                identifiers=[str(problem.problem_id)],
                source_data_hashes=[problem.source_data_hash],
                tag_lists=[["subset:train"]],
                subset_assignments={str(problem.problem_id): "train"},
                creation_metadata={"fixture": True},
                config=splits.SplitConfig(
                    subset_names=["train", "valid", "test"],
                    subset_assign_prob_map={"train": 0.8, "valid": 0.1, "test": 0.1},
                ),
            ).to_file(split_path)
            return source_path, split_path

        return write

    @pytest.mark.parametrize("prompt_lineage", [False, True])
    @pytest.mark.parametrize("bug_kind", ["changed_default", "normalized_payload"])
    def test_bug_pairs_match_supplied_or_normalized_inputs(
        self,
        write_source: typing.Callable[[list[execution.TraceResult]], tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        prompt_lineage: bool,
        bug_kind: str,
    ) -> None:
        code = "def solution(value, offset=1):\n    return value + offset\n"
        hinted_code = code + "# add the offset to the supplied value\n"
        original_id = "TACO/train/p000000/s0000/t0000"
        reference_id = f"{original_id}/a:hints_docs:000" if prompt_lineage else original_id
        bug_id = f"{original_id}/a:issues_iterators:000"
        traces = [_trace(code, 3).model_copy(update={"identifier": original_id, "expected_output": 4})]
        if prompt_lineage:
            other_id = "TACO/train/p000000/s0000/t0001"
            traces.extend(
                [
                    _trace(code, 7).model_copy(update={"identifier": other_id, "expected_output": 8}),
                    _trace(hinted_code, 3).model_copy(update={"identifier": reference_id, "expected_output": 4}),
                    _trace(hinted_code, 7).model_copy(
                        update={"identifier": f"{other_id}/a:hints_docs:000", "expected_output": 8}
                    ),
                ]
            )
        reference_code = hinted_code if prompt_lineage else code
        bugged_code = (
            reference_code.replace("offset=1", "offset=2")
            if bug_kind == "changed_default"
            else reference_code.replace("return value + offset", "return value + offset + 1")
        )
        bug = _trace(bugged_code, 3 if bug_kind == "changed_default" else "3").model_copy(
            update={"identifier": bug_id, "expected_output": 4}
        )
        if prompt_lineage:
            bug.metadata["request_metadata"] = "TACO/train/p000000/s0000/t0001/a:hints_docs:000_fixture"
        traces.append(bug)
        source = write_source(traces)
        recipe = contract.Recipe(
            queries=contract.Queries(max_additional_observed_negatives=0),
            variants=contract.Variants(
                alternate_input_suffix=contract.Splice(enabled=False),
                buggy_reasoning_suffix=contract.Splice(enabled=True, whole_replacement_probability=1),
            ),
            budget=contract.Budget(maximum=None),
        )
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        rows = list(consumer.iter_examples(tmp_path / "author" / "examples.parquet"))
        bug_rows = [row for row in rows if row["author"]["source_identifier"] == bug_id]
        assert {json.loads(row["candidate_output_json"]): row["label"] for row in bug_rows} == {4: False, 5: True}
        assert all(row["author"]["original_identifier"] == reference_id for row in bug_rows)
        spliced_rows = [
            row
            for row in rows
            if row["author"]["source_identifier"] == reference_id
            and row["author"]["requested_variant"] == "whole_replacement"
        ]
        assert {json.loads(row["candidate_output_json"]): row["label"] for row in spliced_rows} == {4: True, 5: False}
        assert all(row["author"]["donor_identifier"] == bug_id for row in spliced_rows)

    def test_bug_pairs_reject_different_supplied_inputs(
        self,
        write_source: typing.Callable[[list[execution.TraceResult]], tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
    ) -> None:
        code = "def solution(value, offset=1):\n    return value + offset\n"
        original_id = "TACO/train/p000000/s0000/t0000"
        bug_id = f"{original_id}/a:issues_iterators:000"
        source = write_source(
            [
                _trace(code, 3).model_copy(update={"identifier": original_id, "expected_output": 4}),
                _trace(code.replace("offset=1", "offset=2"), 9).model_copy(
                    update={"identifier": bug_id, "expected_output": 4}
                ),
            ]
        )
        recipe = contract.Recipe(variants=contract.Variants(buggy_reasoning_suffix=contract.Splice(enabled=True)))
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        rows = list(consumer.iter_examples(tmp_path / "author" / "examples.parquet"))
        assert len(rows) == 1
        assert rows[0]["author"]["source_identifier"] == original_id
        assert rows[0]["author"]["donor_identifier"] is None
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        assert counts["skipped.missing_same_input_original"] == 1
        assert counts["skipped.no_effective_buggy_donor"] == 1

    @pytest.mark.parametrize("search_limit", [1, 2])
    def test_donor_search_tries_another_execution_of_the_same_outcome(
        self,
        write_source: typing.Callable[[list[execution.TraceResult]], tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        search_limit: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        code = (
            "def generator():\n    yield 0\ndef solution(value):\n"
            "    if value == 2:\n        return list(generator())[0]\n"
            "    return 1 if value == 1 else 0\n"
        )
        traces = [
            _trace(code, value).model_copy(
                update={"identifier": f"TACO/train/p000000/s0000/t{value - 1:04d}", "expected_output": int(value == 1)}
            )
            for value in (1, 2, 3)
        ]
        source = write_source(traces)
        recipe = contract.Recipe(
            donor_search_limit=search_limit,
            variants=contract.Variants(alternate_input_suffix=contract.Splice(whole_replacement_probability=1)),
            budget=contract.Budget(maximum=None),
        )
        extraction_calls: collections.Counter[str] = collections.Counter()
        extract_events = events.extract_events

        def counting_extract(
            trace: execution.TraceResult,
            scope: typing.Literal["task_execution", "all_in_program"],
        ) -> events.ExtractedEvents:
            extraction_calls[str(trace.identifier)] += 1
            return extract_events(trace, scope)

        monkeypatch.setattr(events, "extract_events", counting_extract)
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        assert extraction_calls[traces[1].identifier] == 1  # the ineligible generator trace is never reloaded
        assert max(extraction_calls.values()) == 1
        rows = [
            row
            for row in consumer.iter_examples(tmp_path / "author" / "examples.parquet")
            if row["author"]["source_identifier"] == traces[0].identifier
        ]
        faithful_rows = [row for row in rows if row["author"]["requested_variant"] == "faithful"]
        assert {json.loads(row["candidate_output_json"]): row["label"] for row in faithful_rows} == {0: False, 1: True}
        spliced_rows = [row for row in rows if row["author"]["requested_variant"] == "whole_replacement"]
        if search_limit == 1:
            assert not spliced_rows
        else:
            assert len(spliced_rows) == 2
            assert all(row["author"]["donor_identifier"] == traces[2].identifier for row in spliced_rows)
            assert {json.loads(row["candidate_output_json"]): row["label"] for row in spliced_rows} == {
                0: False,
                1: True,
            }

    def test_exit_after_output_survives_a_prepared_metadata_cache(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
    ) -> None:
        code = "value = int(input())\nif value < 0:\n    print(-1)\n    exit()\nprint(value * 2)\n"
        traces = [
            _trace(code, value, None).model_copy(
                update={"identifier": f"TACO/train/p000000/s0000/t{idx:04d}", "expected_output": expected}
            )
            for idx, (value, expected) in enumerate([("-5", "-1\n"), ("4", "8\n")])
        ]
        assert traces[0].exception is not None and traces[0].exception.type == "SystemExit"
        source = write_source(traces)
        readers.DatasetReader(source[0])  # prepares the metadata cache that every later reader loads
        cached = readers.DatasetReader(source[0])
        exit_metadata = cached.get_trace_metadata(cached.trace_keys.index(traces[0].identifier))
        assert exit_metadata.exception == traces[0].exception
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author")
        positives = {
            row["candidate_output_text"]
            for row in consumer.iter_examples(tmp_path / "author" / "examples.parquet")
            if row["author"]["source_identifier"] == traces[0].identifier and row["label"]
        }
        assert positives == {"'-1\\n'"}
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        assert "skipped.failed_certification" not in counts

    @pytest.mark.parametrize("include_reasoning", [False, True])
    def test_hidden_arguments_fall_back_to_distinguishable_payloads(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        include_reasoning: bool,
    ) -> None:
        code = "def solution(value, __offset):\n    return value + __offset\n"
        payloads = [{"value": 3, "__offset": 10}, {"value": 3, "__offset": 20}, {"value": 4, "__offset": 9}]
        traces = [
            _trace(code, payload).model_copy(
                update={
                    "identifier": f"TACO/train/p000000/s0000/t{idx:04d}",
                    "expected_output": payload["value"] + payload["__offset"],
                }
            )
            for idx, payload in enumerate(payloads)
        ]
        source = write_source(traces)
        recipe = contract.Recipe(statements=contract.Statements(include_reasoning=include_reasoning))
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        rows = list(consumer.iter_examples(tmp_path / "author" / "examples.parquet"))
        assert all(
            row["observed_bindings"] == [] and "Stored input payload" in consumer.render_example(row) for row in rows
        )
        positives = {row["author"]["source_identifier"]: row["candidate_output_text"] for row in rows if row["label"]}
        assert positives == {trace.identifier: str(trace.expected_output) for trace in traces}
        assert any(row["candidate_output_text"] == "13" and not row["label"] for row in rows)  # the donor-supplied 13
        displayed_labels: dict[str, set[bool]] = collections.defaultdict(set)
        for row in rows:
            displayed_labels[consumer.render_example({**row, "reasoning_events": None})].add(row["label"])
        assert all(len(labels) == 1 for labels in displayed_labels.values())

    @pytest.mark.parametrize("variant", ["bugged_code", "buggy_reasoning"])
    def test_harness_invocation_failures_never_become_outcomes(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        variant: str,
    ) -> None:
        original_id = "TACO/train/p000000/s0000/t0000"
        bug_id = f"{original_id}/a:issues_iterators:000"
        unmappable = _trace("def solution(value, extra):\n    return value + extra\n", 3)
        assert unmappable.exception is not None and "cannot map" in unmappable.exception.message
        source = write_source(
            [
                _trace("def solution(value):\n    return value\n", 3).model_copy(
                    update={"identifier": original_id, "expected_output": 3}
                ),
                unmappable.model_copy(update={"identifier": bug_id, "expected_output": 3}),
            ]
        )
        variants = (
            contract.Variants(alternate_input_suffix=contract.Splice(enabled=False))
            if variant == "bugged_code"
            else contract.Variants(buggy_reasoning_suffix=contract.Splice(enabled=True))
        )
        recipe = contract.Recipe(
            code_selection=contract.CodeSelection(mode="bugged_only" if variant == "bugged_code" else "original_only"),
            statements=contract.Statements(include_reasoning=variant != "bugged_code"),
            variants=variants,
        )
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        rows = list(consumer.iter_examples(tmp_path / "author" / "examples.parquet"))
        assert not any("cannot map" in row["candidate_output_text"] for row in rows)
        assert not any(
            bug_id in (row["author"]["source_identifier"], row["author"]["donor_identifier"])
            or bug_id in row["author"]["candidate_source_identifiers"]
            for row in rows
        )
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        assert counts["skipped.harness_invocation_failure"] == 1
        if variant == "buggy_reasoning":
            assert counts["skipped.no_effective_buggy_donor"] == 1
            assert [(row["candidate_output_text"], row["label"]) for row in rows] == [("3", True)]

    def test_program_and_library_exceptions_remain_outcomes(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
    ) -> None:
        code = "import json\ndef solution(value):\n    return json.loads(value)\n"
        traces = [
            _trace(code, value).model_copy(update={"identifier": f"TACO/train/p000000/s0000/t{idx:04d}"})
            for idx, value in enumerate(["{", "[1]"])
        ]
        assert traces[0].exception is not None and traces[0].exception.type == "JSONDecodeError"
        source = write_source(traces)
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author")
        positives = {
            row["candidate_output_text"]
            for row in consumer.iter_examples(tmp_path / "author" / "examples.parquet")
            if row["author"]["source_identifier"] == traces[0].identifier and row["label"]
        }
        assert len(positives) == 1 and positives.pop().startswith("'JSONDecodeError(")
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        assert "skipped.harness_invocation_failure" not in counts

    @pytest.mark.parametrize("search_limit, ineligible_first", [(2, True), (1, True), (2, False)])
    def test_buggy_donor_search_skips_ineligible_reasoning(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        search_limit: int,
        ineligible_first: bool,
    ) -> None:
        original_id = "TACO/train/p000000/s0000/t0000"
        # arrange the fixtures around the seeded donor ranking; expected labels do not depend on it
        ranked = sorted(
            [f"{original_id}/a:issues_iterators:{idx:03d}" for idx in range(2)],
            key=lambda identifier: contract.opaque_id("bug_donor", 0, identifier),
        )
        generator_bug = "def values():\n    yield 9\ndef solution(value):\n    return next(values())\n"
        plain_bug = "def solution(value):\n    return value + 1\n"
        codes = [generator_bug, plain_bug] if ineligible_first else [plain_bug, generator_bug]
        traces = [
            _trace("def solution(value):\n    return value\n", 3).model_copy(
                update={"identifier": original_id, "expected_output": 3}
            ),
            *(
                _trace(code, 3).model_copy(update={"identifier": identifier, "expected_output": 3})
                for identifier, code in zip(ranked, codes, strict=True)
            ),
        ]
        source = write_source(traces)
        recipe = contract.Recipe(
            donor_search_limit=search_limit,
            code_selection=contract.CodeSelection(mode="original_only"),
            variants=contract.Variants(
                alternate_input_suffix=contract.Splice(enabled=False),
                buggy_reasoning_suffix=contract.Splice(enabled=True, whole_replacement_probability=1),
            ),
            budget=contract.Budget(maximum=None),
        )
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        groups = collections.defaultdict(list)
        for row in consumer.iter_examples(tmp_path / "author" / "examples.parquet"):
            groups[row["query_group_id"]].append(row)
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        variants = sorted(rows[0]["author"]["requested_variant"] for rows in groups.values())
        if search_limit == 1 and ineligible_first:
            assert variants == ["faithful"]
            assert counts["skipped.no_eligible_buggy_donor"] == 1
            return
        assert variants == ["faithful", "whole_replacement"]
        eligible = ranked[1] if ineligible_first else ranked[0]
        assert {row["author"]["donor_identifier"] for rows in groups.values() for row in rows} == {eligible}
        for rows in groups.values():
            assert {row["candidate_output_text"]: row["label"] for row in rows} == {"3": True, "4": False}
        assert counts.get("donors.ineligible_reasoning_attempts", 0) == int(ineligible_first)

    def test_ineffective_bug_merges_original_into_the_positive(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
    ) -> None:
        code = "def solution(value, offset=1):\n    return value + offset\n"
        original_id = "TACO/train/p000000/s0000/t0000"
        bug_id = f"{original_id}/a:issues_iterators:000"
        bugged_code = code.replace("value + offset", "value + offset if value < 100 else 0")
        source = write_source(
            [
                _trace(code, 3).model_copy(update={"identifier": original_id, "expected_output": 4}),
                _trace(bugged_code, 3).model_copy(update={"identifier": bug_id, "expected_output": 4}),
            ]
        )
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author")
        bug_rows = [
            row
            for row in consumer.iter_examples(tmp_path / "author" / "examples.parquet")
            if row["author"]["source_identifier"] == bug_id
        ]
        assert [(row["candidate_output_text"], row["label"]) for row in bug_rows] == [("4", True)]
        assert bug_rows[0]["author"]["candidate_roles"] == ["recipient", "original"]
        assert bug_rows[0]["author"]["original_identifier"] == original_id
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        assert counts["bugs.ineffective_outcome_contrast"] == 1

    @pytest.mark.parametrize("include_banned", [False, True])
    def test_banned_problems_exclude_recipients_and_donors(
        self,
        write_source: typing.Callable[..., tuple[pathlib.Path, pathlib.Path]],
        tmp_path: pathlib.Path,
        include_banned: bool,
    ) -> None:
        code = "def solution(value):\n    return value * 2\n"
        traces = [
            _trace(code, value).model_copy(
                update={"identifier": f"TACO/train/p000000/s0000/t{idx:04d}", "expected_output": value * 2}
            )
            for idx, value in enumerate((3, 7))
        ]
        source = write_source(traces, banned=True)
        recipe = contract.Recipe(eligibility=contract.Eligibility(include_banned_problems=include_banned))
        exporter.export_statements(_config(source, tmp_path / "public"), tmp_path / "author", recipe)
        rows = list(consumer.iter_examples(tmp_path / "author" / "examples.parquet"))
        counts = json.loads((tmp_path / "author" / "coverage.json").read_text())
        if include_banned:
            assert rows and all(row["author"]["donor_identifier"] is not None for row in rows)
            assert "skipped.banned_problem" not in counts
        else:
            assert not rows
            assert counts["skipped.banned_problem"] == 2


class TestRecipe:
    def test_defaults_and_validation(self) -> None:
        assert contract.Recipe().variants.alternate_input_suffix.whole_replacement_probability == 0.1
        assert contract.Recipe().coverage.minimum_counts == {}
        with pytest.raises(pydantic.ValidationError):
            contract.Queries(max_additional_observed_negatives=-1)
        with pytest.raises(pydantic.ValidationError):
            contract.Recipe.model_validate({"unsupported": True})
        with pytest.raises(pydantic.ValidationError, match="require"):
            contract.Budget(unit="tokens")
        with pytest.raises(pydantic.ValidationError, match="code.buged"):
            contract.CodeSelection(include_categories=["code.buged"])  # pyright: ignore[reportArgumentType]

    def test_shipped_recipes_and_null_sections(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        shipped = {path.name: exporter.load_recipe(path) for path in sorted(RECIPES.glob("*.yaml"))}
        assert shipped["statements-bugged-code.yaml"].code_selection.mode == "bugged_only"
        assert shipped["statements-buggy-reasoning.yaml"].variants.buggy_reasoning_suffix.enabled
        assert shipped["statements-default.yaml"] == contract.Recipe()
        mixed = shipped["statements-mixed.yaml"]
        assert mixed.mixture == contract.Mixture(clean_code=0.75, buggy_reasoning=0.2, bugged_code=0.05)
        assert mixed.variants.original_reasoning_suffix.enabled and mixed.variants.buggy_reasoning_suffix.enabled
        with pytest.raises(pydantic.ValidationError, match="sum to 1"):
            contract.Mixture(clean_code=0.5)
        recipe_path = tmp_path / "recipe.yaml"
        recipe_path.write_text("budget: null\n")
        with pytest.raises(pydantic.ValidationError, match="budget"):
            exporter.load_recipe(recipe_path)

    def test_pinned_local_tokenizer(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        unknown_word = "[UNK]"
        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({unknown_word: 0}, unk_token=unknown_word))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer.enable_truncation(max_length=1)
        tokenizer.enable_padding(length=64)
        resource = tmp_path / "tokenizer.json"
        tokenizer.save(str(resource))
        digest = hashlib.sha256(resource.read_bytes()).hexdigest()
        budget = contract.Budget(unit="tokens", tokenizer_file=resource, tokenizer_sha256=digest)
        counting_tokenizer = exporter._TextBudget(budget).tokenizer
        assert counting_tokenizer is not None
        assert len(counting_tokenizer.encode("one two three", add_special_tokens=False).ids) == 3
        with pytest.raises(pydantic.ValidationError, match="does not match"):
            contract.Budget(unit="tokens", tokenizer_file=resource, tokenizer_sha256="wrong")


class TestQueryBudgets:
    @pytest.fixture()
    def recipient(self) -> exporter._Context:
        trace = _trace(PROGRAM, 3)
        return exporter._Context(
            reference=legacy.TraceReference(trace.identifier, 0, 0, "TACO/train/p000000", "train"),
            outcome=legacy.resolve_trace_outcome(trace),
            task={**exporter._invocation(trace), "problem_statement": None, "source_expected": ("6", "6")},
            input_key="3",
            supplied_input_key="supplied-3",
            pool_key="same-code",
            outcome_evidence="stored_record",
            native_event_count=trace.valid_step_count,
            original_identifier=None,
            categories=("code.original",),
            prompt_parent_id=None,
        )

    def test_positive_only_shortage_and_oversized_query_roles(
        self,
        recipient: exporter._Context,
    ) -> None:
        huge = dataclasses.replace(
            recipient, outcome=legacy.ResolvedOutcome("return_value", "x" * 10000, False, "fixture")
        )
        recipe = contract.Recipe(budget=contract.Budget(maximum=1200))
        counts = collections.Counter()
        bundle = exporter._candidate_bundle(recipient, None, None, [huge], 1)
        rows = exporter._build_group(
            recipient,
            None,
            None,
            bundle,
            None,
            [],
            "fixture",
            "alternate_input",
            "faithful",
            0,
            recipe,
            exporter._TextBudget(recipe.budget),
            counts,
            {},
        )
        assert len(rows) == 1 and rows[0]["label"]
        assert counts["queries.extra_budget_excluded"] == 1
        required = exporter._candidate_bundle(recipient, huge, None, [], 0)
        with pytest.raises(events.IneligibleTraceError, match="required_query_budget"):
            exporter._build_group(
                recipient,
                huge,
                None,
                required,
                None,
                [],
                "fixture",
                "alternate_input",
                "faithful",
                0,
                recipe,
                exporter._TextBudget(recipe.budget),
                collections.Counter(),
                {},
            )
        positive_only = exporter._candidate_bundle(recipient, None, None, [], 1)
        counts = collections.Counter()
        exporter._build_group(
            recipient,
            None,
            None,
            positive_only,
            None,
            [],
            "fixture",
            "alternate_input",
            "faithful",
            0,
            recipe,
            exporter._TextBudget(recipe.budget),
            counts,
            {},
        )
        assert counts["queries.extra_shortage"] == 1

    def test_pruning_has_nested_views_and_reclassifies_empty_donor(
        self,
        recipient: exporter._Context,
    ) -> None:
        reference = events.extract_events(_trace(PROGRAM, 3)).events
        donor = dataclasses.replace(recipient, outcome=legacy.ResolvedOutcome("return_value", 12, False, "fixture"))
        view = events.extract_events(_trace(PROGRAM, 4)).events
        bundle = exporter._candidate_bundle(recipient, donor, None, [], 0)
        masks = []
        for maximum in (1400, 1800, 3000, None):
            recipe = contract.Recipe(budget=contract.Budget(maximum=maximum))
            rows = exporter._build_group(
                recipient,
                donor,
                None,
                bundle,
                view,
                reference,
                "fixture",
                "alternate_input",
                "whole_replacement",
                0,
                recipe,
                exporter._TextBudget(recipe.budget),
                collections.Counter(),
                {},
            )
            masks.append({event["event_id"] for event in rows[0]["reasoning_events"]})
            assert all(row["budget_metadata"]["cost"] == len(consumer.render_example(row)) for row in rows)
            if maximum is None:
                mandatory = max(len(consumer.render_example({**row, "reasoning_events": []})) for row in rows)
        assert all(first <= second for first, second in itertools.pairwise(masks))
        recipe = contract.Recipe(budget=contract.Budget(maximum=mandatory))
        counts = collections.Counter()
        empty = exporter._build_group(
            recipient,
            donor,
            None,
            bundle,
            view,
            reference,
            "fixture",
            "alternate_input",
            "whole_replacement",
            0,
            recipe,
            exporter._TextBudget(recipe.budget),
            counts,
            {},
        )
        assert empty[0]["reasoning_events"] == []
        assert "reasoning.empty" in empty[0]["category_labels"]
        assert "reasoning.whole_replacement" not in empty[0]["category_labels"]
        assert counts["splices.whole_replacement.no_surviving_donor"] == 1

    def test_soft_equal_and_visible_duplicates_share_one_candidate(
        self,
        recipient: exporter._Context,
    ) -> None:
        def execution_of(
            test_idx: int,
            value: typing.Any,
        ) -> exporter._Context:
            reference = dataclasses.replace(recipient.reference, identifier=f"TACO/train/p000000/s0000/t{test_idx:04d}")
            outcome = legacy.ResolvedOutcome("return_value", value, False, "fixture")
            return dataclasses.replace(recipient, reference=reference, outcome=outcome, input_key=f"input-{test_idx}")

        float_twin = execution_of(1, 6.0)
        bundle = exporter._candidate_bundle(recipient, float_twin, None, [], 0)
        assert [candidate.roles for candidate in bundle] == [["recipient", "donor"]]
        distinct = exporter._rank_alternatives(
            recipient, [recipient, float_twin, execution_of(2, 12), execution_of(3, 12.0)], 0
        )
        assert [sorted(context.outcome.value for context in sources) for sources in distinct] == [[12, 12.0]]

    def test_contradicting_an_emitted_displayed_task_skips_the_group(
        self,
        recipient: exporter._Context,
    ) -> None:
        recipe = contract.Recipe(budget=contract.Budget(maximum=None))
        bundle = exporter._candidate_bundle(recipient, None, None, [], 0)

        def build(displayed_labels: dict[bytes, bool]) -> list[dict[str, typing.Any]]:
            return exporter._build_group(
                recipient,
                None,
                None,
                bundle,
                None,
                [],
                "fixture",
                "alternate_input",
                "faithful",
                0,
                recipe,
                exporter._TextBudget(recipe.budget),
                collections.Counter(),
                displayed_labels,
            )

        emitted: dict[bytes, bool] = {}
        build(emitted)
        assert list(emitted.values()) == [True]
        assert build(emitted)  # a consistent repeat of the same displayed task is accepted
        with pytest.raises(events.IneligibleTraceError, match="contradicts_displayed_label"):
            build(dict.fromkeys(emitted, False))
