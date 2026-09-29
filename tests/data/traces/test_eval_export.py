import dataclasses
import hashlib
import json
import pathlib
import typing

import click.testing
import pyarrow.parquet as pq
import pydantic
import pytest

import pyine.apps.traces.eval_exporter
import pyine.data.traces.dataset_reader
import pyine.data.traces.dataset_utils
import pyine.data.traces.eval_export
import pyine.data.utils.lmdb_io
import pyine.data.utils.splits
import pyine.utils.code.execution
import pyine.utils.code.output_compare
import pyine.utils.concurrency
import pyine.utils.reprod
import tests.utils.fake_dataset_readers


def _make_config(tmp_path: pathlib.Path, **updates: typing.Any) -> pyine.data.traces.eval_export.EvalExportConfig:
    """Build a valid exporter config rooted in a temporary test directory."""
    source_path = tmp_path / "source.lmdb"
    source_path.mkdir(exist_ok=True)
    split_file_path = tmp_path / "split.bin"
    if not split_file_path.exists():
        split_file_path.touch()
    values: dict[str, typing.Any] = {
        "source_lmdb_paths": [source_path],
        "split_file_path": split_file_path,
        "output_dir": tmp_path / "export",
    }
    values.update(updates)
    return pyine.data.traces.eval_export.EvalExportConfig(**values)


def _execute(
    code_string: str,
    expected_output: typing.Any,
    entrypoint_name: str | None = None,
    inputs: typing.Any = None,
) -> pyine.utils.code.execution.TraceResult:
    """Execute a compact fixture program and return its native trace result."""
    return pyine.utils.code.execution.execute_and_trace_code(
        code_string=code_string,
        expected_output=expected_output,
        entrypoint_name=entrypoint_name,
        inputs=inputs,
        identifier="TACO/train/p000000/s0000/t0000",
        trace_only_inside_code_string=True,
        use_safe_execution=False,
        seed=0,
    )


def _make_trace_context(
    trace_result: pyine.utils.code.execution.TraceResult,
) -> tuple[pyine.data.traces.dataset_utils.TraceMetadata, pyine.data.traces.dataset_utils.CodingProblem]:
    """Build structurally matching metadata and problem objects for a trace fixture."""
    if trace_result.identifier is None:
        raise ValueError("fixture trace requires an identifier")
    trace_identifier = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_result.identifier)
    problem_identifier = trace_identifier.get_parent_identifier().get_parent_identifier()
    problem = pyine.data.traces.dataset_utils.CodingProblem(
        source_dataset_name=problem_identifier.dataset,
        source_data_path="/test/source",
        source_data_hash="problem-hash",
        problem_id=problem_identifier,
        problem_statement="Test fixture problem",
        problem_tags=["source:test"],
        test_inout_pairs=[(trace_result.inputs, trace_result.expected_output)],
        entrypoint_name=trace_result.entrypoint_name,
        potential_solution_ids=[trace_identifier.get_parent_identifier()],
        parsing_errors=None,
        is_banned=False,
    )
    metadata = pyine.data.traces.dataset_utils.TraceMetadata(
        identifier=trace_result.identifier,
        parent_dataset_hash="trace-shard-hash",
        index=0,
        internal_index=0,
        step_count=trace_result.valid_step_count,
        code_string=trace_result.code_string,
        inputs=trace_result.inputs,
        expected_output=trace_result.expected_output,
        return_value=trace_result.return_value,
        exception=trace_result.exception,
        stdout=trace_result.stdout,
        stderr=trace_result.stderr,
        metadata=trace_result.metadata,
        tags=trace_result.tags,
    )
    return metadata, problem


def _write_real_trace_source(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write one genuine JSON_ZSTD trace LMDB and its matching source split."""
    trace_result = _execute(
        "def solution(value):\n    return (value, value + 1)\n",
        expected_output=[3, 4],
        entrypoint_name="solution",
        inputs=3,
    )
    trace_metadata, problem = _make_trace_context(trace_result)
    source_path = tmp_path / "10s10t.000001of000001.test.lmdb"
    serialization = pyine.data.utils.lmdb_io.SerializationConfig(
        method=pyine.data.utils.lmdb_io.SerializationMethod.JSON_ZSTD,
        compression_kwargs={"level": 1},
    )
    with pyine.data.utils.lmdb_io.LMDBWriter(source_path, serialization_config=serialization) as writer:
        writer.write_metadata(
            {
                "parent_dataset": {
                    "dataset_name": "TACO",
                    "dataset_hash": "source-root-hash",
                },
                "writer_config": {
                    "source_dataset_name": "TACO",
                    "target_problem_ids": "test-shard",
                },
            }
        )
        writer.put(f"{problem.problem_id}{pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX}", problem.model_dump())
        writer.put(trace_metadata.identifier, trace_result.model_dump())
    split_path = tmp_path / "TACO-split.bin"
    split_result = pyine.data.utils.splits.SplitResult(
        source_dataset_name="TACO",
        source_dataset_hash="split-source-hash",
        identifiers=[str(problem.problem_id)],
        tag_lists=[["subset:train"]],
        source_data_hashes=[problem.source_data_hash],
        subset_assignments={str(problem.problem_id): "train"},
        creation_metadata={"fixture": True},
        config=pyine.data.utils.splits.SplitConfig(
            seed=0,
            subset_names=["train", "valid", "test"],
            subset_assign_prob_map={"train": 0.8, "valid": 0.1, "test": 0.1},
        ),
    )
    split_result.to_file(split_path)
    return source_path, split_path


class TestEvalExportConfig:
    def test_rejects_split_fractions_that_do_not_sum_to_one(self, tmp_path: pathlib.Path) -> None:
        """Reject configurations whose derived split fractions are incomplete."""
        with pytest.raises(pydantic.ValidationError, match="must sum to 1.0"):
            _make_config(
                tmp_path,
                train_fraction=0.5,
                validation_fraction=0.2,
                test_fraction=0.2,
            )

    def test_uses_agreed_export_defaults(self, tmp_path: pathlib.Path) -> None:
        """Use the agreed partition fractions and integrity-only default behavior."""
        config = _make_config(tmp_path)
        assert config.train_fraction == 0.50
        assert config.validation_fraction == 0.25
        assert config.test_fraction == 0.25
        assert config.reexecute_max_step_count is None
        assert not config.reexecute_all
        assert config.recheck_timeout_seconds == 60.0

    def test_rejects_conflicting_reexecution_settings(self, tmp_path: pathlib.Path) -> None:
        """Reject simultaneous exhaustive and step-limited re-execution settings."""
        with pytest.raises(pydantic.ValidationError, match="mutually exclusive"):
            _make_config(tmp_path, reexecute_all=True, reexecute_max_step_count=20_000)


class TestValueRendering:
    def test_renders_json_values_canonically(self) -> None:
        """Render JSON-native values with compact deterministic key ordering."""
        rendering = pyine.data.traces.eval_export.render_value({"z": [2, 1], "a": "\u00e9"})
        assert rendering.json_value == '{"a":"\u00e9","z":[2,1]}'
        assert rendering.text == rendering.json_value
        assert rendering.canonical

    @pytest.mark.parametrize("value", [(1, 2), {1, 2}, b"data", float("nan")])
    def test_does_not_normalize_non_json_values(self, value: typing.Any) -> None:
        """Keep non-JSON Python values out of canonical JSON fields."""
        rendering = pyine.data.traces.eval_export.render_value(value)
        assert rendering.json_value is None
        assert rendering.text
        assert not rendering.canonical


class TestOutcomeResolution:
    def test_resolves_entrypoint_return(self) -> None:
        """Select a callable return value as the final outcome."""
        trace_result = _execute(
            "def solution(value):\n    return value * 2\n",
            expected_output=6,
            entrypoint_name="solution",
            inputs=3,
        )
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "return_value"
        assert outcome.value == 6
        assert outcome.expected_matches

    def test_resolves_stdout(self) -> None:
        """Select stdout when a script communicates its outcome by printing."""
        trace_result = _execute("print(3)\n", expected_output="3")
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "stdout"
        assert outcome.value == "3\n"
        assert outcome.expected_matches

    def test_resolves_expected_exception(self) -> None:
        """Select an expected exception as the final outcome."""
        trace_result = _execute("raise ValueError('bad')\n", expected_output="ValueError(bad)")
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "exception"
        assert outcome.value == "ValueError(bad)"
        assert outcome.expected_matches

    def test_resolves_system_exit_stdout(self) -> None:
        """Retain PyINE's stdout fallback for SystemExit programs."""
        trace_result = _execute("import sys\nprint('done')\nsys.exit(13)\n", expected_output="done")
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "stdout"
        assert outcome.value == "done\n"
        assert outcome.expected_matches

    def test_preserves_actual_outcome_on_expected_mismatch(self) -> None:
        """Preserve the actual return value when the source expectation differs."""
        trace_result = _execute(
            "def solution(value):\n    return value * 2\n",
            expected_output=7,
            entrypoint_name="solution",
            inputs=3,
        )
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "return_value"
        assert outcome.value == 6
        assert not outcome.expected_matches

    def test_does_not_replace_callable_return_with_matching_stdout(self) -> None:
        """Keep callable returns authoritative even when stdout matches the expectation."""
        trace_result = _execute(
            "def solution():\n    print(7)\n    return 3\n",
            expected_output=7,
            entrypoint_name="solution",
        )
        outcome = pyine.data.traces.eval_export.resolve_trace_outcome(trace_result)
        assert outcome.kind == "return_value"
        assert outcome.value == 3
        assert not outcome.expected_matches


class TestProblemAssignment:
    def test_split_and_cap_are_order_independent(self, tmp_path: pathlib.Path) -> None:
        """Keep deterministic cap and split results independent of input order."""
        config = _make_config(tmp_path, max_problem_count=3, seed=13)
        problem_identifiers = [f"TACO/train/p{problem_idx:06d}" for problem_idx in range(10)]
        forward = pyine.data.traces.eval_export.select_problem_identifiers(problem_identifiers, config)
        backward = pyine.data.traces.eval_export.select_problem_identifiers(reversed(problem_identifiers), config)
        assert forward == backward
        assert len(forward) == 3
        assert {
            identifier: pyine.data.traces.eval_export.assign_export_split(identifier, config)
            for identifier in sorted(forward)
        } == {
            "TACO/train/p000001": "train",
            "TACO/train/p000003": "train",
            "TACO/train/p000005": "train",
        }


class TestRecordCertification:
    def test_validates_consistent_stored_record(self) -> None:
        """Pass record validation for a structurally consistent native trace."""
        reader = tests.utils.fake_dataset_readers.FakeTraceDatasetReader(
            config=tests.utils.fake_dataset_readers.FakeTraceDataConfig(
                dataset_name="TACO",
                subset_name="train",
                num_problems=1,
            )
        )
        result = pyine.data.traces.eval_export.validate_trace_record(
            reader[0],
            reader.trace_metadata[0],
            reader.get_problem_data(0),
        )
        assert result.method == "record_integrity_checked"
        assert result.outcome == "pass"

    def test_fails_when_stored_metadata_differs(self) -> None:
        """Fail integrity checking when metadata no longer mirrors the stored trace."""
        trace_result = _execute(
            "def solution(value):\n    return value\n",
            expected_output=2,
            entrypoint_name="solution",
            inputs=2,
        )
        trace_metadata, problem = _make_trace_context(trace_result)
        changed_metadata = dataclasses.replace(trace_metadata, return_value=999)
        result = pyine.data.traces.eval_export.validate_trace_record(trace_result, changed_metadata, problem)
        assert result.method == "record_integrity_checked"
        assert result.outcome == "fail"
        assert result.error_phase == "record_integrity"


class TestReexecutionCertification:
    @pytest.fixture(scope="class")
    def fake_record(
        self,
    ) -> tuple[
        pyine.utils.code.execution.TraceResult,
        pyine.data.traces.dataset_utils.TraceMetadata,
        pyine.data.traces.dataset_utils.CodingProblem,
    ]:
        """Provide one internally consistent fake trace, metadata record, and problem."""
        reader = tests.utils.fake_dataset_readers.FakeTraceDatasetReader(
            config=tests.utils.fake_dataset_readers.FakeTraceDataConfig(
                dataset_name="TACO",
                subset_name="train",
                num_problems=1,
            )
        )
        return reader[0], reader.trace_metadata[0], reader.get_problem_data(0)

    def test_reproduces_stored_outcome(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_record: tuple[
            pyine.utils.code.execution.TraceResult,
            pyine.data.traces.dataset_utils.TraceMetadata,
            pyine.data.traces.dataset_utils.CodingProblem,
        ],
    ) -> None:
        """Pass re-execution certification when the stored outcome is reproduced."""
        trace_result, trace_metadata, problem = fake_record
        execution_kwargs: dict[str, typing.Any] = {}

        def _rerun(**kwargs: typing.Any) -> pyine.utils.code.execution.TraceResult:
            """Return an outcome-only copy while recording forwarded execution options."""
            execution_kwargs.update(kwargs)
            return trace_result.model_copy(update={"traced_steps": [], "traced_steps_map": {}})

        monkeypatch.setattr(pyine.utils.code.execution, "execute_and_trace_code", _rerun)
        config = _make_config(tmp_path, reexecute_all=True)
        result = pyine.data.traces.eval_export.certify_trace(trace_result, trace_metadata, problem, config)
        assert result.method == "reexecuted"
        assert result.outcome == "pass"
        assert result.exact_match
        assert result.semantic_match
        assert result.outcome_source == "reexecuted_live"
        assert execution_kwargs["capture_trace_events"] is False
        assert execution_kwargs["use_safe_execution"] is True

    def test_does_not_fallback_after_reexecution_mismatch(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_record: tuple[
            pyine.utils.code.execution.TraceResult,
            pyine.data.traces.dataset_utils.TraceMetadata,
            pyine.data.traces.dataset_utils.CodingProblem,
        ],
    ) -> None:
        """Report a failed re-execution directly instead of silently changing methods."""
        trace_result, trace_metadata, problem = fake_record
        mismatching_code = "def solution(value):\n    return 999\n"
        changed_trace = trace_result.model_copy(update={"code_string": mismatching_code})
        changed_metadata = dataclasses.replace(trace_metadata, code_string=mismatching_code)
        rerun_trace = changed_trace.model_copy(
            update={
                "return_value": 999,
                "traced_steps": [],
                "traced_steps_map": {},
            }
        )
        monkeypatch.setattr(pyine.utils.code.execution, "execute_and_trace_code", lambda **_: rerun_trace)
        config = _make_config(tmp_path, reexecute_all=True)
        result = pyine.data.traces.eval_export.certify_trace(changed_trace, changed_metadata, problem, config)
        assert result.method == "reexecuted"
        assert result.outcome == "fail"

    def test_propagates_process_interrupts(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_record: tuple[
            pyine.utils.code.execution.TraceResult,
            pyine.data.traces.dataset_utils.TraceMetadata,
            pyine.data.traces.dataset_utils.CodingProblem,
        ],
    ) -> None:
        """Propagate process-control exceptions from certification execution."""
        trace_result, trace_metadata, problem = fake_record

        def _interrupt(**_: typing.Any) -> typing.NoReturn:
            """Simulate an operator interrupt inside the execution path."""
            raise KeyboardInterrupt

        monkeypatch.setattr(pyine.utils.code.execution, "execute_and_trace_code", _interrupt)
        config = _make_config(tmp_path, reexecute_all=True)
        with pytest.raises(KeyboardInterrupt):
            pyine.data.traces.eval_export.certify_trace(trace_result, trace_metadata, problem, config)

    def test_real_reexecution_preserves_live_tuple_text(self, tmp_path: pathlib.Path) -> None:
        """Use the native live tuple while reporting its stored list as only semantically equal."""
        live_trace = _execute(
            "def solution(value):\n    return (value, value + 1)\n",
            expected_output=[3, 4],
            entrypoint_name="solution",
            inputs=3,
        )
        stored_trace = live_trace.model_copy(update={"return_value": [3, 4]})
        trace_metadata, problem = _make_trace_context(stored_trace)
        result = pyine.data.traces.eval_export.certify_trace(
            stored_trace,
            trace_metadata,
            problem,
            _make_config(tmp_path, reexecute_all=True),
        )
        assert result.outcome == "pass"
        assert not result.exact_match
        assert result.semantic_match
        assert result.outcome_source == "reexecuted_live"
        assert result.selected_outcome.value == (3, 4)

    def test_real_reexecution_uses_strict_callable_return(self, tmp_path: pathlib.Path) -> None:
        """Certify the callable's return channel rather than matching printed output."""
        trace_result = _execute(
            "def solution():\n    print(7)\n    return 3\n",
            expected_output=7,
            entrypoint_name="solution",
        )
        trace_metadata, problem = _make_trace_context(trace_result)
        result = pyine.data.traces.eval_export.certify_trace(
            trace_result,
            trace_metadata,
            problem,
            _make_config(tmp_path, reexecute_all=True),
        )
        assert result.outcome == "pass"
        assert result.selected_outcome.kind == "return_value"
        assert result.selected_outcome.value == 3
        assert not result.selected_outcome.expected_matches

    @pytest.mark.parametrize(
        ("code_string", "expected_output", "expected_kind"),
        [
            ("print('done')\n", "done", "stdout"),
            ("raise ValueError('bad')\n", "ValueError(bad)", "exception"),
            ("import sys\nprint('done')\nsys.exit(13)\n", "done", "stdout"),
        ],
    )
    def test_real_reexecution_covers_script_outcomes(
        self,
        tmp_path: pathlib.Path,
        code_string: str,
        expected_output: typing.Any,
        expected_kind: str,
    ) -> None:
        """Exercise isolated stdout, exception, and SystemExit script paths."""
        trace_result = _execute(code_string, expected_output=expected_output)
        trace_metadata, problem = _make_trace_context(trace_result)
        result = pyine.data.traces.eval_export.certify_trace(
            trace_result,
            trace_metadata,
            problem,
            _make_config(tmp_path, reexecute_all=True),
        )
        assert result.outcome == "pass"
        assert result.selected_outcome.kind == expected_kind

    def test_real_reexecution_reports_timeout(self, tmp_path: pathlib.Path) -> None:
        """Report an actual isolated timeout as a failed re-execution without fallback."""
        trace_result = _execute(
            "def solution():\n    return 1\n",
            expected_output=1,
            entrypoint_name="solution",
        )
        trace_metadata, problem = _make_trace_context(trace_result)
        hanging_code = "def solution():\n    while True:\n        pass\n"
        hanging_trace = trace_result.model_copy(update={"code_string": hanging_code})
        hanging_metadata = dataclasses.replace(trace_metadata, code_string=hanging_code)
        result = pyine.data.traces.eval_export.certify_trace(
            hanging_trace,
            hanging_metadata,
            problem,
            _make_config(tmp_path, reexecute_all=True, recheck_timeout_seconds=0.05),
        )
        assert result.method == "reexecuted"
        assert result.outcome == "fail"
        assert result.error_phase == "reexecution"
        assert "TimeoutError" in result.reason


class TestSourceValidation:
    def test_rejects_incomplete_numbered_shards(self, tmp_path: pathlib.Path) -> None:
        """Reject missing ordinals from a conventionally numbered source set."""
        shards = [
            pyine.data.traces.eval_export.SourceShardInfo(
                path=str(tmp_path / f"10s10t.{part_idx:06d}of000003.test.lmdb"),
                trace_count=1,
                dataset_hash=f"shard-{part_idx}",
                parent_dataset={"dataset_name": "TACO", "dataset_hash": "source-root"},
                writer_config={"source_dataset_name": "TACO", "target_problem_ids": str(part_idx)},
            )
            for part_idx in (1, 3)
        ]
        config = _make_config(tmp_path)
        with pytest.raises(ValueError, match="incomplete numbered source shard set"):
            pyine.data.traces.eval_export.validate_source_shards(  # type: ignore[reportPrivateUsage]
                shards,
                config,
            )
        partial_config = config.model_copy(update={"allow_partial_source": True})
        pyine.data.traces.eval_export.validate_source_shards(  # type: ignore[reportPrivateUsage]
            shards,
            partial_config,
        )

    def test_rejects_incompatible_writer_configs(self, tmp_path: pathlib.Path) -> None:
        """Reject source shards whose non-selector trace-writer settings differ."""
        shards = [
            pyine.data.traces.eval_export.SourceShardInfo(
                path=str(tmp_path / f"source-{shard_idx}.lmdb"),
                trace_count=1,
                dataset_hash=f"shard-{shard_idx}",
                parent_dataset={"dataset_name": "TACO", "dataset_hash": "source-root"},
                writer_config={
                    "source_dataset_name": "TACO",
                    "target_problem_ids": str(shard_idx),
                    "max_tests_per_solution": shard_idx,
                },
            )
            for shard_idx in (1, 2)
        ]
        with pytest.raises(ValueError, match="incompatible writer configurations"):
            pyine.data.traces.eval_export.validate_source_shards(  # type: ignore[reportPrivateUsage]
                shards,
                _make_config(tmp_path),
            )

    def test_rejects_problem_hash_mismatch(self, tmp_path: pathlib.Path) -> None:
        """Reject source problem content that differs from the supplied split revision."""
        reader = tests.utils.fake_dataset_readers.FakeTraceDatasetReader(
            config=tests.utils.fake_dataset_readers.FakeTraceDataConfig(
                dataset_name="TACO",
                subset_name="train",
                num_problems=1,
            )
        )
        problem_identifier = str(reader.get_problem_data(0).problem_id)
        split_result = pyine.data.utils.splits.SplitResult(
            source_dataset_name="TACO",
            source_dataset_hash="source-hash",
            identifiers=[problem_identifier],
            tag_lists=[[]],
            source_data_hashes=["different-hash"],
            subset_assignments={problem_identifier: "train"},
            creation_metadata={},
            config=pyine.data.utils.splits.SplitConfig(
                subset_names=["train", "valid"],
                subset_assign_prob_map={"train": 0.8, "valid": 0.2},
            ),
        )
        with pytest.raises(ValueError, match="source problem hash does not match"):
            pyine.data.traces.eval_export.validate_source_problem_hashes(  # type: ignore[reportPrivateUsage]
                [reader],
                split_result,
            )

    def test_rejects_source_problem_absent_from_split(self) -> None:
        """Reject trace problems that the supplied split cannot classify for the embargo."""
        reader = tests.utils.fake_dataset_readers.FakeTraceDatasetReader(
            config=tests.utils.fake_dataset_readers.FakeTraceDataConfig(
                dataset_name="TACO",
                subset_name="train",
                num_problems=1,
            )
        )
        split_result = pyine.data.utils.splits.SplitResult(
            source_dataset_name="TACO",
            source_dataset_hash="source-hash",
            identifiers=["TACO/train/p999999"],
            tag_lists=[[]],
            source_data_hashes=["other-hash"],
            subset_assignments={"TACO/train/p999999": "train"},
            creation_metadata={},
            config=pyine.data.utils.splits.SplitConfig(
                subset_names=["train", "valid"],
                subset_assign_prob_map={"train": 0.8, "valid": 0.2},
            ),
        )
        with pytest.raises(ValueError, match="absent from the supplied split"):
            pyine.data.traces.eval_export.validate_source_problem_hashes(  # type: ignore[reportPrivateUsage]
                [reader],
                split_result,
            )


class TestRealLmdbExport:
    def test_preserves_honest_stored_representation_and_promotes_artifact(self, tmp_path: pathlib.Path) -> None:
        """Export a real JSON_ZSTD shard and expose its upstream tuple-to-list normalization."""
        source_path, split_path = _write_real_trace_source(tmp_path)
        output_dir = tmp_path / "export"
        manifest = pyine.data.traces.eval_export.export_eval_traces(
            pyine.data.traces.eval_export.EvalExportConfig(
                source_lmdb_paths=[source_path],
                split_file_path=split_path,
                output_dir=output_dir,
            )
        )
        rows = [
            row
            for split_name in ("train", "validation", "test")
            for row in pq.read_table(output_dir / f"{split_name}.parquet").to_pylist()
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["outcome_text"] == "[3,4]"
        assert row["outcome_json"] == "[3,4]"
        assert row["outcome_source"] == "stored_record"
        assert "values_canonical" not in row
        assert manifest.source_shards[0].writer_config["source_dataset_name"] == "TACO"
        assert output_dir.is_dir()
        assert not output_dir.with_name("export.incomplete").exists()

    def test_refuses_existing_or_incomplete_output(self, tmp_path: pathlib.Path) -> None:
        """Refuse both completed and interrupted output paths without deleting either."""
        source_path, split_path = _write_real_trace_source(tmp_path)
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        config = pyine.data.traces.eval_export.EvalExportConfig(
            source_lmdb_paths=[source_path],
            split_file_path=split_path,
            output_dir=output_dir,
        )
        with pytest.raises(FileExistsError, match="output directory already exists"):
            pyine.data.traces.eval_export.export_eval_traces(config)
        output_dir.rmdir()
        incomplete_dir = output_dir.with_name("export.incomplete")
        incomplete_dir.mkdir()
        with pytest.raises(FileExistsError, match="incomplete export directory already exists"):
            pyine.data.traces.eval_export.export_eval_traces(config)

    def test_failed_final_validation_leaves_only_incomplete_artifact(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Leave staged diagnostics without promoting a failed artifact."""
        source_path, split_path = _write_real_trace_source(tmp_path)
        output_dir = tmp_path / "export"

        def _fail_validation(*_: typing.Any) -> typing.NoReturn:
            """Simulate a final staged-artifact validation failure."""
            raise ValueError("invalid staged artifact")

        monkeypatch.setattr(
            pyine.data.traces.eval_export,
            "_validate_staged_artifact",
            _fail_validation,
        )
        with pytest.raises(ValueError, match="invalid staged artifact"):
            pyine.data.traces.eval_export.export_eval_traces(
                pyine.data.traces.eval_export.EvalExportConfig(
                    source_lmdb_paths=[source_path],
                    split_file_path=split_path,
                    output_dir=output_dir,
                )
            )
        assert not output_dir.exists()
        assert output_dir.with_name("export.incomplete").is_dir()


class TestEvalExportCli:
    def test_rejects_conflicting_reexecution_options(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reject exhaustive and step-limited CLI options before resolving source data."""
        source_path = tmp_path / "source.lmdb"
        source_path.mkdir()
        split_path = tmp_path / "split.bin"
        split_path.touch()
        monkeypatch.setattr(pyine.utils.reprod, "entrypoint_setup", lambda: None)
        result = click.testing.CliRunner().invoke(
            pyine.apps.traces.eval_exporter.main,
            [
                "--lmdb-path",
                str(source_path),
                "--split-file",
                str(split_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--reexecute-all",
                "--reexecute-max-step-count",
                "20",
            ],
        )
        assert result.exit_code == 2
        assert "use either --reexecute-all or --reexecute-max-step-count" in result.output

    def test_writes_self_consistent_local_artifact(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Write and validate a complete synthetic local export artifact."""
        source_path = tmp_path / "source.lmdb"
        source_path.mkdir()
        split_file_path = tmp_path / "TACO-split.bin"
        output_dir = tmp_path / "export"
        fake_config = tests.utils.fake_dataset_readers.FakeTraceDataConfig(
            dataset_name="TACO",
            subset_name="train",
            num_problems=4,
            solutions_per_problem=1,
            tests_per_problem=1,
            augmented_per_solution=1,
        )

        class _ExportFakeReader(tests.utils.fake_dataset_readers.FakeTraceDatasetReader):
            """Expose the source metadata required from a production trace shard."""

            @property
            def metadata(self) -> dict[str, typing.Any]:
                """Return compatible parent-dataset and writer provenance."""
                return {
                    "parent_dataset": {
                        "dataset_name": "TACO",
                        "dataset_hash": "source-root-hash",
                    },
                    "writer_config": {
                        "source_dataset_name": "TACO",
                        "target_problem_ids": "test-shard",
                    },
                }

        def _make_fake_reader(
            lmdb_path: pathlib.Path,
        ) -> tests.utils.fake_dataset_readers.FakeTraceDatasetReader:
            """Create a deterministic in-memory replacement for a native trace shard."""
            reader = _ExportFakeReader(lmdb_path, config=fake_config)
            reader._problems = [  # type: ignore[reportPrivateUsage]
                problem.model_copy(update={"problem_tags": [*problem.problem_tags, "source:code_contests"]})
                for problem in reader._problems  # type: ignore[reportPrivateUsage]
            ]
            return reader

        problem_identifiers = [f"TACO/train/p{problem_idx:06d}" for problem_idx in range(fake_config.num_problems)]
        source_data_hashes = [
            hashlib.sha256(identifier.encode()).hexdigest()[:16] for identifier in problem_identifiers
        ]
        split_config = pyine.data.utils.splits.SplitConfig(
            seed=0,
            subset_names=["train", "valid", "test"],
            subset_assign_prob_map={"train": 0.8, "valid": 0.1, "test": 0.1},
        )
        split_result = pyine.data.utils.splits.SplitResult(
            source_dataset_name="TACO",
            source_dataset_hash="source-hash",
            identifiers=problem_identifiers,
            tag_lists=[["subset:train"] for _ in problem_identifiers],
            source_data_hashes=source_data_hashes,
            subset_assignments=dict.fromkeys(problem_identifiers, "train"),
            creation_metadata={},
            config=split_config,
        )
        split_result.to_file(split_file_path)
        monkeypatch.setattr(pyine.data.traces.dataset_reader, "DatasetReader", _make_fake_reader)
        monkeypatch.setattr(pyine.utils.reprod, "entrypoint_setup", lambda: None)
        certification_batch_sizes: list[int] = []
        original_run_in_parallel = pyine.utils.concurrency.run_in_parallel

        def _run_bounded(
            callables: typing.Sequence[typing.Callable[[], typing.Any]],
            **kwargs: typing.Any,
        ) -> tuple[list[typing.Any], list[BaseException | None]]:
            """Record heavy certification batch sizes before delegating to the real helper."""
            certification_batch_sizes.append(len(callables))
            return original_run_in_parallel(callables, **kwargs)

        monkeypatch.setattr(pyine.utils.concurrency, "run_in_parallel", _run_bounded)
        runner = click.testing.CliRunner()
        result = runner.invoke(
            pyine.apps.traces.eval_exporter.main,
            [
                "--lmdb-path",
                str(source_path),
                "--split-file",
                str(split_file_path),
                "--output-dir",
                str(output_dir),
                "--parquet-batch-size",
                "5",
                "--certification-workers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_dataset_hash"] == "source-hash"
        assert set(manifest["environment"]) == {
            "python_version",
            "platform",
            "uv_lock_sha256",
            "installed_packages_sha256",
        }
        all_rows: list[dict[str, typing.Any]] = []
        problem_splits: dict[int, set[str]] = {}
        for split_name in ("train", "validation", "test"):
            parquet_path = output_dir / f"{split_name}.parquet"
            table = pq.read_table(parquet_path)
            assert table.schema.names == manifest["schema_columns"]
            rows = typing.cast("list[dict[str, typing.Any]]", table.to_pylist())
            assert len(rows) == manifest["partition_row_counts"][split_name]
            assert [row["identifier"] for row in rows] == sorted(row["identifier"] for row in rows)
            for row in rows:
                assert row["pyine_subset"] == "train"
                assert row["export_split"] == split_name
                assert row["source_dataset_names"] == ["code_contests"]
                assert row["recheck_method"] == "record_integrity_checked"
                assert row["recheck_outcome"] == "pass"
                assert row["recheck_exact_match"] is None
                assert row["recheck_semantic_match"] is None
                assert row["outcome_source"] == "stored_record"
                assert row["invocation_kind"] == "callable"
                for value_field in (
                    "inputs_json",
                    "expected_output_json",
                    "outcome_json",
                    "return_value_json",
                ):
                    json.loads(row[value_field])
                comparison = pyine.utils.code.output_compare.compare(
                    json.loads(row["expected_output_json"]),
                    json.loads(row["outcome_json"]),
                    options=pyine.utils.code.output_compare.get_default_comparison_config(),
                )
                assert bool(comparison) == row["expected_matches_outcome"]
                problem_splits.setdefault(row["problem_idx"], set()).add(split_name)
            file_info = manifest["files"][split_name]
            assert file_info["sha256"] == pyine.utils.reprod.compute_hash(parquet_path)
            assert file_info["size_bytes"] == parquet_path.stat().st_size
            all_rows.extend(rows)
        identifiers = [row["identifier"] for row in all_rows]
        assert len(identifiers) == len(set(identifiers)) == 8
        assert all(len(split_names) == 1 for split_names in problem_splits.values())
        certification_lines = (output_dir / "certification" / "certification.jsonl").read_text().splitlines()
        assert len(certification_lines) == len(all_rows)
        certification_total = sum(
            count for method_counts in manifest["certification_counts"].values() for count in method_counts.values()
        )
        assert certification_total == len(all_rows)
        assert manifest["flag_counts"]["has_exception"] == 0
        certification_log_path = output_dir / manifest["certification_log"]["relative_path"]
        assert manifest["certification_log"]["sha256"] == pyine.utils.reprod.compute_hash(certification_log_path)
        assert manifest["source_split"]["file"]["sha256"] == pyine.utils.reprod.compute_hash(split_file_path)
        v1_specification = pathlib.Path(pyine.data.traces.eval_export.__file__).with_name("EVAL_EXPORT_SPEC_V1.md")
        assert (output_dir / "EVAL_EXPORT_SPEC.md").read_bytes() == v1_specification.read_bytes()
        assert (output_dir / "export_schema.json").is_file()
        assert (output_dir / "installed_packages.json").is_file()
        assert not output_dir.with_name(f"{output_dir.name}.incomplete").exists()
        assert certification_batch_sizes
        assert max(certification_batch_sizes) <= 2
