import hashlib
import json
import pathlib
import typing

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyine.data.traces.export_consumer as consumer


def _event(
    kind: str,
    line: int,
    **assertions: typing.Any,
) -> dict[str, typing.Any]:
    return {
        "kind": kind,
        "code_object": "solution:1",
        "line": line,
        "arguments": [],
        "deltas": [],
        "return_value": None,
        "stdout_since_prev_capture": None,
        "stderr_since_prev_capture": None,
        "exception": None,
        **assertions,
    }


class TestRenderExampleWithSpans:
    @pytest.fixture()
    def example(self) -> dict[str, typing.Any]:
        # non-ASCII values check that spans count code points
        return {
            "problem_statement": "Append an accent: \u00e9",
            "code_string": "def solution(value):\n    return value + '\u00e9'\n",
            "invocation_text": "entrypoint solution; observed bindings (not call syntax)",
            "supplied_stdin": None,
            "observed_bindings": [{"name": "value", "value": "'caf\u00e9'"}],
            "inputs_text": "'caf\u00e9'",
            "reasoning_events": [
                _event("call", 1, arguments=[{"name": "value", "value": "'caf\u00e9'"}]),
                _event("line", 2),
                _event("return", 2, return_value="'caf\u00e9\u00e9'"),
            ],
            "candidate_output_text": "'caf\u00e9\u00e9'",
        }

    def test_text_matches_render_example(
        self,
        example: dict[str, typing.Any],
    ) -> None:
        assert consumer.render_example_with_spans(example)[0] == consumer.render_example(example)

    def test_spans_locate_each_step_line(
        self,
        example: dict[str, typing.Any],
    ) -> None:
        text, spans = consumer.render_example_with_spans(example)
        assert len(spans) == len(example["reasoning_events"])
        for event_idx, (event, (start, end)) in enumerate(zip(example["reasoning_events"], spans, strict=True)):
            assert text[start:end] == consumer.render_event(event, event_idx)
            assert text[start - 1] == "\n" and text[end] == "\n"

    @pytest.mark.parametrize("reasoning_events", [None, []])
    def test_no_spans_without_events(
        self,
        example: dict[str, typing.Any],
        reasoning_events: list[dict[str, typing.Any]] | None,
    ) -> None:
        example["reasoning_events"] = reasoning_events
        text, spans = consumer.render_example_with_spans(example)
        assert spans == []
        assert text == consumer.render_example(example)


class TestIterGroups:
    def test_groups_spanning_batches_stay_whole(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        path = tmp_path / "split.parquet"
        group_ids = ["first", "first", "second", "third", "third", "third"]
        pq.write_table(
            pa.table({"query_group_id": group_ids, "row_id": [f"row{idx}" for idx in range(len(group_ids))]}),
            path,
        )
        groups = list(consumer.iter_groups(path, batch_size=2))
        assert [[row["row_id"] for row in rows] for rows in groups] == [
            ["row0", "row1"],
            ["row2"],
            ["row3", "row4", "row5"],
        ]


class TestIterProblems:
    def test_problems_collect_their_adjacent_groups(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        path = tmp_path / "split.parquet"
        group_ids = ["first", "first", "second", "third", "fourth"]
        problem_ids = ["alpha", "alpha", "alpha", "beta", "gamma"]
        pq.write_table(
            pa.table(
                {
                    "problem_id": problem_ids,
                    "query_group_id": group_ids,
                    "row_id": [f"row{idx}" for idx in range(len(group_ids))],
                }
            ),
            path,
        )
        problems = list(consumer.iter_problems(path, batch_size=2))
        assert [[[row["row_id"] for row in rows] for rows in groups] for groups in problems] == [
            [["row0", "row1"], ["row2"]],
            [["row3"]],
            [["row4"]],
        ]


class TestVerifyExport:
    @pytest.fixture()
    def export_dir(
        self,
        tmp_path: pathlib.Path,
    ) -> pathlib.Path:
        (tmp_path / "train.parquet").write_bytes(b"rows")
        (tmp_path / "consumer.py").write_text("print('renderer')\n")
        files = {
            path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
            for path in sorted(tmp_path.iterdir())
        }
        (tmp_path / "export_manifest.json").write_text(json.dumps({"files": files}))
        return tmp_path

    def test_matching_files_pass(
        self,
        export_dir: pathlib.Path,
    ) -> None:
        consumer.verify_export(export_dir)

    def test_author_manifest_is_used_without_public_manifest(
        self,
        export_dir: pathlib.Path,
    ) -> None:
        (export_dir / "export_manifest.json").rename(export_dir / "author_manifest.json")
        consumer.verify_export(export_dir)

    def test_modified_file_fails(
        self,
        export_dir: pathlib.Path,
    ) -> None:
        (export_dir / "train.parquet").write_bytes(b"ROWS")
        with pytest.raises(ValueError, match="train.parquet differs"):
            consumer.verify_export(export_dir)

    def test_missing_file_fails(
        self,
        export_dir: pathlib.Path,
    ) -> None:
        (export_dir / "consumer.py").unlink()
        with pytest.raises(ValueError, match="consumer.py is not a regular file"):
            consumer.verify_export(export_dir)

    def test_nested_manifest_entry_fails(
        self,
        export_dir: pathlib.Path,
    ) -> None:
        manifest = json.loads((export_dir / "export_manifest.json").read_text())
        manifest["files"]["../train.parquet"] = manifest["files"].pop("train.parquet")
        (export_dir / "export_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="is not a regular file directly inside"):
            consumer.verify_export(export_dir)
