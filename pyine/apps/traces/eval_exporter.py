"""Standalone CLI for the local PyINE evaluation-trace export.

The default facts contract is documented in ``pyine/data/traces/EVAL_EXPORT_SPEC_V1.md``.
Explicit statements-v2 exports use ``pyine/data/traces/EVAL_EXPORT_SPEC_V2.md`` and the export guide.

Example:

    ```bash
    python -m pyine.apps.traces.eval_exporter \
        --lmdb-pattern '10s10t-v1.*.lmdb' \
        --output-dir /path/to/pyine-v1-predictor-eval
    ```

Pass ``--lmdb-path`` repeatedly instead when the exact shard paths are already known.
"""

import json
import pathlib

import click

import pyine.data.traces.dataset_utils
import pyine.data.traces.eval_export
import pyine.data.traces.eval_export_v2
import pyine.data.utils.splits
import pyine.utils.reprod


def _resolve_lmdb_paths(
    source_dataset_name: str,
    lmdb_paths: tuple[pathlib.Path, ...],
    lmdb_pattern: str | None,
) -> list[pathlib.Path]:
    """Resolve explicitly provided trace shards or a dataset-root pattern."""
    if lmdb_paths and lmdb_pattern is not None:
        raise click.UsageError("use either --lmdb-path or --lmdb-pattern, not both")
    if lmdb_paths:
        return list(lmdb_paths)
    if lmdb_pattern is None:
        raise click.UsageError("provide at least one --lmdb-path or an --lmdb-pattern")
    resolved_paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
        source_dataset_name=source_dataset_name,
        pattern=lmdb_pattern,
    )
    if not resolved_paths:
        raise click.ClickException(f"no {source_dataset_name} trace dataset paths matched pattern: {lmdb_pattern!r}")
    return resolved_paths


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--export-mode",
    type=click.Choice(["facts-v1", "statements-v2"]),
    default="facts-v1",
    show_default=True,
    help="Export legacy execution facts or coordinated program-outcome classification tasks.",
)
@click.option(
    "--recipe-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="V2 YAML recipe; projection recipes may configure visibility and mixture only.",
)
@click.option(
    "--author-output-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="New private artifact directory, required when constructing a v2 export.",
)
@click.option(
    "--project-from-author",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    help="Project a completed v2 author artifact without accessing native sources.",
)
@click.option(
    "--source-report", is_flag=True, help="Report v2 source eligibility as JSON without exporting or executing."
)
@click.option("--inspect-stored-events", is_flag=True, help="Include native event-scope checks in --source-report.")
@click.option(
    "--lmdb-path",
    "lmdb_paths",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    multiple=True,
    help="Native trace LMDB shard path. Repeat for multiple shards.",
)
@click.option(
    "--lmdb-pattern",
    type=str,
    default=None,
    help="Pattern resolved under PyINE's trace data root for the configured source dataset.",
)
@click.option(
    "--source-dataset-name",
    type=click.Choice(pyine.data.traces.dataset_utils.SUPPORTED_SOURCE_DATASETS, case_sensitive=True),
    default="TACO",
    show_default=True,
    help="Source coding-problem dataset represented by the trace shards.",
)
@click.option(
    "--split-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Original PyINE split file. Defaults to the source dataset's registered split.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    default=None,
    help="New public export directory; not accepted with --source-report.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Deterministic cap and split seed.")
@click.option(
    "--train-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_TRAIN_FRACTION,
    show_default=True,
    help="Whole-problem training fraction; the three split fractions must sum to one.",
)
@click.option(
    "--validation-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_VALIDATION_FRACTION,
    show_default=True,
    help="Whole-problem validation fraction of the selected original training problems.",
)
@click.option(
    "--test-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_TEST_FRACTION,
    show_default=True,
    help="Whole-problem test fraction of the selected original training problems.",
)
@click.option(
    "--max-problem-count",
    type=click.IntRange(min=1),
    default=None,
    help="Optional deterministic whole-problem cap. By default every allowed problem is exported.",
)
@click.option(
    "--reexecute-max-step-count",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Opt into re-execution up to this stored valid-step count. Omitted means integrity-only; "
        "the recommended exhaustive-dataset ceiling is "
        f"{pyine.data.traces.eval_export.DEFAULT_REEXECUTE_MAX_STEP_COUNT}."
    ),
)
@click.option(
    "--reexecute-all",
    is_flag=True,
    help="Re-execute every structurally valid trace, ignoring the stored step count.",
)
@click.option(
    "--recheck-timeout-seconds",
    type=click.FloatRange(min=0.0, min_open=True),
    default=pyine.data.traces.eval_export.DEFAULT_RECHECK_TIMEOUT_SECONDS,
    show_default=True,
    help="Timeout for each outcome-only certification re-execution.",
)
@click.option(
    "--execution-seed-override",
    type=int,
    default=None,
    help="Override the stored execution seed during re-execution.",
)
@click.option(
    "--certification-workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Concurrent certification workers.",
)
@click.option(
    "--parquet-batch-size",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Target Parquet processing batch size; v2 author writes keep query groups intact.",
)
@click.option(
    "--allow-partial-source",
    is_flag=True,
    help="Allow an intentionally incomplete set of conventionally numbered trace shards.",
)
def main(
    export_mode: str,
    recipe_file: pathlib.Path | None,
    author_output_dir: pathlib.Path | None,
    project_from_author: pathlib.Path | None,
    source_report: bool,
    inspect_stored_events: bool,
    lmdb_paths: tuple[pathlib.Path, ...],
    lmdb_pattern: str | None,
    source_dataset_name: str,
    split_file: pathlib.Path | None,
    output_dir: pathlib.Path | None,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    max_problem_count: int | None,
    reexecute_max_step_count: int | None,
    reexecute_all: bool,
    recheck_timeout_seconds: float,
    execution_seed_override: int | None,
    certification_workers: int,
    parquet_batch_size: int,
    allow_partial_source: bool,
) -> None:
    """Export original-train traces as legacy facts or structured classification tasks.

    V2 construction writes separate private author and public artifacts. Alternatively,
    project a completed author artifact or print a source-eligibility report as JSON.
    Normal export reads stored executions; re-execution requires an explicit option.

    Args:
        export_mode: Artifact contract, either the default ``facts-v1`` or opt-in
            ``statements-v2``. Recipe, author, projection, and report options require v2.
        recipe_file: Optional v2 YAML recipe. Omission uses the default construction
            recipe, or the author artifact's visibility and mixture settings during
            projection. Projection accepts only recipe-version, visibility, and mixture settings.
        author_output_dir: New private artifact directory for v2 construction. Must
            not overlap the public output or native sources; rejected by v1 and reports.
        project_from_author: Completed v2 author directory to project instead of
            constructing new examples. Cannot be combined with source, selection,
            certification, report, or author-output options.
        source_report: Print v2 source eligibility instead of creating an export.
            Does not execute programs and rejects output directory options.
        inspect_stored_events: Include stored-record, event-scope, and alternate-input
            shared-cut checks in the source report. Requires ``source_report``.
        lmdb_paths: Explicit native LMDB directories, supplied by repeating
            ``--lmdb-path``. Mutually exclusive with ``lmdb_pattern``.
        lmdb_pattern: Shard-name pattern resolved under the registered source dataset's
            trace root. Supply this or explicit paths for construction and reports.
        source_dataset_name: Registered dataset name used for source and split resolution.
        split_file: Original PyINE split artifact. Omission resolves the dataset's
            registered split; only its original training problems are selected.
        output_dir: New public artifact directory. Required except for source reports,
            which reject it; existing outputs and sibling incomplete directories are not
            overwritten.
        seed: Deterministic problem-selection and split seed, also used for v2 donor
            selection, splicing, and pruning. Does not seed certification re-execution.
        train_fraction: Derived training fraction; the three fractions must sum to one.
        validation_fraction: Derived validation fraction, assigned by whole problem.
        test_fraction: Derived test fraction, assigned by whole problem.
        max_problem_count: Optional positive cap on selected original-training problems.
            Selection is deterministic and keeps each selected problem family together.
        reexecute_max_step_count: Opt into outcome-only re-execution of structurally
            valid traces at or below this stored valid-step count. Mutually exclusive
            with ``reexecute_all``; omission leaves integrity-only certification.
        reexecute_all: Re-execute every structurally valid selected trace regardless
            of its step count. Not available for reports or projection.
        recheck_timeout_seconds: Positive timeout for each certification re-execution.
        execution_seed_override: Optional replacement for the stored execution seed
            during certification. Does not affect problem assignment or v2 view selection.
        certification_workers: Positive concurrency limit for trace certification.
        parquet_batch_size: Positive Parquet processing batch size. V2 author writes
            keep query groups intact, so a write batch may exceed this target.
        allow_partial_source: Accept an incomplete conventionally numbered shard set.
            Does not relax source identities, parent closure, writer compatibility,
            or split problem-hash validation.

    Returns:
        None. Writes the requested artifacts and prints their summary, or prints the
        source report as JSON. Source readers may prepare metadata caches.

    Raises:
        click.UsageError: Required options are absent or an option combination is invalid.
        click.ClickException: A source shard pattern matches no datasets.
        FileExistsError: An output or sibling incomplete directory already exists.
        ValueError: Configuration, source integrity, or artifact validation fails.
    """
    if reexecute_all and reexecute_max_step_count is not None:
        raise click.UsageError("use either --reexecute-all or --reexecute-max-step-count, not both")
    if export_mode == "facts-v1" and any(
        (recipe_file, author_output_dir, project_from_author, source_report, inspect_stored_events)
    ):
        raise click.UsageError("v2-only options require --export-mode statements-v2")
    if inspect_stored_events and not source_report:
        raise click.UsageError("--inspect-stored-events requires --source-report")
    if output_dir is None and not source_report:
        raise click.UsageError("Missing option '--output-dir'.")
    recipe = pyine.data.traces.eval_export_v2.load_recipe(recipe_file) if export_mode == "statements-v2" else None
    if project_from_author is not None:
        if recipe is not None and recipe.model_fields_set - {"recipe_version", "visibility", "mixture"}:
            raise click.UsageError("projection recipes may configure visibility and mixture only")
        context = click.get_current_context()
        construction_options = (
            "source_dataset_name",
            "seed",
            "train_fraction",
            "validation_fraction",
            "test_fraction",
            "max_problem_count",
            "recheck_timeout_seconds",
            "execution_seed_override",
            "certification_workers",
            "allow_partial_source",
        )
        if any(
            context.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
            for name in construction_options
        ):
            raise click.UsageError("--project-from-author cannot change source, selection, or certification settings")
        if any(
            (
                lmdb_paths,
                lmdb_pattern,
                split_file,
                author_output_dir,
                source_report,
                reexecute_all,
                reexecute_max_step_count is not None,
            )
        ):
            raise click.UsageError("--project-from-author cannot be combined with source/export construction options")
        assert output_dir is not None
        projected = pyine.data.traces.eval_export_v2.project_statements(
            project_from_author,
            output_dir,
            recipe.visibility if recipe_file and recipe else None,
            parquet_batch_size,
            recipe.mixture if recipe_file and recipe else "recorded",
        )
        click.echo(f"Projected {sum(projected['partition_row_counts'].values())} rows to {output_dir.resolve()}")
        return
    if source_report and (reexecute_all or reexecute_max_step_count is not None):
        raise click.UsageError("source reports do not execute programs")
    if source_report and (output_dir is not None or author_output_dir is not None):
        raise click.UsageError("source reports write no artifacts; omit --output-dir and --author-output-dir")
    if export_mode == "statements-v2" and not source_report and author_output_dir is None:
        raise click.UsageError("statements-v2 requires --author-output-dir")
    pyine.utils.reprod.entrypoint_setup()
    resolved_lmdb_paths = _resolve_lmdb_paths(source_dataset_name, lmdb_paths, lmdb_pattern)
    resolved_split_file = split_file or pyine.data.utils.splits.get_dataset_split_file_path(
        source_dataset_name,
        must_exist=True,
    )
    config = pyine.data.traces.eval_export.EvalExportConfig(
        source_lmdb_paths=resolved_lmdb_paths,
        split_file_path=resolved_split_file,
        output_dir=output_dir or pathlib.Path("unused-source-report-output"),
        source_dataset_name=source_dataset_name,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        max_problem_count=max_problem_count,
        reexecute_max_step_count=reexecute_max_step_count,
        reexecute_all=reexecute_all,
        recheck_timeout_seconds=recheck_timeout_seconds,
        execution_seed_override=execution_seed_override,
        certification_workers=certification_workers,
        parquet_batch_size=parquet_batch_size,
        allow_partial_source=allow_partial_source,
    )
    if export_mode == "statements-v2":
        if source_report:
            report = pyine.data.traces.eval_export_v2.report_source(config, recipe, inspect_stored_events)
            click.echo(json.dumps(report, indent=2, sort_keys=True))
        else:
            assert author_output_dir is not None
            exported = pyine.data.traces.eval_export_v2.export_statements(config, author_output_dir, recipe)
            click.echo(
                f"Exported {sum(exported['partition_row_counts'].values())} rows to {config.output_dir.resolve()}"
            )
            click.echo(f"Author artifact: {author_output_dir.resolve()}")
        return
    assert output_dir is not None
    manifest = pyine.data.traces.eval_export.export_eval_traces(config)
    total_rows = sum(manifest.partition_row_counts.values())
    click.echo(f"Exported {total_rows} rows to {output_dir.resolve()}")
    click.echo(f"Problem counts: {manifest.partition_problem_counts}")
    click.echo(f"Integrity/re-execution counts: {manifest.certification_counts}")
    click.echo(f"Manifest: {(output_dir / 'export_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
