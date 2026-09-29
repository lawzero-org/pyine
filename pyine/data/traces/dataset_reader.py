"""
This module contains a reader for the PyINE dataset of 'raw' code execution traces.

See the corresponding dataset writer and utility modules for more information on this
dataset format. See also the demo notebook (in the project's root `notebooks` directory)
for an example of how to use this dataset reader.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import fnmatch
import logging
import pathlib
import typing

import msgspec
import pydantic
import torch.utils.data

import pyine.data.traces.dataset_utils
import pyine.data.utils.filter_rules
import pyine.data.utils.lmdb_io
import pyine.utils.code.execution
import pyine.utils.reprod

logger = logging.getLogger(__name__)


@typing.runtime_checkable
class DatasetProtocol(typing.Protocol):
    """Protocol describing the interface expected from trace datasets.

    This protocol defines the common interface that all trace dataset implementations must satisfy,
    enabling structural subtyping and type checking without requiring explicit inheritance. Both
    `DatasetReader` and `DatasetCollection` implement this protocol.
    """

    problem_keys: list[str]
    """List of unique problem identifiers in the dataset."""
    trace_keys: list[str]
    """List of unique trace identifiers in the dataset."""
    trace_key_to_problem_key: dict[str, str]
    """Mapping from trace identifiers to their parent problem identifiers."""
    augment_key_to_parent_trace_key: dict[str, str]
    """Mapping from augmented trace identifiers to their original (non-augmented) parent trace identifiers."""
    trace_metadata: list[pyine.data.traces.dataset_utils.TraceMetadata]
    """List of metadata objects for all traces, indexed by trace position."""

    @property
    def metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        ...

    @property
    def size_on_disk(self) -> int:
        """Returns the total size of the LMDB dataset stored on disk (in bytes)."""
        ...

    @property
    def hash(self) -> str:
        """Returns the hash of this dataset (computed from relevant files on disk)."""
        ...

    @property
    def parent_dataset_name(self) -> str:
        """Returns the name of the parent dataset used to create this traces dataset."""
        ...

    def __len__(self) -> int:
        """Returns the total number of traces in the dataset."""
        ...

    def __getitem__(
        self,
        index_or_key: int | str,
    ) -> pyine.utils.code.execution.TraceResult:
        """Fetches a trace by its index in this dataset (int) or by its identifier (str).

        Args:
            index_or_key: Zero-based index or unique trace identifier.

        Returns:
            The requested trace result containing execution data.
        """
        ...

    def get_problem_data(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches the problem data associated with a trace.

        Args:
            index_or_key: Index or identifier of the trace for which to retrieve problem data.

        Returns:
            The coding problem that generated this trace.
        """
        ...

    def get_trace_metadata(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.TraceMetadata:
        """Returns the metadata associated with a trace.

        Args:
            index_or_key: Index or identifier of the trace.

        Returns:
            Metadata containing trace statistics, tags, and identification info.
        """
        ...

    def get_tags(
        self,
        index_or_key: int | str,
    ) -> list[str]:
        """Returns the list of tags associated with a trace.

        Tags are used for filtering and categorization, combining problem tags, execution tags,
        and augmentation category tags.

        Args:
            index_or_key: Index or identifier of the trace.

        Returns:
            A copy of the tag list for the specified trace.
        """
        ...


DatasetOrDatasetPath = pathlib.Path | str | DatasetProtocol
"""Type used to represent a trace dataset or a path to a trace dataset."""
DatasetOrDatasetPathObjectOrArray = DatasetOrDatasetPath | typing.Sequence[DatasetOrDatasetPath]
"""Type used to represent a trace dataset or a path to a trace dataset or a list of such datasets/paths."""


class DatasetReader(torch.utils.data.Dataset[pyine.utils.code.execution.TraceResult]):
    """PyINE raw trace dataset reader.

    This reader provides execution traces stored in an LMDB dataset. Each trace corresponds to a
    successful code execution made for a specific solution to a coding problem, using a specific
    set of input arguments that are paired with an expected output value. These input/output pairs
    form a 'test', and the execution attempt is considered successful if the output value matches
    the expected value.

    Upon initialization, the reader attempts to load the prepared trace metadata from the dataset's
    LMDB directory. If the metadata is not found, it will prepare the metadata and save it there.

    Note: this reader does NOT attempt to structure the trace steps into deltas, so they will be
    quite verbose, likely too much so for most applications with reasoning models.

    Args:
        lmdb_path: Path to the LMDB database containing code traces.
    """

    def __init__(
        self,
        lmdb_path: pathlib.Path | str,
    ) -> None:
        super().__init__()
        self.path = pathlib.Path(lmdb_path)
        self.reader = pyine.data.utils.lmdb_io.LMDBReader(self.path)
        self._problem_indices: list[int] = []
        self.problem_keys: list[str] = []
        self._trace_indices: list[int] = []
        self.trace_keys: list[str] = []
        self._trace_idx_to_problem_idx: dict[int, int] = {}
        self.trace_key_to_problem_key: dict[str, str] = {}
        self._augment_idx_to_parent_trace_idx: dict[int, int] = {}
        self.augment_key_to_parent_trace_key: dict[str, str] = {}
        self.trace_metadata: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        self._problem_data_cache: collections.OrderedDict[
            int,
            pyine.data.traces.dataset_utils.CodingProblem,
        ] = collections.OrderedDict()
        self._problem_cache_max_size = 512
        self._metadata_cache: dict[str, typing.Any] | None = None
        self._size_on_disk_cache: int | None = None
        self._hash_cache: str | None = None
        self._parent_dataset_name_cache: str | None = None
        self._init_trace_metadata()

    def _is_metadata_prepared(self) -> bool:
        """Returns True if the trace maps and metadata are prepared and ready to be used."""
        return self._get_prepared_metadata_path().is_file()

    def _save_prepared_metadata(self) -> None:
        """Saves the prepared trace maps and metadata to the lmdb directory."""
        trace_maps_and_metadata = {
            "problem_indices": self._problem_indices,
            "problem_keys": self.problem_keys,
            "trace_indices": self._trace_indices,
            "trace_keys": self.trace_keys,
            "trace_idx_to_problem_idx": self._trace_idx_to_problem_idx,
            "trace_key_to_problem_key": self.trace_key_to_problem_key,
            "augment_idx_to_parent_trace_idx": self._augment_idx_to_parent_trace_idx,
            "augment_key_to_parent_trace_key": self.augment_key_to_parent_trace_key,
            "trace_metadata": self.trace_metadata,
        }
        encoded_data = msgspec.msgpack.encode(trace_maps_and_metadata)
        with open(self._get_prepared_metadata_path(), "wb") as fd:
            fd.write(encoded_data)

    def _load_prepared_metadata(self) -> None:
        """Loads the prepared trace maps and metadata from the lmdb directory."""
        with open(self._get_prepared_metadata_path(), "rb") as fd:
            decoded_obj = msgspec.msgpack.decode(fd.read())
        decoded_data = typing.cast("dict[str, typing.Any]", decoded_obj)
        self._problem_indices = list(typing.cast("list[int]", decoded_data["problem_indices"]))
        self.problem_keys = list(typing.cast("list[str]", decoded_data["problem_keys"]))
        self._trace_indices = list(typing.cast("list[int]", decoded_data["trace_indices"]))
        self.trace_keys = list(typing.cast("list[str]", decoded_data["trace_keys"]))
        self._trace_idx_to_problem_idx = dict(typing.cast("dict[int, int]", decoded_data["trace_idx_to_problem_idx"]))
        self.trace_key_to_problem_key = dict(typing.cast("dict[str, str]", decoded_data["trace_key_to_problem_key"]))
        self._augment_idx_to_parent_trace_idx = dict(
            typing.cast("dict[int, int]", decoded_data["augment_idx_to_parent_trace_idx"])
        )
        self.augment_key_to_parent_trace_key = dict(
            typing.cast("dict[str, str]", decoded_data["augment_key_to_parent_trace_key"])
        )
        metadata_payload = typing.cast("list[dict[str, typing.Any]]", decoded_data["trace_metadata"])
        exception_adapter: pydantic.TypeAdapter[pyine.utils.code.execution.TraceException | None] = (
            pydantic.TypeAdapter(pyine.utils.code.execution.TraceException | None)
        )
        for trace_meta in metadata_payload:
            # msgpack decodes the NamedTuple as a plain list, which never equals the stored record
            trace_meta["exception"] = exception_adapter.validate_python(trace_meta["exception"])
        self.trace_metadata = [
            pyine.data.traces.dataset_utils.TraceMetadata(**trace_meta) for trace_meta in metadata_payload
        ]

    def _clear_prepared_metadata(self) -> None:
        """Clears the prepared trace metadata from the lmdb directory."""
        if self._is_metadata_prepared():
            self._get_prepared_metadata_path().unlink()

    def _get_prepared_metadata_path(self) -> pathlib.Path:
        """Returns the file path used to store prepared trace metadata in the lmdb directory."""
        assert self.path.is_dir(), f"unexpected non-directory lmdb path: {self.path}"
        return self.path / "trace_metadata.msgspec"

    def _init_trace_metadata(self, force: bool = False) -> None:
        """Initializes the trace index maps and metadata potentially using already-cached data."""
        # note: the indices kept in the lists below are INTERNAL ones that map to the lmdb content
        if not force and self._is_metadata_prepared():
            self._load_prepared_metadata()
            return
        logger.debug(f"preparing trace metadata for lmdb dataset at: {self.path}")
        self._clear_prepared_metadata()
        problem_indices_result = self.reader.get_indices(
            pattern=pyine.data.traces.dataset_utils.PROBLEM_DATA_PATTERN,
            return_keys=True,
        )
        self._problem_indices, self.problem_keys = typing.cast(
            "tuple[list[int], list[str]]",
            problem_indices_result,
        )
        if len(self._problem_indices) == 0:
            raise ValueError("no problem data found in the dataset")
        if len(self._problem_indices) != len(self.problem_keys):
            raise RuntimeError("problem indices/keys length mismatch")
        self._trace_indices = []
        self.trace_keys = []
        self._trace_idx_to_problem_idx = {}
        self.trace_key_to_problem_key = {}
        self._augment_idx_to_parent_trace_idx = {}
        self.augment_key_to_parent_trace_key = {}
        self.trace_metadata = []
        for prob_iter_idx, (problem_idx, problem_key) in enumerate(
            zip(self._problem_indices, self.problem_keys, strict=False)
        ):
            if not problem_key.endswith(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX):
                raise ValueError(f"malformed problem key: {problem_key}")
            # fix problem key by removing the problem metadata suffix
            problem_key = problem_key[: -len(pyine.data.traces.dataset_utils.PROBLEM_DATA_SUFFIX)]
            self.problem_keys[prob_iter_idx] = problem_key
            problem_data = self.reader.get(problem_idx)
            problem_data = pyine.data.traces.dataset_utils.CodingProblem.model_validate(problem_data)
            curr_trace_data_pattern = problem_key + pyine.data.traces.dataset_utils.TRACE_DATA_SUFFIX
            curr_augm_trace_data_pattern = problem_key + pyine.data.traces.dataset_utils.AUGM_TRACE_DATA_SUFFIX
            trace_indices_result = self.reader.get_indices(
                pattern=curr_trace_data_pattern,
                return_keys=True,
            )
            curr_trace_indices, curr_trace_keys = typing.cast(
                "tuple[list[int], list[str]]",
                trace_indices_result,
            )
            assert len(curr_trace_indices) == len(curr_trace_keys)
            # for 'forward-compatibility' with deltas datasets, remove any elements with the deltas suffix
            filtered_pairs = [
                (idx, key)
                for idx, key in zip(curr_trace_indices, curr_trace_keys, strict=False)
                if not key.endswith(pyine.data.traces.dataset_utils.DELTAS_SUFFIX)
            ]
            if not filtered_pairs:
                raise ValueError(f"no trace data found for problem: {problem_key}")
            curr_trace_indices = [idx for idx, _ in filtered_pairs]
            curr_trace_keys = [key for _, key in filtered_pairs]
            self._trace_indices.extend(curr_trace_indices)
            self.trace_keys.extend(curr_trace_keys)
            curr_augm_key_to_parent_key: dict[str, str] = {}
            for trace_idx, trace_key in zip(curr_trace_indices, curr_trace_keys, strict=False):
                self._trace_idx_to_problem_idx[trace_idx] = problem_idx
                self.trace_key_to_problem_key[trace_key] = problem_key
                if fnmatch.fnmatch(trace_key, curr_augm_trace_data_pattern):
                    augm_trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_key)
                    parent_trace_id = augm_trace_id.get_augmentless_identifier()
                    curr_augm_key_to_parent_key[trace_key] = str(parent_trace_id)
                # note: we combine problem tags, trace (exec) tags, and augmentation tags into a single list
                trace_data = self.reader.get(trace_idx)
                trace_data = pyine.utils.code.execution.TraceResult.model_validate(trace_data)
                assert trace_data.identifier is not None, "trace identifier is required"
                assert trace_data.identifier == trace_key, "trace identifier mismatch"
                trace_id = pyine.data.traces.dataset_utils.TraceIdentifier.from_string(trace_data.identifier)
                curr_trace_tags: list[str] = []
                curr_trace_tags.extend(problem_data.problem_tags)
                curr_trace_tags.extend(trace_data.tags)
                if trace_id.is_augmented:
                    curr_trace_tags.extend({f"augment:{cat}" for cat in trace_id.split_augment_categories})
                self.trace_metadata.append(
                    pyine.data.traces.dataset_utils.TraceMetadata(
                        identifier=trace_data.identifier,
                        parent_dataset_hash=self.hash,
                        index=len(self.trace_metadata),  # increments as we append new traces to this list
                        internal_index=trace_idx,
                        step_count=trace_data.valid_step_count,
                        code_string=trace_data.code_string,
                        inputs=trace_data.inputs,
                        expected_output=trace_data.expected_output,
                        return_value=trace_data.return_value,
                        exception=trace_data.exception,
                        stdout=trace_data.stdout,
                        stderr=trace_data.stderr,
                        metadata=trace_data.metadata,
                        tags=curr_trace_tags,
                    )
                )
            for augm_key, parent_key in curr_augm_key_to_parent_key.items():
                if parent_key not in self.trace_keys:
                    raise KeyError(f"augmentation parent trace key {parent_key} not found in dataset")
                augm_idx = self._trace_indices[self.trace_keys.index(augm_key)]
                parent_idx = self._trace_indices[self.trace_keys.index(parent_key)]
                self._augment_idx_to_parent_trace_idx[augm_idx] = parent_idx
            self.augment_key_to_parent_trace_key.update(curr_augm_key_to_parent_key)
        self._save_prepared_metadata()

    def __len__(self) -> int:
        """Returns the total number of traces in the dataset accessible via ``__getitem__``"""
        return len(self.trace_keys)

    @property
    def metadata(self) -> dict[str, typing.Any]:
        """Returns a dictionary of all metadata stored in the database."""
        if self._metadata_cache is None:
            self._metadata_cache = self.reader.get_metadata()
        return self._metadata_cache

    @property
    def size_on_disk(self) -> int:
        """Returns the total size of the LMDB dataset stored on disk (in bytes)."""
        if self._size_on_disk_cache is None:
            self._size_on_disk_cache = self.reader.get_size_on_disk()
        return self._size_on_disk_cache

    @property
    def hash(self) -> str:
        """Returns the hash of this dataset (computed from relevant lmdb files on disk)."""
        if self._hash_cache is None:
            # note: we don't compute the hash over the entire lmdb dir, just over the file that matters
            # (that folder will likely contain other stuff such as processing logs and metadata caches)
            expected_data_mdb_file = self.reader.path / "data.mdb"
            assert expected_data_mdb_file.is_file(), f"missing data.mdb file: {expected_data_mdb_file}"
            self._hash_cache = pyine.utils.reprod.compute_hash(expected_data_mdb_file)
        return self._hash_cache

    @property
    def parent_dataset_name(self) -> str:
        """Returns the name of the parent dataset used to create this dataset."""
        if self._parent_dataset_name_cache is None:
            parent_info = self.metadata.get("parent_dataset", {})
            assert isinstance(parent_info, dict), "unexpected parent dataset metadata format"
            parent_info = typing.cast("dict[str, typing.Any]", parent_info)
            dataset_name = parent_info.get("dataset_name")
            assert isinstance(dataset_name, str), "missing parent dataset name in metadata"
            self._parent_dataset_name_cache = dataset_name
        return self._parent_dataset_name_cache

    def _resolve_trace_index_or_key(
        self,
        index_or_key: int | str,
    ) -> int:
        """Returns the dataset position for the provided user-facing index or key."""
        if isinstance(index_or_key, int):
            if 0 <= index_or_key < len(self):
                return index_or_key
            raise IndexError(f"index {index_or_key} out of range")
        return self._resolve_trace_key(index_or_key)

    def _resolve_trace_key(
        self,
        trace_key: str,
    ) -> int:
        """Returns the dataset position for the provided user-facing key."""
        if trace_key in self.trace_keys:
            return self.trace_keys.index(trace_key)
        raise KeyError(f"key {trace_key} not found in dataset")

    @typing.override
    def __getitem__(self, index_or_key: int | str) -> pyine.utils.code.execution.TraceResult:
        """Fetches an individual trace data object from the LMDB database by external index or key.

        Args:
            index_or_key: index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        return pyine.utils.code.execution.TraceResult.model_validate(self.reader.get(internal_trace_idx))

    def get_problem_data(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches the problem data associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve parent problem data.

        Returns:
            A `CodingProblem` object containing the problem data associated with the trace.
        """
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        internal_trace_idx = self._trace_indices[trace_idx]
        cached = self._problem_data_cache.get(internal_trace_idx)
        if cached is not None:
            self._problem_data_cache.move_to_end(internal_trace_idx)
            return cached
        problem_idx = self._trace_idx_to_problem_idx[internal_trace_idx]
        problem_data = pyine.data.traces.dataset_utils.CodingProblem.model_validate(self.reader.get(problem_idx))
        self._problem_data_cache[internal_trace_idx] = problem_data
        if len(self._problem_data_cache) > self._problem_cache_max_size:
            self._problem_data_cache.popitem(last=False)
        return problem_data

    def get_trace_metadata(self, index_or_key: int | str) -> pyine.data.traces.dataset_utils.TraceMetadata:
        """Returns the metadata associated with a trace by external index or key."""
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        return self.trace_metadata[trace_idx]

    def get_tags(self, index_or_key: int | str) -> list[str]:
        """Returns a list of tags for a given trace so that we can decide whether to filter it."""
        trace_idx = self._resolve_trace_index_or_key(index_or_key)
        return self.trace_metadata[trace_idx].tags.copy()

    def __str__(self) -> str:
        """Returns a string representation of the dataset reader (for debugging purposes)."""
        return f"{self.__class__.__name__}({self.path}) with {len(self)} instances"


class DatasetCollection(torch.utils.data.Dataset[pyine.utils.code.execution.TraceResult]):
    """Concatenates multiple trace datasets into a single reader-like view.

    Args:
        datasets: sequence of dataset paths or reader instances to combine. Each dataset must expose
            unique trace identifiers so that lookups remain unambiguous.
    """

    def __init__(
        self,
        datasets: typing.Sequence[DatasetOrDatasetPath],
    ) -> None:
        if not datasets:
            raise ValueError("at least one dataset must be provided")
        readers: list[DatasetReader] = []
        for item in datasets:
            if isinstance(item, DatasetReader):
                readers.append(item)
            else:
                if not isinstance(item, (str, pathlib.Path)):
                    raise ValueError("dataset collections can only encapsulate dataset paths or readers")
                readers.append(DatasetReader(item))
        self._readers = tuple(readers)
        self.paths: tuple[pathlib.Path, ...] = tuple(reader.path for reader in self._readers)
        self._component_hashes: tuple[str, ...] = tuple(reader.hash for reader in self._readers)
        self._combined_hash = pyine.utils.reprod.get_params_hash(
            "trace_dataset_collection",
            self._component_hashes,
        )
        self._component_info: list[dict[str, typing.Any]] = []
        self.problem_keys: list[str] = []
        self.trace_keys: list[str] = []
        self.trace_key_to_problem_key: dict[str, str] = {}
        self.augment_key_to_parent_trace_key: dict[str, str] = {}
        self.trace_metadata: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
        self._trace_key_to_index: dict[str, int] = {}
        self._reader_boundaries: list[int] = [0]
        parent_names: set[str] = set()
        cumulative_trace_count = 0
        for reader in self._readers:
            parent_meta: dict[str, typing.Any] = {}
            reader_metadata = getattr(reader, "metadata", {})
            if isinstance(reader_metadata, dict):
                reader_metadata_dict = typing.cast("dict[str, typing.Any]", reader_metadata)
                raw_parent_obj = reader_metadata_dict.get("parent_dataset", {})
                if isinstance(raw_parent_obj, dict):
                    parent_meta = dict(typing.cast("dict[str, typing.Any]", raw_parent_obj))
            parent_name_value = typing.cast("str | None", parent_meta.get("dataset_name"))
            parent_name = parent_name_value or reader.parent_dataset_name
            parent_names.add(parent_name)
            self._component_info.append(
                {
                    "parent_dataset": parent_meta,
                    "hash": reader.hash,
                    "path": str(getattr(reader, "path", "")),
                    "trace_count": len(reader),
                    "fallback_name": reader.parent_dataset_name,
                }
            )
            self.problem_keys.extend(reader.problem_keys)
            self.augment_key_to_parent_trace_key.update(reader.augment_key_to_parent_trace_key)
            for local_idx, trace_key in enumerate(reader.trace_keys):
                if trace_key in self.trace_key_to_problem_key:
                    raise ValueError(f"duplicate trace key across dataset parts: {trace_key}")
                global_idx = cumulative_trace_count + local_idx
                self.trace_keys.append(trace_key)
                self.trace_key_to_problem_key[trace_key] = reader.trace_key_to_problem_key[trace_key]
                self._trace_key_to_index[trace_key] = global_idx
                local_meta = reader.get_trace_metadata(local_idx)
                metadata_payload = dict(local_meta.metadata)
                metadata_payload.setdefault("source_dataset_hash", local_meta.parent_dataset_hash)
                promoted_meta = dataclasses.replace(
                    local_meta,
                    index=global_idx,
                    parent_dataset_hash=self._combined_hash,
                    metadata=metadata_payload,
                )
                self.trace_metadata.append(promoted_meta)
            cumulative_trace_count += len(reader)
            self._reader_boundaries.append(cumulative_trace_count)
        self._length = cumulative_trace_count
        self._size_on_disk = sum(reader.size_on_disk for reader in self._readers)
        self._parent_dataset_name = ",".join(sorted(parent_names)) if parent_names else ""
        self._metadata_cache = self._build_metadata()

    @property
    def component_readers(self) -> tuple[DatasetReader, ...]:
        """Returns the underlying dataset readers."""
        return self._readers

    @property
    def component_hashes(self) -> tuple[str, ...]:
        """Returns the hashes for each component dataset."""
        return self._component_hashes

    @property
    def metadata(self) -> dict[str, typing.Any]:
        """Returns aggregated metadata for the combined dataset."""
        return self._metadata_cache

    def _build_metadata(self) -> dict[str, typing.Any]:
        component_entries: list[dict[str, typing.Any]] = []
        dataset_names: list[str] = []
        dataset_paths: list[str] = []
        for info in self._component_info:
            parent = typing.cast("dict[str, typing.Any]", info.get("parent_dataset", {}))
            name = parent.get("dataset_name")
            path_value = parent.get("dataset_path")
            if isinstance(name, str):
                dataset_names.append(name)
            if isinstance(path_value, (str, pathlib.Path)):
                dataset_paths.append(str(path_value))
            component_entries.append(
                {
                    "dataset_name": name if isinstance(name, str) else info["fallback_name"],
                    "dataset_path": (
                        str(path_value) if isinstance(path_value, (str, pathlib.Path)) else info.get("path")
                    ),
                    "hash": info["hash"],
                    "path": info.get("path"),
                    "trace_count": info["trace_count"],
                }
            )
        fallback_names = {info["fallback_name"] for info in self._component_info}
        combined_name = ",".join(sorted(set(dataset_names) if dataset_names else fallback_names))
        unique_paths = sorted({p for p in dataset_paths if p})
        if not unique_paths:
            dataset_path_value: str | list[str] | None = component_entries[0]["path"] if component_entries else None
        elif len(unique_paths) == 1:
            dataset_path_value = unique_paths[0]
        else:
            dataset_path_value = unique_paths
        return {
            "parent_dataset": {
                "dataset_name": combined_name,
                "dataset_path": dataset_path_value,
                "trace_count": len(self),
                "components": component_entries,
            }
        }

    @property
    def parent_dataset_name(self) -> str:
        """Returns a descriptive parent dataset name for the combined dataset."""
        if self._parent_dataset_name:
            return self._parent_dataset_name
        names = {
            parent_name
            for info in self._component_info
            for parent_name in [info.get("parent_dataset", {}).get("dataset_name")]
            if isinstance(parent_name, str)
        }
        if not names:
            names = {info["fallback_name"] for info in self._component_info}
        return ",".join(sorted(names))

    @property
    def size_on_disk(self) -> int:
        """Returns the total disk footprint across all component datasets."""
        return self._size_on_disk

    @property
    def hash(self) -> str:
        """Returns the combined dataset hash."""
        return self._combined_hash

    def __len__(self) -> int:
        """Returns the total number of traces across all component datasets."""
        return self._length

    def _resolve_index(
        self,
        index_or_key: int | str,
    ) -> int:
        """Resolves a user-facing index or key to a global dataset position."""
        if isinstance(index_or_key, int):
            if 0 <= index_or_key < self._length:
                return index_or_key
            raise IndexError(f"index {index_or_key} out of range")
        key = index_or_key  # at this point, the input should be a str
        if key in self._trace_key_to_index:
            return self._trace_key_to_index[key]
        raise KeyError(f"key {key} not found in dataset collection")

    def _locate_reader(
        self,
        resolved_index: int,
    ) -> tuple[DatasetReader, int]:
        """Locates the component reader and local index for a given global index."""
        reader_pos = bisect.bisect_right(self._reader_boundaries, resolved_index) - 1
        if reader_pos < 0 or reader_pos >= len(self._readers):
            raise IndexError(f"index {resolved_index} out of range")
        reader = self._readers[reader_pos]
        start = self._reader_boundaries[reader_pos]
        return reader, resolved_index - start

    @typing.override
    def __getitem__(
        self,
        index_or_key: int | str,
    ) -> pyine.utils.code.execution.TraceResult:
        """Fetches an individual trace data object by external index or key.

        Args:
            index_or_key: Index or key of the trace to retrieve.

        Returns:
            A `TraceResult` object containing the requested trace data.
        """
        resolved_index = self._resolve_index(index_or_key)
        reader, local_idx = self._locate_reader(resolved_index)
        return reader[local_idx]

    def get_problem_data(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.CodingProblem:
        """Fetches the problem data associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace for which to retrieve parent problem data.

        Returns:
            A `CodingProblem` object containing the problem data associated with the trace.
        """
        resolved_index = self._resolve_index(index_or_key)
        reader, local_idx = self._locate_reader(resolved_index)
        return reader.get_problem_data(local_idx)

    def get_trace_metadata(
        self,
        index_or_key: int | str,
    ) -> pyine.data.traces.dataset_utils.TraceMetadata:
        """Returns the metadata associated with a trace by external index or key.

        Args:
            index_or_key: Index or key of the trace.

        Returns:
            A `TraceMetadata` object containing metadata for the specified trace.
        """
        resolved_index = self._resolve_index(index_or_key)
        return self.trace_metadata[resolved_index]

    def get_tags(
        self,
        index_or_key: int | str,
    ) -> list[str]:
        """Returns a list of tags for a given trace by external index or key.

        Args:
            index_or_key: Index or key of the trace.

        Returns:
            A copy of the tag list for the specified trace.
        """
        resolved_index = self._resolve_index(index_or_key)
        return self.trace_metadata[resolved_index].tags.copy()

    def __str__(self) -> str:
        """Returns a string representation for debugging purposes."""
        paths_repr = ", ".join(str(path) for path in self.paths)
        return f"{self.__class__.__name__}([{paths_repr}]) with {len(self)} instances"


def get_traces_metadata(
    datasets: list[DatasetOrDatasetPath] | DatasetOrDatasetPath,
    base_filter: pyine.data.utils.filter_rules.FilterType | None = None,
) -> list[pyine.data.traces.dataset_utils.TraceMetadata]:
    """Returns a list of TraceMetadata objects for all traces in the provided dataset(s).

    Args:
        datasets: the path or dataset object (or list of) from which to get trace metadata.
        base_filter: filter to apply to the dataset reader(s) to get target traces. If None, then
            no filter will be applied, and metadata for all available traces will be returned.

    Returns:
        A list of TraceMetadata objects for all targeted traces in the provided dataset.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
    readers: list[DatasetProtocol] = []
    for dataset in datasets:
        if isinstance(dataset, DatasetProtocol):
            readers.append(dataset)
        else:
            readers.append(DatasetReader(dataset))
    logger.info(f"preparing traces metadata for {len(readers)} dataset reader(s)...")
    all_trace_keys: list[str] = []
    for reader in readers:
        all_trace_keys.extend(reader.trace_keys)
    if len(set(all_trace_keys)) != len(all_trace_keys):
        raise ValueError("there should be no duplicates in the list of trace keys across all datasets")
    base_traces_meta: list[pyine.data.traces.dataset_utils.TraceMetadata] = []
    for reader in readers:
        for trace_idx in range(len(reader)):
            tags = reader.get_tags(trace_idx)
            is_banned = base_filter(tags) if base_filter is not None else False
            if not is_banned:
                trace_metadata = reader.get_trace_metadata(trace_idx)
                assert reader.trace_keys[trace_idx] == trace_metadata.identifier
                assert trace_idx == trace_metadata.index
                assert reader.hash == trace_metadata.parent_dataset_hash
                base_traces_meta.append(trace_metadata)
    return base_traces_meta
