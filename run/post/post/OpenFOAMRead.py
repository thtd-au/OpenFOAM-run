from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
import os
import pickle
import re

import numpy as np



def _time_key(path: Path):
    try:
        return float(path.name)
    except ValueError:
        return None


def _log(msg, verbose=True):
    if verbose:
        print(f"[OpenFOAMRead] {msg}")


def _read_internal_field(file_path):
    text = Path(file_path).read_text()

    m = re.search(
        r"internalField\s+nonuniform\s+List<(\w+)>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if not m:
        m = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if not m:
            raise ValueError(f"Could not parse internalField in {file_path}")
        return m.group(1).strip()

    value_type = m.group(1)
    n = int(m.group(2))
    body = m.group(3).strip()

    if value_type == "scalar":
        arr = np.fromstring(body, sep=" ")
        if arr.size != n:
            raise ValueError(f"Expected {n} values in {file_path}, got {arr.size}")
        return arr

    if value_type == "vector":
        rows = re.findall(r"\(([^()]+)\)", body)
        arr = np.array([[float(v) for v in row.split()] for row in rows])
        if arr.shape != (n, 3):
            raise ValueError(f"Expected {(n, 3)} vector values in {file_path}, got {arr.shape}")
        return arr

    raise NotImplementedError(f"Unsupported OpenFOAM field type: {value_type}")


_SUPPORTED_VOLUME_CLASSES = {"volScalarField", "volVectorField"}


def _field_class_from_text(text):
    m = re.search(r"class\s+(\w+)\s*;", text)
    return m.group(1) if m else None


def _has_supported_internal_field(text):
    """Return True for scalar/vector internalField formats this reader supports."""
    return bool(
        re.search(r"internalField\s+uniform\s+[^;]+;", text)
        or re.search(
            r"internalField\s+nonuniform\s+List<(?:scalar|vector)>\s+\d+\s*\(",
            text,
            re.S,
        )
    )


def _is_supported_volume_field_file(file_path):
    """
    True only for OpenFOAM volume scalar/vector fields this reader can reshape.

    This deliberately excludes files such as cellToRegion, phi/surfaceScalarField,
    and other auxiliary timestep files that may appear beside regular fields.
    """
    try:
        text = Path(file_path).read_text(errors="ignore")
    except OSError:
        return False

    return (
        _field_class_from_text(text) in _SUPPORTED_VOLUME_CLASSES
        and _has_supported_internal_field(text)
    )


def _load_cell_centres(case_dir):
    zero = Path(case_dir) / "0"

    c_file = zero / "C"
    if c_file.exists():
        return _read_internal_field(c_file)

    cx = _read_internal_field(zero / "Cx")
    cy = _read_internal_field(zero / "Cy")
    cz = _read_internal_field(zero / "Cz")

    return np.column_stack([cx, cy, cz])


def _structured_index(C, decimals=12):
    x = np.round(C[:, 0], decimals)
    y = np.round(C[:, 1], decimals)
    z = np.round(C[:, 2], decimals)

    xs = np.unique(x)
    ys = np.unique(y)
    zs = np.unique(z)

    nx, ny, nz = len(xs), len(ys), len(zs)

    if nx * ny * nz != len(C):
        raise ValueError(
            "Cell centres do not form a complete structured grid. "
            "For unstructured meshes, interpolation is needed before reshaping."
        )

    ix = np.searchsorted(xs, x)
    iy = np.searchsorted(ys, y)
    iz = np.searchsorted(zs, z)

    return ix, iy, iz, nx, ny, nz, xs, ys, zs


def _n_cells(grid):
    return grid[3] * grid[4] * grid[5]


def _expand_uniform(values, grid):
    if not isinstance(values, str):
        return values

    n_cells = _n_cells(grid)
    values = values.strip()

    if values.startswith("(") and values.endswith(")"):
        vec = np.array([float(v) for v in values[1:-1].split()])
        return np.tile(vec, (n_cells, 1))

    return np.full(n_cells, float(values))


def _reshape_to_planes(values, grid, plane="xy"):
    values = _expand_uniform(values, grid)
    ix, iy, iz, nx, ny, nz, *_ = grid

    extra_shape = values.shape[1:] if values.ndim == 2 else ()
    arr3d = np.empty((nz, ny, nx) + extra_shape)
    arr3d[iz, iy, ix] = values

    if plane == "xy":
        return arr3d

    if plane == "xz":
        return np.transpose(arr3d, (1, 0, 2) + tuple(range(3, arr3d.ndim)))

    if plane == "yz":
        return np.transpose(arr3d, (2, 0, 1) + tuple(range(3, arr3d.ndim)))

    raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")


def _strip_foam_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def _read_scalar_dict_block(text, block_name):
    text = _strip_foam_comments(text)
    m = re.search(rf"{block_name}\s*\{{(.*?)\}}", text, re.S)
    if not m:
        return {}

    return {
        name: float(value)
        for name, value in re.findall(r"(\w+)\s+([-+0-9.eE]+)\s*;", m.group(1))
    }


def _read_species_list(text):
    text = _strip_foam_comments(text)
    m = re.search(r"species\s*\((.*?)\)\s*;", text, re.S)
    return re.findall(r"\b\w+\b", m.group(1)) if m else []


def _read_single_scalar(text, name):
    text = _strip_foam_comments(text)
    m = re.search(rf"{name}\s+(?:\[[^\]]+\]\s+)?([-+0-9.eE]+)\s*;", text)
    return float(m.group(1)) if m else None


def _load_reaction_properties(case_dir):
    text = (Path(case_dir) / "constant" / "reactions").read_text()
    return {
        "species": _read_species_list(text),
        "Mi": _read_scalar_dict_block(text, "molarMass"),
        "Hi": _read_scalar_dict_block(text, "henryConstant"),
        "Pref": _read_single_scalar(text, "absolutePressureReference"),
    }


def _load_transport_properties(case_dir):
    text = (Path(case_dir) / "constant" / "transportProperties").read_text()
    return {"rho": _read_single_scalar(text, "rho")}


def _load_properties(case_dir):
    props = {}
    props.update(_load_reaction_properties(case_dir))
    props.update(_load_transport_properties(case_dir))
    return props



def _discover_time_dirs_and_fields(result_dir):
    """
    Find numeric timestep folders and supported volume fields.

    Only files with class volScalarField/volVectorField and a supported
    scalar/vector internalField are returned as available fields. Unsupported
    timestep files are reported separately so fields=["all"] does not try to
    parse auxiliary files such as cellToRegion or phi.
    """
    time_dirs = []
    available_fields = set()
    skipped_fields = set()

    for folder in Path(result_dir).iterdir():
        if not folder.is_dir():
            continue

        timestep = _time_key(folder)
        if timestep is None:
            continue

        time_dirs.append((timestep, folder))

        for field_path in folder.iterdir():
            if not field_path.is_file():
                continue

            if _is_supported_volume_field_file(field_path):
                available_fields.add(field_path.name)
            else:
                skipped_fields.add(field_path.name)

    time_dirs.sort(key=lambda item: item[0])
    return time_dirs, available_fields, skipped_fields



def _discover_field_inventory(time_dirs):
    """
    Build a per-timestep inventory of supported volume fields.

    This uses the same support check as _discover_time_dirs_and_fields, so it is
    the correct source for "what fields exist" instead of inspecting timestep 0.
    """
    fields_by_timestep = {}

    for timestep, folder in time_dirs:
        fields = []
        for field_path in folder.iterdir():
            if field_path.is_file() and _is_supported_volume_field_file(field_path):
                fields.append(field_path.name)
        fields_by_timestep[timestep] = sorted(fields)

    return fields_by_timestep



def _unique_keep_order(items):
    return list(dict.fromkeys(items))


def _as_field_list(fields):
    if isinstance(fields, str):
        return [fields]
    return list(fields)


def _canonical_species_field(field):
    """
    Canonicalize species-field spelling while preserving ordinary field names.

    Examples:
      C_H2  -> c_H2
      c_H2  -> c_H2
      y_H2  -> Y_H2
      peq_H2 / PEQ_H2 -> pEq_H2
      S_H2  -> S_H2
    """
    if not isinstance(field, str) or "_" not in field:
        return field

    prefix, suffix = field.split("_", 1)
    prefix_l = prefix.lower()

    if prefix_l == "y":
        return f"Y_{suffix}"
    if prefix_l == "c":
        return f"c_{suffix}"
    if prefix_l == "s":
        return f"S_{suffix}"
    if prefix_l == "peq":
        return f"pEq_{suffix}"

    return field


def _species_family_from_all_token(field):
    """
    Return the species-family prefix for tokens such as Y_All, Y_all, C_All,
    c_all, S_All, pEq_All. Return None for non-family tokens.
    """
    if not isinstance(field, str) or "_" not in field:
        return None

    prefix, suffix = field.rsplit("_", 1)
    if suffix.lower() != "all":
        return None

    prefix_l = prefix.lower()
    if prefix_l == "y":
        return "Y"
    if prefix_l == "c":
        return "c"
    if prefix_l == "s":
        return "S"
    if prefix_l == "peq":
        return "pEq"

    return None


def _expand_species_family(family, species):
    if family == "Y":
        return [f"Y_{sp}" for sp in species]
    if family == "c":
        return [f"c_{sp}" for sp in species]
    if family == "S":
        return [f"S_{sp}" for sp in species] + ["S_tot"]
    if family == "pEq":
        return [f"pEq_{sp}" for sp in species] + ["pEq_tot"]
    raise ValueError(f"Unsupported species field family: {family}")


def _resolve_requested_fields(fields, available_fields, props):
    """
    Resolve field requests using the common hierarchy:
      1. "all" / "All" / ["all"] -> all supported fields available on disk.
      2. "Y_All", "Y_all", "C_All", "c_All", "S_All", "pEq_All" ->
         all species fields for that family. S and pEq include *_tot.
      3. Explicit names such as "U", "p", "Y_H2", "S_tot" pass through.

    This reader is raw-only, so resolved fields that are not present on disk are
    reported and skipped rather than derived.
    """
    requested = _as_field_list(fields)

    resolved = []
    wants_all = any(isinstance(field, str) and field.lower() == "all" for field in requested)
    if wants_all:
        resolved.extend(sorted(available_fields))

    for field in requested:
        if isinstance(field, str) and field.lower() == "all":
            continue

        family = _species_family_from_all_token(field)
        if family is not None:
            resolved.extend(_expand_species_family(family, props["species"]))
        else:
            resolved.append(_canonical_species_field(field))

    return _unique_keep_order(resolved)



def _required_raw_fields(fields):
    """Fields to attempt loading from each timestep.

    This reader is intentionally raw-only: it loads fields that already exist
    on disk and does not derive missing fields. Derived quantities should be
    added later with OpenfoamPostprocess.py.
    """
    return set(fields)



def _read_raw_fields(folder, required_raw, grid):
    raw_fields = {}
    missing_fields = []
    unsupported_fields = []

    for field in required_raw:
        field_path = folder / field

        if not field_path.exists():
            missing_fields.append(field)
            continue

        if not _is_supported_volume_field_file(field_path):
            unsupported_fields.append(field)
            continue

        values = _read_internal_field(field_path)
        raw_fields[field] = _expand_uniform(values, grid)

    return raw_fields, missing_fields, unsupported_fields


def _process_timestep(args):
    timestep, folder, fields, required_raw, grid, plane = args

    raw_fields, missing_fields, unsupported_fields = _read_raw_fields(folder, required_raw, grid)
    timestep_data = {}

    for field in fields:
        if field not in raw_fields:
            continue

        timestep_data[field] = _reshape_to_planes(raw_fields[field], grid, plane=plane)

    return timestep, timestep_data, sorted(raw_fields), sorted(missing_fields), sorted(unsupported_fields)



def _progress_marks(n_items):
    return {
        max(1, int(0.25 * n_items)),
        max(1, int(0.50 * n_items)),
        max(1, int(0.75 * n_items)),
        n_items,
    }


def _record_timestep_result(data, timestep, timestep_data, completed, n_tasks, marks, verbose):
    data[timestep] = timestep_data

    if completed in marks:
        _log(
            f"Processed {completed}/{n_tasks} timesteps "
            f"({int(100 * completed / n_tasks)}%)",
            verbose,
        )



def _write_pickle(data, output_location, output_filename, verbose=True):
    output_location = Path(output_location)
    output_location.mkdir(parents=True, exist_ok=True)
    out = output_location / output_filename

    _log(f"Writing output file: {out}", verbose)
    with open(out, "wb") as f:
        pickle.dump(data, f)



def _executor_from_backend(parallel_backend):
    backend = parallel_backend.lower()
    if backend == "process":
        return ProcessPoolExecutor
    if backend == "thread":
        return ThreadPoolExecutor
    raise ValueError("parallelBackend must be either 'process' or 'thread'")


def read_openfoam_results(
    result_dir,
    fields,
    output_location=None,
    output_filename=None,
    plane="xy",
    decimals=12,
    deriveMissingFields=False,
    parallel=False,
    nPartitions=None,
    parallelBackend="process",
    verbose=True,
):
    """
    Read OpenFOAM ASCII timestep fields into:
        result["timesteps"][timestep][field] = stacked 2D planes

    The returned object also contains:
        result["coords"]["x"], result["coords"]["y"], result["coords"]["z"]
        result["meta"]
        result["fields"]

    Parameters
    ----------
    result_dir : str or Path
        OpenFOAM case directory containing numeric timestep folders.
    fields : str or list[str]
        Requested fields, e.g. ["U", "p", "Y_H2", "S_tot", "pEq_H2"].
        Special values include "all", "Y_All"/"Y_all", "C_All"/"c_All",
        "S_All", and "pEq_All". "S_All" and "pEq_All" include their
        corresponding total fields. "all" means every supported volume
        scalar/vector field found on disk.
        It skips auxiliary files such as cellToRegion and surface fields such as phi.
    output_location : str or Path, optional
        Directory where pickle output is written.
    output_filename : str, optional
        Pickle filename.
    plane : {"xy", "xz", "yz"}
        2D plane orientation. For plane="xy", output shape is (nz, ny, nx).
    decimals : int
        Coordinate rounding used to identify structured-grid indices.
    deriveMissingFields : bool
        Deprecated and ignored. This reader is raw-only; derive fields later
        using openfoam_postprocess.py.
    parallel : bool
        If True, process timestep folders concurrently.
    nPartitions : int, optional
        Number of parallel workers. Defaults to min(os.cpu_count(), n_timesteps).
    parallelBackend : {"process", "thread"}
        Parallel executor backend used when parallel=True.
    verbose : bool
        Print overview-level runtime messages.

    Notes
    -----
    Existing fields are read from disk only. No derived fields are computed here.
    Use openfoam_postprocess.py for c_*, S_*, S_tot, pEq_*, and pEq_tot.
    """
    result_dir = Path(result_dir)
    props = _load_properties(result_dir)

    _log(f"Reading case: {result_dir}", verbose)

    C = _load_cell_centres(result_dir)
    grid = _structured_index(C, decimals=decimals)
    _log(
        f"Structured grid detected: {grid[3]} x {grid[4]} x {grid[5]} "
        f"({_n_cells(grid):,} cells)",
        verbose,
    )

    time_dirs, available_fields, skipped_fields = _discover_time_dirs_and_fields(result_dir)
    if not time_dirs:
        raise ValueError(f"No timestep folders found in {result_dir}")

    fields_by_timestep = _discover_field_inventory(time_dirs)

    if deriveMissingFields:
        _log(
            "deriveMissingFields is deprecated and ignored. "
            "Use openfoam_postprocess.py to derive fields after reading raw data.",
            verbose,
        )

    fields = _resolve_requested_fields(fields, available_fields, props)

    requested_fields = set(fields)
    missing_requested = requested_fields - available_fields
    required_raw = _required_raw_fields(fields)

    _log("Found supported volume fields: " + ", ".join(sorted(available_fields)), verbose)
    if skipped_fields:
        _log(
            "Skipped unsupported timestep files: " + ", ".join(sorted(skipped_fields)),
            verbose,
        )
    _log("Requested fields: " + ", ".join(sorted(requested_fields)), verbose)
    _log(
        f"Found {len(time_dirs)} timesteps ({time_dirs[0][0]} -> {time_dirs[-1][0]})",
        verbose,
    )

    if missing_requested:
        _log(
            "Requested fields not found on disk and skipped: "
            + ", ".join(sorted(missing_requested)),
            verbose,
        )

    tasks = [
        (timestep, folder, fields, required_raw, grid, plane)
        for timestep, folder in time_dirs
    ]

    data = {}
    loaded_fields_by_timestep = {}
    missing_requested_by_timestep = {}
    unsupported_requested_by_timestep = {}
    marks = _progress_marks(len(tasks))

    if parallel:
        workers = nPartitions or min(os.cpu_count() or 1, len(tasks))
        workers = max(1, min(int(workers), len(tasks)))
        Executor = _executor_from_backend(parallelBackend)

        _log(f"Parallel timestep processing enabled with {workers} partitions", verbose)
        _log(f"Parallel backend: {parallelBackend.lower()}", verbose)

        with Executor(max_workers=workers) as executor:
            futures = [executor.submit(_process_timestep, task) for task in tasks]

            for completed, future in enumerate(as_completed(futures), start=1):
                timestep, timestep_data, loaded, missing, unsupported = future.result()
                loaded_fields_by_timestep[timestep] = loaded
                missing_requested_by_timestep[timestep] = missing
                unsupported_requested_by_timestep[timestep] = unsupported
                _record_timestep_result(
                    data,
                    timestep,
                    timestep_data,
                    completed,
                    len(tasks),
                    marks,
                    verbose,
                )
    else:
        for completed, task in enumerate(tasks, start=1):
            timestep, timestep_data, loaded, missing, unsupported = _process_timestep(task)
            loaded_fields_by_timestep[timestep] = loaded
            missing_requested_by_timestep[timestep] = missing
            unsupported_requested_by_timestep[timestep] = unsupported
            _record_timestep_result(
                data,
                timestep,
                timestep_data,
                completed,
                len(tasks),
                marks,
                verbose,
            )

    data = dict(sorted(data.items(), key=lambda item: item[0]))
    loaded_fields_by_timestep = dict(sorted(loaded_fields_by_timestep.items(), key=lambda item: item[0]))
    missing_requested_by_timestep = dict(sorted(missing_requested_by_timestep.items(), key=lambda item: item[0]))
    unsupported_requested_by_timestep = dict(sorted(unsupported_requested_by_timestep.items(), key=lambda item: item[0]))
    fields_by_timestep = dict(sorted(fields_by_timestep.items(), key=lambda item: item[0]))

    coords = {
        # Unique structured-grid coordinate axes.
        # For plane="xy", field arrays are shaped as (nz, ny, nx, ...),
        # corresponding to coords["z"], coords["y"], coords["x"].
        "x": grid[6],
        "y": grid[7],
        "z": grid[8],
    }

    result = {
        "fields": {
            # Global field inventory based on all numeric timestep folders.
            # Use this instead of list(result["timesteps"][0].keys()) when timestep 0
            # does not contain fields created later in the simulation.
            "available": sorted(available_fields),
            "requested": sorted(requested_fields),
            "skipped_unsupported": sorted(skipped_fields),
            "available_by_timestep": fields_by_timestep,
            "loaded_by_timestep": loaded_fields_by_timestep,
            "missing_requested_by_timestep": missing_requested_by_timestep,
            "unsupported_requested_by_timestep": unsupported_requested_by_timestep,
        },
        "meta": {
            "species": props["species"],
            "constants": {
                "Mi": props["Mi"],
                "Hi": props["Hi"],
                "rho": props["rho"],
                "Pref": props["Pref"],
            },
            "grid": {
                "nx": grid[3],
                "ny": grid[4],
                "nz": grid[5],
                "plane": plane,
            },
        },
        "coords": coords,
        "timesteps": data,
    }

    if output_location and output_filename:
        _write_pickle(result, output_location, output_filename, verbose=verbose)

    _log("Done.", verbose)
    return result
