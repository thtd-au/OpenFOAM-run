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


DERIVABLE_FIELDS = ("c_*", "S_*", "S_tot", "pEq_*", "pEq_tot")


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

    cls = re.search(r"class\s+(\w+);", text)
    field_class = cls.group(1) if cls else ""

    m = re.search(
        r"internalField\s+nonuniform\s+List<(\w+)>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.S,
    )

    if not m:
        m = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if not m:
            raise ValueError(f"Could not parse internalField in {file_path}")
        return m.group(1).strip(), field_class

    value_type = m.group(1)
    n = int(m.group(2))
    body = m.group(3).strip()

    if value_type == "scalar":
        arr = np.fromstring(body, sep=" ")
        if arr.size != n:
            raise ValueError(f"Expected {n} values in {file_path}, got {arr.size}")
        return arr, field_class

    if value_type == "vector":
        rows = re.findall(r"\(([^()]+)\)", body)
        arr = np.array([[float(v) for v in row.split()] for row in rows])
        if arr.shape != (n, 3):
            raise ValueError(f"Expected {(n, 3)} vector values in {file_path}, got {arr.shape}")
        return arr, field_class

    raise NotImplementedError(f"Unsupported OpenFOAM field type: {value_type}")


def _load_cell_centres(case_dir):
    zero = Path(case_dir) / "0"

    c_file = zero / "C"
    if c_file.exists():
        C, _ = _read_internal_field(c_file)
        return C

    cx, _ = _read_internal_field(zero / "Cx")
    cy, _ = _read_internal_field(zero / "Cy")
    cz, _ = _read_internal_field(zero / "Cz")

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


def _derive_missing_field(field, raw_fields, props):
    rho = props["rho"]
    Mi = props["Mi"]
    Hi = props["Hi"]
    Pref = props["Pref"]

    if field.startswith("c_"):
        sp = field[2:]
        Y = raw_fields.get(f"Y_{sp}")
        if Y is None:
            raise ValueError(f"Cannot derive {field}: missing Y_{sp}")
        return rho * Y / Mi[sp]

    if field.startswith("S_") and field != "S_tot":
        p = raw_fields.get("p")
        if p is None:
            raise ValueError(f"Cannot derive {field}: missing p field")

        P_liquid = Pref + rho * p

        sp = field[2:]
        c = _derive_missing_field(f"c_{sp}", raw_fields, props)
        return c / (Hi[sp] * P_liquid)

    if field == "S_tot":
        p = raw_fields.get("p")
        if p is None:
            raise ValueError(f"Cannot derive {field}: missing p field")

        terms = [
            _derive_missing_field(f"S_{sp}", raw_fields, props)
            for sp in props["species"]
            if sp in Hi and f"Y_{sp}" in raw_fields
        ]
        if not terms:
            raise ValueError("Cannot derive S_tot: no gas species with Y_* and Henry constants found")
        return np.sum(terms, axis=0)

    if field.startswith("pEq_") and field != "pEq_tot":
        sp = field[4:]
        c = _derive_missing_field(f"c_{sp}", raw_fields, props)
        return c / Hi[sp]

    if field == "pEq_tot":
        terms = [
            _derive_missing_field(f"pEq_{sp}", raw_fields, props)
            for sp in props["species"]
            if sp in Hi and f"Y_{sp}" in raw_fields
        ]
        if not terms:
            raise ValueError("Cannot derive pEq_tot: no gas species with Y_* and Henry constants found")
        return np.sum(terms, axis=0)

    raise ValueError(f"No derivation rule available for missing field: {field}")


def _discover_time_dirs_and_fields(result_dir):
    time_dirs = []
    available_fields = set()

    for folder in Path(result_dir).iterdir():
        if not folder.is_dir():
            continue

        timestep = _time_key(folder)
        if timestep is None:
            continue

        time_dirs.append((timestep, folder))
        available_fields.update(f.name for f in folder.iterdir() if f.is_file())

    time_dirs.sort(key=lambda item: item[0])
    return time_dirs, available_fields

def _expand_all_species_fields(fields, props):
    expanded = []

    for field in fields:
        if field == "Y_All":
            expanded.extend(f"Y_{sp}" for sp in props["species"])

        elif field == "c_All":
            expanded.extend(f"c_{sp}" for sp in props["species"])

        elif field == "S_All":
            expanded.extend(f"S_{sp}" for sp in props["species"])

        else:
            expanded.append(field)

    return list(dict.fromkeys(expanded))

def _required_raw_fields(fields, derive_missing_fields, reconstruct_fields, props):
    required = set(fields)

    if not derive_missing_fields:
        return required

    if reconstruct_fields:
        required.add("p")

    for field in reconstruct_fields:
        if field in {"S_tot", "pEq_tot"}:
            required.update(f"Y_{sp}" for sp in props["species"])
        elif field.startswith(("S_", "c_", "pEq_")):
            required.add(f"Y_{field.split('_', 1)[1]}")

    return required


def _read_raw_fields(folder, required_raw, grid):
    raw_fields = {}

    for field in required_raw:
        field_path = folder / field
        if field_path.exists():
            values, _ = _read_internal_field(field_path)
            raw_fields[field] = _expand_uniform(values, grid)

    return raw_fields


def _process_timestep(args):
    (
        timestep,
        folder,
        fields,
        required_raw,
        reconstruct_fields,
        derive_missing_fields,
        props,
        grid,
        plane,
    ) = args

    raw_fields = _read_raw_fields(folder, required_raw, grid)
    timestep_data = {}
    derived_here = set()

    for field in fields:
        if field in raw_fields:
            values = raw_fields[field]
        elif derive_missing_fields and field in reconstruct_fields:
            values = _derive_missing_field(field, raw_fields, props)
            derived_here.add(field)
        else:
            continue

        timestep_data[field] = _reshape_to_planes(values, grid, plane=plane)

    # Correction pass:
    # If a requested derived field was skipped because it was considered available
    # globally, derive it for this timestep when it is missing locally.
    if derive_missing_fields:
        for field in fields:
            if field in timestep_data:
                continue

            if field.startswith(("c_", "S_", "pEq_")) or field in {"S_tot", "pEq_tot"}:
                values = _derive_missing_field(field, raw_fields, props)
                timestep_data[field] = _reshape_to_planes(values, grid, plane=plane)
                derived_here.add(field)

    return timestep, timestep_data, derived_here


def _progress_marks(n_items):
    return {
        max(1, int(0.25 * n_items)),
        max(1, int(0.50 * n_items)),
        max(1, int(0.75 * n_items)),
        n_items,
    }


def _write_pickle(data, output_location, output_filename, verbose=True):
    output_location = Path(output_location)
    output_location.mkdir(parents=True, exist_ok=True)
    out = output_location / output_filename

    _log(f"Writing output file: {out}", verbose)
    with open(out, "wb") as f:
        pickle.dump(data, f)


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
    parallelBackend="process", # process or thread 
    verbose=True,
):
    """
    Read OpenFOAM ASCII timestep fields into:
        data[timestep][field] = stacked 2D planes

    Parameters
    ----------
    result_dir : str or Path
        OpenFOAM case directory containing numeric timestep folders.
    fields : list[str]
        Requested fields, e.g. ["U", "p", "Y_H2", "S_tot", "pEq_H2"].
    output_location : str or Path, optional
        Directory where pickle output is written.
    output_filename : str, optional
        Pickle filename.
    plane : {"xy", "xz", "yz"}
        2D plane orientation. For plane="xy", output shape is (nz, ny, nx).
    decimals : int
        Coordinate rounding used to identify structured-grid indices.
    deriveMissingFields : bool
        If True, reconstruct requested derivable fields that are absent from all timesteps.
    verbose : bool
        Print overview-level runtime messages.
    parallel : bool
        If True, process timestep folders concurrently using threads.
    nPartitions : int, optional
        Number of parallel workers. Defaults to min(os.cpu_count(), n_timesteps).

    Notes
    -----
    Existing fields are always read from disk. A requested field is reconstructed only if it
    is absent from all timestep folders. If a field exists in some timesteps but not others,
    it is skipped for the timesteps where it is absent.
    """
    result_dir = Path(result_dir)
    # fields = list(fields)
    fields = list(fields)
    props = _load_properties(result_dir)
    fields = _expand_all_species_fields(fields, props)

    _log(f"Reading case: {result_dir}", verbose)

    C = _load_cell_centres(result_dir)
    grid = _structured_index(C, decimals=decimals)
    _log(
        f"Structured grid detected: {grid[3]} x {grid[4]} x {grid[5]} "
        f"({_n_cells(grid):,} cells)",
        verbose,
    )

    time_dirs, available_fields = _discover_time_dirs_and_fields(result_dir)
    if not time_dirs:
        raise ValueError(f"No timestep folders found in {result_dir}")

    requested_fields = set(fields)
    # props = _load_properties(result_dir) if deriveMissingFields else None
    reconstruct_fields = requested_fields - available_fields if deriveMissingFields else set()
    required_raw = _required_raw_fields(fields, deriveMissingFields, reconstruct_fields, props)

    _log("Found fields: " + ", ".join(sorted(available_fields)), verbose)
    _log("Requested fields: " + ", ".join(sorted(requested_fields)), verbose)
    _log(
        f"Found {len(time_dirs)} timesteps ({time_dirs[0][0]} -> {time_dirs[-1][0]})",
        verbose,
    )

    if deriveMissingFields:
        _log(
            f"Missing-field reconstruction enabled. Available derived fields: "
            + ", ".join(DERIVABLE_FIELDS),
            verbose,
        )
        if reconstruct_fields:
            _log(
                "Fields selected for reconstruction: "
                + ", ".join(sorted(reconstruct_fields)),
                verbose,
            )
        else:
            _log("No requested fields selected for reconstruction.", verbose)

    tasks = [
        (
            timestep,
            folder,
            fields,
            required_raw,
            reconstruct_fields,
            deriveMissingFields,
            props,
            grid,
            plane,
        )
        for timestep, folder in time_dirs
    ]

    data = {}
    reported_derivations = set()
    marks = _progress_marks(len(tasks))

    if parallel:
        workers = nPartitions or min(os.cpu_count() or 1, len(tasks))
        workers = max(1, min(int(workers), len(tasks)))

        Executor = ProcessPoolExecutor if parallelBackend.lower() == "process" else ThreadPoolExecutor

        _log(f"Parallel timestep processing enabled with {workers} partitions", verbose)

        _log(f"Parallel backend: {parallelBackend.lower()}",verbose)
        
        completed = 0

        with Executor(max_workers=workers) as executor:
            future_to_timestep = {executor.submit(_process_timestep, task): task[0] for task in tasks}

            for future in as_completed(future_to_timestep):
                timestep, timestep_data, derived_here = future.result()
                data[timestep] = timestep_data

                new_derivations = derived_here - reported_derivations
                for field in sorted(new_derivations):
                    _log(f"Reconstructing field '{field}'", verbose)
                reported_derivations.update(derived_here)

                completed += 1
                if completed in marks:
                    _log(
                        f"Processed {completed}/{len(tasks)} timesteps "
                        f"({int(100 * completed / len(tasks))}%)",
                        verbose,
                    )
    else:
        for completed, task in enumerate(tasks, start=1):
            timestep, timestep_data, derived_here = _process_timestep(task)
            data[timestep] = timestep_data

            new_derivations = derived_here - reported_derivations
            for field in sorted(new_derivations):
                _log(f"Reconstructing field '{field}'", verbose)
            reported_derivations.update(derived_here)

            if completed in marks:
                _log(
                    f"Processed {completed}/{len(tasks)} timesteps "
                    f"({int(100 * completed / len(tasks))}%)",
                    verbose,
                )

    data = dict(sorted(data.items(), key=lambda item: item[0]))

    result = {
        "meta": {
            "species": props["species"] if props else [],
            "constants": {
                "Mi": props["Mi"] if props else {},
                "Hi": props["Hi"] if props else {},
                "rho": props["rho"] if props else None,
                "Pref": props["Pref"] if props else None,
            },
            "grid": {
                "nx": grid[3],
                "ny": grid[4],
                "nz": grid[5],
                "x": grid[6],
                "y": grid[7],
                "z": grid[8],
                "plane": plane,
            },
        },
        "timesteps": data,
    }

    if output_location and output_filename:
        _write_pickle(result, output_location, output_filename, verbose=verbose)

    _log("Done.", verbose)
    return result
    