from pathlib import Path
import pickle
import numpy as np


DERIVABLE_TOTAL_FIELDS = ("S_tot", "pEq_tot")
COORDINATE_FIELDS = {"C", "Cx", "Cy", "Cz"}


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


def _species_with_molar_mass(meta):
    constants = meta["constants"]
    Mi = constants.get("Mi", {})
    return [sp for sp in meta["species"] if sp in Mi]


def _species_with_henry_constant(meta):
    constants = meta["constants"]
    Mi = constants.get("Mi", {})
    Hi = constants.get("Hi", {})
    return [sp for sp in meta["species"] if sp in Mi and sp in Hi]


def _expand_species_family(family, meta):
    if family == "Y":
        return [f"Y_{sp}" for sp in meta["species"]]
    if family == "c":
        return [f"c_{sp}" for sp in _species_with_molar_mass(meta)]
    if family == "S":
        return [f"S_{sp}" for sp in _species_with_henry_constant(meta)] + ["S_tot"]
    if family == "pEq":
        return [f"pEq_{sp}" for sp in _species_with_henry_constant(meta)] + ["pEq_tot"]
    raise ValueError(f"Unsupported species field family: {family}")


def _all_derivable_fields(meta):
    """Return all fields that can be derived from the stored constants/species."""
    fields = []
    fields.extend(_expand_species_family("c", meta))
    fields.extend(_expand_species_family("S", meta))
    fields.extend(_expand_species_family("pEq", meta))
    return _unique_keep_order(fields)


def _available_loaded_fields(data):
    fields = set()

    inventory = data.get("fields", {})
    fields.update(inventory.get("available", []))
    fields.update(inventory.get("requested", []))

    for timestep_data in data.get("timesteps", {}).values():
        fields.update(timestep_data.keys())

    # Coordinates are stored once in data["coords"], so postprocessing should
    # not request or recreate C/Cx/Cy/Cz as timestep fields.
    return {field for field in fields if field not in COORDINATE_FIELDS}


def _resolve_field_requests(field_requests, data):
    """
    Resolve field requests using the common hierarchy:
      1. "all" / "All" / ["all"] -> all currently available loaded fields
         plus all derivable fields.
      2. "Y_All", "Y_all", "C_All", "c_All", "S_All", "pEq_All" ->
         all species fields for that family. S and pEq include *_tot.
      3. Explicit names such as "U", "p", "Y_H2", "S_tot" pass through.

    The postprocessor only creates fields with derivation rules. Non-derivable
    existing fields such as U, p, Y_* are accepted in the request but left as-is.
    """
    meta = data["meta"]
    requested = _as_field_list(field_requests)

    resolved = []
    wants_all = any(isinstance(field, str) and field.lower() == "all" for field in requested)
    if wants_all:
        resolved.extend(sorted(_available_loaded_fields(data)))
        resolved.extend(_all_derivable_fields(meta))

    for field in requested:
        if isinstance(field, str) and field.lower() == "all":
            continue

        family = _species_family_from_all_token(field)
        if family is not None:
            resolved.extend(_expand_species_family(family, meta))
        else:
            resolved.append(_canonical_species_field(field))

    return _unique_keep_order(resolved)


def _is_derivable_field(field):
    return (
        isinstance(field, str)
        and field not in COORDINATE_FIELDS
        and (
            field.startswith("c_")
            or field.startswith("S_")
            or field.startswith("pEq_")
            or field in DERIVABLE_TOTAL_FIELDS
        )
    )


def _existing_derivable_fields_to_correct(data):
    """
    Derivable fields that were already part of the loaded/read data request.

    These are corrected per timestep so fields that exist later in the solver
    output, but are absent at t=0, are reconstructed without requiring the user
    to list them again in derive_fields.
    """
    candidates = set()
    inventory = data.get("fields", {})

    # Correct fields that were actually requested/read, not every globally
    # available solver field. data["fields"]["available"] may include fields
    # that exist only at later timesteps and whose dependencies were not loaded.
    # Including those here can cause e.g. S_CO2 -> c_CO2 derivation attempts
    # when the user only loaded Y_O2.
    candidates.update(inventory.get("requested", []))

    for timestep_data in data.get("timesteps", {}).values():
        candidates.update(timestep_data.keys())

    return _unique_keep_order(
        _canonical_species_field(field)
        for field in sorted(candidates)
        if _is_derivable_field(_canonical_species_field(field))
    )


def _pressure_reference(constants):
    # Compatible with both older postprocess naming and the reader output.
    Pref = constants.get("Pref")
    if Pref is None:
        Pref = constants.get("absolutePressureReference")
    if Pref is None:
        raise ValueError("Cannot derive S_* fields: missing pressure reference constant 'Pref'")
    return Pref


def _derive_field(field, timestep_data, meta):
    constants = meta["constants"]

    rho = constants["rho"]
    Mi = constants["Mi"]
    Hi = constants.get("Hi", {})

    if field.startswith("c_"):
        sp = field[2:]
        Y_name = f"Y_{sp}"

        if Y_name not in timestep_data:
            raise ValueError(f"Cannot derive {field}: missing {Y_name}")
        if sp not in Mi:
            raise ValueError(f"Cannot derive {field}: missing molar mass for {sp}")

        return rho * timestep_data[Y_name] / Mi[sp]

    if field.startswith("S_") and field != "S_tot":
        sp = field[2:]

        if "p" not in timestep_data:
            raise ValueError(f"Cannot derive {field}: missing p")
        if sp not in Hi:
            raise ValueError(f"Cannot derive {field}: missing Henry constant for {sp}")

        c = timestep_data.get(f"c_{sp}")
        if c is None:
            c = _derive_field(f"c_{sp}", timestep_data, meta)

        Pref = _pressure_reference(constants)
        P_liquid = Pref + rho * timestep_data["p"]
        return c / (Hi[sp] * P_liquid)

    if field == "S_tot":
        terms = []

        for sp in meta["species"]:
            if sp not in Hi:
                continue

            # Prefer an existing S_species field if present. Otherwise only try
            # to derive it when the required Y_species dependency is loaded for
            # this timestep. This avoids accidental attempts to derive S_CO2
            # when only Y_O2 was read.
            existing = timestep_data.get(f"S_{sp}")
            if existing is not None:
                terms.append(existing)
                continue

            if f"Y_{sp}" not in timestep_data and f"c_{sp}" not in timestep_data:
                continue

            try:
                terms.append(_derive_field(f"S_{sp}", timestep_data, meta))
            except ValueError:
                pass

        if not terms:
            raise ValueError("Cannot derive S_tot: no species could be derived from loaded fields")

        return np.sum(terms, axis=0)

    if field.startswith("pEq_") and field != "pEq_tot":
        sp = field[4:]

        if sp not in Hi:
            raise ValueError(f"Cannot derive {field}: missing Henry constant for {sp}")

        c = timestep_data.get(f"c_{sp}")
        if c is None:
            c = _derive_field(f"c_{sp}", timestep_data, meta)

        return c / Hi[sp]

    if field == "pEq_tot":
        terms = []

        for sp in meta["species"]:
            if sp not in Hi:
                continue

            existing = timestep_data.get(f"pEq_{sp}")
            if existing is not None:
                terms.append(existing)
                continue

            if f"Y_{sp}" not in timestep_data and f"c_{sp}" not in timestep_data:
                continue

            try:
                terms.append(_derive_field(f"pEq_{sp}", timestep_data, meta))
            except ValueError:
                pass

        if not terms:
            raise ValueError("Cannot derive pEq_tot: no species could be derived from loaded fields")

        return np.sum(terms, axis=0)

    raise ValueError(f"No derivation rule for {field}")



def _derivation_requirements_message(field, timestep_data, meta):
    """Return None when `field` can be derived for this timestep, else an error reason."""
    constants = meta["constants"]
    Mi = constants.get("Mi", {})
    Hi = constants.get("Hi", {})

    if field.startswith("c_"):
        sp = field[2:]
        missing = []
        if f"Y_{sp}" not in timestep_data:
            missing.append(f"Y_{sp}")
        if sp not in Mi:
            missing.append(f"molar mass Mi['{sp}']")
        return None if not missing else "missing " + ", ".join(missing)

    if field.startswith("S_") and field != "S_tot":
        sp = field[2:]
        missing = []
        if "p" not in timestep_data:
            missing.append("p")
        if f"c_{sp}" not in timestep_data and f"Y_{sp}" not in timestep_data:
            missing.append(f"c_{sp} or Y_{sp}")
        if sp not in Hi:
            missing.append(f"Henry constant Hi['{sp}']")
        try:
            _pressure_reference(constants)
        except ValueError:
            missing.append("pressure reference Pref")
        return None if not missing else "missing " + ", ".join(missing)

    if field == "S_tot":
        missing = []
        if "p" not in timestep_data:
            missing.append("p")
        try:
            _pressure_reference(constants)
        except ValueError:
            missing.append("pressure reference Pref")

        available_terms = [
            sp for sp in meta["species"]
            if sp in Hi and (
                f"S_{sp}" in timestep_data
                or f"c_{sp}" in timestep_data
                or f"Y_{sp}" in timestep_data
            )
        ]
        if not available_terms:
            missing.append("at least one S_<species>, c_<species>, or Y_<species> with a Henry constant")
        return None if not missing else "missing " + ", ".join(missing)

    if field.startswith("pEq_") and field != "pEq_tot":
        sp = field[4:]
        missing = []
        if f"c_{sp}" not in timestep_data and f"Y_{sp}" not in timestep_data:
            missing.append(f"c_{sp} or Y_{sp}")
        if sp not in Hi:
            missing.append(f"Henry constant Hi['{sp}']")
        return None if not missing else "missing " + ", ".join(missing)

    if field == "pEq_tot":
        available_terms = [
            sp for sp in meta["species"]
            if sp in Hi and (
                f"pEq_{sp}" in timestep_data
                or f"c_{sp}" in timestep_data
                or f"Y_{sp}" in timestep_data
            )
        ]
        if not available_terms:
            return "missing at least one pEq_<species>, c_<species>, or Y_<species> with a Henry constant"
        return None

    return f"no derivation rule for {field}"


def _validate_requested_derivations(additional_fields, timesteps, meta, overwrite=False):
    """Fail early for user-requested derived fields that cannot be computed."""
    errors = []

    for field in additional_fields:
        if not _is_derivable_field(field):
            continue

        for timestep, timestep_data in timesteps.items():
            if field in timestep_data and not overwrite:
                continue

            reason = _derivation_requirements_message(field, timestep_data, meta)
            if reason is not None:
                errors.append(f"{field} at timestep {timestep}: {reason}")

    if errors:
        shown = "\n  - " + "\n  - ".join(errors[:20])
        extra = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(
            "Cannot derive the requested field(s). Load the required dependency fields "
            "with read_openfoam_results first, or request a field that can be derived "
            "from the loaded data. Details:" + shown + extra
        )

def derive_fields(
    data,
    derive_fields,
    save=False,
    output_location=None,
    output_filename=None,
    overwrite=False,
    verbose=True,
):
    """
    Add derived OpenFOAM postprocessing fields to an already-read data dictionary.

    Parameters
    ----------
    data : dict
        Expected format from read_openfoam_results:
        data["meta"], data["coords"], data["timesteps"].
    derive_fields : str or list[str]
        Additional fields to derive or correct. Supports explicit fields such
        as "c_H2", "C_H2", "S_tot", species aliases such as
        "Y_All"/"Y_all", "C_All"/"c_All", "S_All", "pEq_All", and
        full expansion with "all", "All", "ALL", ["all"], or ["All"].
        "S_All" and "pEq_All" include their corresponding total fields.
        Existing timestep fields are preserved. Explicitly requested derived
        fields fail early with a clear dependency error if they cannot be
        computed from the loaded data. Derivable fields that were already
        requested/read are corrected best-effort at timesteps where they are
        missing, for example t=0.
    save : bool
        If True, write the updated data dictionary to pickle.
    output_location : str or Path, optional
        Directory where pickle output is written when save=True.
    output_filename : str, optional
        Pickle filename when save=True.
    overwrite : bool
        If False, existing fields are left unchanged.
    verbose : bool
        Print progress messages.
    """
    if "meta" not in data or "timesteps" not in data:
        raise ValueError("Expected data format with data['meta'] and data['timesteps']")

    meta = data["meta"]
    timesteps = data["timesteps"]

    additional_fields = _resolve_field_requests(derive_fields, data)
    correction_fields = _existing_derivable_fields_to_correct(data)
    fields = _unique_keep_order(correction_fields + additional_fields)
    explicitly_requested_derivable = {
        field for field in additional_fields if _is_derivable_field(field)
    }

    _validate_requested_derivations(
        additional_fields,
        timesteps,
        meta,
        overwrite=overwrite,
    )

    if verbose:
        print("[Postprocess] correcting/deriving fields: " + ", ".join(fields))

    for timestep, timestep_data in timesteps.items():
        if verbose:
            print(f"[Postprocess] timestep {timestep}")

        for field in fields:
            # Existing non-derived fields such as U, p, and Y_* are allowed in
            # requests, especially via "all", but the postprocessor only needs
            # to act when a requested field has a derivation rule.
            if not _is_derivable_field(field):
                continue

            # Correction behavior: if the field is missing at a timestep
            # (for example c_* or S_* at t=0), reconstruct it even when the
            # same field exists at later timesteps. Existing fields are kept
            # unless overwrite=True.
            if field in timestep_data and not overwrite:
                continue

            try:
                timestep_data[field] = _derive_field(field, timestep_data, meta)
            except ValueError:
                # User-requested derived fields are validated strictly above, so
                # any error here for them should still be surfaced. Correction
                # fields are best-effort: they fill gaps such as t=0 when the
                # dependencies are present, but they do not fail when the user
                # did not load enough data for that automatic correction.
                if field in explicitly_requested_derivable:
                    raise
                continue

    if save:
        if output_location is None:
            raise ValueError("output_location must be specified when save=True")

        if output_filename is None:
            raise ValueError("output_filename must be specified when save=True")

        output_location = Path(output_location)
        output_location.mkdir(parents=True, exist_ok=True)

        out = output_location / output_filename

        if verbose:
            print(f"[Postprocess] writing {out}")

        with open(out, "wb") as f:
            pickle.dump(data, f)

    return data
