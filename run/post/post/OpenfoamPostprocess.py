# openfoam_postprocess.py

from pathlib import Path
import pickle
import numpy as np


def _expand_all_species_fields(fields, species):
    expanded = []

    for field in fields:
        if field == "Y_All":
            expanded.extend(f"Y_{sp}" for sp in species)
        elif field == "c_All":
            expanded.extend(f"c_{sp}" for sp in species)
        elif field == "S_All":
            expanded.extend(f"S_{sp}" for sp in species)
        elif field == "pEq_All":
            expanded.extend(f"pEq_{sp}" for sp in species)
        else:
            expanded.append(field)

    return list(dict.fromkeys(expanded))


def _derive_field(field, timestep_data, meta):
    constants = meta["constants"]

    rho = constants["rho"]
    Mi = constants["Mi"]
    Hi = constants.get("Hi", {})
    Pref = constants.get("absolutePressureReference")

    if field.startswith("c_"):
        sp = field[2:]
        Y_name = f"Y_{sp}"

        if Y_name not in timestep_data:
            raise ValueError(f"Cannot derive {field}: missing {Y_name}")

        return rho * timestep_data[Y_name] / Mi[sp]

    if field.startswith("S_") and field != "S_tot":
        sp = field[2:]

        if "p" not in timestep_data:
            raise ValueError(f"Cannot derive {field}: missing p")

        c = timestep_data.get(f"c_{sp}")
        if c is None:
            c = _derive_field(f"c_{sp}", timestep_data, meta)

        P_liquid = Pref + rho * timestep_data["p"]
        return c / (Hi[sp] * P_liquid)

    if field == "S_tot":
        terms = []

        for sp in meta["species"]:
            if sp not in Hi:
                continue

            try:
                terms.append(_derive_field(f"S_{sp}", timestep_data, meta))
            except ValueError:
                pass

        if not terms:
            raise ValueError("Cannot derive S_tot: no species could be derived")

        return np.sum(terms, axis=0)

    if field.startswith("pEq_") and field != "pEq_tot":
        sp = field[4:]

        c = timestep_data.get(f"c_{sp}")
        if c is None:
            c = _derive_field(f"c_{sp}", timestep_data, meta)

        return c / Hi[sp]

    if field == "pEq_tot":
        terms = []

        for sp in meta["species"]:
            if sp not in Hi:
                continue

            try:
                terms.append(_derive_field(f"pEq_{sp}", timestep_data, meta))
            except ValueError:
                pass

        if not terms:
            raise ValueError("Cannot derive pEq_tot: no species could be derived")

        return np.sum(terms, axis=0)

    raise ValueError(f"No derivation rule for {field}")


def derive_fields(
    data,
    derive_fields,
    save=False,
    output_location=None,
    output_filename=None,
    overwrite=False,
    verbose=True,
):

    if "meta" not in data or "timesteps" not in data:
        raise ValueError(
            "Expected data format with data['meta'] and data['timesteps']"
        )

    meta = data["meta"]
    timesteps = data["timesteps"]

    fields = _expand_all_species_fields(derive_fields, meta["species"])

    for timestep, timestep_data in timesteps.items():
        if verbose:
            print(f"[Postprocess] timestep {timestep}")

        for field in fields:
            if field in timestep_data and not overwrite:
                continue

            timestep_data[field] = _derive_field(field, timestep_data, meta)

    if save:

        if output_location is None:
            raise ValueError(
                "output_location must be specified when save=True"
            )

        if output_filename is None:
            raise ValueError(
                "output_filename must be specified when save=True"
            )

        output_location = Path(output_location)
        output_location.mkdir(parents=True, exist_ok=True)

        out = output_location / output_filename

        if verbose:
            print(f"[Postprocess] writing {out}")

        with open(out, "wb") as f:
            pickle.dump(data, f)

    return data