from pathlib import Path
import re
import numpy as np

class OpenFOAMRawCase:
    def __init__(self, case_dir, decimals=12, pressure_field="p"):
        self.case_dir = Path(case_dir)
        self.zero_dir = self.case_dir / "0"
        self.constant_dir = self.case_dir / "constant"
        self.pressure_field = pressure_field

        self.C = self._load_coordinates()
        self.grid = self._build_grid(self.C, decimals)

        self.x = self.grid["x"]
        self.y = self.grid["y"]
        self.z = self.grid["z"]

        self.times = self._discover_times()

        self.transport_properties = self._read_openfoam_dictionary(
            self.constant_dir / "transportProperties", required=False
        )
        self.reactions_properties = self._read_openfoam_dictionary(
            self.constant_dir / "reactions", required=False
        )

        self.rho = self._get_scalar_entry(self.transport_properties, "rho")
        self.molar_mass = self._get_subdict(self.reactions_properties, "molarMass")
        self.henry_constant = (self._get_subdict(self.reactions_properties, "henryConstants") or self._get_subdict(self.reactions_properties, "henryConstant"))
        self.p_ref = self._get_pressure_references()

    def is_time_dir(self, path: Path):
        if not path.is_dir():
            return False
        try:
            float(path.name)
            return True
        except ValueError:
            return False


    def _strip_openfoam_comments(self, text):
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r"//.*", "", text)
        return text

    def _read_openfoam_dictionary(self, file_path, required=True):
        file_path = Path(file_path)
        if not file_path.exists():
            if required:
                raise FileNotFoundError(file_path)
            return ""
        return self._strip_openfoam_comments(file_path.read_text())

    def _get_scalar_entry(self, text, name, default=None):
        """
        Reads either
            rho 1000;
        or
            rho [1 -3 0 0 0 0 0] 1000;
        """
        m = re.search(
            rf"(?:^|\n)\s*{re.escape(name)}\s+(?:\[[^\]]+\]\s*)?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;",
            text,
        )
        if not m:
            return default
        return float(m.group(1))

    def _get_subdict(self, text, name):
        m = re.search(rf"(?:^|\n)\s*{re.escape(name)}\s*\{{(.*?)\}}", text, re.S)
        if not m:
            return {}

        out = {}
        for key, value in re.findall(
            r"(\w+)\s+(?:\[[^\]]+\]\s*)?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*;",
            m.group(1),
        ):
            out[key] = float(value)
        return out

    def _get_pressure_references(self):
        """
        Optional species-wise pressure references can be provided in the reactions file as
        one of these dictionaries:
            pressureReference { CO2 101325; }
            pRef              { CO2 101325; }
            P_ref             { CO2 101325; }

        If not provided, S_* and S_tot use the liquid pressure field self.pressure_field.
        If the pressure field is absent, they fall back to absolutePressureReference.
        """
        for name in ("pressureReference", "pRef", "P_ref", "PRef"):
            refs = self._get_subdict(self.reactions_properties, name)
            if refs:
                return refs

        absolute = self._get_scalar_entry(
            self.reactions_properties, "absolutePressureReference", default=None
        )
        return {"__absolute__": absolute} if absolute is not None else {}

    def _need_molar_mass(self, species):
        if species not in self.molar_mass:
            raise KeyError(
                f"No molarMass entry for species '{species}' in constant/reactions"
            )
        return self.molar_mass[species]

    def _need_henry_constant(self, species):
        if species not in self.henry_constant:
            raise KeyError(
                f"No henryConstant entry for species '{species}' in constant/reactions"
            )
        return self.henry_constant[species]

    def _pressure_reference_flat(self, time, species=None):
        n = self.grid["n_cells"]

        p_file = self._time_dir(time) / self.pressure_field

        p_abs_ref = self.p_ref.get("__absolute__", 101325.0)

        if p_file.exists():
            if self.rho is None:
                raise KeyError("No rho entry in constant/transportProperties")
            p_kin = self.read_field_flat(time, self.pressure_field)
            return np.maximum(p_abs_ref + self.rho * p_kin, np.finfo(float).tiny)

        return np.full(n, p_abs_ref)

    def _species_with_henry(self):
        return sorted(set(self.henry_constant).intersection(self.molar_mass))

    def _derive_field_flat(self, time, field):
        if field.startswith("c_"):
            species = field[2:]
            M = self._need_molar_mass(species)
            Y = self.read_field_flat(time, f"Y_{species}")
            if self.rho is None:
                raise KeyError("No rho entry in constant/transportProperties")
            return self.rho * Y / M

        if field.startswith("pEq_") and field != "pEq_tot":
            species = field[4:]
            H = self._need_henry_constant(species)
            return self._derive_field_flat(time, f"c_{species}") / H

        if field.startswith("S_") and field != "S_tot":
            species = field[2:]
            p_eq = self._derive_field_flat(time, f"pEq_{species}")
            p_ref = self._pressure_reference_flat(time, species=species)
            return p_eq / p_ref

        if field == "pEq_tot":
            species_names = self._species_with_henry()
            if not species_names:
                raise KeyError("No species with both molarMass and henryConstant entries")
            return sum(self._derive_field_flat(time, f"pEq_{sp}") for sp in species_names)

        if field == "S_tot":
            p_eq_tot = self._derive_field_flat(time, "pEq_tot")
            p_ref = self._pressure_reference_flat(time, species=None)
            return p_eq_tot / p_ref

        return None

    def read_internal_field(self, file_path):
        text = Path(file_path).read_text()

        m = re.search(
            r"internalField\s+nonuniform\s+List<(\w+)>\s+(\d+)\s*\((.*?)\)\s*;",
            text,
            re.S,
        )

        if m:
            value_type = m.group(1)
            n = int(m.group(2))
            body = m.group(3).strip()

            if value_type == "scalar":
                arr = np.fromstring(body, sep=" ")
                if arr.size != n:
                    raise ValueError(f"Expected {n}, got {arr.size}: {file_path}")
                return arr

            if value_type == "vector":
                rows = re.findall(r"\(([^()]+)\)", body)
                arr = np.array([[float(v) for v in row.split()] for row in rows])
                if arr.shape != (n, 3):
                    raise ValueError(f"Expected {(n, 3)}, got {arr.shape}: {file_path}")
                return arr

            raise NotImplementedError(f"Unsupported field type: {value_type}")

        m = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if m:
            return m.group(1).strip()

        raise ValueError(f"Could not parse internalField: {file_path}")

    def _discover_times(self):
        times = []
        for p in self.case_dir.iterdir():
            if self.is_time_dir(p):
                times.append(float(p.name))
        return sorted(times)

    def _load_coordinates(self):
        C_file = self.zero_dir / "C"

        if C_file.exists():
            return self.read_internal_field(C_file)

        Cx = self.read_internal_field(self.zero_dir / "Cx")
        Cy = self.read_internal_field(self.zero_dir / "Cy")
        Cz = self.read_internal_field(self.zero_dir / "Cz")

        return np.column_stack([Cx, Cy, Cz])

    def _build_grid(self, C, decimals):
        xr = np.round(C[:, 0], decimals)
        yr = np.round(C[:, 1], decimals)
        zr = np.round(C[:, 2], decimals)

        x = np.unique(xr)
        y = np.unique(yr)
        z = np.unique(zr)

        nx, ny, nz = len(x), len(y), len(z)

        if nx * ny * nz != len(C):
            raise ValueError("Cell centres do not form a complete structured grid.")

        ix = np.searchsorted(x, xr)
        iy = np.searchsorted(y, yr)
        iz = np.searchsorted(z, zr)

        return {
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "x": x,
            "y": y,
            "z": z,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "n_cells": len(C),
        }

    def _time_dir(self, time):
        # exact string first
        p = self.case_dir / str(time)
        if p.exists():
            return p

        # otherwise nearest discovered time
        t = min(self.times, key=lambda v: abs(v - float(time)))
        return self.case_dir / f"{t:g}"

    def _expand_uniform(self, values):
        if not isinstance(values, str):
            return values

        values = values.strip()
        n = self.grid["n_cells"]

        if values.startswith("(") and values.endswith(")"):
            vec = np.array([float(v) for v in values[1:-1].split()])
            return np.tile(vec, (n, 1))

        return np.full(n, float(values))

    def read_field_flat(self, time, field):
        file_path = self._time_dir(time) / field

        if file_path.exists():
            values = self.read_internal_field(file_path)
            return self._expand_uniform(values)

        derived = self._derive_field_flat(time, field)
        if derived is not None:
            return derived

        raise FileNotFoundError(f"Missing field file: {file_path}")

    def read_field_3d(self, time, field):
        values = self.read_field_flat(time, field)

        ix = self.grid["ix"]
        iy = self.grid["iy"]
        iz = self.grid["iz"]

        nx = self.grid["nx"]
        ny = self.grid["ny"]
        nz = self.grid["nz"]

        extra_shape = values.shape[1:] if values.ndim == 2 else ()
        arr = np.empty((nz, ny, nx) + extra_shape)

        arr[iz, iy, ix] = values

        return arr

    def _axis_index(self, axis, value):
        coords = {"x": self.x, "y": self.y, "z": self.z}[axis]
        i = np.argmin(np.abs(coords - value))
        return i, coords[i]


    def slice(self, time, field, plane, value):
        arr = self.read_field_3d(time, field)

        if plane == "xy":
            iz, z_actual = self._axis_index("z", value)
            values = arr[iz, :, :]
            return {
                "time": time,
                "field": field,
                "plane": "xy",
                "fixed_axis": "z",
                "fixed_value": z_actual,
                "a": self.x,
                "b": self.y,
                "a_name": "x",
                "b_name": "y",
                "values": values,
            }

        if plane == "xz":
            iy, y_actual = self._axis_index("y", value)
            values = arr[:, iy, :]
            return {
                "time": time,
                "field": field,
                "plane": "xz",
                "fixed_axis": "y",
                "fixed_value": y_actual,
                "a": self.x,
                "b": self.z,
                "a_name": "x",
                "b_name": "z",
                "values": values,
            }

        if plane == "yz":
            ix, x_actual = self._axis_index("x", value)
            values = arr[:, :, ix]
            return {
                "time": time,
                "field": field,
                "plane": "yz",
                "fixed_axis": "x",
                "fixed_value": x_actual,
                "a": self.y,
                "b": self.z,
                "a_name": "y",
                "b_name": "z",
                "values": values,
            }

        raise ValueError("plane must be 'xy', 'xz', or 'yz'")

    def profile(self, time, field, along, fixed):
        arr = self.read_field_3d(time, field)

        if along == "x":
            iy, y_actual = self._axis_index("y", fixed["y"])
            iz, z_actual = self._axis_index("z", fixed["z"])
            values = arr[iz, iy, :]
            coord = self.x
            actual_fixed = {"y": y_actual, "z": z_actual}

        elif along == "y":
            ix, x_actual = self._axis_index("x", fixed["x"])
            iz, z_actual = self._axis_index("z", fixed["z"])
            values = arr[iz, :, ix]
            coord = self.y
            actual_fixed = {"x": x_actual, "z": z_actual}

        elif along == "z":
            ix, x_actual = self._axis_index("x", fixed["x"])
            iy, y_actual = self._axis_index("y", fixed["y"])
            values = arr[:, iy, ix]
            coord = self.z
            actual_fixed = {"x": x_actual, "y": y_actual}

        else:
            raise ValueError("along must be 'x', 'y', or 'z'")

        return {
            "time": time,
            "field": field,
            "along": along,
            "coord": coord,
            "fixed": actual_fixed,
            "values": values,
        }

    def fetch_slices(self, field, times, plane, values):
        out = {}

        for t in times:
            out[t] = {}
            for value in values:
                s = self.slice(t, field, plane=plane, value=value)
                out[t][s["fixed_value"]] = s

        return out


    def fetch_profiles(self, field, times, along, fixed_list):
        out = {}

        for t in times:
            out[t] = []
            for fixed in fixed_list:
                p = self.profile(t, field, along=along, fixed=fixed)
                out[t].append(p)

        return out
    
    def fetch_profiles_multi(self, fields, time, along, fixed):
        if isinstance(fields, str):
            fields = [fields]

        return [
            self.profile(time, field, along=along, fixed=fixed)
            for field in fields
        ]