r"""Precompute the scattering tables read by ccres_disdrometer_processing.scattering.

One table per diameter grid, radar frequency, refractive index, axis ratio method
and beam orientation : the (frequency, index, method) cases are read from every
station configuration, the diameter grids from sample disdrometer files (one per
grid in use). The tables are computed with the package itself, the tables
disabled, so that reading one gives the very values computing it would.

Run it again whenever the scattering code, rustmatrix, a configuration or a
disdrometer diameter grid changes :

    python scripts/build_scattering_tables.py --conf-dir <conf_stations> \\
        --disdro-file <parsivel.nc> --disdro-file <thies.nc> ...
"""

import argparse
import datetime
import importlib.metadata
import os
import tomllib
from pathlib import Path

import numpy as np
import xarray as xr

os.environ["CCRES_NO_SCATTERING_TABLES"] = "1"  # compute, never read the old file

from ccres_disdrometer_processing import scattering  # noqa: E402
from ccres_disdrometer_processing.open_disdro_netcdf import (  # noqa: E402
    read_parsivel_cloudnet_choice,
)

# the reader needs a sampling area, which does not change the diameters
READER_CONFIG = {"instrument_parameters": {"DD_SAMPLING_AREA": 0.0054}}


def cases(conf_dir):
    """Return the (frequency, refractive index, axis ratio method) of every config."""
    out = set()
    for f in sorted(Path(conf_dir).glob("*.toml")):
        m = tomllib.loads(f.read_text())["methods"]
        for freq, (re, im) in zip(  # noqa: B905
            m["RADAR_FREQUENCIES"], m["REFRACTION_INDEX"]
        ):
            out.add((float(freq), complex(re, im), m["AXIS_RATIO_METHOD"]))
    return sorted(out, key=lambda c: (c[2], c[0], c[1].real, c[1].imag))


def grids(files):
    """Return the distinct diameter grids, as the preprocessing passes them."""
    out = []
    for f in files:
        D = np.asarray(
            read_parsivel_cloudnet_choice(f, [35e9], READER_CONFIG).size_classes[0:-5]
        )
        if not any(g.dtype == D.dtype and np.array_equal(g, D) for g in out):
            out.append(D)
    return out


def main():
    """Build the table file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--conf-dir", required=True, type=Path)
    parser.add_argument("--disdro-file", required=True, action="append", type=Path)
    parser.add_argument("--output", type=Path, default=scattering.TABLE_FILE)
    args = parser.parse_args()

    entries = []
    for D in grids(args.disdro_file):
        for freq, e, axr in cases(args.conf_dir):
            for fov in (1, 0):
                s = scattering.scattering_prop(D, fov, freq, e=e, axrMethod=axr)
                entries.append((D, freq, e, axr, fov, s))
                print(f"{len(D)} diameters, {freq / 1e9:g} GHz, e = {e}, fov {fov}")

    n_max = max(len(x[0]) for x in entries)
    dtype = np.result_type(*[x[0].dtype for x in entries])

    def padded(values, dt):
        out = np.full((len(entries), n_max), np.nan, dtype=dt)
        for k, v in enumerate(values):
            out[k, : len(v)] = v
        return out

    ds = xr.Dataset(
        {
            "frequency": ("entry", [x[1] for x in entries], {"units": "Hz"}),
            "e_real": ("entry", [x[2].real for x in entries]),
            "e_imag": ("entry", [x[2].imag for x in entries]),
            "axis_ratio_method": ("entry", np.array([x[3] for x in entries], "U")),
            "beam_orientation": (
                "entry",
                np.array([x[4] for x in entries], "i2"),
                {"comment": "1 : vertical, 0 : horizontal"},
            ),
            "n_diameters": ("entry", np.array([len(x[0]) for x in entries], "i2")),
            "diameter": (
                ("entry", "diameter_index"),
                padded([x[0] for x in entries], dtype),
                {"units": "mm"},
            ),
            **{
                v: (
                    ("entry", "diameter_index"),
                    padded([getattr(x[5], v) for x in entries], "f8"),
                )
                for v in scattering.TABLE_VARS
            },
        },
        attrs={
            "title": "Precomputed scattering of the CCRES disdrometer processing",
            "rustmatrix_version": importlib.metadata.version("rustmatrix"),
            "ndgs": f"{scattering.NDGS} (fallback {scattering.NDGS_FALLBACK})",
            "date_created": datetime.datetime.now(datetime.UTC).isoformat(),
            "source": "scripts/build_scattering_tables.py",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(args.output)
    print(f"{len(entries)} tables written to {args.output}")


if __name__ == "__main__":
    main()
