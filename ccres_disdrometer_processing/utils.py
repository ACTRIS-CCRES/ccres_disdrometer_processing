"""Various utility functions for the disdrometer processing package."""

import logging
import os
from pathlib import Path

import numpy as np
import requests
import xarray as xr

lgr = logging.getLogger(__name__)

CLOUDNET_API_URL = "https://cloudnet.fmi.fi/api/"
CLOUDNET_API_FILE_URL = "https://cloudnet.fmi.fi/api/files/"
CLOUDNET_API_FILES_OPTS = "?site={site:}&date={date:}&instrument={instrument:}&instrumentPid={instrument_pid:}"  # noqa E501


def is_file_available(filename: str, local_dir: Path) -> bool:
    """Check if a file is already downloaded.

    Parameters
    ----------
    file : str
        The file to chack if available.
    local_dir : Path
        The directory where the file should be available.

    Returns
    -------
    bool
        True if the file is already available, False otherwise.

    """
    local_file = local_dir / filename

    file_exists = False
    if local_file.exists():
        file_exists = True

    return file_exists


def get_file_from_cloudnet(
    site: str, date: str, instrument: str, instrument_pid: str, local_dir: Path
) -> None:
    """Download a file from Cloudnet.

    Parameters
    ----------
    site : str
        The site where the file was recorded.
    date : str
        The date when the file was recorded. format %Y-%m-%d.
    instrument : str
        The instrument that recorded the file.
    instrument_pid : str
        The pid of the instrument that recorded the file
    local_dir : Path
        The directory where the file should be downloaded.

    Returns
    -------
    local_file : Path
        The path to the downloaded file.

    """
    # check input
    if not isinstance(local_dir, str):
        local_dir = Path(local_dir)

    request_urls = CLOUDNET_API_FILE_URL + CLOUDNET_API_FILES_OPTS.format(
        **{
            "site": site,
            "date": date,
            "instrument": instrument,
            "instrument_pid": instrument_pid,
        }
    )
    metadata = requests.get(request_urls).json()
    # Cloudnet may expose several files for the same site/date/instrument
    # query (e.g. derived "epsilon-*" turbulence products alongside the raw
    # instrument product). Skip those derived products so the raw file
    # (e.g. containing Zh for a radar) is picked instead.
    candidates = [
        item for item in metadata if not item["product"]["id"].startswith("epsilon")
    ] or metadata
    filename = candidates[0]["filename"]
    local_file = local_dir / filename

    if not is_file_available(filename, local_dir):
        # get url of file
        file_url = candidates[0]["downloadUrl"]
        print(f"Downloading {file_url} into {local_file}")
        # download file
        response = requests.get(file_url, stream=True)
        with open(local_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    return local_file


def instrument_position(position: xr.DataArray) -> xr.DataArray:
    """Return an instrument latitude, longitude or altitude as a scalar.

    Recent Cloudnet files give the position at every timestep instead of once.
    The first finite value is kept (NaN if there is none), with the attributes.

    Parameters
    ----------
    position : xr.DataArray
        latitude, longitude or altitude from a Cloudnet file, scalar or 1D

    Returns
    -------
    xr.DataArray
        The scalar position

    """
    values = np.atleast_1d(position.values).ravel()
    finite = values[np.isfinite(values)]
    value = finite[0] if finite.size else values.dtype.type(np.nan)
    return xr.DataArray(value, attrs=position.attrs)


POSITION_VARS = [
    f"{instrument}_{coord}"
    for instrument in ("disdro", "ams")
    for coord in ("longitude", "latitude", "altitude")
]


def scalar_positions(ds: xr.Dataset) -> xr.Dataset:
    """Make the instrument positions of a preprocessed dataset scalar.

    Older preprocessed files keep a time-dependent disdrometer position on a
    "time_bins" dimension (and the weather station one on "time"). Concatenating
    days would turn each into a (time, time_bins) array of several hundred MB.

    Parameters
    ----------
    ds : xr.Dataset
        One daily preprocessed dataset

    Returns
    -------
    xr.Dataset
        The dataset with scalar positions and without "time_bins"

    """
    for var in POSITION_VARS:
        if var in ds and ds[var].ndim:
            ds[var] = instrument_position(ds[var])
    return ds.drop_dims("time_bins", errors="ignore")


def format_ql_file_prefix(prefix: str):
    """Format and check if the prefix is valid.

    The prefix shouldn't contain any extension.

    Parameters
    ----------
    prefix : str
        Mask to check.

    Returns
    -------
    str
        the checked and formatted prefix.

    """
    name, ext = os.path.splitext(prefix)
    if ext:
        lgr.info("removing extension from prefix")
        prefix = name

    prefix += "_{:02d}.png"

    return prefix
