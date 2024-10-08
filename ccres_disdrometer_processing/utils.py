"""Various utility functions for the disdrometer processing package."""

import logging
import os
from pathlib import Path

import requests

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
    filename = metadata[0]["filename"]
    local_file = local_dir / filename

    if not is_file_available(filename, local_dir):
        # get url of file
        file_url = metadata[0]["downloadUrl"]
        print(f"Downloading {file_url} into {local_file}")
        # download file
        response = requests.get(file_url, stream=True)
        with open(local_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                file.write(chunk)

    return local_file


def build_preprocess_param(test_cases):
    list_params = []
    for case in test_cases:
        site = case["site"]
        date = case["date"]
        radar = case["radar"]
        disdro = case["disdro"]
        use_ams = case["meteo-available"]
        ams = case["meteo"]
        config = case["config"]
        list_params.append((site, date, radar, disdro, use_ams, ams, config))

    return list_params


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
