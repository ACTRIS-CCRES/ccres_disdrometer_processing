"""Various utility functions for the disdrometer processing package."""
import logging
import os

lgr = logging.getLogger(__name__)


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
