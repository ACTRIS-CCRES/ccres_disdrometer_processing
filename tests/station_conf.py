"""Station configuration of the tests : the code holds no default parameter."""

import copy
from pathlib import Path

from ccres_disdrometer_processing.config import load_config

CONF_FILE = (
    Path(__file__).parent / "data" / "conf" / "config_juelich_mira-parsivel.toml"
)
_CONF = load_config(CONF_FILE)


def station_conf(**sections):
    """Return a copy of the test configuration, sections updated with the dicts."""
    conf = copy.deepcopy(_CONF)
    for name, values in sections.items():
        conf[name].update(values)
    return conf
