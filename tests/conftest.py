"""Fixtures for pytest."""

from pathlib import Path

import pytest


@pytest.fixture
def root_dir(request):
    path = request.config.rootdir
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return Path(path)


@pytest.fixture
def data_dir(root_dir):
    path = root_dir / "tests" / "data"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


@pytest.fixture
def data_input_dir(data_dir):
    path = data_dir / "inputs"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


@pytest.fixture
def data_conf_dir(data_dir):
    path: Path = data_dir / "conf"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


@pytest.fixture
def data_out_dir(data_dir):
    path: Path = data_dir / "outputs"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    return path


@pytest.fixture(
    params=[
        # palaiseau
        {
            "site": "palaiseau",
            "date": "2021-02-02",
            "radar": "basta",
            "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.7e13f3f243854ae8",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
            "config_file": "config_palaiseau_basta-parsivel-ws.toml",
        },
        {
            "site": "palaiseau",
            "date": "2024-01-02",
            "radar": "basta",
            "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.7e13f3f243854ae8",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
            "config_file": "config_palaiseau_basta-parsivel-ws.toml",
        },
        {
            "site": "palaiseau",
            "date": "2024-01-02",
            "radar": "basta",
            "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
            "disdro": "thies-lnm",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.11d3217867474e22",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
            "config_file": "config_palaiseau_basta-thies-ws.toml",
        },
        # lindenberg
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "mira",
            "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
            "disdro": "thies-lnm",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.ddeab96e6197478a",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_mira-thies.toml",
        },
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "mira",
            "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.1b0966f63b2d41f2",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_mira-parsivel.toml",
        },
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.70dd09553d13484d",
            "disdro": "thies-lnm",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.ddeab96e6197478a",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_rpg-thies.toml",
        },
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.70dd09553d13484d",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.1b0966f63b2d41f2",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_rpg-parsivel.toml",
        },
        # juelich
        {
            "site": "juelich",
            "date": "2024-02-08",
            "radar": "mira",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
        },
        # hyytiala
        {
            "site": "hyytiala",
            "date": "2023-10-15",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.191564170f8a4686",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.69dddc0004b64b32",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_hyytiala_rpg-parsivel.toml",
        },
    ]
)
def test_data_preprocessing(request):
    param = request.param
    yield param
