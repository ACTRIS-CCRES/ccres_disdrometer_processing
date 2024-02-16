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


@pytest.fixture
def test_data_preprocessing():
    return [
        {
            "site": "palaiseau",
            "date": "2021-02-02",
            "radar": "basta",
            "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.7e13f3f243854ae8",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
            "config_file": "config_palaiseau_basta-parsivel-ws.toml",
        }
    ]
