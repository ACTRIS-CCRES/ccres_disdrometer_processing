"""Fixtures for pytest."""
import pytest
import requests

CLOUDNET_API_URL = "https://cloudnet.fmi.fi/api/"
CLOUDNET_API_RAW_URL = "https://cloudnet.fmi.fi/api/raw-files/"

TEST_DATA = [
    {
        "site": "palaiseau",
        "date": "2021-02-02",
        "radar": "basta",
        "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
        "disdro": "parsivel",
        "disdro-pid": "https://hdl.handle.net/21.12132/3.7e13f3f243854ae8",
        "meteo-availlable": True,
        "meteo": "weather-station",
        "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
    }
]


@pytest.fixture
def root_dir(request):
    return request.config.rootdir


@pytest.fixture
def data_dir(root_dir):
    return root_dir / "tests" / "data"


@pytest.fixture
def data_input_dir(data_dir):
    return data_dir / "inputs"


@pytest.fixture
def data_conf_dir(data_dir):
    return data_dir / "conf"


@pytest.fixture
def data_out_dir(data_dir):
    return data_dir / "outputs"


def pytest_sessionstart() -> None:
    """Will be call before running tests."""
    # Check if the Cloudnet API is available
    try:
        requests.get(CLOUDNET_API_URL)
    except requests.exceptions.ConnectionError:
        pytest.exit("Cloudnet API is not available")
