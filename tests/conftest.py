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
            "output": {
                "preprocess": "palaiseau_2021-02-02_basta-parsivel-ws_preprocessed.nc",  # noqa E501
                "preprocessing_ql": {
                    "weather-overview": "palaiseau_2021-02-02_basta-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "palaiseau_2021-02-02_basta-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "palaiseau_2021-02-02_basta-parsivel-ws_processed.nc",  # noqa E501
                "process_ql": {
                    "summary": "palaiseau_2021-02-02_basta-parsivel-ws_process-summary",
                    "detailled": "palaiseau_2021-02-02_basta-parsivel-ws_process-detailled",  # noqa E501
                },
            },
        },
        # palaiseau
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
            "output": {
                "preprocess": "palaiseau_2024-01-02_basta-parsivel-ws_preprocessed.nc",  # noqa E501
                "preprocessing_ql": {
                    "weather-overview": "palaiseau_2024-01-02_basta-parsivel-ws_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "palaiseau_2024-01-02_basta-parsivel-ws_zh-preproc-overview.png",  # noqa E501
                },
                "process": "palaiseau_2024-01-02_basta-parsivel-ws_processed.nc",  # noqa E501
                "process_ql": {
                    "summary": "palaiseau_2024-01-02_basta-parsivel-ws_process-summary",
                    "detailled": "palaiseau_2024-01-02_basta-parsivel-ws_process-detailled",  # noqa E501
                },
            },
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
            "output": {
                "preprocess": "palaiseau_2024-01-02_basta-thies-ws_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "palaiseau_2024-01-02_basta-thies-ws_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "palaiseau_2024-01-02_basta-thies-ws_zh-preproc-overview.png",  # noqa E501
                },
                "process": "palaiseau_2024-01-02_basta-thies-ws_processed.nc",
                "process_ql": {
                    "summary": "palaiseau_2024-01-02_basta-thies-ws_process-summary",
                    "detailled": "palaiseau_2024-01-02_basta-thies-ws_process-detailled",  # noqa E501
                },
            },
        },
        # test change cloudnet L1 file format lat, lon alt become time dependant in 2025-08-28 # noqa E501
        {
            "site": "palaiseau",
            "date": "2025-09-13",
            "radar": "basta",
            "radar-pid": "https://hdl.handle.net/21.12132/3.643b7b5b43814e6f",
            "disdro": "thies-lnm",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.11d3217867474e22",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.739041931dac4de5",
            "config_file": "config_palaiseau_basta-thies-ws.toml",
            "output": {
                "preprocess": "palaiseau_2025-09-13_basta-thies-ws_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "palaiseau_2025-09-13_basta-thies-ws_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "palaiseau_2025-09-13_basta-thies-ws_zh-preproc-overview.png",  # noqa E501
                },
                "process": "palaiseau_2025-09-13_basta-thies-ws_processed.nc",
                "process_ql": {
                    "summary": "palaiseau_2025-09-13_basta-thies-ws_process-summary",
                    "detailled": "palaiseau_2025-09-13_basta-thies-ws_process-detailled",  # noqa E501
                },
            },
        },
        # lindenberg
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
            "disdro": "thies-lnm",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.ddeab96e6197478a",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_mira-thies.toml",
            "output": {
                "preprocess": "lindenberg_2023-09-22_mira-thies_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "lindenberg_2023-09-22_mira-thies_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "lindenberg_2023-09-22_mira-thies_zh-preproc-overview.png",  # noqa E501
                },
                "process": "lindenberg_2023-09-22_mira-thies_processed.nc",
                "process_ql": {
                    "summary": "lindenberg_2023-09-22_mira-thies_process-summary",
                    "detailled": "lindenberg_2023-09-22_mira-thies_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "lindenberg",
            "date": "2023-09-22",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.1b0966f63b2d41f2",
            "meteo-available": True,
            "meteo": "weather-station",
            "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
            "config_file": "config_lindenberg_mira-parsivel.toml",
            "output": {
                "preprocess": "lindenberg_2023-09-22_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "lindenberg_2023-09-22_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "lindenberg_2023-09-22_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "lindenberg_2023-09-22_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "lindenberg_2023-09-22_mira-parsivel_process-summary",
                    "detailled": "lindenberg_2023-09-22_mira-parsivel_process-detailled",  # noqa E501
                },
            },
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
            "output": {
                "preprocess": "lindenberg_2023-09-22_rpg-thies_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "lindenberg_2023-09-22_rpg-thies_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "lindenberg_2023-09-22_rpg-thies_zh-preproc-overview.png",  # noqa E501
                },
                "process": "lindenberg_2023-09-22_rpg-thies_processed.nc",
                "process_ql": {
                    "summary": "lindenberg_2023-09-22_rpg-thies_process-summary",
                    "detailled": "lindenberg_2023-09-22_rpg-thies_process-detailled",  # noqa E501
                },
            },
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
            "output": {
                "preprocess": "lindenberg_2023-09-22_rpg-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "lindenberg_2023-09-22_rpg-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "lindenberg_2023-09-22_rpg-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "lindenberg_2023-09-22_rpg-parsivel_processed.nc",
                "process_ql": {
                    "summary": "lindenberg_2023-09-22_rpg-parsivel_process-summary",
                    "detailled": "lindenberg_2023-09-22_rpg-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        # juelich
        {
            "site": "juelich",
            "date": "2024-02-08",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2024-02-08_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2024-02-08_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2024-02-08_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2024-02-08_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2024-02-08_mira-parsivel_process-summary",
                    "detailled": "juelich_2024-02-08_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "juelich",
            "date": "2021-12-02",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2021-12-02_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2021-12-02_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2021-12-02_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2021-12-02_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2021-12-02_mira-parsivel_process-summary",
                    "detailled": "juelich_2021-12-02_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "juelich",
            "date": "2021-12-03",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2021-12-03_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2021-12-03_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2021-12-03_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2021-12-03_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2021-12-03_mira-parsivel_process-summary",
                    "detailled": "juelich_2021-12-03_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "juelich",
            "date": "2021-12-04",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2021-12-04_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2021-12-04_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2021-12-04_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2021-12-04_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2021-12-04_mira-parsivel_process-summary",
                    "detailled": "juelich_2021-12-04_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "juelich",
            "date": "2021-12-05",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2021-12-05_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2021-12-05_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2021-12-05_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2021-12-05_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2021-12-05_mira-parsivel_process-summary",
                    "detailled": "juelich_2021-12-05_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        {
            "site": "juelich",
            "date": "2021-12-06",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "output": {
                "preprocess": "juelich_2021-12-06_mira-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "juelich_2021-12-06_mira-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "juelich_2021-12-06_mira-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "juelich_2021-12-06_mira-parsivel_processed.nc",
                "process_ql": {
                    "summary": "juelich_2021-12-06_mira-parsivel_process-summary",
                    "detailled": "juelich_2021-12-06_mira-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        # hyytiala
        {
            "site": "hyytiala",
            "date": "2024-08-02",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.191564170f8a4686",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.69dddc0004b64b32",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_hyytiala_rpg-parsivel.toml",
            "output": {
                "preprocess": "hyytiala_2024-08-02_rpg-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "hyytiala_2024-08-02_rpg-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "hyytiala_2024-08-02_rpg-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "hyytiala_2024-08-02_rpg-parsivel_processed.nc",
                "process_ql": {
                    "summary": "hyytiala_2024-08-02_rpg-parsivel_process-summary",
                    "detailled": "hyytiala_2024-08-02_rpg-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        # bucharest
        {
            "site": "bucharest",
            "date": "2024-03-25",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.90b1e5245b11487d",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.a75d4215f338412e",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_bucharest_rpg-parsivel.toml",
            "output": {
                "preprocess": "bucharest_2024-03-25_rpg-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "bucharest_2024-03-25_rpg-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "bucharest_2024-03-25_rpg-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "bucharest_2024-03-25_rpg-parsivel_processed.nc",
                "process_ql": {
                    "summary": "bucharest_2024-03-25_rpg-parsivel_process-summary",
                    "detailled": "bucharest_2024-03-25_rpg-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        # granada
        {
            "site": "granada",
            "date": "2024-03-09",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.20570e63f7b1496c",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.8f31e16545d14ff3",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_granada_rpg-parsivel.toml",
            "output": {
                "preprocess": "granada_2024-03-09_rpg-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "granada_2024-03-09_rpg-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "granada_2024-03-09_rpg-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "granada_2024-03-09_rpg-parsivel_processed.nc",
                "process_ql": {
                    "summary": "granada_2024-03-09_rpg-parsivel_process-summary",
                    "detailled": "granada_2024-03-09_rpg-parsivel_process-detailled",  # noqa E501
                },
            },
        },
        # galati
        {
            "site": "galati",
            "date": "2024-04-17",
            "radar": "rpg-fmcw-94",
            "radar-pid": "https://hdl.handle.net/21.12132/3.71dad3ea36ab476a",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.070929c5502747f6",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_galati_rpg-parsivel.toml",
            "output": {
                "preprocess": "galati_2024-04-17_rpg-parsivel_preprocessed.nc",
                "preprocessing_ql": {
                    "weather-overview": "galati_2024-04-17_rpg-parsivel_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "galati_2024-04-17_rpg-parsivel_zh-preproc-overview.png",  # noqa E501
                },
                "process": "galati_2024-04-17_rpg-parsivel_processed.nc",
                "process_ql": {
                    "summary": "galati_2024-04-17_rpg-parsivel_process-summary",
                    "detailled": "galati_2024-04-17_rpg-parsivel_process-detailled",  # noqa E501
                },
            },
        },
    ]
)
def test_data_preprocessing(request):
    param = request.param
    yield param


@pytest.fixture(
    params=[
        # juelich
        {
            "site": "juelich",
            "radar": "mira-35",
            "radar-pid": "https://hdl.handle.net/21.12132/3.0366fa69504f4bd6",
            "disdro": "parsivel",
            "disdro-pid": "https://hdl.handle.net/21.12132/3.2a1ca46ed70c4929",
            "meteo-available": False,
            "meteo": "weather-station",
            "meteo-pid": "",
            "config_file": "config_juelich_mira-parsivel.toml",
            "list_dates": ["2021-12-03", "2021-12-04", "2021-12-05", "2021-12-06"],
            "output": {
                "preprocess_tmpl": "juelich_{}_mira-parsivel_preprocessed.nc",
                "process_tmpl": "juelich_{}_mira-parsivel_processed_ndays.nc",
                "process_ql": {
                    "summary_tmpl": "juelich_{}_mira-parsivel_processed_summary",
                    "detailled_tmpl": "juelich_{}_mira-parsivel_processed_detailled",  # noqa E501
                },
            },
        },
    ]
)
def test_data_processing_ndays(request):
    """Data to test processing on several days."""
    param = request.param
    yield param


# test for cli option --no-meteo at processing level
@pytest.fixture(
    params=[
        # palaiseau
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
            "output": {
                "preprocess": "palaiseau_2024-01-02_basta-parsivel-ws_preprocessed.nc",  # noqa E501
                "preprocessing_ql": {
                    "weather-overview": "palaiseau_2021-02-02_basta-parsivel-ws_preproc-weather-overview.png",  # noqa E501
                    "zh-overview": "palaiseau_2021-02-02_basta-parsivel-ws_zh-preproc-overview.png",  # noqa E501
                },
                "process": "palaiseau_2024-01-02_basta-parsivel-ws_processed_no_meteo_cli.nc",  # noqa E501
            },
        },
    ]
)
def test_data_processing_cli_option_no_meteo(request):
    """Data to test processing on several days."""
    param = request.param
    yield param
