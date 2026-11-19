# CCRES Disdrometer processing

[![pypi](https://img.shields.io/pypi/v/ccres_disdrometer_processing.svg)](https://pypi.org/project/ccres_disdrometer_processing)
[![documentation status](https://img.shields.io/readthedocs/ccres-disdrometer-processing/latest)](https://ccres-disdrometer-processing.readthedocs.io)
[![license](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![github actions](https://github.com/ACTRIS-CCRES/ccres_disdrometer_processing/actions/workflows/ci.yaml/badge.svg?branch=main)
[![codecov](https://codecov.io/gh/ACTRIS-CCRES/ccres_disdrometer_processing/graph/badge.svg?token=5KJ06AQE3D)](https://codecov.io/gh/ACTRIS-CCRES/ccres_disdrometer_processing)
[![Issue Badge](https://img.shields.io/github/issues/ACTRIS-CCRES/ccres_disdrometer_processing)](https://github.com/ACTRIS-CCRES/ccres_disdrometer_processing/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/ACTRIS-CCRES/ccres_disdrometer_processing)](https://github.com/ACTRIS-CCRES/ccres_disdrometer_processing/pulls)

# CCRES disdrometer processing

This code provides tools to process disdrometer data from [ACTRIS-CCRES](https://ccres.aeris-data.fr)/[ACTRIS-cloudnet](https://cloudnet.fmi.fi) to monitor the calibration of doppler cloud radars (13, 35, 94 Ghz).

## Installation

```bash
pip install numpy wheels
pip install ccres_disdrometer_processing
```

> [!NOTE]
> Python 3.13 or later is required. The T-matrix computations use [rustmatrix](https://github.com/swnesbitt/rustmatrix), installed from prebuilt wheels on Linux (glibc 2.28 or later), macOS and Windows; other platforms need a Rust toolchain to build it.

## Usage


This code is only compatible with data of Doppler Cloud Radar (DCR), Disdrometer (DD) and weather station using cloudnet netCDF format.

### Configuration file

A configuration file is needed to use the code. The configuration use `toml` format and several examples are available in this [project](https://github.com/ACTRIS-CCRES/ccres_disdro_config) in the `conf_stations` folder.

Every parameter is read in this file, the code holds no default value : each command checks the file first and lists every missing or unknown key. The sections are described in the documentation (*How to*, configuration file), with a complete example.

## Preprocessing

This steps is used to preprocess the DCR and DD data.

```bash
ccres_disdrometer_processing preprocess \
        --disdro-file path/to/disdro.nc \
        --radar-file path/to/radar.nc \
        --ws-file path/to/weather_station.nc \
        --config-file path/to/config.toml \
        path/to/preprocess_file.nc
```

Using the created netCDF, you can create vizualisation of the data.

```bash
ccres_disdrometer_processing preprocess-ql \
        --config-file path/to/config.toml \
        path/to/preprocess_file.nc \
        path/to/weather_overview_ql.png \
        path/to/zh_overview_ql.png

```

# Processing

The processing step allow to identify rain events compatible with the checking of DCR calibration. To get the best results, this step needs the preprocess files of the day before and the day after the day to process.

```bash
ccres_disdrometer_processing process \
        --config-file path/to/config.toml \
        --yesterday path/to/yesterday_preprocess_file.nc \
        --today path/to/today_preprocess_file.nc \
        --tomorrow path/to/tomorrow_preprocess_file.nc \
        path/to/process_file.nc
```

Using the created netCDF, you can create vizualisation of the data. As we don't know how many rain events are detected you need to provide prefix name of the output quicklook files.

```bash
ccres_disdrometer_processing process-ql \
        --config-file path/to/config.toml \
        --preprocess-yesterday path/to/yesterday_preprocess_file.nc \
        --preprocess-today path/to/today_preprocess_file.nc \
        --preprocess-tomorrow path/to/tomorrow_preprocess_file.nc \
        --prefix-output-ql-summary path/to/summary_ql \
        --prefix-output-ql-detailled path/to/detailled_ql \
        --process-today path/to/process_file.nc
```

If at list one event is found in the process file, the script will created the files:
- `path/to/summary_ql_00.png`
- `path/to/detailled_ql_00.png`


## License
- License: GNU Affero General Public License v3.0
- Documentation: https://ccres-disdrometer-processing.readthedocs.io.
