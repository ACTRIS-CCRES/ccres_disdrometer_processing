# CCRES Disdrometer processing

[![pypi](https://img.shields.io/pypi/v/ccres_disdrometer_processing.svg)](https://pypi.org/project/ccres_disdrometer_processing)
[![documentation status](https://readthedocs.org/projects/ccres_disdrometer_processing/badge/?version=latest)](https://ccres_disdrometer_processing.readthedocs.io)
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

> [!WARNING]
> The code is not compatible with python 3.12 because of [pytmatrix](https://github.com/jleinonen/pytmatrix).

## Usage


This code is only compatible with data of Doppler Cloud Radar (DCR), Disdrometer (DD) and weather station using cloudnet netCDF format.

### Configuration file

A configuration file is needed to use the code. The configuration use `toml` format and several examples are available in this [project](https://github.com/ACTRIS-CCRES/ccres_disdro_config) in the `conf_stations` folder.

Here is an example of a configuration file:

```toml
title = "Configuration for the computation of the dcrcc monitoring (preprocessing and processing)"

[location]
SITE = "Jülich"
STATION = "JOYCE" # useful for plots

[methods]
FALL_SPEED_METHOD = "GunAndKinzer"
AXIS_RATIO_METHOD = "BeardChuang_PolynomialFit"
COMPUTE_MIE_METHOD = "pytmatrix"
REFRACTION_INDEX = [2.99645,1.54866] # complex index
RADAR_FREQUENCIES = [10.0e9, 24.0e9, 35.0e9, 94.0e9] # Hz
MAX_ALTITUDE_RADAR_DATA = 2500

[instrument_parameters]
DD_SAMPLING_AREA = 0.0054 # m^2 ; Parsivel2 sampling surface
DCR_DZ_RANGE = 150 # m ; height at which to compute Delta Z ; fill with the real value
RAIN_GAUGE_SAMPLING = 0 # mm ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
DD_ORIENTATION = 0 # degree, from North ; fill with the real value

[plot_parameters]
DCR_PLOTTED_RANGES = [150, 200, 300] # fill with the good values

[thresholds]
MAX_RR = 3  # mm/h
MIN_RAINFALL_AMOUNT = 3 # mm/episode
MAX_MEAN_WS = 7 # m/s ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
MAX_WS = 10 # m/s ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
MIN_TEMP = 2 # °C ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
MIN_HUR = 0 # min relative humidity : avoid cases with evaporation ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
MAX_HUR = 100 # max relative humidity : avoid fog, ... ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
DD_ANGLE = 45 # degree ; keep wind data at DD_ORIENTATION[pi] +- DD_ANGLE
MAX_INTERVAL = 60 # mn ; max interval between two tipping of the pluviometer, to "close" an event
MIN_DURATION = 180 # mn ; min duration of an event
PR_SAMPLING = 15  # mn ; ex CHUNK_THICKNESS ; period of averaging for AMS pr ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
DD_RG_MAX_PR_ACC_RATIO = 0.3 # ex ACCUMULATION_RELATIVE_ERROR ; max relative error in rain accumulation measurement, DD vs Rain gauge ; default value ; no weather available at Juelich but this value is used by the code only if weather is available
DD_FALLSPEED_RATIO = 0.3 # ex FALLSPEED_RELATIVE_ERROR ; relative difference between "theoretical" and DD fall speed

[nc_meta]
title = ""
summary = ""
id = ""
naming_authority = ""
comment = ""
creator_name = "ACTRIS-CCRES"
creator_email = "ccres_contact@listes.ipsl.fr"
creator_url = "https://ccres.aeris-data.fr"
creator_type = "institution"
creator_institution = ""
institution = ""
project = ""
publisher_name = ""
publisher_email = ""
publisher_url = ""
publisher_type = ""
publisher_institution = ""
contributor_name = ""
contributor_role = ""
cdm_data_type = ""
metadata_link = ""
```

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
        path/to/process_file.nc
```

If at list one event is found in the process file, the script will created the files:
- `path/to/summary_ql_00.png`
- `path/to/detailled_ql_00.png`


## License
- License: GNU Affero General Public License v3.0
- Documentation: https://ccres-disdrometer-processing.readthedocs.io.
