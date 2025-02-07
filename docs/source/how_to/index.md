# How to

## General diagram of the code

 ![Schema](../assets/Schema_fonctionnel_dcrcc-Page-1.drawio.png)
General diagram of the DCR calibration constant monitoring code

 <!-- <img src="/source/assets/Schema_fonctionnel_dcrcc-Page-1.drawio.png" alt="General diagram of the DCR calibration constant monitoring code" width="500" height="600">
-->

:::

## Different command lines available in the package

:::

   | **Command** | **Description** |
   |------|------|
   |   `preprocess`   |   Preprocess daily data from CLU daily instruments files : outputs a preprocessing file   |
   |   `preprocess_ql`   |   Generates daily quicklooks at specified paths from daily preprocessing file + a configuration file   |
   |   `process`   |   Outputs a daily processing file from a daily preprocesing file and the neighbouring days' preprocessing files   |
   |   `process_ql`   |   Generates quicklooks focusing on detected rain events at specified paths from daily processing file and the neighbouring day's preprocessing files + configuration file   |
   |      |      |



## Content of the config file that drives the data processing

Here is an example of a configuration file (Palaiseau, with a *BASTA* radar and a *Thies* disdrometer)

```

title = "Configuration for the computation of the dcrcc monitoring (preprocessing and processing) at Palaiseau"

[location]
SITE = "Palaiseau"
STATION = "SIRTA" # useful for plots

[methods]
FALL_SPEED_METHOD = "GunAndKinzer"
AXIS_RATIO_METHOD = "BeardChuang_PolynomialFit"
COMPUTE_MIE_METHOD = "pytmatrix"
REFRACTION_INDEX = [[2.99645,1.54866], [2.99645,1.54866], [2.99645,1.54866], [2.99645,1.54866]] # complex index
RADAR_FREQUENCIES = [10.0e9, 24.0e9, 35.0e9, 94.0e9] # Hz
MAX_ALTITUDE_RADAR_DATA = 2500

[instrument_parameters]
AU = 965
DD_SAMPLING_AREA_DEFAULT = 0.0046
DD_SAMPLING_AREA = 0.004439 # m^2 ; = SAMPLING_AREA_DEFAULT*AU/1000
DCR_DZ_RANGE = 300 # m ; height at which to compute Delta Z
DD_ORIENTATION = 0 # degree, from North

[plot_parameters]

[thresholds]
MAX_RR = 3  # mm/h
MIN_RAINFALL_AMOUNT = 3 # mm/episode
MAX_WS = 10 # m/s ; max wind to keep a timestep
MIN_TEMP = 2 # °C
MIN_HUR = 80 # min relative humidity : avoid cases with evaporation
MAX_HUR = 99 # max relative humidity : avoid fog, ...
DD_ANGLE = 45 # degree ; keep wind data at DD_ORIENTATION[pi] +- DD_ANGLE
MAX_INTERVAL = 60 # mn ; max interval between two tipping of the pluviometer, to "close" an event
MIN_DURATION = 180 # mn ; min duration of an event
PR_SAMPLING = 15  # mn ; ex CHUNK_THICKNESS ; period of averaging for AMS pr
DD_RG_MAX_PR_ACC_RATIO = 0.3 # ex ACCUMULATION_RELATIVE_ERROR ; max relative error in rain accumulation measurement, DD vs Rain gauge
DD_FALLSPEED_RATIO = 0.3 # ex FALLSPEED_RELATIVE_ERROR ; relative difference between "theoretical" and DD fall speed
MIN_POINTS = 50

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


## Preprocessing command


    ccres-disdrometer-processing preprocess --disdro-file DISDRO_FILE [--ws-file WS_FILE] --radar-file RADAR_FILE --config-file CONFIG_FILE OUTPUT_FILE [-v VERBOSITY]

| **Short** | **Long** | **Default** | **Description** |
|------|------|------|------|
|      |   `--disdro-file`   |      |   CLU netCDF disdrometer file for the day to process   |
|      |   `--ws-file`   |   None   |   CLU netCDF weather station file for the day to process   |
|      |   `--radar-file`   |      |   CLU netCDF DCR file for the day to process   |
|      |   `--config-file`   |      |   TOML configuration file suited for the input data (site, instruments, ...)   |
|   `-v`   |      |   0   |   Verbosity   |
|      |      |      |      |

## Preprocessing quicklooks command

    ccres-disdrometer-processing preprocess_ql FILE OUTPUT_QL_OVERVIEW OUTPUT_QL_OVERVIEW_ZH --config-file CONFIG_FILE

| **Short** | **Long** | **Default** | **Description** |
|------|------|------|------|
|      |   `--config-file`   |      |   TOML configuration file suited for the input data (site, instruments, ...)   |
|      |   `file`   |      |   Output of preprocessing for the day to be displayed   |
|      |   `output-ql-overview`   |      |   Path to save the first panel of quicklooks (overview and met variables)   |
|      |   `output-ql-overview-zh`   |      |   Path to save the second panel of quicklooks (time series of DCR and DD reflectivity)   |
|      |      |      |      |


## Processing command


    ccres-disdrometer-processing process [--yesterday YESTERDAY_FILE] --today TODAY_FILE [--tomorrow TOMORROW_FILE] --config-file CONFIG_FILE OUTPUT_FILE [-v VERBOSITY]

| **Short** | **Long** | **Default** | **Description** |
|------|------|------|------|
|      |   `--yesterday`   |   None   |   Output of preprocessing for the day before the day to be processed   |
|      |   `--today`   |      |   Output of preprocessing for the day to be processed   |
|      |   `--tomorrow`   |   None   |   Output of preprocessing for the day before the day to be processed   |
|      |   `--config-file`   |      |   TOML configuration file suited for the input data (site, instruments, ...)   |
|      |   `--no-meteo`   |   False   |   Boolean : set to True to downgrade the processing i.e. to dispense with weather data even if it is provided in input preprocessing files   |
|   `-v`   |      |   0   |   Verbosity   |
|      |      |      |      |

## Processing quicklooks command

    ccres-disdrometer-processing process_ql PROCESS_FILE --preprocess-yesterday PREPROCESS_YESTERDAY --preprocess-today PREPROCESS_TODAY --preprocess-tomorrow PREPROCESS_TOMORROW --prefix-output-ql-summary PREFIX_OUTPUT_QL_SUMMARY --prefix-output-ql-detailled PREFIX_OUTPUT_QL_DETAILLED OUTPUT_QL_OVERVIEW_ZH --config-file CONFIG_FILE

| **Short** | **Long** | **Default** | **Description** |
|------|------|------|------|
|      |   `--process-yesterday`   |      |   Output of processing for the day before the day  for which we want to plot rain events   |
|      |   `--process-today`   |      |   Output of processing for the day for which we want to plot rain events   |
|      |   `--preprocess-yesterday`   |      |   Output of preprocessing for the day before the day  for which we want to plot rain events   |
|      |   `--preprocess-today`   |      |   Output of preprocessing for the day  for which we want to plot rain events   |
|      |   `--preprocess-tomorrow`   |      |   Output of preprocessing for the day before the day  for which we want to plot rain events   |
|      |   `--prefix-output-ql-summary`   |      |   Path to save the first panel of quicklooks (summary panel)   |
|      |   `--prefix-output-ql-detailled`   |      |   Path to save the second panel of quicklooks (panel with detaileld analysis)   |
|      |   `--config-file`   |      |   TOML configuration file suited for the input data (site, instruments, ...)   |
|      |   `--flag`   |      |   If True, plots only events which pass the Quality Flags specified in processing file |
|      |   `--min-points`   |      |   Criterion for minimum number of timesteps on which events statistics are computed to plot the event  |
|      |      |      |      |
