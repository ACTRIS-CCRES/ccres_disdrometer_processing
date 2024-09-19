# How to

## General diagram of the code

![Schema](../assets/Schema_fonctionnel_dcrcc-Page-1.drawio.png)



General diagram of the DCR calibration constant monitoring code
:::

## Different command line interfaces available in the package

.. list-table:: Commands
   :widths: 20 40
   :header-rows: 1

   * - Command
     - Description
   * - `preprocess`
     - Preprocess daily data from CLU daily instruments files : outputs a preprocessing file.
   * - `preprocess_ql`
     - Generates daily quicklooks from daily preprocessing file + other arguments (including conf file)
   * - `process`
     - Outputs a daily processing file from a daily preprocesing file and the neighbouring days' preprocessing files
   * - `process_ql`
     - Generates quicklooks focusing on detected rain events from daily processing file and the neighbouring day's preprocessing files + other arguments

- write a cli for event data extraction ? (for static plots)
- a cli for static plot from an events data collection ?

## Content of the config file that drives the data processing

Here is an example of a configuration file

```

title = "Configuration for the computation of the dcrcc monitoring (preprocessing and processing)"

[location]
SITE = "Palaiseau"
STATION = "SIRTA"  # useful for plots

[methods]
FALL_SPEED_METHOD = "GunAndKinzer"
AXIS_RATIO_METHOD = "BeardChuang_PolynomialFit"
COMPUTE_MIE_METHOD = "pytmatrix"
REFRACTION_INDEX = [2.99645, 1.54866]                # complex refractive index of water
RADAR_FREQUENCIES = [10.0e9, 24.0e9, 35.0e9, 94.0e9] # Hz
MAX_ALTITUDE_RADAR_DATA = 2500

[instrument_parameters]
DD_SAMPLING_AREA = 0.0054 # m^2 ; Parsivel2 sampling surface
DCR_DZ_RANGE = 300        # m ; height at which to compute Delta Z
RAIN_GAUGE_SAMPLING = 0.2 # mm
DD_ORIENTATION = 0        # degree, from North

[plot_parameters]
DCR_PLOTTED_RANGES = [100, 200, 300]

[thresholds]
MAX_RR = 3                   # mm/h
MIN_RAINFALL_AMOUNT = 3      # mm/episode
MAX_MEAN_WS = 7              # m/s ; maximum average wind over a "good" event
MAX_WS = 10                  # m/s ; max wind to keep a timestep
MIN_TEMP = 2                 # °C
MIN_HUR = 0                  # min relative humidity : avoid cases with evaporation
MAX_HUR = 100                # max relative humidity : avoid fog, ...
DD_ANGLE = 45                # degree ; keep wind data at DD_ORIENTATION[pi] +- DD_ANGLE
MAX_INTERVAL = 60            # mn ; max interval between two tipping of the pluviometer, to "close" an event
MIN_DURATION = 180           # mn ; min duration of an event
PR_SAMPLING = 15             # mn ; ex CHUNK_THICKNESS ; period of averaging for AMS pr
DD_RG_MAX_PR_ACC_RATIO = 0.3 # ex ACCUMULATION_RELATIVE_ERROR ; max relative error in rain accumulation measurement, DD vs Rain gauge
DD_FALLSPEED_RATIO = 0.3     # ex FALLSPEED_RELATIVE_ERROR ; relative difference between "theoretical" and DD fall speed

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

.. code-block::

    ccres-disdrometer-processing preprocess --disdro-file DISDRO_FILE [--ws-file WS_FILE] --radar-file RADAR_FILE --config-file CONFIG_FILE OUTPUT_FILE [-v VERBOSITY]

.. list-table:: Arguments
   :widths: 10 20 20 50
   :header-rows: 1

   * - Short
     - Long
     - Default
     - Description
   * -
     - `--disdro-file`
     -
     - CLU netCDF disdrometer file for the day to process
   * -
     - `--ws-file`
     -
     - CLU netCDF weather station file for the day to process
   * -
     - `--radar-file`
     -
     - Single date to be processed. Alternatively, `--start` and `--stop` can be defined
   * -
     - `--config-file`
     -
     - TOML configuration file suited for the input data (site, instruments, ...)
   * - `-v`
     -
     - 0
     - Verbosity
