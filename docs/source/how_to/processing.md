## Further information on processing


### Generalities

Processing consists in identifying significant rain event periods and evaluating reflectivity differences (at a relevant range for the studied station/radar) over these periods in order to monitor them.

Processed files are produced daily, whether there is a rain event identified or not.


Processing step is made by the command line *ccres_disdrometer_processing process* ; it executes the file ***preprocessed_file2processed.py*** that can be found in the *processing* sub-directory.

### Block diagram of the processing

 ![Schema](../assets/Schema_fonctionnel_dcrcc-Page-3.drawio.png)
Focus on the preprocessing step (needs to be slightly modified)




### Recap of the inputs and outputs for the processing :

Recap of the inputs for the processing :
[To process day D ]
- 3 daily outputs from preprocessing for days D-1, D, D+1
- configuration file for the station to process
- Path where to save the output file
- option *no_meteo* to downgrade the processing (i.e. not to use weather station data for some reason, even when it is available and used for preprocessing)

The output is a netCDF file, with three dimensions :
- *range*, the vector of altitudes at which DCR data and Delta between DCR and disdrometer-modeled reflectivity data are provided
- two time bases :
* *time*, the same time vector as the one used in preprocessing output files, from 0:00 to 23:59 (UTC) with a 1-minute resolution
* *events*, a dimension dedicated to the storage of identified significant rain events (see later). The shape corresponds to the number of events identified for the day to process. Events has a dimension of 0 if no event is identified.

### Content of the output file

The following variables can be found in the output processing file :

| Variable                          | Dimensions      | Description |
|-----------------------------------|----------------|-----------|
| time                              | time           | time       |
| range                             | range          | ranges at which to consider DCR/disdrometer data comparison       |
| Zdcr                              | time, range    | DCR reflectivity at the ranges specified in configuration file (3 in general) |
| DVdcr                             | time, range    | DCR Doppler velocity at the ranges specified in configuration file|
| Zdd                               | time           | Disdrometer forward-modeled reflectivity to use for comparison with DCR data |
| fallspeed_dd                      | time           | Average droplet fall speed seen by the disdrometer |
| Delta_Z                           | time, range    | Difference between DCR and disdrometer-modeled reflectivity (dBZ) |
| flag_event                        | time           | Flag to describe if a timestep belongs to a detected rainfall event |
| ams_cp_since_event_begin          | time           | Pluviometer rain accumulation (in mm) since last event start |
| disdro_cp_since_event_begin       | time           | Disdrometer rain accumulation (in mm) since last event start |
| QF_rainfall_amount                | time           | Quality flag for minimum rainfall amount |
| QC_pr                             | time           | Quality check for rainfall rate |
| QC_vdsd_t                         | time           | Quality check for coherence between fall speed and DSD |
| QC_ta                             | time           | Quality check for air temperature |
| QC_ws                             | time           | Quality check for wind speed |
| QC_wd                             | time           | Quality check for wind direction |
| QC_hur                            | time           | Quality check for relative humidity |
| QF_rg_dd                          | time           | Quality flag for discrepancy between rain gauge and disdrometer |
| QC_overall                        | time           | Overall quality check |
| events                            | events         | Dimension for the storage of identified events       |
| start_event                       | events         | Event start epoch |
| end_event                         | events         | Event end epoch |
| event_length                      | events         | Event duration |
| rain_accumulation                 | events         | AMS rain accumulation (in mm) over the whole event |
| QF_rain_accumulation              | events         | Flag on event rain accumulation |
| QF_rg_dd_event                    | events         | Flag on deviation between rain gauge and disdrometer |
| nb_dz_computable_pts              | events         | Number of timesteps at which Delta Z can be computed, for a given event|
| QC_vdsd_t_ratio                   | events         | Ratio of timesteps where check on relationship between fall speed and DSD is good |
| QC_pr_ratio                       | events         | Ratio of timesteps where precipitation rate QC is good |
| QC_ta_ratio                       | events         | Ratio of timesteps where air temperature QC is good |
| QC_ws_ratio                       | events         | Ratio of timesteps where wind speed QC is good |
| QC_wd_ratio                       | events         | Ratio of timesteps where wind direction QC is good |
| QC_hur_ratio                      | events         | Ratio of timesteps where relative humidity QC is good |
| QC_overall_ratio                  | events         | Ratio of timesteps where all checks are good |
| good_points_number                | events         | Number of timesteps where all checks are good |
| dZ_mean                           | events         | Average value of Delta Z for good timesteps |
| dZ_med                            | events         | Median value of Delta Z for good timesteps |
| dZ_q1                             | events         | First quartile of Delta Z distribution for good timesteps |
| dZ_q3                             | events         | Third quartile of Delta Z distribution for good timesteps |
| dZ_min                            | events         | Minimum value of Delta Z for good timesteps |
| dZ_max                            | events         | Maximum value of Delta Z for good timesteps |
| reg_slope                         | events         | Slope of the linear regression Zdd/Zdcr for each event |
| reg_intercept                     | events         | Intercept of the linear regression Zdd/Zdcr for each event |
| reg_score                         | events         | R-squared of the linear regression Zdd/Zdcr for each event |
| reg_rmse                          | events         | RMSE of the linear regression Zdd/Zdcr for each event |
| weather_data_used                 |                | [0 or 1] : use of weather data for processing ? |



### Processing steps


***Step 0*** : concatenation of the preprocessing data files used as input (days D-1, D, D+1)

This operation is made by the function **merge_preprocessed_data()**.

***Step 1*** : significant rain event selection

This selection is made by the function **rain_event_selection()**.

The preprocessed data from day D-1 is included in the input data for rain event selection to avoid that a subset of an event detected during the processing of day D-1 is detected again during the processing of day D.
Two variants of the rain event selection are used, depending on if weather data is available :
- If we have weather data available after preprocessing, the algorithm is based on the rain accumulation data "seen" by the pluviometer.
- Otherwise, or if *no_meteo* option is True, we use disdrometer rain accumulation data. But we have no insight on the revelance of this source of data (in general, pluviometer rain accumulation data is more accurate).

The criteria for rain event selection are the following :

| Variables                            | Thresholds        | Objective(s)                                                                                     |
|--------------------------------------|----------------|-----------------------------------------------------------------------------------------------|
| Event duration                      | > 3h           | Ensure that we have a significant event i.e. robust statistics on Delta Z |
| Rain accumulation                        | > 3mm          | Same                                                                                          |
| Maximum time between two consecutive rain records | 60mn          | Ensure rain event continuity                                                         |

The efficiency of the algorithm as it is implemented now is probablu low. The implementation could be reviewed to enhance the efficiency, hopefully this step is not very time-consuming so this problem is not critical.

The algorithm outputs two lists, containing begin and end dates of the identified events. It outputs empty lists if no events satisfying the criteria are identified.


***Step 2 :*** computation of output variables

-  **Time variables produced from preprocessing file variables :**
    *  **Zdcr**, **Dvdcr** : extraction of DCR data on a small subset of range values (specified in configuration files)
    * **Zdd** : $10 * log(Zdcr)$
    * **fallspeed_dd** : 1-minute average droplet fall speed
    * **Delta_Z** := Zdcr – Zdd (i.e. calculé pour les valeurs de range spécifiées en configuration)
    * **flag_event** : 1 if the timestep belong to an identified event, 0 otherwise
    * **ams_cp_since_event_begin** / **disdro_cp since_event_begin** : rain accumulation since last start of an event ( useful for vizualisations notably). **ams_cp_since_event_begin** is a series of NaN if no weather station data is provided.

- **Computation of "*Quality Checks*" time series**

    These variables are filters that we used to determine at which time steps it is relevant to keep **Delta_Z** values for the monitoring of the calibration. They have a 1-minute sampling.

    * **QC_pr** : Control on the rainfall rate. Aim is to remove time periods with heavy rain to limit the risk or DCR saturation or wet radome. Whether pluviometer data is provided or not, the computation is the same and is based on disdrometer rainfall rate, which is available at 1-minute sampling. The threshold is given in configuration file, by default 3mm/h.


    * **QC_vdsd_t** : Control on the precipitation size distribution (*PSD*) given by the disdrometer. Aim is to check the consistency between droplets fall speed and drop size distribution, to remove snow situations.
    The values compared are :
    - on one hand, the weighted average of droplet fall speed seen by the disdrometer @ 1-minute, computed from drop size distribution and a fall speed model (Gun and Kinzer) ;
    - on the other hand, the average droplet fall speed computed from the disdrometer speed distribution.

    * ***If weather data is provided***, further quality control can be performed on weather variables :
    **QC_ta**, **QC_hur**, **QC_ws**, **QC_wd** : checks based on air temperature, relative humidity, wind speed and wind direction.
    The thresholds used are given in configuration file for each station, for the moment the default values are :
        - température > 2° (remove snow cases)
        - relative humidity ∈ [80,99] % : avoid cases with evaporation (which induce modification of the droplet size distribution between the ground and the radar range) and fog
        - wind speed < 10m/s
        - wind direction : +- 45° from the normal to disdrometer optical axis (because the disdrometer operates optimally for droplets perpendicular to the optical axis)

        To compute these quality controls :

        * (1) ***if weather data has a 1-minute sampling***, the computation is easy, with a direct comparison with the threshold

        * (2) ***if weather data has a lower sampling*** (e.g., 10 minute @ Lindenberg), we use the closest value of the corresponding variable.


    * **Global quality control to sum up all the quality checks : QC_overall**

        with weather data : **QC_overall** = **QC_pr** & **QC_vdsd_t** & **QC_ta** & **QC_ws** & **QC_wd** & **QC_hur**
        without weather data : **QC_overall** = **QC_pr** & **QC_vdsd_t**

The above table sums up the quality controls and thresholds implemented :

| Variables                     | Limits                | With WS and DD | Only with DD | Objectives                               |
|-------------------------------|-----------------------|----------------|--------------|------------------------------------------|
| Air temperature               | > 2°C                | ✅             | ❌           | Remove solid precipitation               |
| Relative humidity             | > 80% and < 98%      | ✅             | ❌           | Avoid fog cases, evaporation             |
| Wind speed                    | < 1 h                | ✅             | ❌           | Ensure rain continuity                   |
| Wind direction                | < 30%                | ✅             | ❌           | Quality control on disdrometer measurement |
| Relationship fall speed/drop size | < 30% vs Gun and Kinzer | ✅          | ✅           | Ensure robustness of Delta Z statistics  |
| Precipitation rate            | < 3 mm/h             | ✅             | ❌           | Remove heavy rain cases                  |



- **Storage of macroscopic information on detected rain event**

    2 possible scenarii :
    1) at least 1 event **ending at day D** is detected ;
    2) no event ending at day D is detected

    In both cases, a dimension "***events***" is initialized in the output Dataset, whose length is the number of detected events.

    1) ***events*** is empty and et related variables (indexed on this dimension) are empty.
    2) The following information is computed and stored :
        - **start_event**, **end_event** : begin and end dates of detected events
        - **event_length** : event duration in minutes
        - **rain_accumulation** / **QF_rain accumulation** : cumulated rainfall amount over the event / flag to ensure that the rainfall amount is significant enough (default threshold : 3mm). In latest version of the code, this control on rainfall amount is directly implemented in the rain event selection algorithm, so that detected events with lower amount than the threshold are thrown automatically. The variable **QF_rain accumulation** could be deleted.
        - **QF_rg_dd_event** : boolean, checks if the relative error between disdrometer and pluviometer rainfall amount is lower than a defined threshold (default : 30%, in configuration file). Aim is to flag events with high discrepancies between pluviometer and disdrometer when weather data is provided, to have a critical look on disdrometer data reliability.
        - **nb_dz_computable_points** : number of timesteps for which the variable **Delta_Z** has a finite value at the comparison range (**DCR_DZ_RANGE**) given in configuration file for each station and instrument setup.  ***The choice of this comparison range follows a logic : it must not be "too high" to ensure representativeness of Droplet Size distribuution and minimize Delta_Z variability, and it must not be "too low" so that we avoid radar antenna near field effects. One needs to look at the DCR and disdrometer-modeled reflectivity data during significant rain periods in order to find a good trade-off for this comparison range.***
        - **QC_[var]_ratio**: ratio of timesteps for which the quality control for the variable *[var]* is satisfied.
        - **good_points_number** : number of timesteps for which **QC_overall** is satisfied and with a finite value of **Delta_Z** (so, we have necessarily **good_points_number**  $\leq$  **nb_dz_cmputable_points**). It corresponds to the number of data points kept for **Delta_Z** analysis and monitoring i.e. the number of timesteps used to compute **Delta_Z** statistics over the event.
        - **dZ_min** / **dZ_max** / **dZ_med** / **dZ_mean** / **dZ_q1** / **dZ_q3** : min and max values, mean and quartiles for **Delta_Z** distribution over the event
        - **reg_slope** / **reg_intercept** / **reg_score** / **reg_rmse** : statistics of the linear regression **Zdcr** vs. **Zdd** (for the subset of points kept for the analysis i.e. with QC OK)



-  **Add of attributes**

    See function *add_attributes()*





### Considerations for the monitoring :

First aim of the code is to provide long-term DCR calibration monitoring over time (for all the period of labeling of a station).

We can sweep all the events detected in daily prcessing files to gather all the significant rain events on the period for which we want to monitor the DCR calibration. Inside this set of events saved in the daily processing outputs, we only keep the subset of events which satisfy two conditions :
- **QF_rg_dd_event** is verified (quality flag on consistency between disdrometer and pluviometer rainfall amounts, when weather is provided )
- **good_points_number**  $\geq 50$ : ensure robustness of **Delta_Z** distribution over an event.

    **Remark :** *We may add a variable to flag this criterion directly in the processing files. A field called* **MIN_POINTS** *is already set in the station configuration files but is not used in the code yet.*


- Get tables for Quality checks and flags from CCRES presentations


- Display a list of variable, dimensions, units in the netCDF daily output files




- Section for monitoring products
