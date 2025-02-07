## Visualizations

### Daily quicklooks

The command line ***preprocess_ql*** enables to display a set of quicklooks from daily preprocessing file.

2 panels are displayed in output :

- A comparison between Z modeled from disdrometer and Z got from DCR at the range specified in configuration (**DCR_DZ_RANGE**)
- A panel with :
* 2D views of **Zdcr** and **DVdcr** up to an altitude given in configuration (**MAX_ALTITUDE_RADAR_DATA**, default = 2500m)
* Time series for met variables if weather data is provided as input of the preprocessing
* Precipitation size distribution and fit curve and comparison with Gun and Kinzer relationship.

Here is an example :

 ![Schema](../assets/_QLdaily.png)

 A downgraded version of the right panel is displayed when met variables are not available.



 ###  "Rain event quicklooks"

 The command line ***process_ql*** enables to display a set of quicklooks from daily preprocessing file.

2 panels are displayed in output :

- A summary of the event, representing a PDF of **Delta_Z** with its median and standard deviation, a scatterplot **Zdcr** vs. **Zdd** with the linear regression between the two data sources over the points which passed the quality controls, and a summary of event characteristics (Duration, rainfall amount and mean reflectivity bias)

- A more detailed panel with 5 subplots :
* 2D Zdcr data with a focus on the event time period ;
* Comparison between disdrometer forward-modeled reflectivity and DCR reflectivity at the comparison range specified in configuration file ;
* Time series for **Delta_Z** ;
* Time series for rainfall accumluation ;
* A time series that represents the periods where the different Quality Controls are verified or not during the event.
All these data are plotted for the time period $ \left[ \text{Event start time - 1h}, \text{Event end time + 1h}\right] $


 Here is an example :

  ![Schema](../assets/_QLprocessed.png)


### Monitoring products

The creation of figures for long-term monitoringis not implemented in this package, but is based on the files created by the processing command.

For a given long time period (e.g. several months, one year), the monitoring consists in sweeping all the daily processing files to gather all the events on the time period, keeping the events which satisfy the Quality Flags and plotting valuable information concerning these events.
We provide two plots :
- A pdf for **Delta_Z** over the whole time period i.e. we display all the timesteps which satisfy the QC inside all the gathered events
- A time series where each identified event is displayed as a boxplot representing its statistics (median, quartiles).

The time series with boxplots is helpful to identify potential drifts or sudden biases in **Delta_Z** and to launch investigation on the instrumental setup to identify a potential problem (see example below for Palaiseau, with a bias identified in 2022. This diagnosis enabled to identify a component dysfunction and to solve it)

For CCRES labeled stations, these two products are shared in the CCRES website : *https://ccres.aeris-data.fr/en/data-visualization/* (in the tab "long-term monitoring")

![Schema](../assets/palaiseau_basta-parsivel-ws_calib-monitoring_all.png)

![Schema](../assets/palaiseau_basta-parsivel-ws_pdf_all.png)
