# Changelog

## 0.2.0 (2024-10-22)

- improvement for getting files from cloudnet api for testing the code + management of several instruments of the same type at a specific site
- corrected bug in rain event selection for days without any rain record (processing file creation was aborted before that)
- integrated the condition of minimum rainfall amount in the rain event selection algorithm
- adjustment of the processing for stations with weather station data @ sampling > 1mn (eg Lindenberg) : specific way to compute some variables and quality checks
- correction for metadata concerning weather data availability in processing output file
- fixed various bugs in preprocessing : wrong behaviours of the code in particular cases (partially missing weather data in Cloudnet files ; downgraded processing when less than three days are given as input ; bordering events ; abortion of the code when a detected event has no timestep with QC OK ; ...)
- additional QC on relative humidity when weather is available
- disdrometer orientation is now taken into account to compute QC on wind direction when weather is provided

## 0.1.0 (2023-07-10)

- First release on PyPI.
