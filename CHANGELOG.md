# Changelog

## v1.0.2 (2026-10-23)

### Fix

- preprocess without weather data when the weather file is incomplete
- plot events that have exactly --min-points good timesteps
- correct the other flag meanings of the processed files
- give QC_hur flag meanings that match its values
- store instrument positions as scalars and read older files lightly
- robust AMS sampling detection and a single weather QC path
- close netCDF files after reading them
- pin xarray concat defaults that are about to change
- let process-ql --no-flag turn the quality flag off

### Refactor

- move the remaining standalone processing scripts to scripts/legacy
- remove functions nothing calls
- move unused standalone scripts out of the package

## v1.0.1 (2026-10-20)

### Fix

- remove invalid escape sequences and deprecated datetime.utcnow()

## v1.0.0 (2026-10-20)

### BREAKING CHANGE

- Python 3.11 and 3.12 are no longer supported (requires-python >= 3.13).
- NumPy 1.x is no longer supported.
- Installing pytmatrix-lte needs a Fortran compiler (gfortran).

### Feat

- require Python 3.13 or later
- pin numpy to >=2 now that pytmatrix-lte supports it
- replace pytmatrix with pytmatrix-lte to fix NumPy 2 compatibility

### Fix

- do not transpose Parsivel2 data_raw from files not written by cloudnetpy
- transpose Parsivel2 data_raw only for cloudnetpy < 1.94
- bundle logo PNG assets into the installed package
- reorder changelog entry and pin commitizen <4.11 for Python 3.9 support

## v0.3.5 (2026-09-17)

### Fix

- use correct  kwarg for curve_fit in get_y_fit_dd
- replace deprecated pandas frequency alias S with s
- exclude derived epsilon-* products when fetching files from Cloudnet

## 0.2.2 (2024-11-05)

- fix bug when weather station data have a 10 minutes resolution

## 0.2.1 (2024-10-30)

- update plots

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
