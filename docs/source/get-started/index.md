# Usefulness of the code

***ccres_disdrometer_processing*** is a Python package used to process Doppler Cloud Radar (DCR) and disdrometer data, implemented in the framework of Centre for Cloud REmote Sensing (CCRES), within the infrastructure ACTRIS.

The package is aimed to monitor Doppler Cloud Radar calibration constant over time. This monitoring consists in a comparison between DCR reflectivity data at a certain range (that is chosen depending on the instrument as a tradeoff between rain distribution representativity and radar saturation effects) and disdrometer forward-modeled reflectivity data during well chosen rain events.

The forward-modeling is based on a scattering model that converts, at each time period, the drop size distribution (DSD) measured by the disdrometer into reflectivity measurements.

The methodology implemented is based on the work of (Myagkov et al. 2020) : *https://doi.org/10.5194/amt-13-5799-2020*



# Get started

```python
pip install ccres_disdrometer_processing
```
