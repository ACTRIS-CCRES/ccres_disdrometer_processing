import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import datetime

f = xr.open_dataset(
    "/home/ygrit/Documents/dcrcc_data/lindenberg/RPG_file_model/230929_040001_P00_ZEN.LV1.NC"
)
print(f)
print(f.Time.values[0:50])

epoch_time = datetime.datetime(2001, 1, 1)
# print(f.Time.values[1], type(f.Time.values[1]))
# time = epoch_time + datetime.timedelta(seconds=f.Time.values[1])
# print(time)

print(epoch_time + datetime.timedelta(seconds=717652801))
print(f.Time.attrs)
print(list(f.keys()))
