import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

dd_old = dd_prepro = xr.open_dataset(
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/20210202_palaiseau_preprocessing.nc"
)
dd_prepro = xr.open_dataset(
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/20210202_palaiseau_preprocessing_vnew.nc"
)
print(dd_old.time.values.shape)
print(dd_prepro.time.values.shape)


plt.figure()
plt.plot(dd_old.time, dd_old.VD.values[:, 5], color="blue")
plt.plot(
    dd_prepro.time, dd_prepro.VD.values[:, 5] - dd_old.VD.values[:, 5], color="green"
)
plt.show()
