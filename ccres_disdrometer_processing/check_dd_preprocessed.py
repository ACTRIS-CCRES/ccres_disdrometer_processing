import matplotlib.pyplot as plt
import xarray as xr

dd_old = dd_prepro = xr.open_dataset(
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/20210202_palaiseau_preprocessing.nc"
)
dd_prepro = xr.open_dataset(
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/20210202_palaiseau_preprocessing_VNEW.nc"
)
print(dd_old.time.values.shape)
print(dd_prepro.time.values.shape)

print(dd_old.psd.values)

print(dd_prepro.VD.dims)

from scattering import compute_fallspeed

print(compute_fallspeed(dd_prepro.size_classes)[5])


plt.figure()
plt.plot(dd_old.time, dd_old.VD.values[:, 5], color="blue")
plt.plot(dd_prepro.time, dd_prepro.VD.values[:, 5], color="red")
plt.plot(
    dd_prepro.time, dd_prepro.VD.values[:, 5] - dd_old.VD.values[:, 5], color="green"
)
plt.show()
