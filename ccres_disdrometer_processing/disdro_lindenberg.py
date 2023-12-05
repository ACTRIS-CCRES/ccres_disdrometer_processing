import glob

import numpy as np
import pandas as pd
import xarray as xr
from scattering import compute_fallspeed

ds_prepro = xr.open_dataset(
    "/home/ygrit/Documents/dcrcc_data/juelich/disdrometer_preprocessed/20210504_juelich_preprocessed_degrade.nc"
)
print(ds_prepro.attrs)
print("###############")

ds = xr.open_dataset(
    glob.glob(
        # "/home/ygrit/Documents/dcrcc_data/lindenberg/disdrometer/*20230630*thies*.nc"
        "/home/ygrit/Documents/dcrcc_data/lindenberg/disdrometer/*20230731*thies*.nc"
    )[0]
)
print(ds.attrs)
# print(
#     list(ds.keys()),
#     len(list(ds.keys())),
# )
for key in list(ds.keys()):
    print(key, ds[key].attrs)
print("############################")
parsivel_ds = xr.open_dataset(
    glob.glob(
        "/home/ygrit/Documents/dcrcc_data/lindenberg/disdrometer/*20230731*parsivel*.nc"
    )[0]
)
for k in parsivel_ds.keys():
    print(k, parsivel_ds[k].attrs)
# print(parsivel_ds.sig_laser.attrs)
# print(parsivel_ds.time.values[0:10])


start_time = pd.Timestamp(parsivel_ds.time.values[0]).replace(
    hour=0, minute=0, second=0
)
end_time = pd.Timestamp(parsivel_ds.time.values[0]).replace(
    hour=23, minute=59, second=0
)
print(type(start_time), start_time, end_time)
time_index = pd.date_range(start_time, end_time + pd.Timedelta(minutes=1), freq="1T")
time_index_offset = time_index - pd.Timedelta(30, "sec")
print(time_index.shape)

time_var = []
notime_var = []
for var in parsivel_ds.keys():
    if "time" in parsivel_ds[var].coords.dims:
        time_var.append(var)
    else:
        notime_var.append(var)
print(time_var)


parsivel_ds_time_resampled = (
    parsivel_ds[time_var]
    .groupby_bins("time", time_index_offset, labels=time_index[:-1])
    .first()
)
parsivel_ds_notime = parsivel_ds[notime_var]

print(type(parsivel_ds_time_resampled.time_bins.values[0]))
print(parsivel_ds_time_resampled.time_bins.values[0:10])

final_ds = xr.merge((parsivel_ds_time_resampled, parsivel_ds_notime))
print(final_ds.attrs)


# print(ds.visibility.values[700:900])
filt = np.where(ds.n_particles.values == np.max(ds.n_particles.values))
print(ds.time.values[filt])
# LES DONNEES PSD SONT A L'ENDROIT !!!!
print(ds.dims, ds.data_raw.values.shape)

A = 0.0046
t = 60
k = 3 / (np.pi / 2 * 1 / (A * t))

VD = np.zeros((len(ds.time.values), len(ds.diameter.values)))
for ii in range(len(ds.time.values)):
    for i in range(len(ds.diameter.values)):
        VD[ii, i] = np.nansum(
            ds.data_raw.values[ii, i, :] * ds.velocity.values
        ) / np.nansum(ds.data_raw.values[ii, i, :])
VDmodel = compute_fallspeed(ds.diameter.values * 10**3)
print(VD[filt])

dsd = np.nansum(ds.data_raw.values, axis=2)
print(dsd.shape, ds.diameter.values.shape, VD.shape)

vmor_calc = np.zeros(len(ds.time.values))
for x in range(len(vmor_calc)):
    vmor_calc[x] = k / np.nansum(
        dsd[x, :] * ds.diameter.values**2 / VDmodel[:]
    )  # VD[x,:]
print(k, vmor_calc[filt], ds.visibility.values[filt])

AU = 1000 * (vmor_calc / ds.visibility.values)
# print(AU[720:800])
print(AU[400:500])

# print(len(ds.time.values), ds.time.values[0:10], ds.time.values[-10:])
# print(ds.attrs)

# radar_ds = xr.open_dataset(
#     glob.glob("/home/ygrit/Documents/dcrcc_data/lindenberg/radar/*2022*mira*.nc")[0]
# )
# print(
#     len(radar_ds.time.values),
#     radar_ds.time.values[0:10],
#     radar_ds.time.values[-10:],  # no_fmt
# )


# print(list(ds_juelich.keys()), len(list(ds_juelich.keys())))
