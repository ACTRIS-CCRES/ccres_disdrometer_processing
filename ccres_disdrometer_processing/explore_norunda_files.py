import glob

import xarray as xr

a = glob.glob("/home/ygrit/Documents/dcrcc_data/norunda/disdrometer/*")[100]
a = xr.open_dataset(a)
print(len(list(a.keys())), list(a.keys()))

b = glob.glob("/home/ygrit/Documents/dcrcc_data/norunda/disdrometer/*")[-100]
b = xr.open_dataset(b)
print(len(list(b.keys())), list(b.keys()))

# b = glob.glob("/home/ygrit/Documents/dcrcc_data/norunda/radar/*")[100]
# b = xr.open_dataset(b)
# print(b.attrs)
# print(list(b.keys()))
# print(b.altitude.values, b.range.values[0:10], b.range_resolution.values)
# print(b.rainfall_rate.values[0:20])

# plt.figure()
# plt.plot(b.time.values, b.rainfall_rate.values)
# plt.show()
