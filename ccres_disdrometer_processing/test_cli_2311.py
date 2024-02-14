import os 
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

MAIN_DIR = Path(__file__).parent.parent
TEST_DIR = MAIN_DIR / "tests"
TEST_INPUT = TEST_DIR / "data/inputs"
TEST_OUT_DIR = TEST_DIR / "data/outputs"

disdro_file = f"{TEST_INPUT}/20210202_palaiseau_parsivel.nc"
ws_file = f"{TEST_INPUT}/20210202_palaiseau_weather-station.nc"
radar_file = f"{TEST_INPUT}/20210202_palaiseau_basta.nc"
config_file = f"{TEST_INPUT}/CONFIG_preprocessing_processing.toml"
output_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v0802.nc"
print(config_file)


do = 0
open = 1
rad = 0
dd = 0
weather = 0
compare_versions = 0

if do :
    os.system("ccres_disdrometer_processing preprocess --disdro-file {} --ws-file {} --radar-file {} --config-file {} {}".format(disdro_file, ws_file, radar_file, config_file, output_file))
    # os.system("ccres_disdrometer_processing preprocess --disdro-file {} --radar-file {} --config-file {} {}".format(disdro_file, radar_file, config_file, output_file))
    #subprocess.call / checkrun -> récupérer le retour du code
    print("DONE")

if open : 
    ds = xr.open_dataset(output_file)
    # print(ds.attrs)
    print(ds.dims)
    print(ds.attrs)
    for var in list(ds.keys()):
        try :
            x = ds[var].attrs["units"]
        except KeyError :
            print(var)

if rad : 
    radar = xr.open_dataset(radar_file)
    print(list(radar.keys()))
    print(radar.attrs)
    print(radar.latitude.attrs)

if dd :
    dd = xr.open_dataset(disdro_file)
    print("### DISDROMETER FILE ###")
    # print(dd.attrs)
    # print(list(dd.keys()))
    # print(len(dd.time))
    # print(dd.rainfall_rate.values[0:50])
    print(dd.longitude.attrs)
    print(dd.data_raw.attrs)

if weather : 
    ws = xr.open_dataset(ws_file)
    print(list(ws.keys()))
    print(ws.wind_direction.attrs)

if compare_versions :
    ds = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1612.nc")
    ds_fov = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1101.nc")
    ds_old = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_monolambda.nc")
    plt.figure()
    # plt.plot(ds.time[100:300], ds.Ze_tm.values[100:300,-1], label='old file multilambda', lw=3)
    # plt.plot(ds.time[100:300], ds.Ze_tm.values[100:300,-1], label='old file multilambda', lw=3)
    plt.plot(ds_fov.time[100:300], ds_fov.attd_vfov_modv[100:300,-1], label='new file multilambda')
    plt.plot(ds.time[100:300], ds.attenuation.values[100:300,-1], label='first file multilambda')
    #plt.plot(ds_old.time[100:300], ds_old.M2.values[100:300], label="old monolambda ds")
    plt.legend()
    plt.show(block=True)
    print(ds.Ze_tm.dims, ds.computed_frequencies.values)
    print(ds_fov.Zdlin_vfov_modv_tm.dims, ds_fov.computed_frequencies.values)

    print(ds.Ze_tm.values[150,:])
    print(ds_fov.Zdlin_vfov_modv_tm.values[150,:])

# ds_lamb = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1612.nc")
# ds = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v0812.nc")
# met = xr.open_dataset(f"{TEST_INPUT}/20210202_palaiseau_weather-station.nc")
# print(ds.Ze_tm.values[200:250])
# print(ds_lamb.Ze_tm.values[200:250,3]) # 95GHz for BASTA

# print(list(ds_lamb.keys()))
# keys = ["ws","wd","ta","hur","ps","pr", "ams_latitude"]
# for k in keys :
#     # print(ds_lamb[k].values[0:5])
#     print(ds_lamb[k].attrs)
#     # print(ds_lamb[k].values[0])