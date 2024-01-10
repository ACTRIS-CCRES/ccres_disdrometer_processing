import os 
from pathlib import Path
import xarray as xr

MAIN_DIR = Path(__file__).parent.parent
TEST_DIR = MAIN_DIR / "tests"
TEST_INPUT = TEST_DIR / "data/inputs"
TEST_OUT_DIR = TEST_DIR / "data/outputs"

disdro_file = f"{TEST_INPUT}/20210202_palaiseau_parsivel.nc"
ws_file = f"{TEST_INPUT}/20210202_palaiseau_weather-station.nc"
radar_file = f"{TEST_INPUT}/20210202_palaiseau_basta.nc"
config_file = f"{TEST_INPUT}/CONFIG_test.toml"
output_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1612.nc"


do = True
open = True

if do :
    os.system("ccres_disdrometer_processing preprocess --disdro-file {} --ws-file {} --radar-file {} --config-file {} {}".format(disdro_file, ws_file, radar_file, config_file, output_file))
    # os.system("ccres_disdrometer_processing preprocess --disdro-file {} --radar-file {} --config-file {} {}".format(disdro_file, radar_file, config_file, output_file))
    #subprocess.call / checkrun -> récupérer le retour du code
    print("DONE")

if open : 
    ds = xr.open_dataset(output_file)
    # print(ds.attrs)
    print(list(ds.keys()))
    
    radar = xr.open_dataset(radar_file)
    print(list(radar.keys()))
    print(radar.attrs)
    print(radar.latitude.attrs)


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