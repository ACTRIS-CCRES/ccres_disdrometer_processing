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
output_file = f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v1112_multilambda.nc"

# print(MAIN_DIR)
# print(TEST_DIR)

do = True
open = False

if do :
    os.system("ccres_disdrometer_processing preprocess --disdro-file {} --ws-file {} --radar-file {} --config-file {} {}".format(disdro_file, ws_file, radar_file, config_file, output_file))
    # os.system("ccres_disdrometer_processing preprocess --disdro-file {} --radar-file {} --config-file {} {}".format(disdro_file, radar_file, config_file, output_file))
    #subprocess.call / checkrun -> récupérer le retour du code

if open : 
    ds = xr.open_dataset(output_file)
    # print(ds.attrs)
    # print(ds.coords)
    # print(list(ds.keys()))
    # print(ds.wd.dims)
    print(ds.Ze_tm.dims)


ds_lamb = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v0812_multilambda.nc")
ds = xr.open_dataset(f"{TEST_OUT_DIR}/20210202_palaiseau_preprocessed_v0812.nc")
print(ds.Ze_tm.values[200:250])
print(ds_lamb.Ze_tm.values[200:250,2])