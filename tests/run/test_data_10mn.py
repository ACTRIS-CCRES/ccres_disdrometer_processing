import os

import xarray as xr

from ccres_disdrometer_processing.processing.preprocessed_file2processed import process


def test_data_lindenberg():
    yesterday = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240208_lindenberg_preprocessed_ws.nc"  # noqa E501
    today = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240209_lindenberg_preprocessed_ws.nc"  # noqa E501
    tomorrow = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240210_lindenberg_preprocessed_ws.nc"  # noqa E501
    conf = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/conf/config_lindenberg_mira-parsivel.toml"  # noqa E501
    output = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/ccres_disdrometer_processing/processing/Lindenberg_test_weather.nc"  # noqa E501
    output_noweather = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/ccres_disdrometer_processing/processing/Lindenberg_test_noweather.nc"  # noqa E501
    if os.path.exists(output):
        os.remove(output)
    # Processing with weathers
    process(yesterday, today, tomorrow, conf, output, no_meteo=False, verbosity=1)
    ds = xr.open_dataset(output)
    print("number of events : ", len(ds.events))
    for k in range(len(ds.events)):
        print(
            f"event {k+1} : {ds.start_event.values[k]} - {ds.end_event.values[k]}, {ds.rain_accumulation.values[k]:.4f}mm"  # noqa E501
        )
        print("good pts number : ", ds.good_points_number.values[k])
    # Processing without weather
    process(
        yesterday, today, tomorrow, conf, output_noweather, no_meteo=True, verbosity=1
    )
    ds = xr.open_dataset(output_noweather)
    print("number of events noweather : ", len(ds.events))
    for k in range(len(ds.events)):
        print(
            f"event {k+1} : {ds.start_event.values[k]} - {ds.end_event.values[k]}, {ds.rain_accumulation.values[k]:.4f}mm"  # noqa E501
        )
        print("good pts number : ", ds.good_points_number.values[k])
