import os

import xarray as xr

from ccres_disdrometer_processing.processing.preprocessed_file2processed import process


def test_hur_low_spl():
    yesterday = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240418_lindenberg_preprocessed_ws.nc"  # noqa E501
    today = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240419_lindenberg_preprocessed_ws.nc"  # noqa E501
    tomorrow = "/homedata/ygrit/tests_production_data/preprocessing_outputs/lindenberg/20240420_lindenberg_preprocessed_ws.nc"  # noqa E501
    conf = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/conf/config_lindenberg_mira-parsivel.toml"  # noqa E501
    output = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/ccres_disdrometer_processing/processing/Lindenberg_low_sampling_test.nc"  # noqa E501
    if os.path.exists(output):
        os.remove(output)
    process(yesterday, today, tomorrow, conf, output, no_meteo=False, verbosity=1)
    ds = xr.open_dataset(output)
    print("number of events : ", len(ds.events))
    for k in range(len(ds.events)):
        print(f"event {k+1} : {ds.start_event.values[k]} - {ds.end_event.values[k]}")
    print(ds.QC_hur.attrs)
    print("QC hur ratio(s) : ", ds.QC_hur_ratio.values)
    print("QC overall ratio(s) : ", ds.QC_overall_ratio.values)
    print(ds.QC_overall.attrs["comment"])
