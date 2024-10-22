import pandas as pd
import xarray as xr
from scipy import constants

LIST_VARIABLES = ["Zh", "v", "radar_frequency", "latitude", "longitude", "altitude"]


def read_radar_cloudnet(filename, max_radar_alt=2500):
    """Open, extract and resample data from CLU daily DCR files.

    Parameters
    ----------
    filename : Union[str, Path]
        The considered daily DCR file from CLU
    max_radar_alt : int, optional
        Altitude until which to extract 2D data (Z, DV), by default 2500m

    Returns
    -------
    xr.Dataset
        A formatted dataset for radar data (specs, Z, DV and range vector)
        with a 1-minute regular sampling

    """
    range_bounds = [0, max_radar_alt]
    data_nc = xr.open_dataset(filename)[LIST_VARIABLES].sel(
        range=slice(range_bounds[0], range_bounds[1])
    )

    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0, microsecond=0, nanosecond=0
    )
    end_time = pd.Timestamp(data_nc.time.values[-1]).replace(
        hour=23, minute=59, second=0, microsecond=0, nanosecond=0
    )
    time_index = pd.date_range(
        start_time, end_time + pd.Timedelta(minutes=1), freq="1min"
    )
    radar_ds = xr.Dataset(
        coords=dict(
            time=(["time"], time_index[:-1]), range=(["range"], data_nc.range.data)
        )
    )
    radar_ds.range.attrs = {"units": "m", "long_name": "Range of each gate"}

    radar_ds["radar_longitude"] = xr.DataArray(
        data_nc.longitude.values, attrs=data_nc.longitude.attrs
    )
    radar_ds["radar_latitude"] = xr.DataArray(
        data_nc.latitude.values, attrs=data_nc.latitude.attrs
    )
    radar_ds["radar_altitude"] = xr.DataArray(
        data_nc.altitude.values, attrs=data_nc.altitude.attrs
    )
    radar_ds["radar_altitude"].attrs["positive"] = "up"

    radar_ds["radar_model"] = data_nc.attrs["source"]
    radar_ds["radar_model"].attrs = {
        "long_name": "Radar model",
        "comment": "Radar model",
    }

    radar_ds["radar_frequency"] = (
        data_nc.radar_frequency * 10**9
    )  # in GHz, so * 10**9 to get Hz
    radar_ds["radar_frequency"].attrs["units"] = "Hz"
    radar_ds["radar_frequency"].attrs["long_name"] = "Frequency of the DCR, in Hertz"

    radar_ds["radar_wavelength"] = constants.c / radar_ds["radar_frequency"]
    radar_ds["radar_wavelength"].attrs["units"] = "m"
    radar_ds["radar_wavelength"].attrs["long_name"] = "Wavelength of the DCR, in meter"

    time_index_offset = time_index - pd.Timedelta(30, "sec")

    Z_dcr_resampled = data_nc.Zh.groupby_bins(
        "time", time_index_offset, labels=time_index[:-1]
    ).median(dim="time", keep_attrs=True)

    Doppler_resampled = data_nc.v.groupby_bins(
        "time", time_index_offset, labels=time_index[:-1]
    ).mean(dim="time", keep_attrs=True)

    radar_ds["alt"] = xr.DataArray(
        data_nc.range.values, dims=["range"], attrs=data_nc["range"].attrs
    )
    radar_ds["alt"].attrs["Comment"] = "Altitude above ground level"
    radar_ds["alt"].attrs["positive"] = "up"
    radar_ds["Zdcr"] = xr.DataArray(
        Z_dcr_resampled.values,
        dims=["time", "range"],
        attrs=data_nc["Zh"].attrs,
    )

    radar_ds["DVdcr"] = xr.DataArray(
        Doppler_resampled.values,
        dims=["time", "range"],
        attrs=data_nc["v"].attrs,
    )
    radar_ds["DVdcr"].attrs["Comment"] = "Negative downward"

    radar_ds.attrs["radar_source"] = data_nc.attrs["source"]
    radar_ds.attrs["radar_pid"] = data_nc.attrs["instrument_pid"]

    data_nc.close()

    return radar_ds
