import pandas as pd
import xarray as xr

LIST_VARIABLES = ["Zh", "v", "radar_frequency"]
RANGE_BOUNDS = [0, 3000]


def read_radar_cloudnet(filename):  # daily radar file from cloudnet
    data_nc = xr.open_dataset(filename)[LIST_VARIABLES].sel(
        range=slice(RANGE_BOUNDS[0], RANGE_BOUNDS[1])
    )

    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0, microsecond=0, nanosecond=0
    )
    end_time = pd.Timestamp(data_nc.time.values[-1]).replace(
        hour=23, minute=59, second=0, microsecond=0, nanosecond=0
    )
    time_index = pd.date_range(
        start_time, end_time + pd.Timedelta(minutes=1), freq="1T"
    )
    radar_ds = xr.Dataset(
        coords=dict(
            time=(["time"], time_index[:-1]), range=(["range"], data_nc.range.data)
        )
    )
    radar_ds["frequency"] = data_nc.radar_frequency  # * 10**9

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

    radar_ds.attrs["radar_source"] = data_nc.attrs["source"]

    data_nc.close()

    return radar_ds
