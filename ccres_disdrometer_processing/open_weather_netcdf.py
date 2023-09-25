import numpy as np
import xarray as xr
import pandas as pd


def read_weather_cloudnet(filename):
    data_nc = xr.open_dataset(filename)

    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0
    )
    end_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=23, minute=59, second=0
    )
    time_index = pd.date_range(
        start_time, end_time + pd.Timedelta(minutes=1), freq="1T"
    )
    time_index_offset = time_index - pd.Timedelta(30, "sec")

    vars = [
        "wind_speed",
        "wind_direction",
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "rainfall_rate",
        "rainfall_amount",
    ]
    data_nc_resampled = (
        data_nc[vars]
        .groupby_bins("time", time_index_offset, labels=time_index[:-1])
        .first()
    )

    data = xr.Dataset(coords=dict(time=(["time"], data_nc_resampled.time_bins.data)))

    if data_nc.source == "Generic weather-station":
        data["ws"] = xr.DataArray(
            data_nc_resampled["wind_speed"].values,
            dims=["time"],
            attrs=data_nc["wind_speed"].attrs,
        )
        data["wd"] = xr.DataArray(
            data_nc_resampled["wind_direction"].values,
            dims=["time"],
            attrs=data_nc["wind_direction"].attrs,
        )
        data["temp"] = xr.DataArray(
            data_nc_resampled["air_temperature"].values,
            dims=["time"],
            attrs=data_nc["air_temperature"].attrs,
        )
        data["rh"] = xr.DataArray(
            data_nc_resampled["relative_humidity"].values,
            dims=["time"],
            attrs=data_nc["relative_humidity"].attrs,
        )
        data["pres"] = xr.DataArray(
            data_nc_resampled["air_pressure"].values,
            dims=["time"],
            attrs=data_nc["air_pressure"].attrs,
        )
        data["rain"] = xr.DataArray(
            data_nc_resampled["rainfall_rate"].values * 60 * 1000,
            dims=["time"],
            attrs={
                "units": "mm/mn",
                "long_name": "Rainfall rate",
                "standard_name": "rainfall_rate",
            },
        )
        data["rain_sum"] = xr.DataArray(
            data_nc_resampled["rainfall_amount"].values * 1000,
            dims=["time"],
            attrs={
                "units": "mm",
                "long_name": "Rainfall amount",
                "standard_name": "thickness_of_rainfall_amount",
                "comment": "Cumulated precipitation since 00:00 UTC",
            },
        )

        data["u"] = xr.DataArray(
            -data["ws"] * np.sin(data["wd"] * np.pi / 180),
            dims=["time"],
            attrs={
                "units": "m/s",
                "long_name": "Zonal wind",
            },
        )
        data["v"] = xr.DataArray(
            -data["ws"] * np.cos(data["wd"] * np.pi / 180),
            dims=["time"],
            attrs={
                "units": "m/s",
                "long_name": "Meridional wind",
            },
        )

    return data
