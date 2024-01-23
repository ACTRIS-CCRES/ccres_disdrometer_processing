import numpy as np
import pandas as pd
import xarray as xr


def read_weather_cloudnet(filename):
    data_nc = xr.open_dataset(filename)

    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0, microsecond=0, nanosecond=0
    )
    end_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=23, minute=59, second=0, microsecond=0, nanosecond=0
    )
    time_index = pd.date_range(
        start_time, end_time + pd.Timedelta(minutes=1), freq="1T"
    )
    time_index_offset = time_index - pd.Timedelta(30, "sec")

    vars_timedep = ["wind_speed",
        "wind_direction",
        "air_temperature",
        "relative_humidity",
        "air_pressure",
        "rainfall_rate",
        "rainfall_amount",
    ]

    # vars_notimedep = ["longitude", "latitude", "altitude"]

    data_nc_resampled = (
        data_nc[vars_timedep]
        .groupby_bins("time", time_index_offset, labels=time_index[:-1])
        .first()
    )

    data = xr.Dataset(coords=dict(time=(["time"], data_nc_resampled.time_bins.data)))

    if data_nc.source == "Generic weather-station" or data_nc.source == "":
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
        data["ta"] = xr.DataArray(
            data_nc_resampled["air_temperature"].values - 273.15,
            dims=["time"],
            attrs=data_nc["air_temperature"].attrs,
        )
        data["ta"].attrs["units"] = "Celsius"
        data["ta"].attrs["Comment"] = "Inside no ventilated shelter"
        data["hur"] = xr.DataArray(
            data_nc_resampled["relative_humidity"].values * 100,
            dims=["time"],
            attrs=data_nc["relative_humidity"].attrs,
        )
        data["hur"].attrs["units"] = "%"
        data["hur"].attrs["Comment"] = "Inside no ventilated shelter"
        data["ps"] = xr.DataArray(
            data_nc_resampled["air_pressure"].values / 100,
            dims=["time"],
            attrs=data_nc["air_pressure"].attrs,
        )
        data["ps"].attrs["units"] = "hPa"
        data["ams_pr"] = xr.DataArray(
            data_nc_resampled["rainfall_rate"].values * 60 * 60 * 1000, # * 3600 to convert m/s to m/hr
            dims=["time"],
            attrs={
                "units": "mm/hr",
                "long_name": "Met station precipitation rate at 1m agl",
                "standard_name": "rainfall_rate",
                "Comment":"The abbreviation 'lwe' means liquid water equivalent. 'Precipitation rate' means the depth or thickness of the layer formed by precipitation per unit time."
            },
        )
        data["ams_cp"] = xr.DataArray(
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

    data.attrs["weather_source"] = data_nc_resampled.attrs["source"]
    data.attrs["weather-station_pid"] = data_nc_resampled.attrs["instrument_pid"]

    data["ams_longitude"] = data_nc["longitude"]
    data["ams_longitude"].attrs["Comment"] = "AMS = Atmospheric Meteorological Station"
    data["ams_latitude"] = data_nc["latitude"]
    data["ams_altitude"] = data_nc["altitude"]

    return data
