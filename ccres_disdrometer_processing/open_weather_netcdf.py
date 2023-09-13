import numpy as np
import xarray as xr


def read_weather_cloudnet(filename):
    data_nc = xr.open_dataset(filename)
    data = xr.Dataset(coords=dict(time=(["time"], data_nc.time.data)))

    if data_nc.source == "Generic weather-station":
        data["ws"] = xr.DataArray(
            data_nc["wind_speed"].values,
            dims=["time"],
            attrs=data_nc["wind_speed"].attrs,
        )
        data["wd"] = xr.DataArray(
            data_nc["wind_direction"].values,
            dims=["time"],
            attrs=data_nc["wind_direction"].attrs,
        )
        data["temp"] = xr.DataArray(
            data_nc["air_temperature"].values,
            dims=["time"],
            attrs=data_nc["air_temperature"].attrs,
        )
        data["rh"] = xr.DataArray(
            data_nc["relative_humidity"].values,
            dims=["time"],
            attrs=data_nc["relative_humidity"].attrs,
        )
        data["pres"] = xr.DataArray(
            data_nc["air_pressure"].values,
            dims=["time"],
            attrs=data_nc["air_pressure"].attrs,
        )
        data["rain"] = xr.DataArray(
            data_nc["rainfall_rate"].values * 60 * 1000,
            dims=["time"],
            attrs={
                "units": "mm/mn",
                "long_name": "Rainfall rate",
                "standard_name": "rainfall_rate",
            },
        )
        data["rain_sum"] = xr.DataArray(
            data_nc["rainfall_amount"].values * 1000,
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

        data["time"] = data.time.dt.round(freq="S")

    return data
