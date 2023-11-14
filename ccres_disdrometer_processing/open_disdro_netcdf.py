from pathlib import Path
from typing import Union

import constants
import numpy as np
import pandas as pd
import xarray as xr
from scattering import DATA, compute_fallspeed
from scipy import constants as cst

F = {
    "OTT HydroMet Parsivel2": constants.F_PARSIVEL,
    "Thies Clima LNM": constants.F_THIES,
}

KEYS = [
    "visibility",
    "sig_laser",
    "n_particles",
    "T_sensor",
    "I_heating",
    "V_power_supply",
    "kinetic_energy",
    "snowfall_rate",
    "synop_WaWa",
    # "diameter_spread",
    # "velocity_spread",
]
NEW_KEYS = [
    "visi",
    "sa",
    "particles_count",
    "sensor_temp",
    "heating_current",
    "sensor_volt",
    "KE",
    "sr",
    "SYNOP_code",
    # "size_classes_width",
    # "speed_classes_width",
]


def resample_data_perfect_timesteps(filename: Union[str, Path]) -> xr.Dataset:
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
    time_var, notime_var = [], []
    for var in data_nc.keys():
        if "time" in data_nc[var].coords.dims:
            time_var.append(var)
        else:
            notime_var.append(var)
    data_time_resampled = (
        data_nc[time_var]
        .groupby_bins("time", time_index_offset, labels=time_index[:-1])
        .first()
    )
    data_notime = data_nc[notime_var]
    data_perfect_timesteps = xr.merge((data_time_resampled, data_notime))
    data_perfect_timesteps["time_bins"] = data_perfect_timesteps.time_bins.dt.round(
        freq="1S"
    )

    for key in ["year", "month", "day", "location"]:
        data_perfect_timesteps.attrs[key] = data_nc.attrs[key]
    data_perfect_timesteps.attrs["disdrometer_source"] = data_nc.attrs["source"]
    return data_perfect_timesteps


def read_parsivel_cloudnet(
    data_nc: xr.Dataset,
) -> xr.Dataset:  # Read Parsivel file from CLU resampled file
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time_bins.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )
    if data_nc.source == "OTT HydroMet Parsivel2":
        data["F"] = F[data_nc.source]
        data["pr"] = xr.DataArray(
            data_nc["rainfall_rate"].values * 1000 * 3600,
            dims=["time"],
            attrs={"units": "mm/h"},
        )
        data["cp"] = xr.DataArray(
            np.cumsum(data_nc["rainfall_rate"].values * 60 * 1000),
            dims=["time"],
            attrs={"units": "mm"},
        )
        data["Z"] = xr.DataArray(
            data_nc["radar_reflectivity"].values, dims=["time"], attrs={"units": "dBZ"}
        )
        data["psd"] = xr.DataArray(
            np.transpose(data_nc["data_raw"].values, axes=(0, 2, 1)),
            dims=["time", "size_classes", "speed_classes"],
        )
        data["time_resolution"] = (
            data.time.values[1] - data.time.values[0]
        ) / np.timedelta64(1, "s")

        for i in range(len(KEYS)):
            if KEYS[i] in list(data_nc.keys()):
                data[NEW_KEYS[i]] = xr.DataArray(data_nc[KEYS[i]].values, dims=["time"])


        data["size_classes_width"] = xr.DataArray(
            data_nc["diameter_spread"].values * 1000, dims=["size_classes"]
        )  # mm
        data["speed_classes_width"] = xr.DataArray(
            data_nc["velocity_spread"].values, dims=["speed_classes"]
        )

        # VD = np.empty((data.time.size, data.size_classes.size))
        # for t in range(data.time.size):
        #     for s in range(data.size_classes.size):
        #         VD[t, s] = (
        #             np.nansum(data.psd.values[t, s, :] * data.speed_classes.values)
        #         ) / np.nansum(data.psd.values[t, s, :])
        # data["VD"] = xr.DataArray(VD, dims=["time", "size_classes"])

    return data


def read_parsivel_cloudnet_bis(
    data_nc: xr.Dataset,
) -> xr.Dataset:  # Read Parsivel file from CLU resampled file, Jülich specificities
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time_bins.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )

    if data_nc.source == "OTT HydroMet Parsivel2":
        data["F"] = F[data_nc.source]
        data["pr"] = xr.DataArray(
            data_nc["rainfall_rate"].values * 1000 * 3600,
            dims=["time"],
            attrs={"units": "mm/h"},
        )
        data["cp"] = xr.DataArray(
            np.cumsum(data_nc["rainfall_rate"].values * 60 * 1000),
            dims=["time"],
            attrs={"units": "mm"},
        )
        data["Z"] = xr.DataArray(
            data_nc["radar_reflectivity"].values, dims=["time"], attrs={"units": "dBZ"}
        )
        data["visi"] = xr.DataArray(data_nc["visibility"].values, dims=["time"])
        data["sa"] = xr.DataArray(data_nc["sig_laser"].values, dims=["time"])
        data["particles_count"] = xr.DataArray(
            data_nc["n_particles"].values, dims=["time"]
        )
        data["sensor_temp"] = xr.DataArray(data_nc["T_sensor"].values, dims=["time"])
        data["heating_current"] = xr.DataArray(
            data_nc["I_heating"].values, dims=["time"]
        )
        data["sensor_volt"] = xr.DataArray(
            data_nc["V_power_supply"].values, dims=["time"]
        )
        data["SYNOP_code"] = xr.DataArray(data_nc["synop_WaWa"].values, dims=["time"])
        data["time_resolution"] = (
            data.time.values[1] - data.time.values[0]
        ) / np.timedelta64(1, "s")
        data["psd"] = xr.DataArray(
            np.transpose(data_nc["data_raw"].values, axes=(0, 2, 1)),
            dims=["time", "size_classes", "speed_classes"],
        )
        data["size_classes_width"] = xr.DataArray(
            data_nc["diameter_spread"].values * 1000, dims=["size_classes"]
        )
        data["speed_classes_width"] = xr.DataArray(
            data_nc["velocity_spread"].values, dims=["speed_classes"]
        )
        # data["VD"] = np.sum(
        #     data.psd * data.speed_classes.values.reshape(1, data.size_classes.size, 1),
        #     axis=1,
        # ) / np.sum(data.psd, axis=1)

    return data


def read_thies_cloudnet(
    data_nc: xr.Dataset,
) -> xr.Dataset:  # Read Parsivel file from CLU resampled file
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time_bins.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )

    data["F"] = F[data_nc.source]
    data["pr"] = xr.DataArray(
        data_nc["rainfall_rate"].values * 1000 * 3600,
        dims=["time"],
        attrs={"units": "mm/h"},
    )
    data["cp"] = xr.DataArray(
        np.cumsum(data_nc["rainfall_rate"].values * 60 * 1000),
        dims=["time"],
        attrs={"units": "mm"},
    )
    data["Z"] = xr.DataArray(
        data_nc["radar_reflectivity"].values, dims=["time"], attrs={"units": "dBZ"}
    )
    data["visi"] = xr.DataArray(data_nc["visibility"].values, dims=["time"])
    # data["sa"] = xr.DataArray(data_nc["sig_laser"].values, dims=["time"])
    data["particles_count"] = xr.DataArray(data_nc["n_particles"].values, dims=["time"])
    data["sensor_temp"] = xr.DataArray(data_nc["T_interior"].values, dims=["time"])
    data["heating_current"] = xr.DataArray(
        data_nc["I_heating_laser_head"].values, dims=["time"]
    )
    data["sensor_volt"] = xr.DataArray(data_nc["V_sensor_supply"].values, dims=["time"])
    data["SYNOP_code"] = xr.DataArray(data_nc["synop_WaWa"].values, dims=["time"])
    data["time_resolution"] = (
        data.time.values[1] - data.time.values[0]
    ) / np.timedelta64(1, "s")
    data["psd"] = xr.DataArray(
        data_nc["data_raw"].values, dims=["time", "size_classes", "speed_classes"]
    )
    data["size_classes_width"] = xr.DataArray(
        data_nc["diameter_spread"].values * 1000, dims=["size_classes"]
    )
    data["speed_classes_width"] = xr.DataArray(
        data_nc["velocity_spread"].values, dims=["speed_classes"]
    )
    # data["VD"] = np.sum(
    #     data.psd * data.speed_classes.values.reshape(1, 1, -1),
    #     axis=2,
    # ) / np.sum(data.psd, axis=2)

    return data


def read_parsivel_cloudnet_choice(filename: Union[str, Path]) -> xr.Dataset:
    data_nc = resample_data_perfect_timesteps(filename=filename)
    station = data_nc.location
    source = data_nc.source
    if station == "Palaiseau":
        data = read_parsivel_cloudnet(data_nc)
    elif station == "Jülich" or station == "Norunda":
        # data = read_parsivel_cloudnet_bis(data_nc)
        data = read_parsivel_cloudnet(data_nc)
    elif source == "Thies Clima LNM":
        data = read_thies_cloudnet(data_nc)
    else:
        data = None
    if not (data is None):
        data.attrs = data_nc.attrs
    return data


def reflectivity_model(
    mparsivel,
    scatt,
    n,
    freq,
    strMethod="GunAndKinzer",
    mieMethod="pymiecoated",
    normMethod="model",
):
    # integration time (note: there is an issue with the 10s files -
    # dt must remain at 60s)
    t = mparsivel.time_resolution.values  # s
    # wavelength
    lambda_m = cst.c / freq
    F = mparsivel.F.data

    model = DATA()

    model.RR = np.zeros([len(mparsivel.time)])
    model.VD = np.zeros([len(mparsivel.time), len(mparsivel.size_classes)])
    model.M2 = np.zeros([len(mparsivel.time)])
    model.M3 = np.zeros([len(mparsivel.time)])
    model.M4 = np.zeros([len(mparsivel.time)])
    model.Ze_ray = np.zeros([len(mparsivel.time)])
    model.Ze_mie = np.zeros([len(mparsivel.time)])
    model.Ze_tm = np.zeros([len(mparsivel.time)])
    model.attenuation = np.zeros([len(mparsivel.time)])
    model.V_tm = np.zeros([len(mparsivel.time)])
    model.V_mie = np.zeros([len(mparsivel.time)])
    model.dsd = np.zeros([len(mparsivel.time), len(mparsivel.size_classes)])

    model.diameter_bin_width_mm = mparsivel.size_classes_width.values

    for ii in range(len(mparsivel.time)):
        Ni = np.nansum(
            mparsivel.psd.values[ii, :, :], 1
        )  # sum over speed axis -> number of drops per time and size # replace axis=1 by 0 if not transposed in parsivel

        model.RR[ii] = (
            (np.pi / 6.0)
            * (1.0 / (F * t))
            * np.nansum(Ni * (mparsivel.size_classes.values**3))
            * (3.6 * 1e-3)  # get mm/h from m/s : k = 1e-9 * 3.6 * 1e6
        )
        # we need to derive V(D)
        for i in range(len(mparsivel.size_classes)):
            model.VD[ii, i] = np.nansum(
                mparsivel.psd.values[ii, i, :] * mparsivel.speed_classes.values
            ) / np.nansum(mparsivel.psd.values[ii, i, :])

        # parameterisation for the velocity (Gun and Kinzer)
        VDmodel = compute_fallspeed(mparsivel.size_classes.values, strMethod=strMethod)

        if normMethod == "measurement":
            VDD = model.VD[ii, :]

        elif normMethod == "model":
            VDD = VDmodel

        # with velocity parameterisation
        model.dsd[ii, :] = (
            Ni / VDD / F / t / (model.diameter_bin_width_mm * 1.0e-3)
        )  # #particles/m3 normalised per diameter bin width

        model.M2[ii] = (
            (np.nansum(Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 2) / VDD))
            / F
            / t
        )
        model.M3[ii] = (
            (np.nansum(Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 3) / VDD))
            / F
            / t
        )
        model.M4[ii] = (
            (np.nansum(Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 4) / VDD))
            / F
            / t
        )
        model.Ze_ray[ii] = (
            (np.nansum(Ni * (mparsivel.size_classes.values**6) / VDD)) / F / t
        )

        if mieMethod == "pymiecoated":
            model.Ze_mie[ii] = (
                1e18
                # because we want mm6 instead of m6 ; when pytmatrix,
                # input is in mm so we don'ty have to apply this scale factor
                * np.nansum(Ni[0:n] * scatt.bscat_mie / VDD[0:n])
                * (lambda_m**4 / (np.pi) ** 5.0)
                / 0.93  # squared water dielectric constant
                / F
                / t
            )  # mm6/m3
        elif mieMethod == "pytmatrix":
            model.Ze_mie[ii] = (
                np.nansum(Ni[0:n] * scatt.bscat_mie / VDD[0:n]) / F / t
            )  # mm6/m3

        model.Ze_tm[ii] = (
            np.nansum(Ni[0:n] * scatt.bscat_tmatrix / VDD[0:n]) / F / t
        )  # mm6/m3

        model.attenuation[ii] = (
            np.nansum(Ni[0:n] * scatt.att_tmatrix / VDD[0:n]) / F / t
        )  # dB/km
        model.V_mie[ii] = np.nansum(Ni[0:n] * scatt.bscat_mie) / np.nansum(
            Ni[0:n] * scatt.bscat_tmatrix / VDD[0:n]
        )  # m/s
        model.V_tm[ii] = np.nansum(Ni[0:n] * scatt.bscat_tmatrix) / np.nansum(
            Ni[0:n] * scatt.bscat_tmatrix / VDD[0:n]
        )  # m/s

        # with velocity from parsivel
        model.dsd[ii, :] = (
            Ni / model.VD[ii, :] / F / t / (model.diameter_bin_width_mm * 1.0e-3)
        )  # particles/m3 #normalised per diameter bin width

    DensityLiquidWater = 1000.0e3  # g/m3

    # store results in parsivel object
    mparsivel["dsd"] = xr.DataArray(model.dsd, dims=["time", "size_classes"])
    mparsivel["RR"] = xr.DataArray(model.RR, dims=["time"])
    mparsivel["VD"] = xr.DataArray(model.VD, dims=["time", "size_classes"])
    mparsivel["M2"] = xr.DataArray(model.M2, dims=["time"])
    mparsivel["M3"] = xr.DataArray(model.M3, dims=["time"])
    mparsivel["M4"] = xr.DataArray(model.M4, dims=["time"])
    mparsivel["Ze_ray"] = xr.DataArray(model.Ze_ray, dims=["time"])
    mparsivel["Ze_mie"] = xr.DataArray(model.Ze_mie, dims=["time"])
    mparsivel["Ze_tm"] = xr.DataArray(model.Ze_tm, dims=["time"])
    mparsivel["attenuation"] = xr.DataArray(model.attenuation, dims=["time"])
    mparsivel["V_tm"] = xr.DataArray(model.V_tm, dims=["time"])
    mparsivel["V_mie"] = xr.DataArray(model.V_mie, dims=["time"])

    # additional parameters
    mparsivel["Dm"] = xr.DataArray(model.M4 / model.M3, dims=["time"])
    mparsivel["LWC"] = xr.DataArray(
        DensityLiquidWater * (1.0 / 6.0) * np.pi * model.M3, dims=["time"]
    )
    mparsivel["N0"] = xr.DataArray(
        (4.0**4 / (np.pi * DensityLiquidWater))
        * mparsivel.LWC.values
        / (mparsivel.Dm.values) ** 4,
        dims=["time"],
    )
    mparsivel["re"] = xr.DataArray(0.5 * model.M3 / model.M2, dims=["time"])

    return mparsivel
