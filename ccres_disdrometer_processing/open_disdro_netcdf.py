from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from scattering import DATA, compute_fallspeed
from scipy import constants as cst
import constants

F = {"OTT HydroMet Parsivel2": constants.F_PARSIVEL}


def read_parsivel_cloudnet(
    filename: Union[str, Path]
) -> xr.Dataset:  # Read Parsivel file from CLU
    data_nc = xr.open_dataset(filename)
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time.data),
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
        data["KE"] = xr.DataArray(data_nc["kinetic_energy"].values, dims=["time"])
        data["sr"] = xr.DataArray(data_nc["snowfall_rate"].values, dims=["time"])
        data["SYNOP_code"] = xr.DataArray(data_nc["synop_WaWa"].values, dims=["time"])
        data["time_resolution"] = (
            data.time.values[1] - data.time.values[0]
        ) / np.timedelta64(1, "s")
        data["psd"] = xr.DataArray(
            data_nc["data_raw"].values, dims=["time", "size_classes", "speed_classes"]
        )  # En vérité axis = 1 correspond aux vitesses et 2 aux diamètres ...
        data["size_classes_width"] = xr.DataArray(
            data_nc["diameter_spread"].values * 1000, dims=["size_classes"]
        )
        data["speed_classes_width"] = xr.DataArray(
            data_nc["velocity_spread"].values, dims=["speed_classes"]
        )
        data["VD"] = np.sum(
            data.psd * data.speed_classes.values.reshape(1, data.size_classes.size, 1),
            axis=1,
        ) / np.sum(data.psd, axis=1)

        data["time"] = data.time.dt.round(freq="S")

    return data


def read_parsivel_cloudnet_juelich(
    filename: Union[str, Path]
) -> xr.Dataset:  # Read Parsivel file from CLU
    data_nc = xr.open_dataset(filename)
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )

    if data_nc.source == "OTT HydroMet Parsivel2":
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
            data_nc["data_raw"].values, dims=["time", "size_classes", "speed_classes"]
        )  # En vérité axis = 1 correspond aux vitesses et 2 aux diamètres ...
        data["size_classes_width"] = xr.DataArray(
            data_nc["diameter_spread"].values * 1000, dims=["size_classes"]
        )
        data["speed_classes_width"] = xr.DataArray(
            data_nc["velocity_spread"].values, dims=["speed_classes"]
        )
        data["VD"] = np.sum(
            data.psd * data.speed_classes.values.reshape(1, data.size_classes.size, 1),
            axis=1,
        ) / np.sum(data.psd, axis=1)

        data["time"] = data.time.dt.round(freq="S")

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
            mparsivel.psd.values[ii, :, :], 0
        )  # sum over speed axis -> number of drops per time and size

        model.RR[ii] = (
            (np.pi / 6.0)
            * (1.0 / (F * t))
            * np.nansum(Ni * (mparsivel.size_classes.values**3))
            * (3.6 * 1e-3)  # get mm/h from m/s : k = 1e-9 * 3.6 * 1e6
        )
        # we need to derive V(D)
        for i in range(len(mparsivel.size_classes)):
            model.VD[ii, i] = np.nansum(
                mparsivel.psd.values[ii, :, i] * mparsivel.speed_classes.values
            ) / np.nansum(mparsivel.psd.values[ii, :, i])

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
