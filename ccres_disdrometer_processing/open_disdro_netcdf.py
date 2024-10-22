from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ccres_disdrometer_processing.scattering import DATA, compute_fallspeed

KEYS = [
    # "visibility",
    # "sig_laser",
    "n_particles",
    # "T_sensor",
    # "I_heating",
    # "V_power_supply",
    # "kinetic_energy",
    # "snowfall_rate",
    # "synop_WaWa",
    # "diameter_spread",
    # "velocity_spread",
]
NEW_KEYS = [
    # "visi",
    # "sa",
    "particles_count",
    # "sensor_temp",
    # "heating_current",
    # "sensor_volt",
    # "KE",
    # "sr",
    # "SYNOP_code",
    # "size_classes_width",
    # "speed_classes_width",
]


def resample_data_perfect_timesteps(filename: Union[str, Path], config) -> xr.Dataset:
    """Open and resample CLU daily disdrometer files.

    CLU daily files can have an irregular timestamping. This function resamples data
    at a 1mn frequency. When several value are present in a minute,
    only the value from the first timestamp is kept.

    Parameters
    ----------
    filename : Union[str, Path]
        The considered daily disdrometer file from CLU
    config : dict (output from toml.load())
        The corresponding toml configuration file, already loaded

    Returns
    -------
    xr.Dataset
        A dataset with a 1-minute regular sampling

    """
    data_nc = xr.open_dataset(filename)
    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0, microsecond=0, nanosecond=0
    )
    end_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=23, minute=59, second=0, microsecond=0, nanosecond=0
    )
    time_index = pd.date_range(
        start_time, end_time + pd.Timedelta(minutes=1), freq="1min"
    )
    time_index_offset = time_index - pd.Timedelta(30, "sec")
    time_var, notime_var = [], []
    for var in data_nc.keys():  # noqa SIM118
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
    data_perfect_timesteps = xr.merge(
        (data_time_resampled, data_notime), combine_attrs="drop_conflicts"
    )
    data_perfect_timesteps.attrs = {}
    data_perfect_timesteps["time_bins"] = data_perfect_timesteps.time_bins.dt.round(
        freq="1S"
    )
    # TODO: to fix. variable in other option in config file
    data_perfect_timesteps["F"] = config["instrument_parameters"]["DD_SAMPLING_AREA"]

    for key in ["year", "month", "day", "location"]:
        data_perfect_timesteps.attrs[key] = data_nc.attrs[key]
    data_perfect_timesteps.attrs["disdrometer_source"] = data_nc.attrs["source"]
    data_perfect_timesteps.attrs["disdrometer_pid"] = data_nc.attrs["instrument_pid"]
    return data_perfect_timesteps


def read_parsivel_cloudnet(
    data_nc: xr.Dataset,
) -> xr.Dataset:  # Read Parsivel file from CLU resampled file
    """Create a dataset from CLU resampled data.

    Input the result of resample_data_perfect_timesteps() ;
    Read and select useful data. Case of a Parsivel disdrometer.
    Store the result in a new dataset, add some metadata.

    Parameters
    ----------
    data_nc : xr.Dataset
        output from resample_data_perfect_timesteps()

    Returns
    -------
    xr.Dataset
        Dataset for Parsivel daily disdrometer data, with only
    the useful variables kept (after being renamed)

    """
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time_bins.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )
    data.size_classes.attrs = {
        "units": "mm",
        "long_name": "Center of the diameter bins",
    }
    data.speed_classes.attrs = {
        "units": "m/s",
        "long_name": "Center of the velocity bins",
    }

    if data_nc.disdrometer_source == "OTT HydroMet Parsivel2":
        data["F"] = data_nc["F"]
        data["F"].attrs["long_name"] = "Disdrometer sampling area"
        data["F"].attrs["units"] = "m^2"
        data["disdro_pr"] = xr.DataArray(
            data_nc["rainfall_rate"].values * 1000 * 3600,
            dims=["time"],
            attrs={
                "units": "mm/h",
                "long_name": "Disdrometer-based precipitation rate",
                "comment": "Calculated with disdrometer, from rainfall rate data",
            },
        )
        data["disdro_cp"] = xr.DataArray(
            np.nancumsum(data_nc["rainfall_rate"].values * 60 * 1000),
            dims=["time"],
            attrs={
                "units": "mm",
                "long_name": "Disdrometer-based cumulated precipitation since 00:00 UTC",  # noqa E501
                "comment": "Calculated with disdrometer",
            },
        )

        data["psd"] = xr.DataArray(
            np.transpose(data_nc["data_raw"].values, axes=(0, 2, 1)),
            dims=["time", "size_classes", "speed_classes"],
            attrs={
                "units": "#/bin/mn",
                "long_name": "Number of droplets per diameter and fall speed bins",
                "comment": "Unit not recognized by UDUNITS ; does not account for the size of the bins",  # noqa E501
            },
        )

        for i in range(len(KEYS)):
            if KEYS[i] in list(data_nc.keys()):
                data[NEW_KEYS[i]] = xr.DataArray(
                    data_nc[KEYS[i]].values, dims=["time"], attrs=data_nc[KEYS[i]].attrs
                )

        data["size_classes_width"] = xr.DataArray(
            data_nc["diameter_spread"].values * 1000,
            dims=["size_classes"],
            attrs={"units": "mm", "long_name": "Width of the diameter bins"},
        )  # mm
        data["speed_classes_width"] = xr.DataArray(
            data_nc["velocity_spread"].values,
            dims=["speed_classes"],
            attrs={"units": "m/s", "long_name": "Width of the speed bins"},
        )
    return data


def read_thies_cloudnet(
    data_nc: xr.Dataset,
) -> xr.Dataset:
    """Create a dataset from CLU resampled data.

    Input the result of resample_data_perfect_timesteps() ;
    Read and select useful data. Case of a Parsivel disdrometer.
    Store the result in a new dataset, add some metadata.

    Parameters
    ----------
    data_nc : xr.Dataset
        output from resample_data_perfect_timesteps()

    Returns
    -------
    xr.Dataset
        Dataset for Thies daily disdrometer data, with only
    the useful variables kept (after being renamed)

    """
    data = xr.Dataset(
        coords=dict(
            time=(["time"], data_nc.time_bins.data),
            size_classes=(["size_classes"], data_nc.diameter.data * 1000),
            speed_classes=(["speed_classes"], data_nc.velocity.data),
        )
    )
    data.size_classes.attrs = {
        "units": "mm",
        "long_name": "Center of the diameter bins",
    }
    data.speed_classes.attrs = {
        "units": "m/s",
        "long_name": "Center of the velocity bins",
    }

    data["F"] = data_nc["F"]
    data["F"].attrs["long_name"] = "Disdrometer sampling area"
    data["F"].attrs["units"] = "m^2"
    data["F"].attrs["comment"] = (
        "Varies from one instrument to another for Thies LNM disdrometers"
    )
    data["disdro_pr"] = xr.DataArray(
        data_nc["rainfall_rate"].values * 1000 * 3600,
        dims=["time"],
        attrs={
            "units": "mm/h",
            "long_name": "Disdrometer-based precipitation rate",
            "comment": "Calculated with disdrometer, from rainfall rate data",
        },
    )
    data["disdro_cp"] = xr.DataArray(
        np.cumsum(data_nc["rainfall_rate"].values * 60 * 1000),
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Disdrometer-based cumulated precipitation since 00:00 UTC",
            "comment": "Calculated with disdrometer",
        },
    )
    # data["Z"] = xr.DataArray(
    #     data_nc["radar_reflectivity"].values, dims=["time"], attrs={"units": "dBZ"}
    # )
    # data["visi"] = xr.DataArray(data_nc["visibility"].values, dims=["time"])
    # data["sa"] = xr.DataArray(data_nc["sig_laser"].values, dims=["time"])

    # data["sensor_temp"] = xr.DataArray(data_nc["T_interior"].values, dims=["time"])
    # data["heating_current"] = xr.DataArray(
    #     data_nc["I_heating_laser_head"].values, dims=["time"]
    # )
    # data["sensor_volt"] = xr.DataArray(data_nc["V_sensor_supply"].values, dims=["time"]) # noqa E501

    # data["SYNOP_code"] = xr.DataArray(data_nc["synop_WaWa"].values, dims=["time"])

    data["psd"] = xr.DataArray(
        data_nc["data_raw"].values,
        dims=["time", "size_classes", "speed_classes"],
        attrs={
            "units": "#/bin/mn",
            "long_name": "Number of droplets per diameter and fall speed bins",
            "comment": "Unit not recognized by UDUNITS ; does not account for the size of the bins",  # noqa E501
        },
    )

    data["particles_count"] = xr.DataArray(
        data_nc["n_particles"].values, dims=["time"], attrs=data_nc["n_particles"].attrs
    )

    data["size_classes_width"] = xr.DataArray(
        data_nc["diameter_spread"].values * 1000,
        dims=["size_classes"],
        attrs={"units": "mm", "long_name": "Width of the diameter bins"},
    )
    data["speed_classes_width"] = xr.DataArray(
        data_nc["velocity_spread"].values,
        dims=["speed_classes"],
        attrs={"units": "m/s", "long_name": "Width of the speed bins"},
    )
    return data


def read_parsivel_cloudnet_choice(
    filename: Union[str, Path], radar_frequencies: list, config
) -> xr.Dataset:
    """Format a daily nc file from any daily disdrometer file from CLU.

    Input a CLU daily disdrometer file (either Parsivel or Thies) and output
    a formatted dataset with the useful renamed data variables and metadata.

    Parameters
    ----------
    filename : Union[str, Path]
        CLU daily disdrometer file
    radar_frequencies : list
        List of frequencies at which scattering variables will be computed later
    config : dict
        The configuration file corresponding to the disdrometer data file
        (station, instruments, ...)

    Returns
    -------
    xr.Dataset
        A formatted dataset with disdrometer data, with a structure independent
        from the disdrometer model used for the data acquisition

    """
    data_nc = resample_data_perfect_timesteps(filename=filename, config=config)
    source = data_nc.disdrometer_source

    if source == "OTT HydroMet Parsivel2":
        data = read_parsivel_cloudnet(data_nc)
    elif source == "Thies Clima LNM":
        data = read_thies_cloudnet(data_nc)
    else:
        data = None

    if data is not None:
        for latlon_nc, latlon in zip(
            ["longitude", "latitude", "altitude"],
            ["disdro_longitude", "disdro_latitude", "disdro_altitude"],
        ):
            data[latlon] = data_nc[latlon_nc]
            data[latlon].attrs["long_name"] = f"{latlon_nc} of disdrometer set-up"
        data["disdro_altitude"].attrs["positive"] = "up"

        data.attrs = data_nc.attrs
        data["disdro_model"] = source
        data["disdro_model"].attrs = {
            "long_name": "Disdrometer model",
            "comment": "Disdrometer model",
        }
        data["time_resolution"] = (
            data.time.values[1] - data.time.values[0]
        ) / np.timedelta64(1, "s")
        data["time_resolution"].attrs = {
            "units": "s",
            "long_name": "Time resolution of the preprocessed file",
        }

    data = data.assign_coords({"radar_frequencies": np.array(radar_frequencies)})
    data["radar_frequencies"].attrs.update(
        {
            "units": "Hz",
            "long_name": "Frequencies at which scattering-related variables are computed in the file",  # noqa E501
        }
    )

    return data


def reflectivity_model_multilambda_measmodV_hvfov(
    mparsivel,
    scatt_list,
    n,
    strMethod="GunAndKinzer",
):
    """Add computed scattering-related variables to formatted disdro dataset.

    Input the formatted dataset got with read_cloudnet_choice() ;
    compute the reflectivity in different cases and other variables
    which characterize the DSD ;
    add them in the formatted dataset previously created.

    Parameters
    ----------
    mparsivel : xr.Dataset
        formatted dataset with daily raw disdrometer data
    scatt_list : list
        list of outputs from scattering.scattering_prop() method :
        each output contains the Mie and t-matrix backscattering coefficients for
        all the disdrometer droplet diameter classes, for a specifig configuration
    n : int
        number of diameter classes to be considered in the computation
    strMethod : str, optional
        Method used to model droplet fall speed from disdrometer droplet distribution,
        by default "GunAndKinzer"

    Returns
    -------
    xr.Dataset
        The formatted disdrometer dataset enhanced with the additional variables
        (reflectivity in the different computed configurations, ...).

    """
    # scatt_list : list of scattering_prop() objects :
    # [(lambda1, vert), (lambda2, vert), ...(lambda1, hori), ...] -> 4 lambdas = 8 scatt objects # noqa E501

    # integration time
    t = mparsivel.time_resolution.values  # s

    F = mparsivel.F.data

    fov = 2
    meas_modV = 2
    fr = len(mparsivel.radar_frequencies)

    model = DATA()

    model.RR = np.zeros([len(mparsivel.time)])
    model.VD = np.zeros([len(mparsivel.time), len(mparsivel.size_classes)])
    model.M2 = np.zeros([len(mparsivel.time), meas_modV])
    model.M3 = np.zeros([len(mparsivel.time), meas_modV])
    model.M4 = np.zeros([len(mparsivel.time), meas_modV])
    model.Ze_ray = np.zeros([len(mparsivel.time), meas_modV])

    model.Ze_mie = np.zeros(
        [len(mparsivel.time), len(mparsivel.radar_frequencies), fov, meas_modV]
    )
    model.Ze_tm = np.zeros(
        [len(mparsivel.time), len(mparsivel.radar_frequencies), fov, meas_modV]
    )
    model.attenuation = np.zeros(
        [len(mparsivel.time), len(mparsivel.radar_frequencies), fov, meas_modV]
    )
    model.V_tm = np.zeros(
        [len(mparsivel.time), len(mparsivel.radar_frequencies), fov, meas_modV]
    )
    model.V_mie = np.zeros(
        [len(mparsivel.time), len(mparsivel.radar_frequencies), fov, meas_modV]
    )
    model.psd_sum = np.zeros(
        [len(mparsivel.size_classes), len(mparsivel.speed_classes)]
    )  # time-integrated psd
    model.psd_sum_n = np.zeros(
        [len(mparsivel.size_classes), len(mparsivel.speed_classes)]
    )  # time-integrated psd

    model.diameter_bin_width_mm = mparsivel.size_classes_width.values

    # parameterisation for the velocity (Gun and Kinzer)
    VDmodel = compute_fallspeed(mparsivel.size_classes.values, strMethod=strMethod)

    for ii in range(len(mparsivel.time)):
        # sum over speed axis -> number of drops per time and size
        # replace axis=1 by 0 if not transposed in parsivel
        Ni = np.nansum(mparsivel.psd.values[ii, :, :], 1)

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

        VDD = np.vstack(
            (model.VD[ii, :], VDmodel)
        ).T  # 1st col : VD measured, 2nd col : VD from model law

        model.M2[ii, :] = (
            (
                np.nansum(
                    np.tile(
                        Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 2), (2, 1)
                    ).T
                    / VDD,
                    axis=0,
                )
            )
            / F
            / t
        )
        model.M3[ii, :] = (
            (
                np.nansum(
                    np.tile(
                        Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 4), (2, 1)
                    ).T
                    / VDD,
                    axis=0,
                )
            )
            / F
            / t
        )
        model.M4[ii, :] = (
            (
                np.nansum(
                    np.tile(
                        Ni * ((mparsivel.size_classes.values * 1.0e-3) ** 4), (2, 1)
                    ).T
                    / VDD,
                    axis=0,
                )
            )
            / F
            / t
        )
        model.Ze_ray[ii, :] = (
            (
                np.nansum(
                    np.tile(Ni * (mparsivel.size_classes.values**6), (2, 1)).T / VDD,
                    axis=0,
                )
            )
            / F
            / t
        )

        model.psd_sum = (
            mparsivel.psd.sum(dim="time").values
            / F
            / t
            / np.tile(VDmodel.flatten(), (len(mparsivel.speed_classes), 1)).T
            / 1e6
        )  # / ) ? Particles/cm3
        model.psd_sum_n = (
            model.psd_sum
            / np.tile(model.diameter_bin_width_mm, (len(mparsivel.speed_classes), 1)).T
        )
        # # dsd with velocity from parsivel
        # model.dsd[ii, :] = (nul
        #     Ni / model.VD[ii, :] / F / t / (model.diameter_bin_width_mm * 1.0e-3)
        # )  # particles/m3 #normalised per diameter bin width

        ## Here begins the multilambda specific part
        for k, scatt in enumerate(scatt_list):
            model.Ze_mie[ii, k % fr, k // fr, :] = (
                np.nansum(
                    np.tile(Ni[0:n] * scatt.bscat_mie, (2, 1)).T / VDD[0:n, :],
                    axis=0,
                )
                / F
                / t
            )  # mm6/m3

            model.Ze_tm[ii, k % fr, k // fr, :] = (
                np.nansum(
                    np.tile(Ni[0:n] * scatt.bscat_tmatrix, (2, 1)).T / VDD[0:n, :],
                    axis=0,
                )
                / F
                / t
            )  # mm6/m3
            model.Ze_tm[np.where(model.Ze_tm == 0)] = np.nan

            model.attenuation[ii, k % fr, k // fr, :] = (
                np.nansum(
                    np.tile(Ni[0:n] * scatt.att_tmatrix, (2, 1)).T / VDD[0:n, :], axis=0
                )
                / F
                / t
            )  # dB/km

            model.V_mie[ii, k % fr, k // fr, :] = np.nansum(
                Ni[0:n] * scatt.bscat_mie
            ) / np.nansum(
                np.tile(Ni[0:n] * scatt.bscat_mie, (2, 1)).T / VDD[0:n, :], axis=0
            )  # m/s

            model.V_tm[ii, k % fr, k // fr, :] = np.nansum(
                Ni[0:n] * scatt.bscat_tmatrix
            ) / np.nansum(
                np.tile(Ni[0:n] * scatt.bscat_tmatrix, (2, 1)).T / VDD[0:n, :], axis=0
            )  # m/s

    DensityLiquidWater = 1000.0e3  # g/m3

    # store results in parsivel object
    mparsivel["disdro_pr_from_raw"] = xr.DataArray(
        model.RR,
        dims=["time"],
        attrs={
            "units": "mm/h",
            "long_name": "Disdrometer-based precipitation rate from raw data",
            "comment": "Calculated with disdrometer, rainfall rate is computed from droplet diameter distribution",  # noqa E501
        },
    )
    mparsivel["measV"] = xr.DataArray(
        model.VD,
        dims=["time", "size_classes"],
        attrs={
            "units": "m/s",
            "long_name": "measured fall speed at each timestep as a function of the diameter",  # noqa E501
        },
    )
    mparsivel["modV"] = xr.DataArray(
        VDmodel,
        dims=["size_classes"],
        attrs={
            "units": "m/s",
            "long_name": "modeled droplet fall speed as a function of the diameter",
        },
    )

    mparsivel["psd_sum_n"] = xr.DataArray(
        model.psd_sum_n,
        dims=["size_classes", "speed_classes"],
        attrs={
            "units": "#/cm^3/mm",
            "long_name": "Normalized precipitation size distribution",
        },
    )
    mparsivel["psd_sum"] = xr.DataArray(
        model.psd_sum,
        dims=["size_classes", "speed_classes"],
        attrs={
            "units": "#/cm^3/bin",
            "long_name": "Normalized precipitation size distribution",
            "comment": "Unit not recognized by UDUNITS ; does not account for the size of the bins",  # noqa E501
        },
    )

    mparsivel["Zdlin_hfov_measv_mie"] = xr.DataArray(
        model.Ze_mie[:, :, 1, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Mie reflectivity in lin scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_hfov_measv_mie"] = xr.DataArray(
        10 * np.log10(model.Ze_mie[:, :, 1, 0]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Mie reflectivity in log scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_hfov_modv_mie"] = xr.DataArray(
        model.Ze_mie[:, :, 1, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Mie reflectivity in lin scale for modeled fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_hfov_modv_mie"] = xr.DataArray(
        10 * np.log10(model.Ze_mie[:, :, 1, 1]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Mie reflectivity in log scale for modeled fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_vfov_measv_mie"] = xr.DataArray(
        model.Ze_mie[:, :, 0, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Mie reflectivity in lin scale for measured fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_vfov_measv_mie"] = xr.DataArray(
        10 * np.log10(model.Ze_mie[:, :, 0, 0]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Mie reflectivity in log scale for measured fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_vfov_modv_mie"] = xr.DataArray(
        model.Ze_mie[:, :, 0, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Mie reflectivity in lin scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_vfov_modv_mie"] = xr.DataArray(
        10 * np.log10(model.Ze_mie[:, :, 0, 1]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Mie reflectivity in log scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )

    mparsivel["Zdlin_hfov_measv_tm"] = xr.DataArray(
        model.Ze_tm[:, :, 1, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer geometric reflectivity in lin scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_hfov_measv_tm"] = xr.DataArray(
        10 * np.log10(model.Ze_tm[:, :, 1, 0]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer geometric reflectivity in log scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_hfov_modv_tm"] = xr.DataArray(
        model.Ze_tm[:, :, 1, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer geometric reflectivity in lin scale for modeled fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_hfov_modv_tm"] = xr.DataArray(
        10 * np.log10(model.Ze_tm[:, :, 1, 1]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer geometric reflectivity in log scale for modeled fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_vfov_measv_tm"] = xr.DataArray(
        model.Ze_tm[:, :, 0, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer geometric reflectivity in lin scale for measured fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_vfov_measv_tm"] = xr.DataArray(
        10 * np.log10(model.Ze_tm[:, :, 0, 0]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer geometric reflectivity in log scale for measured fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_vfov_modv_tm"] = xr.DataArray(
        model.Ze_tm[:, :, 0, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer geometric reflectivity in lin scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_vfov_modv_tm"] = xr.DataArray(
        10 * np.log10(model.Ze_tm[:, :, 0, 1]),
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer geometric reflectivity in log scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
            "coverage_content_type": "modelResult",
        },
    )

    mparsivel["Zdlin_measv_ray"] = xr.DataArray(
        model.Ze_ray[:, 0],
        dims=["time"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Rayleigh reflectivity in lin scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_measv_ray"] = xr.DataArray(
        10 * np.log10(model.Ze_ray[:, 0]),
        dims=["time"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Rayleigh reflectivity in log scale for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlin_modv_ray"] = xr.DataArray(
        model.Ze_ray[:, 1],
        dims=["time"],
        attrs={
            "units": "mm^6.m^-3",
            "long_name": "Disdrometer Rayleigh reflectivity in lin scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )
    mparsivel["Zdlog_modv_ray"] = xr.DataArray(
        10 * np.log10(model.Ze_ray[:, 1]),
        dims=["time"],
        attrs={
            "units": "dBZ",
            "long_name": "Disdrometer Rayleigh reflectivity in log scale for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer",
        },
    )

    mparsivel["attd_hfov_measv"] = xr.DataArray(
        model.attenuation[:, :, 1, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dB/km",
            "long_name": "Disdrometer attenuation for measured fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer ; unit not recognized by UDUNITS",
        },
    )
    mparsivel["attd_hfov_modv"] = xr.DataArray(
        model.attenuation[:, :, 1, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dB/km",
            "long_name": "Disdrometer attenuation for modeled fall drop velocity and horizontal field of view",  # noqa E501
            "comment": "Calculated with disdrometer ; unit not recognized by UDUNITS",
        },
    )
    mparsivel["attd_vfov_measv"] = xr.DataArray(
        model.attenuation[:, :, 0, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dB/km",
            "long_name": "Disdrometer attenuation for measured fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer ; unit not recognized by UDUNITS",
        },
    )
    mparsivel["attd_vfov_modv"] = xr.DataArray(
        model.attenuation[:, :, 0, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "dB/km",
            "long_name": "Disdrometer attenuation for modeled fall drop velocity and vertical field of view",  # noqa E501
            "comment": "Calculated with disdrometer ; unit not recognized by UDUNITS",
        },
    )

    mparsivel["DVd_hfov_measv_mie"] = xr.DataArray(
        model.V_mie[:, :, 1, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer Mie Doppler velocity for measured fall drop velocity and horizontal field of view",  # noqa E501
        },
    )
    mparsivel["DVd_hfov_modv_mie"] = xr.DataArray(
        model.V_mie[:, :, 1, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer Mie Doppler velocity for modeled fall drop velocity and horizontal field of view",  # noqa E501
        },
    )
    mparsivel["DVd_vfov_measv_mie"] = xr.DataArray(
        model.V_mie[:, :, 0, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer Mie Doppler velocity for measured fall drop velocity and vertical field of view",  # noqa E501
        },
    )
    mparsivel["DVd_vfov_modv_mie"] = xr.DataArray(
        model.V_mie[:, :, 0, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer Mie Doppler velocity for modeled fall drop velocity and vertical field of view",  # noqa E501
        },
    )

    mparsivel["DVd_hfov_measv_tm"] = xr.DataArray(
        model.V_tm[:, :, 1, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer geometric Doppler velocity for measured fall drop velocity and horizontal field of view",  # noqa E501
        },
    )
    mparsivel["DVd_hfov_modv_tm"] = xr.DataArray(
        model.V_tm[:, :, 1, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer geometric Doppler velocity for modeled fall drop velocity and horizontal field of view",  # noqa E501
        },
    )
    mparsivel["DVd_vfov_measv_tm"] = xr.DataArray(
        model.V_tm[:, :, 0, 0],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer geometric Doppler velocity for measured fall drop velocity and vertical field of view",  # noqa E501
        },
    )
    mparsivel["DVd_vfov_modv_tm"] = xr.DataArray(
        model.V_tm[:, :, 0, 1],
        dims=["time", "radar_frequencies"],
        attrs={
            "units": "m.s^-1",
            "long_name": "Disdrometer geometric Doppler velocity for modeled fall drop velocity and vertical field of view",  # noqa E501
        },
    )

    mparsivel["m2_measv"] = xr.DataArray(
        model.M2[:, 0],
        dims=["time"],
        attrs={
            "units": "mm^2",
            "long_name": "Second order momentum for measured fall drop velocity",
        },
    )
    mparsivel["m2_modv"] = xr.DataArray(
        model.M2[:, 1],
        dims=["time"],
        attrs={
            "units": "mm^2",
            "long_name": "Second order momentum for modeled fall drop velocity",
        },
    )
    mparsivel["m3_measv"] = xr.DataArray(
        model.M3[:, 0],
        dims=["time"],
        attrs={
            "units": "mm^3",
            "long_name": "Third order momentum for measured fall drop velocity",
        },
    )
    mparsivel["m3_modv"] = xr.DataArray(
        model.M3[:, 1],
        dims=["time"],
        attrs={
            "units": "mm^3",
            "long_name": "Third order momentum for modeled fall drop velocity",
        },
    )
    mparsivel["m4_measv"] = xr.DataArray(
        model.M4[:, 0],
        dims=["time"],
        attrs={
            "units": "mm^4",
            "long_name": "Fourth order momentum for measured fall drop velocity",
        },
    )
    mparsivel["m4_modv"] = xr.DataArray(
        model.M4[:, 1],
        dims=["time"],
        attrs={
            "units": "mm^4",
            "long_name": "Fourth order momentum for modeled fall drop velocity",
        },
    )

    # additional parameters

    mparsivel["dm_measv"] = xr.DataArray(
        model.M4[:, 0] / model.M3[:, 0],
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Mean diameter of the precipitation for measured fall drop velocity",  # noqa E501
        },
    )
    mparsivel["dm_modv"] = xr.DataArray(
        model.M4[:, 1] / model.M3[:, 1],
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Mean diameter of the precipitation for modeled fall drop velocity",  # noqa E501
        },
    )
    mparsivel["re_measv"] = xr.DataArray(
        0.5 * model.M3[:, 0] / model.M2[:, 0],
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Effective radius of the precipitation for measured fall drop velocity",  # noqa E501
        },
    )
    mparsivel["re_modv"] = xr.DataArray(
        0.5 * model.M3[:, 1] / model.M2[:, 1],
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Effective radius of the precipitation for modeled fall drop velocity",  # noqa E501
        },
    )
    mparsivel["lwc_measv"] = xr.DataArray(
        DensityLiquidWater * (1.0 / 6.0) * np.pi * model.M3[:, 0],
        dims=["time"],
        attrs={
            "units": "g/cm^3",
            "long_name": "Liquid Water Content for measured fall drop velocity",
        },
    )
    mparsivel["lwc_modv"] = xr.DataArray(
        DensityLiquidWater * (1.0 / 6.0) * np.pi * model.M3[:, 1],
        dims=["time"],
        attrs={
            "units": "g/cm^3",
            "long_name": "Liquid Water Content for modeled fall drop velocity",
        },
    )
    mparsivel["n0_measv"] = xr.DataArray(
        (4.0**4 / (np.pi * DensityLiquidWater))
        * mparsivel.lwc_measv.values
        / (mparsivel.dm_measv.values) ** 4,
        dims=["time"],
        attrs={
            "units": "#/cm^3",
            "long_name": "Total number concentration for measured fall drop velocity",
        },
    )
    mparsivel["n0_modv"] = xr.DataArray(
        (4.0**4 / (np.pi * DensityLiquidWater))
        * mparsivel.lwc_modv.values
        / (mparsivel.dm_modv.values) ** 4,
        dims=["time"],
        attrs={
            "units": "#/cm^3",
            "long_name": "Total number concentration for modeled fall drop velocity",
        },
    )

    return mparsivel
