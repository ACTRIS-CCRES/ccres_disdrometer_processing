import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import constants

from ccres_disdrometer_processing.saturation import radar_saturation
from ccres_disdrometer_processing.utils import NOMINAL_INTERVAL

LIST_VARIABLES = ["Zh", "v", "radar_frequency", "latitude", "longitude", "altitude"]

# Alignment of the radar on the disdrometer telegrams (Zdcr_aligned) :
# the radar window covers the measurement interval of the telegram, advanced by
# the effective fall time of the drops from each gate to the ground,
# FALL_TIME_FACTOR x range / |v| : the drops advected by the wind reach the
# disdrometer sooner than their fall time (factor fitted on 17 radar-disdrometer
# combinations of the network)
FALL_TIME_FACTOR = 0.55
# end : the time written marks the end of the measurement interval (Parsivel and
# Thies loggers read by cloudnetpy) ; start : its beginning (MeteoSwiss files)
TIMESTAMP_CONVENTIONS = ("end", "start")
RADAR_WINDOW = NOMINAL_INTERVAL  # s, duration of the radar window
MIN_FALL_SPEED = 0.5  # m/s, slower gates (no rain) are not used for |v|
MIN_WINDOW_COVERAGE = 0.6  # share of the radar profiles expected in a window


def read_radar_frequency(filename):
    """Return the frequency of a CLU daily DCR file (Hz)."""
    with xr.open_dataset(filename) as raw:
        return float(np.nanmedian(raw["radar_frequency"].values)) * 1e9


def fall_time_lag(rng, dv, rain, factor=FALL_TIME_FACTOR):
    """Return the lag of the drops from each gate to the ground (s).

    factor x tau, tau = r / |v| with |v| the median over the gates up to r of the
    median Doppler velocity of each gate during the rain minutes of the day
    (gates slower than MIN_FALL_SPEED left out). 0 without rain.

    Parameters
    ----------
    rng : np.ndarray
        Range of the gates (m)
    dv : np.ndarray
        Doppler velocity, (time, range) (m/s)
    rain : np.ndarray
        Rain minutes, (time,) (bool)
    factor : float, optional
        Share of the fall time, by default FALL_TIME_FACTOR

    Returns
    -------
    np.ndarray
        Lag of each gate (s)

    """
    lag = np.zeros(rng.size)
    if not np.any(rain):
        return lag
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmedian(np.abs(dv[rain]), axis=0)
    v[~(v >= MIN_FALL_SPEED)] = np.nan
    for j in range(rng.size):
        if np.isfinite(v[: j + 1]).any():
            lag[j] = factor * rng[j] / np.nanmedian(v[: j + 1])
    return lag


def window_median(rt, z, lo, hi, nmin):
    """Return the median of z over the windows (lo, hi] of the times rt.

    Parameters
    ----------
    rt : np.ndarray
        Sorted times of the samples (s)
    z : np.ndarray
        Values of the samples, (time,) ; NaN are ignored
    lo, hi : np.ndarray
        Bounds of the windows (s)
    nmin : int
        Windows holding fewer samples are NaN

    Returns
    -------
    np.ndarray
        Median of each window

    """
    i0 = np.searchsorted(rt, lo, side="right")
    i1 = np.searchsorted(rt, hi, side="right")
    i0[~np.isfinite(lo)] = 0
    i1[~np.isfinite(lo)] = 0
    count = i1 - i0
    out = np.full(lo.size, np.nan)
    if rt.size == 0 or count.max() <= 0:
        return out
    idx = i0[:, None] + np.arange(count.max())[None, :]
    inside = idx < i1[:, None]
    g = np.where(inside, z[np.clip(idx, 0, rt.size - 1)], np.nan)
    with warnings.catch_warnings():  # windows without echo are NaN, as expected
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(g, axis=1)
    out[count < nmin] = np.nan
    return out


def aligned_reflectivity(
    data_nc,
    labels,
    telegram_time,
    rain,
    dv,
    factor=FALL_TIME_FACTOR,
    convention="end",
):
    """Return Zh aligned on the disdrometer telegrams, and the lag of each gate.

    For the telegram written at s, the radar window is (s - W, s] for the "end"
    convention, (s, s + W] for "start", advanced by the fall lag of the gate ; for
    a minute without telegram, (t - W/2, t + W/2] around its label t, advanced the
    same way. W is RADAR_WINDOW. Median in dBZ, as for Zdcr.

    Parameters
    ----------
    data_nc : xr.Dataset
        Radar data, with Zh (time, range)
    labels : pd.DatetimeIndex
        Minutes of the preprocessed file
    telegram_time : np.ndarray
        Time of the disdrometer telegram of each minute (datetime64, NaT if none)
    rain : np.ndarray
        Rain minutes (bool), for the fall speed
    dv : np.ndarray
        Doppler velocity per minute, (time, range) (m/s)
    factor : float, optional
        Share of the fall time, by default FALL_TIME_FACTOR
    convention : str, optional
        What the telegram time marks, "end" (default) or "start" of its interval

    Returns
    -------
    np.ndarray, np.ndarray
        Aligned Zh (time, range) and lag of each gate (s)

    """
    if convention not in TIMESTAMP_CONVENTIONS:
        raise ValueError(
            f"timestamp convention {convention!r} not in {TIMESTAMP_CONVENTIONS}"
        )
    t0 = labels[0].to_datetime64()

    def seconds(t):
        return (np.asarray(t, dtype="datetime64[ns]") - t0) / np.timedelta64(1, "s")

    rt = seconds(data_nc.time.values)
    lab = seconds(labels.values)
    s = seconds(telegram_time)
    if convention == "end":
        lo, hi = s - RADAR_WINDOW, s
    else:
        lo, hi = s, s + RADAR_WINDOW
    none = ~np.isfinite(s)
    lo[none], hi[none] = lab[none] - RADAR_WINDOW / 2, lab[none] + RADAR_WINDOW / 2

    rng = data_nc.range.values.astype(float)
    lag = fall_time_lag(rng, dv, rain, factor)
    dt = float(np.median(np.diff(rt))) if rt.size > 1 else RADAR_WINDOW
    nmin = max(1, int(MIN_WINDOW_COVERAGE * RADAR_WINDOW / dt))
    zh = data_nc.Zh.transpose("time", "range").values.astype(float)
    out = np.full((lab.size, rng.size), np.nan)
    for j in range(rng.size):
        out[:, j] = window_median(rt, zh[:, j], lo - lag[j], hi - lag[j], nmin)
    return out, lag


def read_radar_cloudnet(
    filename,
    max_radar_alt=2500,
    disdro=None,
    fall_time_factor=FALL_TIME_FACTOR,
    timestamp_convention="end",
):
    """Open, extract and resample data from CLU daily DCR files.

    Parameters
    ----------
    filename : Union[str, Path]
        The considered daily DCR file from CLU
    max_radar_alt : int, optional
        Altitude until which to extract 2D data (Z, DV), by default 2500m
    disdro : xr.Dataset, optional
        Disdrometer data of the day (disdro_telegram_time, disdro_pr) : when
        given, Zh is also aligned on its telegrams (Zdcr_aligned)
    fall_time_factor : float, optional
        Share of the fall time for the alignment, by default FALL_TIME_FACTOR
    timestamp_convention : str, optional
        What the disdrometer telegram time marks, "end" (default) or "start" of
        its measurement interval

    Returns
    -------
    xr.Dataset
        A formatted dataset for radar data (specs, Z, DV and range vector)
        with a 1-minute regular sampling

    """
    range_bounds = [0, max_radar_alt]
    # in memory first : grouping a lazily read file reads it piece by piece, slowly
    with xr.open_dataset(filename) as raw:
        # the SNR, when the file has it (MIRA), reveals the receiver saturation
        variables = LIST_VARIABLES + (["SNR"] if "SNR" in raw else [])
        data_nc = (
            raw[variables].sel(range=slice(range_bounds[0], range_bounds[1])).load()
        )

    start_time = pd.Timestamp(data_nc.time.values[0]).replace(
        hour=0, minute=0, second=0, microsecond=0, nanosecond=0
    )
    # the day of the file, from its first sample as for the disdrometer and
    # weather station : some radar files end with a sample at 00:00 the next
    # day, which would otherwise add a whole second day to the time grid
    end_time = pd.Timestamp(data_nc.time.values[0]).replace(
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

    # manage case were lat/lon/alt are 1D arrays instead of scalars
    # linked to change of file format in cloudnet L1
    if data_nc.longitude.values.size > 1:
        radar_longitude = data_nc.longitude.values[0]
    else:
        radar_longitude = data_nc.longitude.values
    if data_nc.latitude.values.size > 1:
        radar_latitude = data_nc.latitude.values[0]
    else:
        radar_latitude = data_nc.latitude.values
    if data_nc.altitude.values.size > 1:
        radar_altitude = data_nc.altitude.values[0]
    else:
        radar_altitude = data_nc.altitude.values

    radar_ds["radar_longitude"] = xr.DataArray(
        radar_longitude, attrs=data_nc.longitude.attrs
    )
    radar_ds["radar_latitude"] = xr.DataArray(
        radar_latitude, attrs=data_nc.latitude.attrs
    )
    radar_ds["radar_altitude"] = xr.DataArray(
        radar_altitude, attrs=data_nc.altitude.attrs
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

    for name, var in radar_saturation(
        data_nc, time_index_offset, time_index[:-1]
    ).items():
        radar_ds[name] = var

    if disdro is not None and "disdro_telegram_time" in disdro:
        on_grid = disdro.reindex(time=radar_ds.time.values)
        rain = np.nan_to_num(on_grid["disdro_pr"].values) > 0
        z_aligned, lag = aligned_reflectivity(
            data_nc,
            time_index[:-1],
            on_grid["disdro_telegram_time"].values,
            rain,
            radar_ds["DVdcr"].values,
            factor=fall_time_factor,
            convention=timestamp_convention,
        )
        radar_ds["Zdcr_aligned"] = xr.DataArray(
            z_aligned.astype(np.float32),
            dims=["time", "range"],
            attrs={
                **data_nc["Zh"].attrs,
                "comment": (
                    "Median of Zh over the measurement interval of the "
                    f"disdrometer telegram of the minute ({RADAR_WINDOW:.0f} s, "
                    f"timestamp marking its {timestamp_convention}), advanced by "
                    "fall_time_lag ; around the minute label when no telegram"
                ),
            },
        )
        radar_ds["fall_time_lag"] = xr.DataArray(
            lag.astype(np.float32),
            dims=["range"],
            attrs={
                "units": "s",
                "long_name": "Lag of the drops from each gate to the disdrometer",
                "comment": (
                    f"{fall_time_factor} x range / |v|, |v| the median over the "
                    "gates up to the range of the median Doppler velocity of each "
                    "gate during the rain minutes of the day"
                ),
            },
        )

    radar_ds.attrs["radar_source"] = data_nc.attrs["source"]
    radar_ds.attrs["radar_pid"] = data_nc.attrs["instrument_pid"]

    data_nc.close()

    return radar_ds
