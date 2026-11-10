"""Detection of the radar receiver saturation (MIRA ceiling).

In heavy rain the receiver of some radars saturates: the SNR stops growing and
piles up just below a fixed value, so the reflectivity is capped. The ceiling is
found day by day and gate by gate from the SNR of the Cloudnet radar file:

- the day's cap is the CAP_RANK-th largest SNR of the gate;
- the gate saturates that day when many profiles sit within NEAR_DB of the cap
  (N_STRONG, or N_MIN with RATIO_MIN times more profiles there than between
  BAND_DB below the cap) and the cap is high (SNR_CAP_MIN): without saturation
  only a few profiles reach the top of the distribution.

Each minute is then flagged at each gate (QF_saturation):
1 when at least MINUTE_FRACTION of its profiles are at the ceiling,
2 when the radar is not at the ceiling but the disdrometer reflectivity is
within ZDD_MARGIN of it: the radar could not have measured that minute fully,
and keeping it would bias Delta Z low, the minutes above the ceiling being
removed while those just under it are kept.

Radars without SNR in their file (RPG, BASTA) are not assessed.
"""

import numpy as np
import xarray as xr

CAP_RANK = 20
NEAR_DB = 1.0
BAND_DB = (2.0, 4.0)
MIN_PROFILES = 200
N_STRONG = 300
N_MIN = 100
RATIO_MIN = 1.5
SNR_CAP_MIN = 50.0
MINUTE_FRACTION = 0.1
ZDD_MARGIN = 4.0

QF_ATTRS = {
    "long_name": "Quality flag for radar receiver saturation",
    "flag_values": np.array([0, 1, 2], dtype="i1"),
    "flag_meanings": "no_saturation radar_at_ceiling disdrometer_reflectivity_near_ceiling",  # noqa E501
    "comment": (
        f"1: at least {MINUTE_FRACTION:.0%} of the profiles of the minute within "
        f"{NEAR_DB:g} dB of the day's SNR ceiling of the gate ; 2: Zdd within "
        f"{ZDD_MARGIN:g} dB of the day's reflectivity ceiling of the gate. "
        "Ceiling detected from the pile-up of SNR values below their maximum."
    ),
}


def _kth_largest(values, k=CAP_RANK):
    return np.partition(values, values.size - k)[values.size - k]


def detect_ceiling(snr, zh):
    """Find the day's receiver ceiling of each gate.

    Parameters
    ----------
    snr, zh : np.ndarray
        SNR (dB) and reflectivity (dBZ) of every profile, shape (time, range).

    Returns
    -------
    snr_cap, zh_ceiling : np.ndarray
        Ceiling of each gate in SNR and in reflectivity, NaN where the gate does
        not saturate that day.
    at_ceiling : np.ndarray of bool
        Profiles within NEAR_DB of the ceiling, shape (time, range).

    """
    nrange = snr.shape[1]
    snr_cap = np.full(nrange, np.nan)
    zh_ceiling = np.full(nrange, np.nan)
    at_ceiling = np.zeros(snr.shape, dtype=bool)
    for j in range(nrange):
        s = snr[:, j]
        finite = np.isfinite(s)
        if finite.sum() < MIN_PROFILES:
            continue
        cap = _kth_largest(s[finite])
        if cap < SNR_CAP_MIN:
            continue
        near = finite & (s >= cap - NEAR_DB)
        n_near = int(near.sum())
        n_band = int(((s >= cap - BAND_DB[1]) & (s < cap - BAND_DB[0])).sum())
        if not (
            n_near >= N_STRONG
            or (n_near >= N_MIN and n_near >= RATIO_MIN * max(n_band, 1))
        ):  # noqa E501
            continue
        snr_cap[j] = cap
        at_ceiling[:, j] = near
        z = zh[:, j][np.isfinite(zh[:, j])]
        if z.size >= CAP_RANK:
            zh_ceiling[j] = _kth_largest(z)
    return snr_cap, zh_ceiling, at_ceiling


def radar_saturation(data_nc, time_bins, labels):
    """Saturation variables of the radar dataset, on the 1-minute grid.

    Parameters
    ----------
    data_nc : xr.Dataset
        Radar data at full resolution (Zh, and SNR when the file has it).
    time_bins, labels :
        Bins and labels of the 1-minute resampling of the reflectivity.

    Returns
    -------
    dict of xr.DataArray
        QF_saturation (time, range), saturation_snr_cap and
        Zdcr_saturation_ceiling (range).

    """
    nrange = data_nc.range.size
    if "SNR" not in data_nc:
        comment = "not assessed : no SNR in the radar file"
        qf = np.zeros((labels.size, nrange), dtype="i1")
        snr_cap = zh_ceiling = np.full(nrange, np.nan)
    else:
        comment = None
        snr_cap, zh_ceiling, at_ceiling = detect_ceiling(
            data_nc["SNR"].values.astype(float), data_nc["Zh"].values.astype(float)
        )
        fraction = (
            xr.DataArray(
                at_ceiling.astype(float),
                dims=["time", "range"],
                coords={"time": data_nc.time},
            )
            .groupby_bins("time", time_bins, labels=labels)
            .mean(dim="time")
        )
        qf = (fraction.fillna(0).values >= MINUTE_FRACTION).astype("i1")
    out = {
        "QF_saturation": xr.DataArray(qf, dims=["time", "range"], attrs=dict(QF_ATTRS)),
        "saturation_snr_cap": xr.DataArray(
            snr_cap.astype(np.float32),
            dims=["range"],
            attrs={
                "long_name": "SNR ceiling of the radar receiver detected for the day",
                "units": "dB",
                "comment": "NaN where the gate does not saturate that day",
            },
        ),
        "Zdcr_saturation_ceiling": xr.DataArray(
            zh_ceiling.astype(np.float32),
            dims=["range"],
            attrs={
                "long_name": "Reflectivity ceiling of the radar receiver detected for the day",  # noqa E501
                "units": "dBZ",
                "comment": "NaN where the gate does not saturate that day",
            },
        ),
    }
    if comment is not None:
        for var in out.values():
            var.attrs["comment"] = comment
    return out


def flag_disdrometer_near_ceiling(ds):
    """Set QF_saturation to 2 where Zdd is within ZDD_MARGIN of the ceiling."""
    if "QF_saturation" not in ds or "Zdlog_vfov_modv_tm" not in ds:
        return ds
    zdd = ds["Zdlog_vfov_modv_tm"].sel(
        radar_frequencies=ds["radar_frequency"], method="nearest"
    )
    near = zdd >= ds["Zdcr_saturation_ceiling"] - ZDD_MARGIN  # NaN ceiling: False
    qf = ds["QF_saturation"]
    flagged = xr.where((qf == 0) & near, 2, qf).transpose(*qf.dims).astype("i1")
    flagged.attrs = qf.attrs
    ds["QF_saturation"] = flagged
    return ds


def saturation_qc(ds, conf):
    """QC_sat at the Delta Z gate (1 = no saturation)."""
    r = conf["instrument_parameters"]["DCR_DZ_RANGE"]
    if "QF_saturation" in ds:
        qf = ds["QF_saturation"].sel(range=r, method="nearest")
        data = (qf.fillna(0) == 0).astype("i2").values
        comment = (
            f"from QF_saturation at the gate closest to {r} m (DCR_DZ_RANGE) : "
            "radar at its receiver ceiling or Zdd within "
            f"{ZDD_MARGIN:g} dB of it"
        )
    else:
        data = np.ones(ds.time.size, dtype="i2")
        comment = "not assessed : preprocessed file without QF_saturation"
    return xr.DataArray(
        data=data,
        dims=["time"],
        attrs={
            "long_name": "Quality check for radar receiver saturation",
            "comment": comment,
            "flag_values": np.array([0, 1]).astype("i2"),
            "flag_meanings": "radar_saturated_or_zdd_near_ceiling no_saturation",
        },
    )
