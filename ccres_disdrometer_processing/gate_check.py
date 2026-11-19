r"""Optimal Delta Z gate of a radar-disdrometer combination, and its stability.

Run it from time to time (yearly, after an intervention on the radar), on the
preprocessed and processed files of one combination :

    python -m ccres_disdrometer_processing.gate_check \
        --combo juelich_mira-35_0366fa69_parsivel_2a1ca46e \
        --preprocessed-dir preprocessing_outputs/juelich \
        --processed-dir processing_outputs/juelich \
        --config conf_stations/conf_juelich_mira-parsivel.toml \
        --period year --output-dir gate_check/juelich

Method (reference line) :

1. good events of the processed files (rain accumulation and rain gauge checks
   passed, at least MIN_POINTS QC-passed minutes). QC_overall only checks the
   saturation at DCR_DZ_RANGE : a minute flagged by QF_saturation at any gate up
   to R_SEARCH is dropped as well, the MIRA reference layer being the first to
   saturate. The whole minute is dropped, not only its saturated gates : that
   would take the strongest minutes out of the farther gates alone ;
2. a selection of steady stratiform rain, from criteria that never use Delta Z
   (LEVELS, nested from "large" to "strict") : the most demanding level holding
   at least MIN_EVENTS events is used. Convective events have a near-radar excess
   reaching higher, which blurs the gate ;
3. per event and gate, the median Delta Z of the QC-passed minutes, corrected for
   the two-way rain attenuation ; its deviation from a straight line fitted over
   R_REF, a layer assumed free of near-range effects (its slope absorbs the
   residual attenuation, gas included) ;
4. the composite deviation (median over the events) : the optimal gate is the
   lowest one above which every gate up to R_REF stays within TOL of the line,
   a deficit (incomplete overlap) as well as an excess ;
5. N_BOOT bootstrap draws of the events : the recommended gate is the
   BOOT_QUANTILE quantile of the drawn gates (between two gates, the higher one :
   a systematic excess costs more than a little precision). Bistatic radars
   (separate transmit and receive antennas : RPG-FMCW, BASTA) have a near-range
   deficit that fades slowly, still about -0.2 dB at the first gate within TOL :
   their recommended gate is at least the gate above it. Monostatic radars (MIRA)
   have an excess that fades within one gate, and their farther gates saturate
   first : no extra gate ;
6. by period (year, or instrument log epochs) : the deviation at the recommended
   gate tells whether a single gate holds within TOL over the whole history ;
7. double check from the radar alone : on the same minutes, the reflectivity
   Zdcr of each minute is normalised by its mean over [R_REF, R_SEARCH] (in dBZ),
   and the median normalised profile of each event goes through steps 3 to 6
   (reference line, composite, optimal gate, bootstrap and bistatic rule) without
   the disdrometer. The two recommended gates are compared : equal within a gate,
   the near-range effect is the radar's ; a higher Delta Z gate points to the
   disdrometer side (alignment, fall time, simulated Z), which is no reason to
   raise the gate. A real vertical gradient of the rain below R_REF shows in both.

The parameters (R_MAX, R_SEARCH, R_REF, TOL, MIN_EVENTS, MIN_EVENTS_PERIOD,
N_BOOT, BOOT_QUANTILE, DV_RATIO and the LEVELS, from the most lenient to the most
demanding) are read in the [gate_check] section of the station configuration,
MIN_POINTS in its [thresholds]. The configuration files are left untouched : the
recommendation is to be checked before updating DCR_DZ_RANGE.
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ccres_disdrometer_processing.config import GATE_LEVEL_KEYS, load_config
from ccres_disdrometer_processing.utils import rain_attenuation

lgr = logging.getLogger(__name__)

# antenna layout from the radar type of the combination name
MONOSTATIC = ("mira",)
BISTATIC = ("rpg", "basta")

# Stratiform selection descriptors of each level of [gate_check.levels]
# (n_qc : QC-passed minutes ; rain_frac, qc_frac : share of the event minutes
# with rain, passing the QC ; lpr_std : std of log10(rain rate) ; pr_p95 : 95th
# percentile of the rain rate (mm/h) ; z5_jump : median minute-to-minute change of
# Z at 500 m (dB) ; dv_ratio : |v| at 900 m / near the ground ; dv_lo : |v| near
# the ground (m/s) ; z_grad : |Z(900 m) - Z(500 m)| (dB)) ; MINIMA are lower
# bounds, the others upper bounds
MINIMA = ("n_qc", "rain_frac", "qc_frac", "dv_ratio", "dv_lo")
LOG_EVENTS = {"installation", "maintenance", "calibration", "removal"}
PRE_VARS = (
    "Zdcr",
    "Zdcr_aligned",
    "Zdlog_vfov_modv_tm",
    "attd_vfov_modv",
    "DVdcr",
    "disdro_pr",
    "radar_frequency",
    "QF_saturation",
)


@dataclass(frozen=True)
class GateParameters:
    """Parameters of gate_check, from the station configuration."""

    r_max: float  # m, highest gate read (the selection uses Z and |v| at 900 m)
    r_search: float  # m, top of the reference layer
    r_ref: float  # m, bottom of the reference layer
    tol: float  # dB
    min_points: int  # QC-passed minutes for an event, and a gate of it, to count
    min_events: int  # events wanted in the selection
    min_events_period: int  # events for a period to be assessed
    n_boot: int
    boot_quantile: float
    dv_ratio: float  # gates slower than this share of the lowest ones (melting)
    levels: dict  # nested levels, from the most lenient to the most demanding

    @classmethod
    def from_config(cls, conf):
        """Build them from a (checked) station configuration."""
        g = conf["gate_check"]
        return cls(
            r_max=float(g["R_MAX"]),
            r_search=float(g["R_SEARCH"]),
            r_ref=float(g["R_REF"]),
            tol=float(g["TOL"]),
            min_points=int(conf["thresholds"]["MIN_POINTS"]),
            min_events=int(g["MIN_EVENTS"]),
            min_events_period=int(g["MIN_EVENTS_PERIOD"]),
            n_boot=int(g["N_BOOT"]),
            boot_quantile=float(g["BOOT_QUANTILE"]),
            dv_ratio=float(g["DV_RATIO"]),
            levels={k: dict(v) for k, v in g["levels"].items()},
        )


def _quiet_nanmedian(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(x, axis=axis)


# reading -------------------------------------------------------------------------


def _day_files(directory, combo, kind):
    """Return {day: file} of the combination's daily files."""
    out = {}
    for f in Path(directory).rglob(f"{combo}_{'[0-9]' * 8}_{kind}.nc"):
        out[f.name[len(combo) + 1 : len(combo) + 9]] = f
    return dict(sorted(out.items()))


def read_processed(files, min_points):
    """Return the events (one row each) and the QC_overall series."""
    events, qc = [], []
    for day, f in files.items():
        with xr.open_dataset(f) as p:
            qc.append(pd.Series(p["QC_overall"].values == 1, index=p.time.values))
            if not p.sizes.get("events"):
                continue
            ev = pd.DataFrame(
                {
                    v: p[v].values
                    for v in (
                        "start_event",
                        "end_event",
                        "QF_rain_accumulation",
                        "QF_rg_dd_event",
                        "good_points_number",
                        "dZ_med",
                    )
                    if v in p
                }
            )
            ev["day"] = day
            events.append(ev)
    qc = pd.concat(qc).sort_index() if qc else pd.Series(dtype=bool)
    qc = qc[~qc.index.duplicated()]
    if not events:
        return pd.DataFrame(), qc
    ev = pd.concat(events).drop_duplicates("start_event").reset_index(drop=True)
    good = (
        (ev.get("QF_rain_accumulation", 1) > 0)
        & (ev.get("QF_rg_dd_event", 1) != 0)
        & (ev["good_points_number"] >= min_points)
        & np.isfinite(ev["dZ_med"])
    )
    return ev[good].reset_index(drop=True), qc


class Preprocessed:
    """Lazy reader of the daily preprocessed files (one day kept at a time)."""

    def __init__(self, files, r_max):
        self.files = files
        self.r_max = r_max
        self.cache = {}

    def day(self, day):
        if day not in self.cache:
            self.cache = {}  # events are read in time order
            f = self.files.get(day)
            if f is None:
                self.cache[day] = None
            else:
                with xr.open_dataset(f) as raw:
                    ds = raw[[v for v in PRE_VARS if v in raw]]
                    self.cache[day] = ds.sel(range=slice(0, self.r_max)).load()
        return self.cache[day]

    def event(self, start, end):
        """Return the preprocessed data of [start, end], or None."""
        days = pd.date_range(pd.Timestamp(start).normalize(), end, freq="1D")
        parts = [self.day(d.strftime("%Y%m%d")) for d in days]
        parts = [p.sel(time=slice(start, end)) for p in parts if p is not None]
        parts = [p for p in parts if p.sizes["time"]]
        if not parts:
            return None
        if len(parts) > 1 and any(
            p.range.size != parts[0].range.size
            or not np.allclose(p.range.values, parts[0].range.values)
            for p in parts
        ):
            parts = [p.interp(range=parts[0].range.values) for p in parts]
        return xr.concat(parts, dim="time", data_vars="all", join="override")


# per-event quantities ------------------------------------------------------------


def saturated_minutes(ds, r_top):
    """Minutes flagged by QF_saturation at any gate up to r_top (None without it)."""
    if "QF_saturation" not in ds:
        return None
    qf = ds["QF_saturation"].sel(range=slice(0, r_top))
    return (qf.fillna(0) > 0).any("range").to_numpy()


def event_profile(ds, qc, layer):
    """Return the stratiform descriptors, the Delta Z and normalised Z profiles.

    Parameters
    ----------
    ds : xr.Dataset
        Preprocessed data of the event
    qc : np.ndarray
        QC_overall of its minutes (bool), saturated minutes removed
    layer : tuple
        (bottom, top) of the normalisation layer (m)

    Returns
    -------
    dict
        Descriptors (see GATE_LEVEL_KEYS), and range, n, med (Delta Z), medc (corrected
        for the rain attenuation), dv, zn (normalised Z), n_zn : per gate

    """
    r = ds.range.values.astype(float)
    zdcr = ds["Zdcr_aligned" if "Zdcr_aligned" in ds else "Zdcr"].values.astype(float)
    zdd = (
        ds["Zdlog_vfov_modv_tm"]
        .sel(radar_frequencies=ds.radar_frequency, method="nearest")
        .values.astype(float)
    )
    att = rain_attenuation(ds, r) if "attd_vfov_modv" in ds else np.zeros(zdcr.shape)
    dv = ds["DVdcr"].values.astype(float)
    pr = ds["disdro_pr"].values.astype(float)
    dz = zdcr - zdd[:, None]
    ok = np.isfinite(dz) & qc[:, None]
    n = ok.sum(axis=0)
    med = _quiet_nanmedian(np.where(ok, dz, np.nan), axis=0)
    medc = _quiet_nanmedian(np.where(ok, dz + att, np.nan), axis=0)
    dvq = np.where(qc[:, None], np.abs(dv), np.nan)
    v = _quiet_nanmedian(dvq, axis=0)

    # normalised reflectivity, radar alone (Zdcr, not aligned on the disdrometer) :
    # minus its mean over the layer, on the minutes with the whole layer measured
    zr = ds["Zdcr"].values.astype(float)
    in_layer = (r >= layer[0]) & (r <= layer[1])
    full = qc & in_layer.any() & np.isfinite(zr[:, in_layer]).all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zn = zr - np.nanmean(zr[:, in_layer], axis=1, keepdims=True)
    ok_zn = np.isfinite(zn) & full[:, None]

    # stratiform descriptors, never from Delta Z
    j5, j9 = int(np.argmin(abs(r - 500))), int(np.argmin(abs(r - 900)))
    rain = np.nan_to_num(pr) > 0
    lpr = np.log10(pr[qc & rain]) if (qc & rain).sum() > 10 else np.array([np.nan])
    jump = np.abs(np.diff(zdcr[:, j5]))
    zq = np.where(qc[:, None], zdcr, np.nan)
    dv_lo = _quiet_nanmedian(dvq[:, :3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        desc = dict(
            n_qc=int(qc.sum()),
            qc_frac=float(qc.mean()),
            rain_frac=float(rain.mean()),
            lpr_std=float(np.nanstd(lpr)),
            pr_p95=float(np.nanpercentile(pr[qc], 95)) if qc.any() else np.nan,
            z5_jump=float(np.nanmedian(jump)) if np.isfinite(jump).any() else np.nan,
            dv_ratio=float(v[j9] / dv_lo) if dv_lo > 0 else np.nan,
            dv_lo=float(dv_lo),
            z_grad=float(abs(np.nanmedian(zq[:, j9] - zq[:, j5]))),
        )
    return dict(
        **desc,
        range=r,
        n=n,
        med=med,
        medc=medc,
        dv=v,
        zn=_quiet_nanmedian(np.where(ok_zn, zn, np.nan), axis=0),
        n_zn=ok_zn.sum(axis=0),
    )


def running_median(x, w=3):
    """Return the running median over w points, NaN ignored."""
    h = w // 2
    out = np.full(x.shape, np.nan)
    for i in range(x.size):
        seg = x[max(0, i - h) : i + h + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def deviation(r, medc, n, v, gp):
    """Return the deviation of a profile from its reference line, and the line.

    The line is fitted over [gp.r_ref, gp.r_search] on the gates with
    gp.min_points minutes and a rain-like fall speed (|v| above gp.dv_ratio times
    that of the lowest gates : a melting layer is left out), the profile being
    smoothed by a 3-gate running median. NaN without at least 4 such gates.
    """
    valid = (r <= gp.r_search) & (n >= gp.min_points) & np.isfinite(medc)
    if np.isfinite(v[valid]).sum() >= 3:
        ref_v = np.nanmedian(v[valid][:3])
        valid &= np.isfinite(v) & (v >= gp.dv_ratio * ref_v)
    sm = running_median(np.where(valid, medc, np.nan))
    fit = valid & (r >= gp.r_ref) & np.isfinite(sm)
    if fit.sum() < 4:
        return np.full(r.size, np.nan), (np.nan, np.nan)
    a, b = np.polyfit(r[fit], sm[fit], 1)
    dev = np.where(valid, medc - (a * r + b), np.nan)
    return dev, (a, b)


# selection and gate --------------------------------------------------------------


def selection_level(features, levels, min_events):
    """Return the levels each event passes, and the level to use.

    The most demanding level holding at least min_events events ; the most
    lenient one if none does, "tous" (every good event) if it holds fewer than 5.
    """
    passes = {}
    lenient = next(iter(levels))
    for name, th in levels.items():
        ok = np.ones(len(features), bool)
        for k, x in th.items():
            col = features[k].to_numpy(dtype=float)
            ok &= (col >= x) if k in MINIMA else (col <= x)
        passes[name] = ok
    passes = pd.DataFrame(passes, index=features.index)
    level = "tous"
    for name in levels:
        if passes[name].sum() >= min_events or (
            name == lenient and passes[name].sum() >= 5
        ):
            level = name
    return passes, level


def optimal_gate(gates, devs, tol, r_ref):
    """Return the optimal gate of a set of deviation profiles (NaN if none).

    The composite is the median over the events ; the gate is the lowest one
    above the last gate below r_ref whose composite leaves +-tol.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        comp = np.nanmedian(devs, axis=0)
    usable = np.isfinite(comp)
    if not usable[gates < r_ref].any():
        return np.nan, comp
    bad = np.where((gates < r_ref) & usable & (np.abs(comp) > tol))[0]
    above = np.where(usable & (np.arange(gates.size) > (bad.max() if bad.size else -1)))
    return (float(gates[above[0][0]]) if above[0].size else np.nan), comp


def bootstrap_gates(gates, devs, gp, seed=0):
    """Return the optimal gates of gp.n_boot draws of the events."""
    rng = np.random.default_rng(seed)
    k = devs.shape[0]
    return np.array(
        [
            optimal_gate(gates, devs[rng.integers(0, k, k)], gp.tol, gp.r_ref)[0]
            for _ in range(gp.n_boot)
        ]
    )


def antenna_layout(combo):
    """Return "monostatique", "bistatique" or None from the radar type of a combo."""
    rtype = combo.split("_")[1].lower()
    if rtype.startswith(MONOSTATIC):
        return "monostatique"
    if rtype.startswith(BISTATIC):
        return "bistatique"
    return None


def gate_above(gates, comp, g):
    """Return the first gate above g with a composite deviation (NaN if none)."""
    if not np.isfinite(g):
        return np.nan
    up = np.where((gates > g) & np.isfinite(comp))[0]
    return float(gates[up[0]]) if up.size else np.nan


def value_at(gates, comp, g):
    """Return the composite at the gate nearest to g (NaN without g)."""
    if g is None or not np.isfinite(g):
        return np.nan
    return float(comp[int(np.argmin(abs(gates - g)))])


def summarise(gates, devs, conf_gate, label, gp, gate_ref=None, bistatic=False):
    """Optimal gate, bootstrap and deviation at given gates, for one event set.

    The recommended gate is the bootstrap quantile, and for a bistatic radar at
    least the gate above the composite optimal gate.
    """
    g_opt, comp = optimal_gate(gates, devs, gp.tol, gp.r_ref)
    boot = bootstrap_gates(gates, devs, gp) if devs.shape[0] >= 5 else np.array([g_opt])
    boot = boot[np.isfinite(boot)]
    g_boot = (
        float(np.quantile(boot, gp.boot_quantile, method="higher"))
        if boot.size
        else np.nan
    )
    g_plus = gate_above(gates, comp, g_opt) if bistatic else np.nan
    rec_gate = (
        float(np.nanmax([g_boot, g_plus]))
        if np.isfinite([g_boot, g_plus]).any()
        else np.nan
    )
    counts = pd.Series(boot).round().value_counts(normalize=True).sort_index()

    def dev_at(g):
        return value_at(gates, comp, g)

    return dict(
        set=label,
        n_events=int(devs.shape[0]),
        gate_composite=g_opt,
        gate_bootstrap=g_boot,
        gate_composite_plus_one=g_plus,
        gate_recommended=rec_gate,
        bootstrap=" ".join(f"{g:.0f}:{p:.0%}" for g, p in counts.items() if p >= 0.05),
        dev_at_conf_gate=dev_at(conf_gate),
        dev_at_recommended=dev_at(rec_gate),
        dev_at_reference=dev_at(gate_ref),
    ), comp


ZN_KEYS = (
    "gate_composite",
    "gate_bootstrap",
    "gate_composite_plus_one",
    "gate_recommended",
    "bootstrap",
    "dev_at_conf_gate",
    "dev_at_recommended",
    "dev_at_reference",
)


def compare_methods(gates, row, comp, row_zn, comp_zn, n_zn):
    """Add the normalised Z results (suffix _zn) and their comparison to a row.

    gate_shift is the normalised Z gate minus the Delta Z gate, in gates ;
    methods_agree when they are at most one gate apart.
    """
    out = dict(row, n_events_zn=int(n_zn))
    out.update({f"{k}_zn": row_zn[k] for k in ZN_KEYS})
    g, g_zn = row["gate_recommended"], row_zn["gate_recommended"]
    if np.isfinite([g, g_zn]).all():
        shift = int(np.argmin(abs(gates - g_zn))) - int(np.argmin(abs(gates - g)))
        out.update(gate_shift=shift, methods_agree=abs(shift) <= 1)
    else:
        out.update(gate_shift=np.nan, methods_agree=np.nan)
    out["dev_at_zn_gate"] = value_at(gates, comp, g_zn)
    out["dev_zn_at_dz_gate"] = value_at(gates, comp_zn, g)
    return out


def log_breaks(log_file, combo):
    """Dates of the radar's instrument log entries that may change its overlap."""
    site, rtype, rpid = combo.split("_")[:3]
    if log_file is None or not Path(log_file).exists():
        return []
    entries = json.loads(Path(log_file).read_text()).get("entries", [])
    out = []
    for e in entries:
        if e.get("instrument") == f"{rtype} {rpid}" and e.get("event") in LOG_EVENTS:
            out.append(pd.Timestamp(e.get("end_date") or e["date"]))
    return sorted(set(out))


def periods(starts, how, breaks=()):
    """Label each event with its period : calendar year or instrument log epoch."""
    starts = pd.DatetimeIndex(starts)
    if how == "year":
        return np.asarray(starts.year.astype(str))
    if how == "log":
        edges = [pd.Timestamp.min, *sorted(breaks), pd.Timestamp.max]
        lab = []
        for t in starts:
            k = np.searchsorted(
                np.array(edges[1:-1], dtype="datetime64[ns]"), t.to_datetime64()
            )
            lo = "début" if k == 0 else edges[k].strftime("%Y-%m-%d")
            lab.append(f"depuis {lo}")
        return np.array(lab)
    return np.full(len(starts), "tout")


# main ----------------------------------------------------------------------------


def evaluate(combo, preprocessed_dir, processed_dir, conf, period="year", breaks=()):
    """Return the per-event table, the summary rows and the composite profiles.

    conf is the station configuration : DCR_DZ_RANGE and the gate_check
    parameters (GateParameters). The composites are {"dz": {set: profile},
    "zn": {set: profile}}, of the Delta Z and the normalised Z deviations.
    """
    gp = GateParameters.from_config(conf)
    conf_gate = float(conf["instrument_parameters"]["DCR_DZ_RANGE"])
    layout = antenna_layout(combo)
    if layout is None:
        lgr.warning("antenna layout unknown for %s : no extra gate", combo)
    bistatic = layout == "bistatique"
    ev, qc = read_processed(
        _day_files(processed_dir, combo, "processed"), gp.min_points
    )
    if ev.empty:
        raise ValueError(f"no good event for {combo} in {processed_dir}")
    pre = Preprocessed(_day_files(preprocessed_dir, combo, "preprocessed"), gp.r_max)
    profs, no_flag = [], 0
    for _, e in ev.sort_values("start_event").iterrows():
        ds = pre.event(e.start_event, e.end_event)
        if ds is None:
            continue
        q = qc.reindex(ds.time.values).fillna(False).to_numpy(dtype=bool)
        sat = saturated_minutes(ds, gp.r_search)
        if sat is None:
            no_flag += 1
            sat = np.zeros(q.size, bool)
        n_sat = int((q & sat).sum())
        q = q & ~sat
        if q.sum() < gp.min_points:
            continue
        p = event_profile(ds, q, (gp.r_ref, gp.r_search))
        p["n_sat"] = n_sat
        p["dev"], p["line"] = deviation(p["range"], p["medc"], p["n"], p["dv"], gp)
        p["dev_zn"], _ = deviation(p["range"], p["zn"], p["n_zn"], p["dv"], gp)
        p["start"], p["end"] = pd.Timestamp(e.start_event), pd.Timestamp(e.end_event)
        profs.append(p)
    if no_flag:
        lgr.warning(
            "%s : %d events without QF_saturation (preprocessed files before the "
            "saturation check) : saturation above DCR_DZ_RANGE not removed",
            combo,
            no_flag,
        )
    if not profs:
        raise ValueError(f"no event with preprocessed data for {combo}")

    # common gate grid : the most frequent one (a RPG chirp change moves gates)
    grids = pd.Series([tuple(np.round(p["range"], 1)) for p in profs])
    gates = np.array(grids.value_counts().index[0])
    gates = gates[gates <= gp.r_search]
    devs, meds, devs_zn = [], [], []
    for p in profs:
        for key, out in (("dev", devs), ("medc", meds), ("dev_zn", devs_zn)):
            y, r = p[key], np.round(p["range"], 1)
            ok = np.isfinite(y)
            out.append(
                np.interp(gates, r[ok], y[ok], left=np.nan, right=np.nan)
                if ok.sum() >= 2
                else np.full(gates.size, np.nan)
            )
    devs, meds, devs_zn = np.array(devs), np.array(meds), np.array(devs_zn)

    features = pd.DataFrame(
        [{k: p[k] for k in (*GATE_LEVEL_KEYS, "n_sat", "start", "end")} for p in profs]
    )
    passes, level = selection_level(features, gp.levels, gp.min_events)
    sel = passes[level].to_numpy() if level != "tous" else np.ones(len(profs), bool)
    features = pd.concat([features, passes], axis=1)
    features["selected"] = sel
    features["period"] = periods(features.start, period, breaks)
    j_conf = int(np.argmin(abs(gates - conf_gate)))
    features["dZc_conf_gate"] = meds[:, j_conf]
    features["fit_ok"] = np.isfinite(devs).any(axis=1)
    features["zn_fit_ok"] = np.isfinite(devs_zn).any(axis=1)

    summary, comps = [], {"dz": {}, "zn": {}}
    g_ref = {"dz": None, "zn": None}

    def assess(mask, label, lev):
        """Both methods on the same events ; the row is appended to the summary."""
        rows = {}
        for key, d in (("dz", devs), ("zn", devs_zn)):
            rows[key], comps[key][label] = summarise(
                gates, d[mask], conf_gate, label, gp, g_ref[key], bistatic=bistatic
            )
        n_zn = np.isfinite(devs_zn[mask]).any(axis=1).sum()
        row = compare_methods(
            gates,
            rows["dz"],
            comps["dz"][label],
            rows["zn"],
            comps["zn"][label],
            n_zn,
        )
        row["level"] = lev
        summary.append(row)
        return row

    fit_ok = features.fit_ok.to_numpy()
    row = assess(sel & fit_ok, f"{level} (toute la période)", level)
    g_ref = {"dz": row["gate_recommended"], "zn": row["gate_recommended_zn"]}
    assess(fit_ok, "tous les événements", "tous")
    if period != "none":
        order = [*reversed(gp.levels), "tous"]  # from the most demanding
        for lab in sorted(set(features.period)):
            in_p = (features.period == lab).to_numpy() & features.fit_ok.to_numpy()
            # the global level, or looser ones when it leaves too few events
            for lev in order[order.index(level) :]:
                m = in_p & (passes[lev].to_numpy() if lev != "tous" else True)
                if m.sum() >= gp.min_events_period:
                    break
            if m.sum() < gp.min_events_period:
                summary.append(dict(set=lab, level="-", n_events=int(in_p.sum())))
                continue
            assess(m, f"{lab} ({lev})", lev)
    summary = pd.DataFrame(summary)
    summary.insert(0, "antenna", layout)
    for suffix in ("", "_zn"):
        ref = summary[f"dev_at_reference{suffix}"]
        summary[f"single_gate_ok{suffix}"] = (ref.abs() <= gp.tol).where(ref.notna())
    return features, summary, gates, comps


def plot(gates, comps, conf_gate, title, path, gp, recommended=None):
    """Composite deviation profiles of both methods, with the tolerance and gates.

    comps is {"dz": {set: profile}, "zn": {set: profile}}, recommended the
    gates {"dz": g, "zn": g} of the first set.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recommended = recommended or {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    panels = (
        ("dz", "écart du profil ΔZ à sa droite (dB)", "ΔZ = Z radar − Z disdro"),
        (
            "zn",
            "écart du profil Z normalisé à sa droite (dB)",
            "Z normalisé, radar seul",
        ),
    )
    for ax, (key, xlabel, name) in zip(axes, panels, strict=True):
        ax.axvspan(-gp.tol, gp.tol, color="#d9eadf", label=f"±{gp.tol} dB")
        ax.axhspan(gp.r_ref, gp.r_search, color="#eeeeee", label="couche de référence")
        for k, (label, comp) in enumerate(comps[key].items()):
            style = dict(lw=2.2, color="k") if k == 0 else dict(lw=1, alpha=0.7)
            ax.plot(comp, gates, marker="o", ms=3, label=label, **style)
        ax.axhline(
            conf_gate, ls=":", color="#c8641e", label=f"gate TOML {conf_gate:g} m"
        )
        for other, color in (("dz", "#1f5fa8"), ("zn", "#7a3d9a")):
            g = recommended.get(other, np.nan)
            if g is not None and np.isfinite(g):
                lab = "ΔZ" if other == "dz" else "Z normalisé"
                ax.axhline(
                    g,
                    ls="-" if other == key else "--",
                    lw=1,
                    color=color,
                    label=f"gate {lab} {g:.0f} m",
                )
        ax.set_xlabel(xlabel)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("altitude de la gate (m)")
    axes[0].set_ylim(0, gp.r_search + 20)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--combo", required=True, help="file prefix, before the date")
    parser.add_argument("--preprocessed-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="station TOML")
    parser.add_argument("--period", choices=("none", "year", "log"), default="year")
    parser.add_argument("--instrument-log", type=Path, help="local log (JSON)")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    conf = load_config(args.config)
    conf_gate = float(conf["instrument_parameters"]["DCR_DZ_RANGE"])
    breaks = log_breaks(args.instrument_log, args.combo)
    features, summary, gates, comps = evaluate(
        args.combo,
        args.preprocessed_dir,
        args.processed_dir,
        conf,
        args.period,
        breaks,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / f"{args.combo}_gate_events.csv", index=False)
    summary.insert(0, "combo", args.combo)
    summary.insert(1, "conf_gate", conf_gate)
    summary.to_csv(out / f"{args.combo}_gate_summary.csv", index=False)
    first = summary.iloc[0]
    plot(
        gates,
        comps,
        conf_gate,
        args.combo,
        out / f"{args.combo}_gate.png",
        GateParameters.from_config(conf),
        {"dz": first["gate_recommended"], "zn": first["gate_recommended_zn"]},
    )
    pd.set_option("display.width", 220)
    print(summary.drop(columns=["combo"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
