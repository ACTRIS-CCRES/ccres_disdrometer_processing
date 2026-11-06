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
   passed, at least MIN_POINTS QC-passed minutes) ;
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
   a systematic excess costs more than a little precision) ;
6. by period (year, or instrument log epochs) : the deviation at the recommended
   gate tells whether a single gate holds within TOL over the whole history.

The configuration files are left untouched : the recommendation is to be checked
before updating DCR_DZ_RANGE.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import toml
import xarray as xr

from ccres_disdrometer_processing.utils import rain_attenuation

lgr = logging.getLogger(__name__)

R_MAX = 1000.0  # m, highest gate read (the selection uses Z and |v| at 900 m)
R_SEARCH = 600.0  # m, top of the reference layer
R_REF = 350.0  # m, bottom of the reference layer
TOL = 0.25  # dB
MIN_POINTS = 50  # QC-passed minutes for a gate of an event to count
MIN_EVENTS = 15  # events wanted in the selection
MIN_EVENTS_PERIOD = 8  # events for a period to be assessed (looser level if need be)
N_BOOT = 200
BOOT_QUANTILE = 0.8
DV_RATIO = 0.7  # gates slower than this share of the lowest ones (melting layer)

# Stratiform selection, nested levels ; thresholds of the Juelich study
# (n_qc : QC-passed minutes ; rain_frac, qc_frac : share of the event minutes
# with rain, passing the QC ; lpr_std : std of log10(rain rate) ; pr_p95 : 95th
# percentile of the rain rate (mm/h) ; z5_jump : median minute-to-minute change of
# Z at 500 m (dB) ; dv_ratio : |v| at 900 m / near the ground ; dv_lo : |v| near
# the ground (m/s) ; z_grad : |Z(900 m) - Z(500 m)| (dB))
LEVELS = {
    "large": dict(
        n_qc=180,
        rain_frac=0.70,
        qc_frac=0.55,
        lpr_std=0.60,
        pr_p95=2.7,
        z5_jump=1.5,
        dv_ratio=0.85,
        dv_lo=2.5,
        z_grad=1.5,
    ),  # fmt: skip
    "moyen": dict(
        n_qc=240,
        rain_frac=0.80,
        qc_frac=0.65,
        lpr_std=0.55,
        pr_p95=2.5,
        z5_jump=1.0,
        dv_ratio=0.90,
        dv_lo=3.0,
        z_grad=1.0,
    ),  # fmt: skip
    "exigeant": dict(
        n_qc=270,
        rain_frac=0.825,
        qc_frac=0.70,
        lpr_std=0.50,
        pr_p95=2.25,
        z5_jump=1.0,
        dv_ratio=0.90,
        dv_lo=3.0,
        z_grad=1.0,
    ),  # fmt: skip
    "strict": dict(
        n_qc=300,
        rain_frac=0.85,
        qc_frac=0.75,
        lpr_std=0.45,
        pr_p95=2.0,
        z5_jump=1.0,
        dv_ratio=0.90,
        dv_lo=3.0,
        z_grad=1.0,
    ),  # fmt: skip
}
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


def read_processed(files):
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
        & (ev["good_points_number"] >= MIN_POINTS)
        & np.isfinite(ev["dZ_med"])
    )
    return ev[good].reset_index(drop=True), qc


class Preprocessed:
    """Lazy reader of the daily preprocessed files (one day kept at a time)."""

    def __init__(self, files):
        self.files = files
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
                    self.cache[day] = ds.sel(range=slice(0, R_MAX)).load()
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


def event_profile(ds, qc):
    """Return the stratiform descriptors and the Delta Z profile of one event.

    Parameters
    ----------
    ds : xr.Dataset
        Preprocessed data of the event
    qc : np.ndarray
        QC_overall of its minutes (bool)

    Returns
    -------
    dict
        Descriptors (see LEVELS), and range, n, med (Delta Z), medc (corrected
        for the rain attenuation), dv : per gate

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
    return dict(**desc, range=r, n=n, med=med, medc=medc, dv=v)


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


def deviation(r, medc, n, v, r_ref=R_REF, r_search=R_SEARCH):
    """Return the deviation of a profile from its reference line, and the line.

    The line is fitted over [r_ref, r_search] on the gates with MIN_POINTS
    minutes and a rain-like fall speed (|v| above DV_RATIO times that of the
    lowest gates : a melting layer is left out), the profile being smoothed by a
    3-gate running median. NaN without at least 4 such gates.
    """
    valid = (r <= r_search) & (n >= MIN_POINTS) & np.isfinite(medc)
    if np.isfinite(v[valid]).sum() >= 3:
        ref_v = np.nanmedian(v[valid][:3])
        valid &= np.isfinite(v) & (v >= DV_RATIO * ref_v)
    sm = running_median(np.where(valid, medc, np.nan))
    fit = valid & (r >= r_ref) & np.isfinite(sm)
    if fit.sum() < 4:
        return np.full(r.size, np.nan), (np.nan, np.nan)
    a, b = np.polyfit(r[fit], sm[fit], 1)
    dev = np.where(valid, medc - (a * r + b), np.nan)
    return dev, (a, b)


# selection and gate --------------------------------------------------------------


def selection_level(features, min_events=MIN_EVENTS):
    """Return the levels each event passes, and the level to use.

    The most demanding level holding at least min_events events ; "large" if none
    does, "tous" (every good event) if "large" holds fewer than 5.
    """
    passes = {}
    for name, th in LEVELS.items():
        ok = np.ones(len(features), bool)
        for k, x in th.items():
            col = features[k].to_numpy(dtype=float)
            ok &= (col >= x) if k in MINIMA else (col <= x)
        passes[name] = ok
    passes = pd.DataFrame(passes, index=features.index)
    level = "tous"
    for name in LEVELS:
        if passes[name].sum() >= min_events or (
            name == "large" and passes[name].sum() >= 5
        ):
            level = name
    return passes, level


def optimal_gate(gates, devs, tol=TOL, r_ref=R_REF):
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


def bootstrap_gates(gates, devs, n_boot=N_BOOT, seed=0):
    """Return the optimal gates of n_boot draws of the events."""
    rng = np.random.default_rng(seed)
    k = devs.shape[0]
    return np.array(
        [optimal_gate(gates, devs[rng.integers(0, k, k)])[0] for _ in range(n_boot)]
    )


def summarise(gates, devs, conf_gate, label, gate_ref=None):
    """Optimal gate, bootstrap and deviation at given gates, for one event set."""
    g_opt, comp = optimal_gate(gates, devs)
    boot = bootstrap_gates(gates, devs) if devs.shape[0] >= 5 else np.array([g_opt])
    boot = boot[np.isfinite(boot)]
    rec_gate = (
        float(np.quantile(boot, BOOT_QUANTILE, method="higher"))
        if boot.size
        else np.nan
    )
    counts = pd.Series(boot).round().value_counts(normalize=True).sort_index()

    def dev_at(g):
        if g is None or not np.isfinite(g):
            return np.nan
        return float(comp[int(np.argmin(abs(gates - g)))])

    return dict(
        set=label,
        n_events=int(devs.shape[0]),
        gate_composite=g_opt,
        gate_recommended=rec_gate,
        bootstrap=" ".join(f"{g:.0f}:{p:.0%}" for g, p in counts.items() if p >= 0.05),
        dev_at_conf_gate=dev_at(conf_gate),
        dev_at_recommended=dev_at(rec_gate),
        dev_at_reference=dev_at(gate_ref),
    ), comp


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
            k = np.searchsorted(np.array(edges[1:-1], dtype="datetime64[ns]"), t)
            lo = "début" if k == 0 else edges[k].strftime("%Y-%m-%d")
            lab.append(f"depuis {lo}")
        return np.array(lab)
    return np.full(len(starts), "tout")


# main ----------------------------------------------------------------------------


def evaluate(
    combo, preprocessed_dir, processed_dir, conf_gate, period="year", breaks=()
):
    """Return the per-event table, the summary rows and the composite profiles."""
    ev, qc = read_processed(_day_files(processed_dir, combo, "processed"))
    if ev.empty:
        raise ValueError(f"no good event for {combo} in {processed_dir}")
    pre = Preprocessed(_day_files(preprocessed_dir, combo, "preprocessed"))
    profs = []
    for _, e in ev.sort_values("start_event").iterrows():
        ds = pre.event(e.start_event, e.end_event)
        if ds is None:
            continue
        q = qc.reindex(ds.time.values).fillna(False).to_numpy(dtype=bool)
        if q.sum() < MIN_POINTS:
            continue
        p = event_profile(ds, q)
        p["dev"], p["line"] = deviation(p["range"], p["medc"], p["n"], p["dv"])
        p["start"], p["end"] = pd.Timestamp(e.start_event), pd.Timestamp(e.end_event)
        profs.append(p)
    if not profs:
        raise ValueError(f"no event with preprocessed data for {combo}")

    # common gate grid : the most frequent one (a RPG chirp change moves gates)
    grids = pd.Series([tuple(np.round(p["range"], 1)) for p in profs])
    gates = np.array(grids.value_counts().index[0])
    gates = gates[gates <= R_SEARCH]
    devs, meds = [], []
    for p in profs:
        for key, out in (("dev", devs), ("medc", meds)):
            y, r = p[key], np.round(p["range"], 1)
            ok = np.isfinite(y)
            out.append(
                np.interp(gates, r[ok], y[ok], left=np.nan, right=np.nan)
                if ok.sum() >= 2
                else np.full(gates.size, np.nan)
            )
    devs, meds = np.array(devs), np.array(meds)

    features = pd.DataFrame(
        [{k: p[k] for k in (*LEVELS["large"], "start", "end")} for p in profs]
    )
    passes, level = selection_level(features)
    sel = passes[level].to_numpy() if level != "tous" else np.ones(len(profs), bool)
    features = pd.concat([features, passes], axis=1)
    features["selected"] = sel
    features["period"] = periods(features.start, period, breaks)
    j_conf = int(np.argmin(abs(gates - conf_gate)))
    features["dZc_conf_gate"] = meds[:, j_conf]
    features["fit_ok"] = np.isfinite(devs).any(axis=1)

    use = sel & features.fit_ok.to_numpy()
    summary, comps = [], {}
    row, comp = summarise(gates, devs[use], conf_gate, f"{level} (toute la période)")
    row["level"] = level
    summary.append(row)
    comps[row["set"]] = comp
    g_ref = row["gate_recommended"]
    row_all, comp_all = summarise(
        gates,
        devs[features.fit_ok.to_numpy()],
        conf_gate,
        "tous les événements",
        g_ref,
    )
    row_all["level"] = "tous"
    summary.append(row_all)
    comps[row_all["set"]] = comp_all
    if period != "none":
        order = [*reversed(LEVELS), "tous"]  # from the most demanding
        for lab in sorted(set(features.period)):
            in_p = (features.period == lab).to_numpy() & features.fit_ok.to_numpy()
            # the global level, or looser ones when it leaves too few events
            for lev in order[order.index(level) :]:
                m = in_p & (passes[lev].to_numpy() if lev != "tous" else True)
                if m.sum() >= MIN_EVENTS_PERIOD:
                    break
            if m.sum() < MIN_EVENTS_PERIOD:
                summary.append(dict(set=lab, level="-", n_events=int(in_p.sum())))
                continue
            r, c = summarise(gates, devs[m], conf_gate, f"{lab} ({lev})", g_ref)
            r["level"] = lev
            summary.append(r)
            comps[r["set"]] = c
    summary = pd.DataFrame(summary)
    if "dev_at_reference" in summary:
        ref = summary["dev_at_reference"]
        summary["single_gate_ok"] = (ref.abs() <= TOL).where(ref.notna())
    return features, summary, gates, comps


def plot(gates, comps, conf_gate, title, path):
    """Composite deviation profiles, with the tolerance and the gates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axvspan(-TOL, TOL, color="#d9eadf", label=f"±{TOL} dB")
    ax.axhspan(R_REF, R_SEARCH, color="#eeeeee", label="couche de référence")
    for k, (name, comp) in enumerate(comps.items()):
        style = dict(lw=2.2, color="k") if k == 0 else dict(lw=1, alpha=0.7)
        ax.plot(comp, gates, marker="o", ms=3, label=name, **style)
    ax.axhline(conf_gate, ls=":", color="#c8641e", label=f"gate TOML {conf_gate:g} m")
    ax.set_xlabel("écart du profil ΔZ à sa droite de référence (dB)")
    ax.set_ylabel("altitude de la gate (m)")
    ax.set_ylim(0, R_SEARCH + 20)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
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

    conf_gate = float(toml.load(args.config)["instrument_parameters"]["DCR_DZ_RANGE"])
    breaks = log_breaks(args.instrument_log, args.combo)
    features, summary, gates, comps = evaluate(
        args.combo,
        args.preprocessed_dir,
        args.processed_dir,
        conf_gate,
        args.period,
        breaks,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    features.to_csv(out / f"{args.combo}_gate_events.csv", index=False)
    summary.insert(0, "combo", args.combo)
    summary.insert(1, "conf_gate", conf_gate)
    summary.to_csv(out / f"{args.combo}_gate_summary.csv", index=False)
    plot(gates, comps, conf_gate, args.combo, out / f"{args.combo}_gate.png")
    pd.set_option("display.width", 220)
    print(summary.drop(columns=["combo"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
