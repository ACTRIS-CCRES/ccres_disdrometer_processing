import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import xarray as xr

event_stats = [
    "start_event",
    "end_event",
    "event_length",
    "rain_accumulation",
    "QF_rain_accumulation",
    "QF_rg_dd_event",
    "nb_dz_computable_pts",
    "QC_vdsd_t_ratio",
    "QC_pr_ratio",
    "QC_ta_ratio",
    "QC_ws_ratio",
    "QC_wd_ratio",
    "QC_overall_ratio",
    "good_points_number",
    "dZ_mean",
    "dZ_med",
    "dZ_q1",
    "dZ_q3",
    "dZ_min",
    "dZ_max",
    "reg_slope",
    "reg_intercept",
    "reg_score",
    "reg_rmse",
]

MIN_TIMESTEPS = 50  # minimum number of good timesteps to consider the event statistics are robust enough for being plotted in the monitoring timeseries  # noqa


def monitoring_timeseries(
    folder,
    conf,
    showmeans=True,
    showfliers=True,
    showcaps=True,
):
    files = sorted(glob.glob(folder))
    print("number of files : ", len(files))
    f0 = xr.open_dataset(files[0])
    print(f0[event_stats].dims)
    ds = xr.concat([xr.open_dataset(file)[event_stats] for file in files], dim="events")
    print(ds.dims)
    print(ds.events.values)
    print(ds.dZ_max.values)

    fig, ax = plt.subplots(figsize=((20, 6)))
    ax.axhline(y=0, color="blue")

    # Moving average of the median bias
    n = 3  # avg(T) given by T, T-1, T-2
    f = np.intersect1d(
        np.where(ds.QF_rain_accumulation > 0)[0],
        np.where(np.isfinite(ds.dZ_med) * 1 == 1)[0],
    )
    f = np.intersect1d(f, np.where(ds.good_points_number >= MIN_TIMESTEPS))
    f = np.arange(len(ds.events))  # TODO : remove this line after tests are OK

    dZ_good, t_good = (
        ds.dZ_med[f],
        (
            (ds.start_event[f].astype("int64") + ds.end_event[f].astype("int64")) / 2
        ).astype("datetime64[ns]"),
    )
    dZ_moving_avg = np.convolve(dZ_good[:], np.ones(n) / n, mode="valid")
    (moving_avg,) = ax.plot(t_good[n - 1 :], dZ_moving_avg, color="red")

    # Custom boxplot properties
    mean_shape = dict(markeredgecolor="green", marker="v", markerfacecolor="green")
    med_shape = dict(
        marker="*",
        markersize=6,
        markerfacecolor="darkorange",
        markeredgecolor="darkorange",
    )
    boxprops = dict(color="green")
    flierprops = dict(
        markerfacecolor="none",
        marker="o",
        markeredgecolor="green",
    )
    whiskerprops = dict(lw=1.0, color="green")  # hide whiskers ? (lw=0)

    # Boxplot stats for each eligible event
    for k in range(len(f)):  # should it be in range(len(dZ_good)) ?
        bxp_stats = [
            {
                "mean": ds.dZ_mean[f][k].values,
                "med": ds.dZ_med[f][k].values,
                "q1": ds.dZ_q1[f][k].values,
                "q3": ds.dZ_q3[f][k].values,
                "fliers": [ds.dZ_min[f][k].values, ds.dZ_max[f][k].values],
                "whishi": np.minimum(
                    ds.dZ_q3[f][k].values
                    + 1 * np.abs(ds.dZ_q3[f][k].values - ds.dZ_q1[f][k].values),
                    ds.dZ_max[f][k].values,
                ),  # min(Q3 + 1*IQR, max_value)
                "whislo": np.maximum(
                    ds.dZ_q1[f][k].values
                    - 1 * np.abs(ds.dZ_q3[f][k].values - ds.dZ_q1[f][k].values),
                    ds.dZ_min[f][k].values,
                ),  # max(Q1 - 1*IQR, min_value)
            }
        ]

        box = ax.bxp(
            bxp_stats,
            showmeans=showmeans,
            meanline=False,
            showfliers=showfliers,
            showcaps=showcaps,
            meanprops=mean_shape,
            medianprops=med_shape,
            positions=[mpl.dates.date2num(t_good[k])],
            widths=[1],
            flierprops=flierprops,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
        )
        for cap in box["caps"]:
            cap.set(color="green")

    if showmeans:
        ax.legend(
            [moving_avg, box["medians"][0], box["means"][0]],
            ["bias moving avg (3 values)", "median", "mean"],
            loc="upper right",
        )
    else:
        ax.legend(
            [moving_avg, box["medians"][0]],
            ["bias moving avg (3 values)", "median"],
            loc="upper right",
        )

    plt.grid()
    plt.xlabel("Date", fontsize=15, fontweight=300)
    plt.ylabel("$Z_{DCR} - Z_{disdrometer}$ (dBZ)", fontsize=15, fontweight=300)
    ax.set_ylim(bottom=-30, top=30)
    # locator = mpl.dates.MonthLocator(interval=1)
    locator = mpl.dates.DayLocator(interval=1)
    formatter = mpl.dates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # dates_axe = mpl.dates.num2date(np.array(ax.get_xticks()))
    # dates_axe = [d.date() for d in dates_axe]
    plt.xticks(rotation=45, fontsize=13, fontweight="semibold")
    ax.set_yticklabels(ax.get_yticks(), fontsize=18, fontweight="semibold")

    plt.title(
        "{} - {} Time series of {} @ {} CC variability \n".format(
            pd.to_datetime(ds.start_event.values[0]).strftime("%Y/%m"),
            pd.to_datetime(ds.start_event.values[-1]).strftime("%Y/%m"),
            ds.radar_source,
            ds.location,
        )
        + r"({} good events -- more than {:.0f}mm of rain and {} timesteps to compute ".format(  # noqa
            len(f),
            conf["thresholds"]["MIN_RAINFALL_AMOUNT"],
            int(MIN_TIMESTEPS),  # TODO : get min rain acc value properly
        )
        + r"$\Delta Z$)",
        fontsize=13,
        fontweight="semibold",
    )

    centered_time = pd.Timestamp(
        (
            (
                ds.start_event[0].values.astype("int64")
                + ds.start_event[-1].values.astype("int64")
            )
            / 2
        ).astype("datetime64[ns]")
    )

    plt.text(
        x=centered_time,
        y=26,
        s="disdrometer used as a reference : " + ds.disdrometer_source,
        fontsize=14,
        ha="center",
    )
    if f0.weather_data_avail.values[0] > 0:
        plt.text(
            x=centered_time,
            y=23,
            s="Weather station data used for QC computation",
            fontsize=14,
            ha="center",
        )
    else:
        plt.text(
            x=centered_time,
            y=23,
            s="no Weather station data available for QC",
            fontsize=14,
            ha="center",
        )
    print(f0.range.shape)
    plt.text(
        x=centered_time,
        y=20,
        s=r"$\Delta Z$ computation @ {:.0f}m AGL".format(
            f0.range.sel(
                {"range": conf["instrument_parameters"]["DCR_DZ_RANGE"]},
                method="nearest",
            ).values
        ),
        fontsize=14,
        ha="center",
    )

    plt.savefig("plot_test_template_timeseries.png", dpi=400)
    plt.close()

    return fig, ax


if __name__ == "__main__":
    # folder = "/bdd/ACTRIS/CCRES/pub/disdro/juelich/2024/*/*/*proc*.nc"
    folder = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/juelich_2021-12*_processed.nc"  # noqa
    conf = toml.load(
        "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/inputs/config_files/conf_stations/conf_juelich_mira-parsivel.toml"
    )
    monitoring_timeseries(folder, conf)