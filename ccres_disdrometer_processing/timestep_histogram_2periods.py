import datetime as dt
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path = "/homedata/ygrit/disdro_processing/juelich/csv2/*TZZ*202208*.csv"
data = pd.read_csv(glob.glob(path)[0], index_col=0)
time = data.iloc[:, 2]
Z_dd = data.iloc[:, 0]
Z_dcr = data.iloc[:, 1]

t1 = [time[0], time[int(len(time) / 3)]]
t2 = [time[int(len(time) * 3 / 4)], time[len(time) - 1]]


def histos(data, p1=t1, p2=t2):
    time = data.iloc[:, 2]

    fig, ax = plt.subplots()
    ax.set_xlim(left=-30, right=30)

    f1 = np.where((time >= p1[0]) & (time <= p1[1]))
    f2 = np.where((time >= p2[0]) & (time <= p2[1]))

    d1 = data.iloc[f1[0], :]
    d2 = data.iloc[f2[0], :]
    dfs = [d1, d2]
    ps = [p1, p2]
    colors = ["red", "blue"]

    for i in range(2):
        df = dfs[i]
        ax.hist(
            df.iloc[:, 1] - df.iloc[:, 0],
            alpha=0.4,
            bins=np.arange(-30, 31, 1),
            color=colors[i],
            density=True,
            label="{} - {}, {} timesteps".format(
                dt.datetime.strptime(ps[i][0], "%Y/%m/%d %H:%M:%S").strftime(
                    "%Y/%m/%d"
                ),
                dt.datetime.strptime(ps[i][1], "%Y/%m/%d %H:%M:%S").strftime(
                    "%Y/%m/%d"
                ),
                len(df),
            ),
        )
        median_dz = np.nanmedian(df.iloc[:, 1] - df.iloc[:, 0])
        ax.axvline(
            x=median_dz,
            color=colors[i],
            label=r"Median $\Delta Z$ = {:.2f} dBZ, $\sigma$ = {:.2f} dBZ".format(
                median_dz, np.nanstd(df.iloc[:, 1] - df.iloc[:, 0])
            ),
        )
        ax.grid()
        ax.set_xlabel(r"$Z_{radar} - Z_{disdrometer}$ [dBZ]")
        ax.set_ylabel("% of values")
        ax.set_ylim(top=0.15)
        ax.set_yticklabels(np.round(100 * np.array(ax.get_yticks()), decimals=0))

    ax.legend()
    ax.set_title(
        "PDF of $\Delta Z$ timestep by timestep \n" + "Comparison of two periods",
        fontsize=11,
        fontweight="semibold",
    )
    plt.savefig(
        "/homedata/ygrit/disdro_processing/juelich/timeseries2/histo_compare_2periods_example.png",
        transparent=False,
        dpi=500,
    )
    plt.show()


if __name__ == "__main__":
    histos(data)
