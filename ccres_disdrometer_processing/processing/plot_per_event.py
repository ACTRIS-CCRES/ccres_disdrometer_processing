import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import xarray as xr


def plot_from_preprocessed_noweather(preprocessed, processed, conf):
    preprocessed = xr.open_dataset(preprocessed)
    processed = xr.open_dataset(processed)

    print(f"Number of events : {len(processed.events)}")

    if len(processed.events) != 0:
        for k in range(len(processed.events)):
            preprocessed_event = preprocessed.sel(
                {
                    "time": slice(
                        pd.to_datetime(processed.start_event.values[k])
                        - pd.Timedelta(1, "hour"),
                        pd.to_datetime(processed.end_event.values[k])
                        + pd.Timedelta(1, "hour"),
                    )
                }
            )
            print("OK1")
            processed_event = processed.sel(
                {
                    "time": slice(
                        pd.to_datetime(processed.start_event.values[k]),
                        pd.to_datetime(processed.end_event.values[k]),
                    )
                }
            )
            print("OK2")

            # QL plot 2d Z
            cmap = plt.get_cmap("rainbow").copy()
            cmap.set_under("w")

            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            (ax, ax2, ax3, ax4) = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            locator = mpl.dates.AutoDateLocator()
            formatter = mpl.dates.ConciseDateFormatter(locator)
            for axe in (ax, ax2, ax3, ax4):
                axe.xaxis.set_major_formatter(formatter)
                axe.set_xlim(
                    left=preprocessed_event.time.values[0],
                    right=preprocessed_event.time.values[-1],
                )

            pc = ax.pcolormesh(
                pd.to_datetime(preprocessed_event.time.values),
                preprocessed_event.range.values,
                preprocessed_event.Zdcr.T,
                vmin=0,
                cmap=cmap,
                shading="nearest",
            )
            ax.set_ylabel(preprocessed_event.range.attrs["long_name"])
            ax.set_title(preprocessed_event.Zdcr.attrs["long_name"])
            pos = ax.get_position()
            cb_ax = fig.add_axes([0.91, pos.y0, 0.02, pos.y1 - pos.y0])
            cb1 = fig.colorbar(pc, orientation="vertical", cax=cb_ax)
            cb1.set_label(
                f"{preprocessed_event.Zdcr.attrs['long_name']} [{preprocessed_event.Zdcr.attrs['units']}]"  # noqa
            )

            # QL Plot Zbasta vs Zdisdro
            for i in range(len(processed_event.range.values)):
                rng = processed_event.range.values[i]
                ax2.plot(
                    processed_event.time.values,
                    processed_event.Zdcr[:, i].values,
                    label=f"radar @ {rng:.0f} m",
                    linewidth=1,
                )
            ax2.plot(
                processed_event.time.values,
                processed_event["Zdd"],
                color="green",
                lw=2,
                label="disdrometer reflectivity",
            )
            ax2.set_ylabel("Z [dBZ]")
            ax2.set_xlabel("Time (UTC)")
            ax2.grid()
            ax2.legend()
            ax2.set_title("Disdrometer and DCR reflectivity")

            # QL Plot Disdro rain
            ax3.plot(
                processed_event.time.values,
                processed_event.disdro_cum_since_event_begin,
                color="green",
            )
            ax3.set_ylabel("Rain accumulation [mm]")
            ax3.set_xlabel("Time (UTC)")
            ax3.grid()
            ax3.set_title("Disdrometer rain accumulation")

            ax4.plot(
                processed_event.time.values,
                processed_event["Delta_Z"].sel(
                    {"range": conf["instrument_parameters"]["DCR_DZ_RANGE"]}
                ),
                color="green",
                label=rf"$\Delta Z$ @ {conf['instrument_parameters']['DCR_DZ_RANGE']:.0f}m",  # noqa
            )
            ax4.set_ylabel(r"$\Delta Z$ [dBZ]")
            ax4.set_xlabel("Time (UTC)")
            ax4.grid()
            ax4.legend()
            ax4.set_title(
                "Differences between DCR and disdrometer-modeled reflectivity"
            )

            # Save
            plt.suptitle(
                "{} - start : {} - end : {}".format(
                    preprocessed_event.attrs["location"],
                    pd.to_datetime(processed_event.start_event[k]).strftime(
                        "%Y-%m-%d %H:%m"
                    ),
                    pd.to_datetime(processed_event.end_event[k]).strftime(
                        "%Y-%m-%d %H:%m"
                    ),
                ),
            )
            plt.savefig(
                "./{}_{}_QL1.png".format(
                    preprocessed.attrs["station_name"],
                    pd.to_datetime(processed.start_event.values[k]).strftime(
                        "%Y-%m-%d-%T"
                    ),
                ),
                dpi=500,
                transparent=False,
                edgecolor="white",
            )
            plt.close()

            # Quality checks visualization
            Quality_matrix = np.hstack(
                (
                    processed_event.QC_pr.values.reshape((-1, 1)),
                    processed_event.QC_vdsd_t.values.reshape((-1, 1)),
                )
            )
            Quality_sum = Quality_matrix.all(axis=1)

            Quality_matrix_sum = np.flip(
                np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
            )
            Quality_matrix_sum.astype(int)
            print(Quality_matrix_sum.shape)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.xaxis.set_major_formatter(formatter)
            cmap = mpl.colors.ListedColormap(["red", "green"])
            ax.pcolormesh(
                processed_event.time.values,
                np.arange(len(Quality_matrix_sum.T)),
                Quality_matrix_sum.T,
                cmap=cmap,
                shading="nearest",
            )
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_yticks([q for q in range(len(Quality_matrix_sum.T))])
            ax.set_yticklabels(
                np.array(
                    [
                        "QC rain rate",
                        "QC V(D)",
                        "Overall QC",
                    ]
                )[::-1],
                fontsize=8,
            )
            ax.set_title(
                "{} - {} - QF / QC timeseries".format(
                    preprocessed_event.attrs["location"],
                    pd.to_datetime(processed_event.start_event.values[k]).strftime(
                        "%Y-%m-%d %H:%m"
                    ),
                )
            )
            plt.savefig(
                "./{}_{}_QL2.png".format(
                    preprocessed.attrs["station_name"],
                    pd.to_datetime(processed.start_event.values[k]).strftime(
                        "%Y-%m-%d-%T"
                    ),
                ),
                dpi=500,
                transparent=False,
                edgecolor="white",
            )
            plt.close()

    return


if __name__ == "__main__":
    preprocessed = "/home/ygrit/disdro_processing/ccres_disdrometer_processing/tests/data/outputs/juelich_2021-12-04_mira-parsivel_preprocessed.nc"  # noqa
    processed = "./JOYCE_2021-12-04_processed.nc"
    conf = toml.load("../../tests/data/conf/config_juelich_mira-parsivel.toml")

    pro = xr.open_dataset(processed)
    print(pro.start_event.attrs)
    plot_from_preprocessed_noweather(preprocessed, processed, conf)

    # # Total QC/QF :
    # print(QC_RR_disdro.shape, QC_vdsd_t.shape)
    # Quality_matrix = np.hstack((QC_RR_disdro.reshape((-1, 1)), QC_vdsd_t))
    # Quality_sum = Quality_matrix.all(axis=1)

    # Quality_matrix_sum = np.flip(
    #     np.hstack((Quality_matrix, Quality_sum.reshape((-1, 1)))), axis=1
    # )
    # Quality_matrix_sum.astype(int)

    # # Plot QC/QF
    # fig, ax = plt.subplots(figsize=(10, 5))
    # t = disdro.time
    # ax.xaxis.set_major_formatter(formatter)
    # cmap = colors.ListedColormap(["red", "green"])
    # ax.pcolormesh(
    #     t,
    #     np.arange(len(Quality_matrix_sum.T)),
    #     Quality_matrix_sum.T,
    #     cmap=cmap,
    #     shading="nearest",
    # )
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # ax.set_yticks([q for q in range(len(Quality_matrix_sum.T))])
    # ax.set_yticklabels(
    #     np.array(
    #         [
    #             "QC Rain Rate",
    #             "QC V(D)",
    #             r"$\Pi$",
    #         ]
    #     )[::-1],
    #     fontsize=8,
    # )
    # ax.set_title("{} - QF / QC timeseries".format(dcr.attrs["location"]))
    # plt.savefig(
    #     data_dir
    #     + "/QL/{}_{}_{}_Quality_checks.png".format(
    #         start_time.strftime("%Y%m%d%T"),
    #         preprocessed_ds.radar_source,
    #         preprocessed_ds.disdrometer_source,
    #     ),
    #     dpi=500,
    #     transparent=False,
    #     edgecolor="white",
    # )
    # plt.close()
