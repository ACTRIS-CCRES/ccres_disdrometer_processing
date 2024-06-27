import datetime
import glob
import os

import template_static_monitoring_timeseries_pdf as plot_timeseries_pdf


def plot_timeseries(site, conf, input_dir, save_plot_dir, start_date, end_date):
    files = sorted(glob.glob(input_dir))
    print(files)
    print("number of files : ", len(files))
    files_date = []
    for fi in files:
        filename = os.path.basename(fi).split("/")[-1]
        day_of_file = datetime.datetime.strptime(filename[0:8], "%Y%m%d")
        if (day_of_file >= start_date) and (day_of_file <= end_date):
            print(day_of_file)
            files_date.append(fi)
    print(files_date[0], files_date[-1])

    fig, ax = plot_timeseries_pdf.monitoring_timeseries(files_date, conf=conf)
    start_str = datetime.datetime.strftime(start_date, "%Y%m%d")
    end_str = datetime.datetime.strftime(end_date, "%Y%m%d")
    path_tosave = os.path.join(
        save_plot_dir,
        f"{start_str}_{end_str}_{site}_timeseries.png",
    )
    # maybe get date edges from xarray concatenated dataset,
    # if processing files are available in the folder in a shorter subset of dates than the whole asked time period  # noqa
    if os.path.isfile(path_tosave):
        os.remove(path_tosave)
    fig.savefig(
        path_tosave,
        dpi=500,
    )
    return


def plot_dz_pdf(site, conf, input_dir, save_plot_dir, start_date, end_date):
    files = sorted(glob.glob(input_dir))
    files_date = []
    for fi in files:
        filename = os.path.basename(fi).split("/")[-1]
        day_of_file = datetime.datetime.strptime(filename[0:8], "%Y%m%d")
        if (day_of_file >= start_date) and (day_of_file <= end_date):
            # print(day_of_file)
            files_date.append(fi)
    fig, ax = plot_timeseries_pdf.timestep_pdf(files_date, conf=conf)
    start_str = datetime.datetime.strftime(start_date, "%Y%m%d")
    end_str = datetime.datetime.strftime(end_date, "%Y%m%d")
    path_tosave = os.path.join(
        save_plot_dir,
        f"{start_str}_{end_str}_{site}_pdf.png",
    )
    if os.path.isfile(path_tosave):
        os.remove(path_tosave)
    fig.savefig(
        path_tosave,
        dpi=500,
    )
    return
