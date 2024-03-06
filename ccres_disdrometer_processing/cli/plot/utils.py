import datetime as dt
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as stats
import xarray as xr


def load_module(name, path):
    """Load python file as module.

    Notes
    -----
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    Parameters
    ----------
    name : str
        The name of the module.
    path : str
        The path to the python file to load.

    Returns
    -------
    module
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (OSError, ImportError, FileNotFoundError) as err:
        print(f"ERROR: impossible to load module {path}")
        print(err)
        sys.exit(1)

    return module


def read_nc(file_: str):
    """_summary_

    Args:
        conf (_type_): _description_
        file_ (_type_): _description_

    Returns:
        _type_: _description_
    """
    return xr.open_dataset(file_)


def add_logo():
    """add logos to the current plot on top right corner.

    Parameters
    ----------
    dirname: str
        directory name
    station: str
        station name
    """

    plt.axes([0.76, 0.9, 0.2, 0.1])  # left, bottom, width, height
    plt.axis("off")

    try:
        logo = plt.imread("ccres_disdrometer_processing/cli/assets/logo/logo_CCRES.png")
        plt.imshow(logo, origin="upper")
    except OSError:
        print("graphreobs file : Impossible to include the logo !!!!!")

    return


def npdt64_to_datetime(dt64):
    """_summary_

    Args:
        dt64 (_type_): _description_

    Returns:
        _type_: _description_
    """
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (dt64 - unix_epoch) / one_second
    return dt.datetime.utcfromtimestamp(seconds_since_epoch)


def f_th(x):
    """_summary_

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 9.40 * (
        1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
    )  # Gun and Kinzer (th.)


def f_fit(x, a, b, c):
    """_summary_

    Args:
        x (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    return a * (1 - np.exp(-b * np.power(x * (10**-3), c)))  # target shape


def get_size_and_classe_to_fit(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    drop_density = np.nansum(data["psd"].values, axis=0)  # sum over time dim
    psd_nonzero_indexes = np.where(drop_density != 0)
    list_sizes, list_classes = [], []

    for k in range(
        len(psd_nonzero_indexes[0])
    ):  # add observations (size, speed) in the proportions described by the diameter/velocity distribution
        list_sizes += [data["size_classes"][psd_nonzero_indexes[0][k]]] * int(
            drop_density[psd_nonzero_indexes[0][k], psd_nonzero_indexes[1][k]]
        )
        list_classes += [data["speed_classes"][psd_nonzero_indexes[1][k]]] * int(
            drop_density[psd_nonzero_indexes[0][k], psd_nonzero_indexes[1][k]]
        )

    sizes, classes = np.array(list_sizes), np.array(list_classes)
    return sizes, classes, drop_density


def get_y_fit_dd(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    sizes, classes, drop_density = get_size_and_classe_to_fit(data)  # disdrometer
    #
    if classes.size != 0:
        try:
            popt, pcov = so.curve_fit(f_fit, sizes, classes)
            y_hat = f_fit(data["size_classes"], popt[0], popt[1], popt[2])
            y_th = f_th(data["size_classes"])
            return y_hat, y_th, sizes, classes, drop_density, 1
        except Exception as e:
            print(e)
            return (
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            )
    else:
        return (
            0,
            0,
            0,
            0,
            0,
            0,
        )


def get_cdf(delta_ZH, nbins=100):
    """_summary_

    Args:
        delta_ZH (_type_): _description_
        nbins (int, optional): _description_. Defaults to 100.
    """
    cdf = stats.cumfreq(delta_ZH, numbins=100)
    x_ = cdf.lowerlimit + np.linspace(
        0, cdf.binsize * cdf.cumcount.size, cdf.cumcount.size
    )
    return cdf, x_


def get_min_max_limits(zh_dd, zh_gate):
    """_summary_

    Args:
        zh_dd (_type_): _description_
        zh_gate (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.DataFrame.from_dict({"dd": zh_dd, "dcr": zh_gate})
    df = df.replace(-np.inf, np.nan)
    df = df.dropna()
    if df is not None:
        # round up/down 5
        lim_min, lim_max = (
            np.floor((df.min().min() - 1) / 5) * 5,
            np.ceil((df.max().max() + 1) / 5) * 5,
        )

        return lim_min, lim_max
    else:
        return 0, 10
