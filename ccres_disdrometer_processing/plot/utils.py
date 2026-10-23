from importlib import resources as impresources
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as stats
import xarray as xr

from ccres_disdrometer_processing.assets import logo as logo_dir
from ccres_disdrometer_processing.utils import scalar_positions


def read_nc(file_: str):
    """Read a netCDF file in memory with xarray, then close it.

    Files left open break later HDF5 reopenings of the same file in the same
    process ("NetCDF: HDF error", or a crash), e.g. a preprocessed file read
    by process-ql after preprocess-ql and process.

    Parameters
    ----------
    file_ : str or pathlib.Path
        The path to the file to read.

    Returns
    -------
    xarray.Dataset
        The xarray object containing the data.

    """
    if not isinstance(file_, Path):
        file_ = Path(file_)

    return xr.load_dataset(file_)


def add_logo():
    """Add logos to the current plot on top right corner.

    Parameters
    ----------
    dirname: str
        directory name
    station: str
        station name

    """
    plt.axes([0.76, 0.9, 0.2, 0.1])  # left, bottom, width, height
    plt.axis("off")

    with impresources.as_file(
        impresources.files(logo_dir).joinpath("logo_CCRES.png")
    ) as img_path:
        logo = plt.imread(img_path)

    plt.imshow(logo, origin="upper")

    return


def npdt64_to_datetime(dt64):
    """Convert np.datetime64 to datetime object.

    Parameters
    ----------
    dt64 : numpy.datetime64
        date store in numpy datetime64 format.

    Returns
    -------
    datetime.datetime
        The date converted as a datetime.datetime object.

    """
    return pd.Timestamp(dt64).to_pydatetime()


def f_th(x):
    """Compute the theoretical fall speed using Gun and Kinzer formula.

    Parameters
    ----------
    x: numpy.ndarray
        The disdrometer size classes distribution.


    Returns
    -------
    numpy.ndarray
        The theoretical fall speed.

    """
    return 9.40 * (
        1 - np.exp(-1.57 * (10**3) * np.power(x * (10**-3), 1.15))
    )  # Gun and Kinzer (th.)


def f_fit(x, a, b, c):
    """Fit the disdrometer size classes distribution.

    Parameters
    ----------
    x : numpy.ndarray
        The disdrometer size classes distribution.
    a : numpy.ndarray
        Coefficient a of Gun and Kinzer formula.
    b : numpy.ndarray
        Coefficient b of Gun and Kinzer formula.
    c : numpy.ndarray
        Coefficient c of Gun and Kinzer formula.

    Returns
    -------
    numpy.ndarray
        The fitted size classes distribution.

    """
    return a * (1 - np.exp(-b * np.power(x * (10**-3), c)))  # target shape


def get_size_and_classe_to_fit(data):
    """Extract size and speed classes and density.

    Parameters
    ----------
    data : xarray.Dataset
        The disdrometer psd variable.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The size classes, the speed classes and the drop density.

    """
    drop_density = np.nansum(data["psd"].values, axis=0)  # sum over time dim
    psd_nonzero_indexes = np.where(drop_density != 0)
    list_sizes, list_classes = [], []

    for k in range(len(psd_nonzero_indexes[0])):
        # add observations (size, speed) in the proportions described
        # by the diameter/velocity distribution
        list_sizes += [data["size_classes"][psd_nonzero_indexes[0][k]]] * int(
            drop_density[psd_nonzero_indexes[0][k], psd_nonzero_indexes[1][k]]
        )
        list_classes += [data["speed_classes"][psd_nonzero_indexes[1][k]]] * int(
            drop_density[psd_nonzero_indexes[0][k], psd_nonzero_indexes[1][k]]
        )

    sizes, classes = np.array(list_sizes), np.array(list_classes)
    return sizes, classes, drop_density


def get_y_fit_dd(data):
    """Fit the disdrometer size classes distribution.

    Parameters
    ----------
    data : xarray.Dataset
        Data read from disdrometer.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray,
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        The fitted size classes distribution, the theoretical size classes distribution,
        the size classes, the speed classes, the drop density and the flag.

    """
    sizes, classes, drop_density = get_size_and_classe_to_fit(data)  # disdrometer
    #
    if classes.size != 0:
        try:
            popt, pcov = so.curve_fit(f_fit, sizes, classes, maxfev=5000)
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


def linear_reg_scipy(x, y):
    """Do linear regression.

    Parameters
    ----------
    x: np.array()
        var 1 (if time -> int or float)
    y: np.array()
        var 2

    Returns
    -------
    slope: float
        Slope of the regression line.
    intercept: float
        Intercept of the regression line.
    r_value: float
        Correlation coefficient.
    p_value: float
        The p-value for a hypothesis test whose null hypothesis is that the slope
        is zero, using Wald Test with t-distribution of the test statistic.
    std_err: float
        Standard error of the estimated slope (gradient), under the assumption of
        residual normality.

    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


def read_and_concatenante_preprocessed_ds(
    ds_pro: xr.Dataset, preprocessing_files: list[Path]
):
    """Read and concatenate preprocessed file.

    Parameters
    ----------
    ds_pro : xr.Dataset
        The process dataset.
    preprocessing_files : list[Path]
        The list of preprocessing files to read and concatenate.

    """
    tmp_ds = []
    for file in preprocessing_files:
        tmp_ds.append(scalar_positions(read_nc(file)))

    ds_prepro = xr.concat(tmp_ds, dim="time", data_vars="all", join="outer")

    return ds_prepro
