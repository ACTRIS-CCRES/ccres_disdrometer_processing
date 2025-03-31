"""Tests preprocessing."""

import xarray as xr
from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_run_one_day(
    test_data_preprocessing, data_input_dir, data_conf_dir, data_out_dir
) -> None:
    """Test the preprocessing for a specific test case."""
    conf = test_data_preprocessing["config_file"]
    preprocess_file = data_out_dir / test_data_preprocessing["output"]["preprocess"]
    process_file = data_out_dir / test_data_preprocessing["output"]["process"]
    summary_png = (
        data_out_dir / test_data_preprocessing["output"]["process_ql"]["summary"]
    )
    detail_png = (
        data_out_dir / test_data_preprocessing["output"]["process_ql"]["detailled"]
    )
    print("HELLO 1")
    processed_nc = xr.open_dataset(process_file)
    print(processed_nc.events.size)

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

    # run the preprocessing
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        "--preprocess-today",
        str(preprocess_file),
        "--process-today",
        str(process_file),
        "--config-file",
        str(conf),
        "--prefix-output-ql-summary",
        str(summary_png),
        "--prefix-output-ql-detailled",
        str(detail_png),
    ]

    runner = CliRunner()
    result = runner.invoke(
        cli.process_ql,
        args,
        catch_exceptions=False,
    )

    print("HELLO")

    assert result.exit_code == 0, result.output
