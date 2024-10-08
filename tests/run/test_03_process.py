"""Tests processing."""

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_run_one_day(
    test_data_preprocessing, data_input_dir, data_conf_dir, data_out_dir
) -> None:
    """Test the processing for a specific test case."""
    conf = test_data_preprocessing["config_file"]
    today_file = data_out_dir / test_data_preprocessing["output"]["preprocess"]
    output_file = data_out_dir / test_data_preprocessing["output"]["process"]

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

    # run the processing
    # ---------------------------------------------------------------------------------
    # required args
    args = ["--today", str(today_file), "--config-file", str(conf), str(output_file)]

    runner = CliRunner()
    result = runner.invoke(
        cli.process,
        args,
        catch_exceptions=False,
    )

    assert result.exit_code == 0


def test_run_one_day_no_meteo_cli(
    test_data_processing_cli_option_no_meteo,
    data_input_dir,
    data_conf_dir,
    data_out_dir,
) -> None:
    """Test the processing for a specific test case."""
    conf = test_data_processing_cli_option_no_meteo["config_file"]
    today_file = (
        data_out_dir / test_data_processing_cli_option_no_meteo["output"]["preprocess"]
    )
    output_file = (
        data_out_dir / test_data_processing_cli_option_no_meteo["output"]["process"]
    )

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

    # run the processing
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        "--today",
        str(today_file),
        "--config-file",
        str(conf),
        "--no-meteo",
        str(output_file),
    ]

    runner = CliRunner()
    result = runner.invoke(
        cli.process,
        args,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
