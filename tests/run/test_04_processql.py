"""Tests preprocessing."""

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

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

    # run the preprocessing
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        "--config-file",
        str(conf),
        "--prefix-output-ql-summary",
        str(summary_png),
        "--prefix-output-ql-detailled",
        str(detail_png),
        "--preprocess-today",
        str(preprocess_file),
        str(process_file),
    ]

    runner = CliRunner()
    result = runner.invoke(
        cli.process_ql,
        args,
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
