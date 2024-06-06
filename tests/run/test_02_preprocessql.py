"""Tests preprocessing."""

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_run(test_data_preprocessing, data_conf_dir, data_out_dir) -> None:
    """Test the creation of preprocessing QLs for a specific test case."""
    conf = test_data_preprocessing["config_file"]
    input_file = data_out_dir / test_data_preprocessing["output"]["preprocess"]
    output_ql_1 = (
        data_out_dir
        / test_data_preprocessing["output"]["preprocessing_ql"]["weather-overview"]
    )
    output_ql_2 = (
        data_out_dir
        / test_data_preprocessing["output"]["preprocessing_ql"]["zh-overview"]
    )

    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf

    # run the preprocessing
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        str(input_file),
        str(output_ql_1),
        str(output_ql_2),
        "--config-file",
        str(conf),
    ]

    runner = CliRunner()
    result = runner.invoke(
        cli.preprocess_ql,
        args,
        catch_exceptions=False,
    )

    assert result.exit_code == 0
