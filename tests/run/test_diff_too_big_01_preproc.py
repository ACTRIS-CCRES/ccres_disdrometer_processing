"""Test to check cases with big differences between disdrometers and radars.."""

from click.testing import CliRunner

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.cli import cli


def test_preproc_big_diff(
    test_data_proc_big_diff, data_input_dir, data_conf_dir, data_out_dir_big_diff
) -> None:
    """Test the preprocessing for a specific test case."""
    site = test_data_proc_big_diff["site"]
    dates = test_data_proc_big_diff["list_dates"]
    radar = test_data_proc_big_diff["radar"]
    radar_pid = test_data_proc_big_diff["radar-pid"]
    disdro = test_data_proc_big_diff["disdro"]
    disdro_pid = test_data_proc_big_diff["disdro-pid"]
    has_meteo = test_data_proc_big_diff["meteo-available"]
    meteo = test_data_proc_big_diff["meteo"]
    conf = test_data_proc_big_diff["config_file"]

    # conf path
    conf = data_conf_dir / conf

    output_code = []
    for date in dates:
        # get the data if needed
        # ------------------------------------------------------------------------------
        # radar
        radar_file = utils.get_file_from_cloudnet(
            site, date, radar, radar_pid, data_input_dir
        )
        # disdro
        disdro_file = utils.get_file_from_cloudnet(
            site, date, disdro, disdro_pid, data_input_dir
        )
        # meteo
        if test_data_proc_big_diff["meteo-available"]:
            meteo_pid = test_data_proc_big_diff["meteo-pid"]
            meteo_file = utils.get_file_from_cloudnet(
                site, date, meteo, meteo_pid, data_input_dir
            )

        # output file
        output_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date)  # noqa E501

        # preprocessing QL output files
        output_ql_weather = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocessing_ql"
        ]["weather-overview_tmpl"].format(date)
        output_ql_zh = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocessing_ql"
        ]["zh-overview_tmpl"].format(date)

        # run the preprocessing
        # ------------------------------------------------------------------------------
        # required args
        args = [
            "-vvvvv",
            "preprocess",
            "--disdro-file",
            str(disdro_file),
            "--radar-file",
            str(radar_file),
            "--config-file",
            str(conf),
        ]
        # add meteo if available
        if has_meteo:
            args += [
                "--ws-file",
                str(meteo_file),
            ]

        args += [str(output_file)]

        print("\nPreprocess args:")
        print(args, "\n")

        runner = CliRunner()
        result = runner.invoke(
            cli.cli,
            args,
            catch_exceptions=False,
        )

        output_code.append((result.exit_code, date, result.output, "preprocess"))

        # run the preprocessing-QL
        # ------------------------------------------------------------------------------
        # required args
        args = [
            "-vvvvv",
            "preprocess-ql",
            str(output_file),
            str(output_ql_weather),
            str(output_ql_zh),
            "--config-file",
            str(conf),
        ]

        print("Preprocess-ql args:")
        print(args, "\n")

        runner = CliRunner()
        result = runner.invoke(
            cli.cli,
            args,
            catch_exceptions=False,
        )

        output_code.append((result.exit_code, date, result.output, "preprocess-ql"))

    for ret in output_code:
        assert ret[0] == 0, f"test failed for {ret[3]} {ret[1]}: {ret[2]}"
