"""Tests preprocessing."""

import datetime as dt

from click.testing import CliRunner

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.cli import cli


def test_run_ndays(
    test_data_processing_ndays, data_input_dir, data_conf_dir, data_out_dir
) -> None:
    """Test the preprocessing for a specific test case."""
    site = test_data_processing_ndays["site"]
    dates = test_data_processing_ndays["list_dates"]
    radar = test_data_processing_ndays["radar"]
    radar_pid = test_data_processing_ndays["radar-pid"]
    disdro = test_data_processing_ndays["disdro"]
    disdro_pid = test_data_processing_ndays["disdro-pid"]
    has_meteo = test_data_processing_ndays["meteo-available"]
    meteo = test_data_processing_ndays["meteo"]
    conf = test_data_processing_ndays["config_file"]

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
        if test_data_processing_ndays["meteo-available"]:
            meteo_pid = test_data_processing_ndays["meteo-pid"]
            meteo_file = utils.get_file_from_cloudnet(
                site, date, meteo, meteo_pid, data_input_dir
            )

        # output file
        output_file = data_out_dir / test_data_processing_ndays["output"][
            "preprocess_tmpl"
        ].format(date)  # noqa E501

        # run the preprocessing
        # ------------------------------------------------------------------------------
        # required args
        args = [
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

        runner = CliRunner()
        result = runner.invoke(
            cli.preprocess,
            args,
            catch_exceptions=False,
        )

        output_code.append((result.exit_code, date, result.output, "preprocess"))

    # run processing
    # ---------------------------------------------------------------------------------
    for date in dates:
        date_dt = dt.datetime.strptime(date, "%Y-%m-%d")
        date_dt_dm1 = date_dt - dt.timedelta(days=1)
        date_dm1 = date_dt_dm1.strftime("%Y-%m-%d")
        data_dt_dp1 = date_dt + dt.timedelta(days=1)
        date_dp1 = data_dt_dp1.strftime("%Y-%m-%d")

        # get input files
        dm1_file = data_out_dir / test_data_processing_ndays["output"][
            "preprocess_tmpl"
        ].format(date_dm1)
        if not dm1_file.exists():
            dm1_file = None
        dp1_file = data_out_dir / test_data_processing_ndays["output"][
            "preprocess_tmpl"
        ].format(date_dp1)
        if not dp1_file.exists():
            dp1_file = None

        d_file = data_out_dir / test_data_processing_ndays["output"][
            "preprocess_tmpl"
        ].format(date)

        # process nc file
        process_file = data_out_dir / test_data_processing_ndays["output"][
            "process_tmpl"
        ].format(date)

        # prefix output process QL
        prefix_output_ql_summary = data_out_dir / test_data_processing_ndays["output"][
            "process_ql"
        ]["summary_tmpl"].format(date)
        prefix_output_ql_detailled = data_out_dir / test_data_processing_ndays[
            "output"
        ]["process_ql"]["detailled_tmpl"].format(date)

        # run the processing
        # ------------------------------------------------------------------------------
        args = []
        if dm1_file is not None:
            args += ["--yesterday", str(dm1_file)]
        if dp1_file is not None:
            args += ["--tomorrow", str(dp1_file)]

        # required args
        args += [
            "--today",
            str(d_file),
            "--config-file",
            str(conf),
            str(process_file),
        ]
        runner = CliRunner()
        result = runner.invoke(
            cli.process,
            args,
            catch_exceptions=False,
        )

        output_code.append((result.exit_code, date, result.output, "process"))

        # run the processing ql
        # ------------------------------------------------------------------------------
        args = []
        if dm1_file is not None:
            args += ["--preprocess-yesterday", str(dm1_file)]
        if dp1_file is not None:
            args += ["--preprocess-tomorrow", str(dp1_file)]

        # required args
        args += [
            "--process-today",
            str(process_file),
            "--preprocess-today",
            str(d_file),
            "--config-file",
            str(conf),
            "--prefix-output-ql-summary",
            str(prefix_output_ql_summary),
            "--prefix-output-ql-detailled",
            str(prefix_output_ql_detailled),
        ]
        print(args)
        runner = CliRunner()
        result = runner.invoke(
            cli.process_ql,
            args,
            catch_exceptions=False,
        )

        output_code.append((result.exit_code, date, result.output, "process-ql"))

    for ret in output_code:
        assert ret[0] == 0, f"test failed for {ret[3]} {ret[1]}: {ret[2]}"
