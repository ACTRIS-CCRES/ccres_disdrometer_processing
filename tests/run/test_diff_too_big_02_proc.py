"""Test to check cases with big differences between disdrometers and radars.."""

import datetime as dt

from click.testing import CliRunner

from ccres_disdrometer_processing.cli import cli


def test_proc_big_diff(
    test_data_proc_big_diff, data_conf_dir, data_out_dir_big_diff
) -> None:
    """Test the preprocessing for a specific test case."""
    dates = test_data_proc_big_diff["list_dates"]
    conf = test_data_proc_big_diff["config_file"]

    # conf path
    conf = data_conf_dir / conf

    output_code = []

    # run processing
    # ---------------------------------------------------------------------------------
    runner = CliRunner()
    for date in dates:
        date_dt = dt.datetime.strptime(date, "%Y-%m-%d")
        date_dt_dm1 = date_dt - dt.timedelta(days=1)
        date_dm1 = date_dt_dm1.strftime("%Y-%m-%d")
        data_dt_dp1 = date_dt + dt.timedelta(days=1)
        date_dp1 = data_dt_dp1.strftime("%Y-%m-%d")

        # get input files
        dm1_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date_dm1)
        if not dm1_file.exists():
            dm1_file = None
        dp1_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date_dp1)
        if not dp1_file.exists():
            dp1_file = None

        d_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date)

        # process nc file
        process_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "process_tmpl"
        ].format(date)

        # prefix output process QL
        prefix_output_ql_summary = data_out_dir_big_diff / test_data_proc_big_diff[
            "output"
        ]["process_ql"]["summary_tmpl"].format(date)
        prefix_output_ql_detailled = data_out_dir_big_diff / test_data_proc_big_diff[
            "output"
        ]["process_ql"]["detailled_tmpl"].format(date)

        # run the processing
        # ------------------------------------------------------------------------------
        args = [
            "-vvvvv",
            "process",
        ]
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

        print("Process args:")
        print(args, "\n")

        result = runner.invoke(
            cli.cli,
            args,
            catch_exceptions=True,
        )

        print(result.exit_code, date, result.output, f"process {date}")
        output_code.append((result.exit_code, date, result.output, "process"))

    # run the processing ql
    # ------------------------------------------------------------------------------
    runner = CliRunner()
    for date in dates:
        date_dt = dt.datetime.strptime(date, "%Y-%m-%d")
        date_dt_dm1 = date_dt - dt.timedelta(days=1)
        date_dm1 = date_dt_dm1.strftime("%Y-%m-%d")
        data_dt_dp1 = date_dt + dt.timedelta(days=1)
        date_dp1 = data_dt_dp1.strftime("%Y-%m-%d")

        # get input files
        dm1_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date_dm1)
        if not dm1_file.exists():
            dm1_file = None
        dp1_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date_dp1)
        if not dp1_file.exists():
            dp1_file = None

        d_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "preprocess_tmpl"
        ].format(date)

        # process nc file
        process_file = data_out_dir_big_diff / test_data_proc_big_diff["output"][
            "process_tmpl"
        ].format(date)

        # prefix output process QL
        prefix_output_ql_summary = data_out_dir_big_diff / test_data_proc_big_diff[
            "output"
        ]["process_ql"]["summary_tmpl"].format(date)
        prefix_output_ql_detailled = data_out_dir_big_diff / test_data_proc_big_diff[
            "output"
        ]["process_ql"]["detailled_tmpl"].format(date)

        args = [
            "-vvvvv",
            "process-ql",
        ]
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

        print("Process-ql args:")
        print(" ".join(args), "\n")

        result = runner.invoke(
            cli.cli,
            args,
            catch_exceptions=True,
        )

        print(result.exit_code, date, result.output, f"process-ql {date}")
        output_code.append((result.exit_code, date, result.output, "process-ql"))

    for ret in output_code:
        assert ret[0] == 0, f"test failed for {ret[3]} {ret[1]}: {ret[2]}"
