from pathlib import Path

from click.testing import CliRunner

from ccres_disdrometer_processing import utils
from ccres_disdrometer_processing.cli import cli

params_list = [
    {
        "site": "lindenberg",
        "date_yesterday": "2024-09-14",
        "date_today": "2024-09-15",
        "date_tomorrow": "2024-09-16",
        "radar": "mira-35",
        "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
        "disdro": "thies-lnm",
        "disdro-pid": "https://hdl.handle.net/21.12132/3.ddeab96e6197478a",
        "meteo-available": True,
        "meteo": "weather-station",
        "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
        "config_file": "config_lindenberg_mira-thies.toml",
        "output": {
            "preprocess_yesterday": "lindenberg_2024-09-14_mira-thies_preprocessed.nc",
            "preprocess_today": "lindenberg_2024-09-15_mira-thies_preprocessed.nc",
            "preprocess_tomorrow": "lindenberg_2024-09-16_mira-thies_preprocessed.nc",
            "preprocessing_ql": {
                "weather-overview": "lindenberg_2024-09-15_mira-thies_preproc-weather-overview.png",  # noqa E501
                "zh-overview": "lindenberg_2024-09-15_mira-thies_zh-preproc-overview.png",  # noqa E501
            },
            "process": "lindenberg_2024-09-15_mira-thies_processed.nc",
            "process_ql": {
                "summary": "lindenberg_2024-09-15_mira-thies_process-summary",
                "detailled": "lindenberg_2024-09-15_mira-thies_process-detailled",  # noqa E501
            },
        },
    },
    {
        "site": "lindenberg",
        "date_yesterday": "2024-09-15",
        "date_today": "2024-09-16",
        "date_tomorrow": "2024-09-17",
        "radar": "mira-35",
        "radar-pid": "https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b",
        "disdro": "thies-lnm",
        "disdro-pid": "https://hdl.handle.net/21.12132/3.ddeab96e6197478a",
        "meteo-available": True,
        "meteo": "weather-station",
        "meteo-pid": "https://hdl.handle.net/21.12132/3.ffb25f43330f4793",
        "config_file": "config_lindenberg_mira-thies.toml",
        "output": {
            "preprocess_yesterday": "lindenberg_2024-09-15_mira-thies_preprocessed.nc",
            "preprocess_today": "lindenberg_2024-09-16_mira-thies_preprocessed.nc",
            "preprocess_tomorrow": "lindenberg_2024-09-17_mira-thies_preprocessed.nc",
            "preprocessing_ql": {
                "weather-overview": "lindenberg_2024-09-16_mira-thies_preproc-weather-overview.png",  # noqa E501
                "zh-overview": "lindenberg_2024-09-16_mira-thies_zh-preproc-overview.png",  # noqa E501
            },
            "process": "lindenberg_2024-09-16_mira-thies_processed.nc",
            "process_ql": {
                "summary": "lindenberg_2024-09-16_mira-thies_process-summary",
                "detailled": "lindenberg_2024-09-16_mira-thies_process-detailled",  # noqa E501
            },
        },
    },
]


def test_overlap(data_input_dir, data_conf_dir, data_dir):
    path: Path = data_dir / "outputs_test_ql_overlapping_event"
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data_overlap_dir = Path(path)
    for params in params_list:
        site = params["site"]
        radar = params["radar"]
        radar_pid = params["radar-pid"]
        disdro = params["disdro"]
        disdro_pid = params["disdro-pid"]
        has_meteo = params["meteo-available"]
        meteo = params["meteo"]
        conf = data_conf_dir / params["config_file"]
        prepro_file_yesterday = (
            data_overlap_dir / params["output"]["preprocess_yesterday"]
        )
        prepro_file_today = data_overlap_dir / params["output"]["preprocess_today"]
        prepro_file_tomorrow = (
            data_overlap_dir / params["output"]["preprocess_tomorrow"]
        )
        output_process = data_overlap_dir / params["output"]["process"]
        # do the preprocessing for the three input days :
        for date, output in zip(
            [params["date_yesterday"], params["date_today"], params["date_tomorrow"]],
            [prepro_file_yesterday, prepro_file_today, prepro_file_tomorrow],
        ):
            print(date, output)
            if not output.exists():
                # get the data if needed
                # ---------------------------------------------------------------------------------  # noqa E501
                # radar
                radar_file = utils.get_file_from_cloudnet(
                    site, date, radar, radar_pid, data_input_dir
                )
                # disdro
                disdro_file = utils.get_file_from_cloudnet(
                    site, date, disdro, disdro_pid, data_input_dir
                )
                # meteo
                if params["meteo-available"]:
                    meteo_pid = params["meteo-pid"]
                    meteo_file = utils.get_file_from_cloudnet(
                        site, date, meteo, meteo_pid, data_input_dir
                    )
                # other parameters
                # ---------------------------------------------------------------------------------  # noqa E501
                # conf
                conf = data_conf_dir / conf
                # run the preprocessing
                # ---------------------------------------------------------------------------------  # noqa E501
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
                args += [str(output)]
                runner = CliRunner()
                result_preprocess = runner.invoke(
                    cli.preprocess,
                    args,
                    catch_exceptions=False,
                )
                assert result_preprocess.exit_code == 0
        # do the processing
        if output_process.exists():
            print("File exists, remove and resave")
            output_process.unlink()
        process_args = [
            "--yesterday",
            str(prepro_file_yesterday),
            "--today",
            str(prepro_file_today),
            "--tomorrow",
            str(prepro_file_tomorrow),
            "--config-file",
            str(conf),
            str(output_process),
            "-v",
        ]
        runner = CliRunner()
        result_process = runner.invoke(
            cli.process,
            process_args,
            catch_exceptions=False,
        )
        assert result_process.exit_code == 0
    # Create the processing Quicklooks for 15/09/2024
    params = params_list[1]
    conf = params["config_file"]
    preprocess_file_yesterday = (
        data_overlap_dir / params["output"]["preprocess_yesterday"]
    )
    preprocess_file_today = data_overlap_dir / params["output"]["preprocess_today"]
    preprocess_file_tomorrow = (
        data_overlap_dir / params["output"]["preprocess_tomorrow"]
    )
    process_file_yesterday = (
        data_overlap_dir / params_list[0]["output"]["process"]
    )  # 15/09
    process_file_today = data_overlap_dir / params_list[1]["output"]["process"]  # 16/09
    summary_png = data_overlap_dir / params["output"]["process_ql"]["summary"]
    detail_png = data_overlap_dir / params["output"]["process_ql"]["detailled"]
    # other parameters
    # ---------------------------------------------------------------------------------
    # conf
    conf = data_conf_dir / conf
    # run the process quicklooks
    # ---------------------------------------------------------------------------------
    # required args
    args = [
        "--process-yesterday",
        str(process_file_yesterday),
        "--process-today",
        str(process_file_today),
        "--preprocess-yesterday",
        str(preprocess_file_yesterday),
        "--preprocess-today",
        str(preprocess_file_today),
        "--preprocess-tomorrow",
        str(preprocess_file_tomorrow),
        "--config-file",
        str(conf),
        "--prefix-output-ql-summary",
        str(summary_png),
        "--prefix-output-ql-detailled",
        str(detail_png),
    ]
    runner = CliRunner()
    result_processql = runner.invoke(
        cli.process_ql,
        args,
        catch_exceptions=False,
    )
    assert result_processql.exit_code == 0
