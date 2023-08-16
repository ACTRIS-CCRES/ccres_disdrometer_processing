from compute_dz_1event import dz_per_event

thresholds = []
filtered = True


def dz_timeseries(start_list, end_list):
    for i in range(len(start_list)):
        (
            start_time,
            end_time,
            nb_points,
            dZ_med_quartiles,
            Temp_criterion_ratio,
            Wind_criterion_ratio,
            RainRate_criterion_ratio,
            QC_vdsd_t_ratio,
            Accumulation_flag,
            Accumulation_relative_error_flag,
            good_points_ratio,
            good_points_number,
            AVG_RAINDROP_DIAMETER,
        ) = dz_per_event(start_list[i], end_list[i])
