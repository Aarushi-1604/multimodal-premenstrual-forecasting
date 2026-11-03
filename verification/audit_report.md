# mcPHASES Phase 1 Audit

## active_minutes.csv

- Rows: 5552, Cols: 8
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'sedentary', 'lightly', 'moderately', 'very']
- Missingness % per column:
    - sedentary: 33.4%

## active_zone_minutes.csv

- Rows: 154482, Cols: 7
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'heart_zone_id', 'total_minutes']
- Missingness % per column:

## altitude.csv

- Rows: 90878, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'altitude']
- Missingness % per column:

## calories.csv

- Rows: 20166975, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'calories']
- Missingness % per column:

## computed_temperature.csv

- Rows: 5575, Cols: 14
- Participant column: id (n unique: 42)
- Date column: sleep_start_day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'sleep_start_day_in_study', 'sleep_start_timestamp', 'sleep_end_day_in_study', 'sleep_end_timestamp', 'type', 'temperature_samples', 'nightly_temperature', 'baseline_relative_sample_sum', 'baseline_relative_sample_sum_of_squares', 'baseline_relative_nightly_standard_deviation', 'baseline_relative_sample_standard_deviation']
- Missingness % per column:
    - baseline_relative_sample_sum: 6.8%
    - baseline_relative_sample_sum_of_squares: 6.8%
    - baseline_relative_nightly_standard_deviation: 6.8%
    - baseline_relative_sample_standard_deviation: 6.8%

## demographic_vo2_max.csv

- Rows: 11482, Cols: 8
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'demographic_vo2_max', 'demographic_vo2_max_error', 'filtered_demographic_vo2_max', 'filtered_demographic_vo2_max_error']
- Missingness % per column:

## distance.csv

- Rows: 7666949, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'distance']
- Missingness % per column:

## estimated_oxygen_variation.csv

- Rows: 3070312, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'infrared_to_red_signal_ratio']
- Missingness % per column:

## exercise.csv

- Rows: 7282, Cols: 26
- Participant column: id (n unique: 31)
- Date column: start_day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'start_day_in_study', 'start_timestamp', 'last_modified_day_in_study', 'last_modified_timestamp', 'original_start_day_in_study', 'original_start_timestamp', 'originalduration', 'activityname', 'activitytypeid', 'activitylevel', 'averageheartrate', 'calories', 'duration', 'activeduration', 'steps', 'logtype', 'manualvaluesspecified', 'heartratezones', 'activezoneminutes', 'elevationgain', 'hasgps', 'shouldfetchdetails', 'hasactivezoneminutes']
- Missingness % per column:
    - averageheartrate: 0.7%
    - steps: 2.4%
    - heartratezones: 0.7%
    - elevationgain: 9.8%

## glucose.csv

- Rows: 837130, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '90')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'glucose_value']
- Missingness % per column:
    - glucose_value: 0.1%

## heart_rate.csv

- Rows: 63100276, Cols: 7
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'bpm', 'confidence']
- Missingness % per column:

## heart_rate_variability_details.csv

- Rows: 436262, Cols: 9
- Participant column: id (n unique: 40)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'rmssd', 'coverage', 'low_frequency', 'high_frequency']
- Missingness % per column:

## height_and_weight.csv

- Rows: 42, Cols: 5
- Participant column: id (n unique: 42)
- Date column: None, range: None
- Columns: ['id', 'height_2022', 'weight_2022', 'height_2024', 'weight_2024']
- Missingness % per column:
    - height_2022: 50.0%
    - weight_2022: 42.9%
    - height_2024: 81.0%
    - weight_2024: 73.8%

## hormones_and_selfreport.csv

- Rows: 5659, Cols: 22
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'phase', 'lh', 'estrogen', 'pdg', 'flow_volume', 'flow_color', 'appetite', 'exerciselevel', 'headaches', 'cramps', 'sorebreasts', 'fatigue', 'sleepissue', 'moodswing', 'stress', 'foodcravings', 'indigestion', 'bloating']
- Missingness % per column:
    - lh: 5.7%
    - estrogen: 5.7%
    - pdg: 67.1%
    - flow_volume: 43.6%
    - flow_color: 43.6%
    - appetite: 41.2%
    - exerciselevel: 41.2%
    - headaches: 41.2%
    - cramps: 41.2%
    - sorebreasts: 41.2%
    - fatigue: 41.1%
    - sleepissue: 41.2%
    - moodswing: 41.3%
    - stress: 41.2%
    - foodcravings: 41.2%
    - indigestion: 41.2%
    - bloating: 41.2%

## respiratory_rate_summary.csv

- Rows: 6301, Cols: 17
- Participant column: id (n unique: 40)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'full_sleep_breathing_rate', 'full_sleep_standard_deviation', 'full_sleep_signal_to_noise', 'deep_sleep_breathing_rate', 'deep_sleep_standard_deviation', 'deep_sleep_signal_to_noise', 'light_sleep_breathing_rate', 'light_sleep_standard_deviation', 'light_sleep_signal_to_noise', 'rem_sleep_breathing_rate', 'rem_sleep_standard_deviation', 'rem_sleep_signal_to_noise']
- Missingness % per column:

## resting_heart_rate.csv

- Rows: 13737, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'value', 'error']
- Missingness % per column:

## sleep.csv

- Rows: 14765, Cols: 18
- Participant column: id (n unique: 42)
- Date column: sleep_start_day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'sleep_start_day_in_study', 'sleep_start_timestamp', 'sleep_end_day_in_study', 'sleep_end_timestamp', 'duration', 'minutestofallasleep', 'minutesasleep', 'minutesawake', 'minutesafterwakeup', 'timeinbed', 'efficiency', 'type', 'infocode', 'levels', 'mainsleep']
- Missingness % per column:

## sleep_score.csv

- Rows: 5308, Cols: 12
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'overall_score', 'composition_score', 'revitalization_score', 'duration_score', 'deep_sleep_in_minutes', 'resting_heart_rate', 'restlessness']
- Missingness % per column:
    - composition_score: 30.0%
    - duration_score: 30.0%
    - deep_sleep_in_minutes: 0.1%

## steps.csv

- Rows: 7666949, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'steps']
- Missingness % per column:

## stress_score.csv

- Rows: 7932, Cols: 14
- Participant column: id (n unique: 36)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'stress_score', 'sleep_points', 'max_sleep_points', 'responsiveness_points', 'max_responsiveness_points', 'exertion_points', 'max_exertion_points', 'status', 'calculation_failed']
- Missingness % per column:

## subject-info.csv

- Rows: 42, Cols: 8
- Participant column: id (n unique: 42)
- Date column: None, range: None
- Columns: ['id', 'birth_year', 'gender', 'ethnicity', 'education', 'sexually_active', 'self_report_menstrual_health_literacy', 'age_of_first_menarche']
- Missingness % per column:
    - self_report_menstrual_health_literacy: 2.4%

## time_in_heart_rate_zones.csv

- Rows: 5553, Cols: 8
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'in_default_zone_3', 'in_default_zone_2', 'in_default_zone_1', 'below_default_zone_1']
- Missingness % per column:

## wrist_temperature.csv

- Rows: 6856019, Cols: 6
- Participant column: id (n unique: 42)
- Date column: day_in_study, range: ('1', '1004')
- Columns: ['id', 'study_interval', 'is_weekend', 'day_in_study', 'timestamp', 'temperature_diff_from_baseline']
- Missingness % per column:
