import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from src.SARIMAX_model import run_sarimax_forecast_with_outputs


def run_sarimax_comparison():
    """
    Runs SARIMAX base and optimized models for specific day types.
    Saves forecast plots, metrics, and a comparison summary CSV.
    """
    BASE_CONFIG = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 48)
    }

    OPTIMIZED_CONFIGS = {
        "Weekday": {"order": (2, 1, 0), "seasonal_order": (0, 0, 0, 48)},
        "Saturday": {"order": (2, 0, 0), "seasonal_order": (1, 1, 0, 48)},
        "Sunday": {"order": (2, 0, 1), "seasonal_order": (1, 1, 0, 48)},
        "CNY": {"order": (1, 0, 1), "seasonal_order": (2, 1, 0, 48)}
    }

    TRAIN_DATES = {
        "Weekday": ('2019-01-01', '2020-01-01'),
        "Saturday": ('2019-01-01', '2020-01-01'),
        "Sunday": ('2019-01-01', '2020-01-01'),
        "CNY": ('2016-01-01', '2020-01-01')
    }

    TEST_START = '2023-01-01'
    
    exog_cols = [
        'NEM Demand (Forecast)',
        'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3',
        'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3'
    ]
    target_col = 'NEM Demand (Actual)'

    df = pd.read_csv("Data/EMA_Demand_Lagged.csv", parse_dates=['Datetime'])
    df = df.set_index('Datetime')

    SCENARIO_MASKS = {
        "Weekday": df[df['TreatAs_DayType'].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])],
        "Saturday": df[df['TreatAs_DayType'] == 'Sat'],
        "Sunday": df[df['TreatAs_DayType'] == 'Sun'],
        "CNY": df[df['TreatAs_DayType'] == 'CNY']
    }

    results = []

    for daytype, sub_df in SCENARIO_MASKS.items():
        train_range = TRAIN_DATES[daytype]

        # Base run
        base_out_dir = f"Output/SARIMAX_Comparison/{daytype}/Base"
        _, _, base_metrics, _ = run_sarimax_forecast_with_outputs(
            df=sub_df.copy(),
            target_col=target_col,
            exog_cols=exog_cols,
            output_dir=base_out_dir,
            train_range=train_range,
            test_start=TEST_START,
            order=BASE_CONFIG['order'],
            seasonal_order=BASE_CONFIG['seasonal_order']
        )
        base_metrics.update({"Scenario": daytype, "Model": "Base"})
        results.append(base_metrics)

        # Optimized run
        opt_conf = OPTIMIZED_CONFIGS[daytype]
        opt_out_dir = f"Output/SARIMAX_Comparison/{daytype}/Optimized"
        _, _, opt_metrics, _ = run_sarimax_forecast_with_outputs(
            df=sub_df.copy(),
            target_col=target_col,
            exog_cols=exog_cols,
            output_dir=opt_out_dir,
            train_range=train_range,
            test_start=TEST_START,
            order=opt_conf['order'],
            seasonal_order=opt_conf['seasonal_order']
        )
        opt_metrics.update({"Scenario": daytype, "Model": "Optimized"})
        results.append(opt_metrics)

    # Save full comparison
    results_df = pd.DataFrame(results)
    results_df.to_csv("Output/SARIMAX_Comparison/sarimax_comparison_results.csv", index=False)
    print("\n✅ SARIMAX base vs optimized comparison complete.")