# === main.py ===

import pandas as pd
import os
from Model.XGBoost import (
    add_time_features,
    filter_periods,
    evaluate_period,
    evaluate_combined_model,
    plot_model_comparison
)

# === CONFIGURATION ===
DATASETS = {
    "Scenario1": "Data/Combined_Demand_Data.xlsx",
    "Scenario2": "Data/EMA_Demand Data (2015-2025).xlsx"
}

# === FEATURE SETUP ===
features = ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code']
target = 'NEM Demand (Actual)'

# === PROCESS EACH SCENARIO ===
for scenario_label, filepath in DATASETS.items():
    input_path = os.path.abspath(filepath)
    output_dir = f"Output/{scenario_label}"
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess
    df = pd.read_excel(input_path).drop(index=0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = add_time_features(df)

    # Filter by time periods
    df_covid, df_cny, df_typical = filter_periods(df)
    df_combined = pd.concat([df_covid, df_cny, df_typical], axis=0)

    # Evaluate each period
    results_covid = evaluate_period(df_covid, features, target, f"{scenario_label}_COVID")
    results_cny = evaluate_period(df_cny, features, target, f"{scenario_label}_CNY")
    results_typical = evaluate_period(df_typical, features, target, f"{scenario_label}_Typical_Day")
    results_combined = evaluate_combined_model(df_combined, features, target, f"{scenario_label}_Combined")

    # Combine and save results
    period_results = pd.concat([results_covid, results_cny, results_typical])
    all_results = pd.concat([period_results, results_combined])
    
    # Save results
    period_results.to_csv(f"{output_dir}/xgboost_demand_evaluation_results.csv", index=True)
    all_results.to_csv(f"{output_dir}/xgboost_all_periods_evaluation_summary.csv", index=True)

    # Plot performance comparison
    plot_model_comparison(all_results, output_path=f"{output_dir}/comparison_chart.png")

    print(f"✅ {scenario_label} completed. Results saved to: {output_dir}/")