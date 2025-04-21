# === main.py ===

import pandas as pd
import os
from src.XGBoost import (
    add_time_features,
    filter_periods,
    evaluate_period,
    evaluate_combined_model,
    plot_model_comparison
)

# === CONFIGURATION ===
DATASETS = {
    "XGBoost Scenario 1": "Data/Combined_Demand_Data.xlsx",
    "XGBoost Scenario 2": "Data/EMA_Demand Data (2015-2025).xlsx",
    "XGBoost Scenario 3": "Data/EMA_Demand_Lagged.csv"
}

# === FEATURE SETUP PER SCENARIO ===
FEATURE_SETS = {
    "XGBoost Scenario 1": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
    "XGBoost Scenario 2": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
    "XGBoost Scenario 3": [
        'NEM Demand (Forecast)',
        'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3',
        'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3',
        'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'
    ]
}
target = 'NEM Demand (Actual)'

# === PROCESS EACH SCENARIO ===
for scenario_label, filepath in DATASETS.items():
    input_path = os.path.abspath(filepath)
    output_dir = f"Output/{scenario_label.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Processing {scenario_label}...")

    # Load data
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(input_path).drop(index=0)
    else:
        df = pd.read_csv(input_path)

    # Parse datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Period Ending Time'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M", errors='coerce')

    # Add features
    df = add_time_features(df)

    # Filter periods
    df_covid, df_cny, df_typical = filter_periods(df, frac=0.2)
    df_combined = pd.concat([df_covid, df_cny, df_typical], axis=0)

    # Get feature list for this scenario
    features = FEATURE_SETS[scenario_label]

    # Evaluate periods
    results_covid = evaluate_period(df_covid, features, target, f"{scenario_label}_COVID")
    results_cny = evaluate_period(df_cny, features, target, f"{scenario_label}_CNY")
    results_typical = evaluate_period(df_typical, features, target, f"{scenario_label}_Typical_Day")
    results_combined = evaluate_combined_model(df_combined, features, target, f"{scenario_label}_Combined")

    # Combine and save results
    period_results = pd.concat([results_covid, results_cny, results_typical])
    all_results = pd.concat([period_results, results_combined])

    period_results.to_csv(f"{output_dir}/xgboost_demand_evaluation_results.csv", index=True)
    all_results.to_csv(f"{output_dir}/xgboost_all_periods_evaluation_summary.csv", index=True)

    # Plot comparison
    plot_model_comparison(all_results, output_path=f"{output_dir}/comparison_chart.png")

    print(f"✅ {scenario_label} completed. Results saved to: {output_dir}/")
    
# === FINAL STEP: COMBINE 'Combined-XGBoost' RESULTS FROM ALL SCENARIOS ===
def combine_combined_xgboost_results():
    print("\n📊 Combining Combined-XGBoost results from all scenarios...")

    scenario_files = {
        "Scenario 1": "Output/XGBoost_Scenario_1/xgboost_all_periods_evaluation_summary.csv",
        "Scenario 2": "Output/XGBoost_Scenario_2/xgboost_all_periods_evaluation_summary.csv",
        "Scenario 3": "Output/XGBoost_Scenario_3/xgboost_all_periods_evaluation_summary.csv"
    }

    combined_rows = []

    for scenario, path in scenario_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df_combined = df[df.index.str.contains("_Combined-XGBoost")].copy()
            df_combined.insert(0, "Scenario", scenario)
            df_combined.insert(1, "Label", df_combined.index)
            combined_rows.append(df_combined.reset_index(drop=True))
        else:
            print(f"⚠️ File not found: {path}")

    if combined_rows:
        final_df = pd.concat(combined_rows, axis=0).reset_index(drop=True)
        os.makedirs("Output", exist_ok=True)
        output_path = "Output/final_combined_xgboost_results.csv"
        final_df.to_csv(output_path, index=False)
        print(f"✅ Combined results saved to: {output_path}")
    else:
        print("❌ No combined results found.")

# === Run the final combination step ===
combine_combined_xgboost_results()