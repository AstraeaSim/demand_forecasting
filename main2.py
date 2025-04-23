#from src.time_series_analysis import run_sarimax_forecast
from src.lstm_model import run_lstm_shap_analysis
from src.eda_demand_forecasting import run_eda_pipeline
from src.models import (
    add_time_features,
    filter_periods,
    evaluate_period,
    evaluate_combined_model,
    plot_model_comparison
)

import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("Output/Charts", exist_ok=True)
    os.makedirs("Output/Configuration", exist_ok=True)

    print("🚀 Running EDA pipeline...")
    run_eda_pipeline()

    # print("🚀 Running SARIMAX forecasting pipeline...")
    # run_sarimax_forecast()

    print("🧠 Running LSTM + SHAP interpretability pipeline...")
    run_lstm_shap_analysis()

    # === CONFIGURATION ===
    DATASETS = {
        "Scenario 1": "Data/Combined_Demand_Data.xlsx",
        "Scenario 2": "Data/EMA_Demand Data (2015-2025).xlsx",
        "Scenario 3": "Data/EMA_Demand_Lagged.csv"
    }

    FEATURE_SETS = {
        "Scenario 1": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
        "Scenario 2": ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'],
        "Scenario 3": [
            'NEM Demand (Forecast)',
            'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3',
            'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3',
            'Hour', 'DayOfWeek', 'TreatAs_DayType_Code'
        ]
    }
    target = 'NEM Demand (Actual)'
    MODEL_TYPES = ["xgboost", "random_forest", "lightgbm", "catboost"]

    # Save configuration to file
    config_path = os.path.join("Output/Configuration", "model_config.txt")
    with open(config_path, "w") as f:
        f.write("Model Configuration:\n")
        f.write(f"Target: {target}\n")
        f.write(f"Model Types: {MODEL_TYPES}\n")
        for scenario, features in FEATURE_SETS.items():
            f.write(f"\n{scenario} Features: {features}\n")
        f.write(f"\nDatasets:\n")
        for scenario, path in DATASETS.items():
            f.write(f"{scenario}: {path}\n")
    print(f"📝 Configuration saved to {config_path}")

    final_results = []

    for model_type in MODEL_TYPES:
        for scenario_label, filepath in DATASETS.items():
            input_path = os.path.abspath(filepath)
            output_dir = f"Output/{scenario_label.replace(' ', '_')}"
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n🚀 Processing {scenario_label} with {model_type}...")

            if filepath.endswith(".xlsx"):
                df = pd.read_excel(input_path).drop(index=0)
            else:
                df = pd.read_csv(input_path)

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Period Ending Time'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M", errors='coerce')
            df = add_time_features(df)
            df_covid, df_cny, df_typical = filter_periods(df, frac=0.2)
            df_combined = pd.concat([df_covid, df_cny, df_typical], axis=0)

            features = FEATURE_SETS[scenario_label]

            results_covid = evaluate_period(df_covid, features, target, f"{scenario_label}_COVID", model_type=model_type)
            results_cny = evaluate_period(df_cny, features, target, f"{scenario_label}_CNY", model_type=model_type)
            results_typical = evaluate_period(df_typical, features, target, f"{scenario_label}_Typical_Day", model_type=model_type)
            results_combined = evaluate_combined_model(df_combined, features, target, f"{scenario_label}_Combined", model_type=model_type)

            period_results = pd.concat([results_covid, results_cny, results_typical])
            all_results = pd.concat([period_results, results_combined])
            all_results['Model_Type'] = model_type

            period_results.to_csv(f"{output_dir}/{model_type}_demand_evaluation_results.csv", index=True)
            all_results.to_csv(f"{output_dir}/{model_type}_all_periods_evaluation_summary.csv", index=True)

            plot_model_comparison(all_results, model_type=model_type, output_path=f"{output_dir}/comparison_chart_{model_type}.png")

            final_results.append(all_results)

    print("\n📊 Combining Combined results from all scenarios and models...")
    if final_results:
        combined_df = pd.concat(final_results, axis=0).reset_index()
        combined_df.to_csv("Output/final_combined_all_models_results.csv", index=False)

        is_combined = combined_df['index'].str.contains("Combined", case=False)
        summary_df = combined_df[is_combined].copy()

        def parse_scenario_model(index_str):
            parts = index_str.strip().split('-')
            if len(parts) == 2:
                scenario_part = parts[0].strip()
                model_part = parts[1].strip().lower()
                match = re.match(r"(Scenario\s*\d+)", scenario_part, flags=re.IGNORECASE)
                if match:
                    scenario = match.group(1).title()
                    return scenario, model_part
            return "Unknown", "Unknown"

        summary_df[['Scenario', 'Model']] = summary_df['index'].apply(lambda x: pd.Series(parse_scenario_model(x)))
        summary_df = summary_df[['Scenario', 'Model', 'MAE', 'RMSE', 'R²']]
        summary_df = summary_df.sort_values(by=['Scenario', 'Model'])

        summary_path = "Output/final_model_performance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Final model performance summary saved to: {summary_path}")

        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            charts_dir = os.path.join("Output", "Charts")
            os.makedirs(charts_dir, exist_ok=True)

            metrics = ["MAE", "RMSE", "R²"]
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x="Scenario", y=metric, hue="Model")
                plt.title(f"{metric} Comparison Across Models and Scenarios", fontsize=14)
                plt.ylabel(metric)
                plt.xlabel("Scenario")
                plt.legend(title="Model")
                plt.tight_layout()
                chart_path = os.path.join(charts_dir, f"{metric.lower()}_model_comparison.png")
                plt.savefig(chart_path, dpi=300)
                plt.close()
                print(f"📊 Chart saved: {chart_path}")
    else:
        print("❌ No results found to combine.")

if __name__ == "__main__":
    main()