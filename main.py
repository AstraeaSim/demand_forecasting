# === main.py ===

import pandas as pd
from Model.XGBoost import add_time_features, filter_periods, evaluate_period
from Model.XGBoost import evaluate_combined_model, plot_model_comparison
import os 
input_path = os.path.abspath("Data/Combined_Demand_Data.xlsx")


# Load cleaned dataset
df = pd.read_excel(input_path)

# Drop metadata row and process timestamps
df = df.drop(index=0)
df['Date'] = pd.to_datetime(df['Date'])  # Ensure datetime for date column
df = add_time_features(df)

# Define features and target
features = ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code']
target = 'NEM Demand (Actual)'

# Filter periods
df_covid, df_cny, df_typical = filter_periods(df)

# Evaluate each period
results_covid = evaluate_period(df_covid, features, target, "COVID")
results_cny = evaluate_period(df_cny, features, target, "CNY")
results_typical = evaluate_period(df_typical, features, target, "Typical Day")

# Combine and show results
combined_results = pd.concat([results_covid, results_cny, results_typical])
# print("Model Evaluation:\n")
# print(combined_results)

combined_results.to_csv("Output/xgboost_demand_evaluation_results.csv", index=True)
print("Results saved to 'xgboost_demand_evaluation_results.csv'")


# Evaluate combined model (COVID + CNY + Typical Day samples)
df_combined = pd.concat([df_covid, df_cny, df_typical], axis=0)
results_combined = evaluate_combined_model(df_combined, features, target, "Combined")

# Combine all evaluation results into a single DataFrame
final_results = pd.concat([
    results_covid,
    results_cny,
    results_typical,
    results_combined
])

# Print and save to CSV
#print("Full Model Evaluation Summary:\n")
#print(final_results)
final_results.to_csv("Output/xgboost_all_periods_evaluation_summary.csv", index=True)
# Plot
plot_model_comparison(final_results)