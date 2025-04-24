import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# === Data Preprocessing ===
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M").dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayType_Hour'] = df['TreatAs_DayType_Code'] * df['Hour']
    return df

# === Model Training and Evaluation ===
def run_model(df, condition, label, output_dir, results_list):
    if condition != "True":
        filtered_df = df.query(condition).sample(frac=0.2, random_state=42)
    else:
        filtered_df = df.sample(frac=0.2, random_state=42)

    features = ['NEM Demand (Forecast)', 'Hour', 'DayOfWeek', 'TreatAs_DayType_Code', 'DayType_Hour']
    target = 'NEM Demand (Actual)'

    X = filtered_df[features]
    y = filtered_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"--- {label} ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.5f}")

    # Append results to list
    results_list.append({
        "Label": label,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 5)
    })

    # Save scatter plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual NEM Demand")
    plt.ylabel("Predicted NEM Demand")
    plt.title(f"{label} - Linear Regression + Interaction")
    plt.grid(True)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{label.lower().replace(' ', '_')}_scatter.png")
    plt.savefig(fig_path)
    plt.close()

# === Main Script ===
def run_linear_regression_analysis():
    df = pd.read_csv("Output/EDA/demand_feature_dataset.csv")
    df = preprocess_data(df)
    os.makedirs("images", exist_ok=True)

    results = []

    run_model(df, "Year in [2020, 2021]", "COVID", "images", results)

    cny_mask = ((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | \
               ((df['Month'] == 2) & (df['Date'].dt.day <= 10))
    run_model(df[cny_mask], "True", "CNY", "images", results)

    run_model(df, "Year == 2019 and TreatAs_DayType_Code == 0", "Typical 2019", "images", results)

    combined_mask = (
        (df['Year'].isin([2020, 2021])) |
        (((df['Month'] == 1) & (df['Date'].dt.day >= 20)) | ((df['Month'] == 2) & (df['Date'].dt.day <= 10))) |
        ((df['Year'] == 2019) & (df['TreatAs_DayType_Code'] == 0))
    )
    run_model(df[combined_mask], "True", "Combined", "images", results)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("Output/Linear", exist_ok=True)
    results_df.to_csv("Output/Linear/linear_regression_metrics.csv", index=False)
    print("📁 Linear regression metrics saved to images/linear_regression_metrics.csv")

if __name__ == "__main__":
    run_linear_regression_analysis()