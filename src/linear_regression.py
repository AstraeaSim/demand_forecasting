import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = pd.to_datetime(df['Period Ending Time'], format="%H:%M").dt.hour
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['TreatAs_DayType_Code'] = np.where(df['DayOfWeek'] >= 5, 1, 0)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayType_Hour'] = df['TreatAs_DayType_Code'] * df['Hour']
    return df

def run_model(df, condition, label, output_dir):
    filtered_df = df.query(condition).sample(frac=0.2, random_state=42)
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

def main():
    df = pd.read_csv("data/demand_feature_dataset.csv")
    df = preprocess_data(df)
    os.makedirs("reports/figures", exist_ok=True)

    run_model(df, "Year in [2020, 2021]", "COVID", "images")
    run_model(df, "(Month == 1 and Date.dt.day >= 20) or (Month == 2 and Date.dt.day <= 10)", "CNY", "images")
    run_model(df, "Year == 2019 and TreatAs_DayType_Code == 0", "Typical 2019", "images")
    run_model(df, "Year in [2020, 2021] or ((Month == 1 and Date.dt.day >= 20) or (Month == 2 and Date.dt.day <= 10)) or (Year == 2019 and TreatAs_DayType_Code == 0)", "Combined", "images")

if __name__ == "__main__":
    main()
