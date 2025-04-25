import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# === Sequence Creation ===
def create_sequences(X, y, time_steps=3):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# === LSTM Training and Evaluation ===
def train_and_evaluate_lstm(df, features, target, label, time_steps=3, epochs=50, batch_size=32):
    df_clean = df[features + [target]].dropna()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df_clean[features])
    y_scaled = scaler_y.fit_transform(df_clean[[target]])

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)

    print(f"{label} MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return {"Label": label, "MAE": mae, "RMSE": rmse, "R2": r2}

# === Main Runner for All Periods ===
def run_lstm_analysis_all_periods(demand_lagged):
    results = []
    periods = {
        "Weekdays": demand_lagged[demand_lagged['TreatAs_DayType'].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])],
        "Saturday": demand_lagged[demand_lagged['TreatAs_DayType'] == 'Sat'],
        "Sunday": demand_lagged[demand_lagged['TreatAs_DayType'] == 'Sun'],
        "CNY": demand_lagged[demand_lagged['TreatAs_DayType'] == 'CNY'],
        "COVID": demand_lagged[demand_lagged['TreatAs_DayType'] == 'COVID']
    }

    features = [
        'NEM Demand (Forecast)', 
        'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3', 
        'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3'
    ]
    target = "NEM Demand (Actual)"

    for label, df in periods.items():
        result = train_and_evaluate_lstm(df, features, target, label)
        results.append(result)

    return pd.DataFrame(results)

# === SHAP Explanation ===
def run_lstm_with_shap_interpretation(input_path="Data/EMA_Demand_Lagged.csv", output_dir="Output/LSTM/SHAP"):
    print("🧠 Running LSTM with SHAP interpretation...")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    features = [
        'NEM Demand (Actual)_lag1', 'NEM Demand (Actual)_lag2', 'NEM Demand (Actual)_lag3',
        'NEM Demand (Forecast)_lag1', 'NEM Demand (Forecast)_lag2', 'NEM Demand (Forecast)_lag3'
    ]
    target = 'NEM Demand (Actual)'

    df_clean = df[features + [target]].dropna()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df_clean[features])
    y_scaled = scaler_y.fit_transform(df_clean[[target]])

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, verbose=0)

    X_test_flat = X_test[:100].reshape(100, -1)

    def model_predict(X_flat):
        return model.predict(X_flat.reshape((X_flat.shape[0], 3, 6))).flatten()

    shap.initjs()
    explainer = shap.KernelExplainer(model_predict, X_test_flat[:10])
    shap_values = explainer.shap_values(X_test_flat)

    # Save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_flat, feature_names=features * 3, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches='tight')
    plt.close()

    # Save interactive SHAP force plot
    force_plot = shap.force_plot(
        explainer.expected_value, shap_values[0], X_test_flat[0], feature_names=features * 3
    )
    shap.save_html(f"{output_dir}/shap_force_plot_sample.html", force_plot)

    print("✅ SHAP outputs saved in:", output_dir)
