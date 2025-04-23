import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def run_lstm_shap_analysis():
    os.makedirs('Output/LSTM_SHAP', exist_ok=True)

    # Load and preprocess
    df = pd.read_csv('Data/EMA_Demand_Lagged.csv')
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

    def create_sequences(X, y, time_steps=3):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # Build and train LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64)

    # SHAP KernelExplainer
    X_test_flat = X_test[:100].reshape(100, -1)
    def model_predict(X_flat):
        X_reshaped = X_flat.reshape((X_flat.shape[0], 3, 6))
        return model.predict(X_reshaped).flatten()

    explainer = shap.KernelExplainer(model_predict, X_test_flat[:10])
    shap_values = explainer.shap_values(X_test_flat)

    # Save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test_flat, feature_names=features * 3, show=False)
    plt.savefig('Output/LSTM_SHAP/shap_summary_plot.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    r2 = r2_score(y_test_inverse, y_pred_inverse)
    print(f'MAE: {mae}, RMSE: {rmse}, R²: {r2}')
    pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "R²": [r2]}).to_csv("Output/LSTM_SHAP/lstm_metrics.csv", index=False)

    # Prepare for SHAP comparison by day type
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Day'] = df['Datetime'].dt.day_name()
    df['Day_Type'] = df['Day'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

    def process_day_type(day_type, filename):
        time_steps = 3
        required_sequences = 5
        required_rows = time_steps + required_sequences

        subset = df[df['Day_Type'] == day_type].sample(required_rows)
        X = subset[features].values
        X_scaled = scaler_X.transform(X)

        X_seq, _ = create_sequences(X_scaled, np.zeros((len(X_scaled), 1)))
        X_flat = X_seq.reshape((X_seq.shape[0], -1))

        y_pred = model.predict(X_seq)
        y_pred_inverse = scaler_y.inverse_transform(y_pred)

        shap_output = explainer.shap_values(X_flat)
        shap_vals = shap_output[0] if isinstance(shap_output, list) else shap_output
        shap_sum = np.sum(shap_vals, axis=1)

        result = subset.iloc[-len(y_pred):].copy()
        result['Predicted'] = y_pred_inverse.flatten()
        result['SHAP_sum'] = shap_sum

        result[['Datetime', 'Predicted', 'SHAP_sum']].to_csv(f'Output/LSTM_SHAP/{filename}', index=False)
        print(f"[{day_type}] ✅ Saved to Output/LSTM_SHAP/{filename}")

    # Run for weekday and weekend
    process_day_type('Weekday', 'weekdays_predictions.csv')
    process_day_type('Weekend', 'weekends_predictions.csv')

    # Summary Table 1
    weekdays_df = pd.read_csv('Output/LSTM_SHAP/weekdays_predictions.csv')
    weekends_df = pd.read_csv('Output/LSTM_SHAP/weekends_predictions.csv')

    weekdays_mean_shap = weekdays_df['SHAP_sum'].mean()
    weekends_mean_shap = weekends_df['SHAP_sum'].mean()
    weekdays_mean_pred = weekdays_df['Predicted'].mean()
    weekends_mean_pred = weekends_df['Predicted'].mean()

    print("\n📊 Table 1 Summary")
    print(f"Weekdays Mean SHAP: {weekdays_mean_shap:.4f}")
    print(f"Weekends Mean SHAP: {weekends_mean_shap:.4f}")
    print(f"Weekdays Mean Prediction: {weekdays_mean_pred:.2f}")
    print(f"Weekends Mean Prediction: {weekends_mean_pred:.2f}")

if __name__ == "__main__":
    run_lstm_shap_analysis()
