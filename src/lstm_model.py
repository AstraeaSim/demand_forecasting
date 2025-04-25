import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# === Sequence Creation ===
def regenerate_demand_lagged_csv():
    import pandas as pd
    import os

    print("🔄 Regenerating EMA_Demand_Lagged.csv...")

    demand = pd.read_excel("Data/EMA_Demand Data (2015-2025).xlsx")
    holiday = pd.read_excel("Data/sg_holiday_cny_covid_2015_2025.xlsx")

    demand['Date'] = pd.to_datetime(demand['Date'])
    holiday['Date'] = pd.to_datetime(holiday['Date'])

    treat_as_map = dict(zip(holiday['Date'], holiday['Treat As']))
    holiday_name_map = dict(zip(holiday['Date'], holiday['Holiday Name']))

    covid_start = pd.to_datetime("2020-04-07")
    covid_end = pd.to_datetime("2021-08-10")

    def classify_daytype(row):
        date = row['Date']
        if covid_start <= date <= covid_end:
            return 'COVID'
        elif date in treat_as_map:
            return treat_as_map[date]
        elif row['Day'] == 'Sun':
            return 'Sun'
        elif row['Day'] == 'Sat':
            return 'Sat'
        else:
            return row['Day']

    demand['TreatAs_DayType'] = demand.apply(classify_daytype, axis=1)
    demand['Holiday Name'] = demand['Date'].apply(lambda x: holiday_name_map.get(x, 'Normal Day'))
    demand['Datetime'] = pd.to_datetime(demand['Date'].astype(str) + ' ' + demand['Period Ending Time'])

    for col in ['NEM Demand (Actual)', 'NEM Demand (Forecast)']:
        for lag in range(1, 4):
            demand[f'{col}_lag{lag}'] = demand[col].shift(lag)

    demand_lagged = demand.dropna()
    os.makedirs("Data", exist_ok=True)
    demand_lagged.to_csv("Data/demand_feature_dataset_LSTM.csv", index=False)

    print("✅ File saved: Data/demand_feature_dataset_LSTM.csv")



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


