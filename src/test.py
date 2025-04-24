import pandas as pd

# Load lagged dataset set
df_lagged = pd.read_csv("load data/EMA_Demand_Lagged.csv", parse_dates=['Datetime'], index_col='Datetime')
df_lagged = df_lagged.reset_index()  # This makes 'Datetime' a regular column
# Load feature dataset set
df_feature = pd.read_csv("load data/demand_feature_dataset.csv")

# Convert to datetime key
df_feature['Datetime'] = pd.to_datetime(df_feature['Date'].astype(str) + ' ' + df_feature['Period Ending Time'])
# Merge on 'Datetime'
merged_df = pd.merge(df_lagged, df_feature, on='Datetime', how='outer')
merged_df.to_csv("SARIMAX_merged.csv")
#Drop column Date_y, Day_y, Period Ending Time_y, System Demand (Actual)_y, NEM Demand (Actual)_y, NEM Demand (Forecast)_y
merged_df = merged_df.drop(columns=['Date_y', 'Day_y', 'Period Ending Time_y', 'System Demand (Actual)_y', 'NEM Demand (Actual)_y', 'NEM Demand (Forecast)_y'])
merged_df.rename(columns={'Date_x': 'Date', 'Day_x': 'Day', 'Period Ending Time_x': 'Period Ending Time', 'System Demand (Actual)_x':'System Demand (Actual)',
                          'NEM Demand (Actual)_x':'NEM Demand (Actual)', 'NEM Demand (Forecast)_x':'NEM Demand (Forecast)'}, inplace=True)
merged_df.isnull().sum().sort_values(ascending=False)
merged_df = merged_df.dropna() #Drop rows with null values
merged_df.isnull().sum().sort_values(ascending=False) #reset index
merged_df.head()
# Convert boolean columns to float
merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']] = merged_df[['IsWeekend', 'IsHoliday', 'IsCOVID']].astype(int)
merged_df.head()
y = merged_df['NEM Demand (Actual)']
X = merged_df[['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]] 

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

merged_df.index = pd.to_datetime(merged_df.index) # Set index if not already
merged_df = merged_df.sort_index() # Sort index
merged_df = merged_df.set_index('Datetime')  # replace the incorrect numeric index
train = merged_df.loc['2019-01-01':'2020-01-01']
test = merged_df.loc['2023-01-01':]

target_col = 'NEM Demand (Actual)'
exog_cols = ['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]
y_train = train[target_col]
X_train = train[exog_cols]
y_test = test[target_col]
X_test = test[exog_cols]
model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),              # You can tune this
    seasonal_order=(1, 1, 1, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
#print(results.summary())
# Assume y_train and X_train have been used to fit SARIMAX
n_test = len(X_test)

# Forecast using exact number of steps
pred = results.predict(
    start=len(X_train),
    end=len(X_train) + n_test - 1,
    exog=X_test
)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(pred.index, pred, label='Predicted', linestyle='--')
plt.title('SARIMAX Forecast vs Actual (2023)')
plt.xlabel('Date')
plt.ylabel('System Demand (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Split by daytype
weekday_df = merged_df[merged_df['TreatAs_DayType'].isin(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])]
weekend_df = merged_df[merged_df['TreatAs_DayType'].isin(['Sat'])]
sunday_df = merged_df[merged_df['TreatAs_DayType'] == 'Sun']
cny_df = merged_df[merged_df['TreatAs_DayType'] == 'CNY']

y = weekday_df['NEM Demand (Actual)']
X = weekday_df[['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]] 

weekday_df = weekday_df.sort_index()
weekday_df.head()

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

weekday_df.index = pd.to_datetime(weekday_df.index) # Set index if not already
weekday_df = weekday_df.sort_index() # Sort index

train = weekday_df.loc['2019-01-01':'2020-01-01']
test = weekday_df.loc['2023-01-01':]

target_col = 'NEM Demand (Actual)'
exog_cols = ['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]
y_train = train[target_col]
X_train = train[exog_cols]
y_test = test[target_col]
X_test = test[exog_cols]

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),              # You can tune this
    seasonal_order=(1, 1, 1, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
#print(results.summary())
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
from pmdarima import auto_arima

stepwise_model = auto_arima(
    y_train,
    seasonal=False,  # 🔥 turn off seasonal completely
    d=None,
    start_p=0, start_q=0,
    max_p=2, max_q=2,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True
)

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(2, 1, 0),              # You can tune this
    seasonal_order=(0, 0, 0, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
#print(results.summary())
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

y = weekend_df['NEM Demand (Actual)']
X = weekend_df[['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]] 

weekend_df = weekend_df.sort_index()
weekend_df.head()

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

weekend_df.index = pd.to_datetime(weekend_df.index) # Set index if not already
weekend_df = weekend_df.sort_index() # Sort index

train = weekend_df.loc['2019-01-01':'2020-01-01']
test = weekend_df.loc['2023-01-01':]

target_col = 'NEM Demand (Actual)'
exog_cols = ['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]
y_train = train[target_col]
X_train = train[exog_cols]
y_test = test[target_col]
X_test = test[exog_cols]

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),              # You can tune this
    seasonal_order=(1, 1, 1, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
#print(results.summary())
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
from pmdarima import auto_arima

# Fit auto_arima on your training target with exogenous features
stepwise_model = auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=2, max_q=2,
    start_P=0, max_P=1,
    start_Q=0, max_Q=0,
    m=48,
    seasonal=True,
    d=None, D=1,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True,
    n_fits=15
)

#print(stepwise_model.summary())

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(2, 0, 0),              # You can tune this
    seasonal_order=(1, 1, 0, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
#print(results.summary())

pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


y = sunday_df['NEM Demand (Actual)']
X = sunday_df[['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]] 

sunday_df = sunday_df.sort_index()
sunday_df.head()

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

sunday_df.index = pd.to_datetime(sunday_df.index) # Set index if not already
sunday_df = sunday_df.sort_index() # Sort index

train = sunday_df.loc['2019-01-01':'2020-01-01']
test = sunday_df.loc['2023-01-01':]

target_col = 'NEM Demand (Actual)'
exog_cols = ['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]
y_train = train[target_col]
X_train = train[exog_cols]
y_test = test[target_col]
X_test = test[exog_cols]

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),              # You can tune this
    seasonal_order=(1, 1, 1, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
#print(results.summary())
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

from pmdarima import auto_arima

# Fit auto_arima on your training target with exogenous features
stepwise_model = auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=2, max_q=2,
    start_P=0, max_P=1,
    start_Q=0, max_Q=0,
    m=48,
    seasonal=True,
    d=None, D=1,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True,
    n_fits=15
)


model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(2, 0, 1),              # You can tune this
    seasonal_order=(1, 1, 0, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

y = cny_df['NEM Demand (Actual)']
X = cny_df[['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3']] 

cny_df = cny_df.sort_index()

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

cny_df.index = pd.to_datetime(cny_df.index) # Set index if not already
cny_df = cny_df.sort_index() # Sort index

train = cny_df.loc['2016-01-01':'2020-01-01']
test = cny_df.loc['2023-01-01':]

target_col = 'NEM Demand (Actual)'
exog_cols = ['NEM Demand (Forecast)',
    'NEM Demand (Actual)_lag1',
    'NEM Demand (Actual)_lag2',
    'NEM Demand (Actual)_lag3'
]
y_train = train[target_col]
X_train = train[exog_cols]
y_test = test[target_col]
X_test = test[exog_cols]
model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 1, 1),              # You can tune this
    seasonal_order=(1, 1, 1, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
pred = results.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, exog=X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

from pmdarima import auto_arima

# Fit auto_arima on your training target with exogenous features
stepwise_model = auto_arima(
    y_train,
    exogenous=X_train,
    start_p=1, start_q=1,
    max_p=3, max_q=3,
    seasonal=True,
    m=48,  # 48 = seasonality of 1 day (30min intervals)
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    d=None, D=1,  # Let it determine 'd'; use seasonal differencing (D=1)
    trace=True,  # Print the steps
    error_action='ignore',  
    suppress_warnings=True,
    stepwise=True
)
#print(stepwise_model.summary())

model = SARIMAX(
    endog=y_train,
    exog=X_train,
    order=(1, 0, 1),              # You can tune this
    seasonal_order=(2, 1, 0, 48), # 48 = 1 day of 30-min intervals
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
#print(results.summary())

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")