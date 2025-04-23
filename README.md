## Course: DSS5105 — Data Science Projects in Practice  
# Semester: AY2024/25 S2

# Demand Forecasting and SHAP Analysis Project

This repository contains the full pipeline for analyzing and forecasting Singapore's electricity demand using machine learning models, SARIMAX, and LSTM-based approaches with SHAP interpretability.

## 📁 Project Structure

```
├── Data/
│   ├── EMA_Demand Data (2015-2025).xlsx
│   ├── EMA_Demand_Lagged.csv
│   └── sg_holiday_cny_covid_2015_2025.xlsx
├── Output/
│   ├── Charts/
│   ├── LSTM_SHAP/
│   ├── SARIMAX/
│   ├── Configuration/
│   ├── Scenario_1/, Scenario_2/, Scenario_3/
├── src/
│   ├── eda_demand_forecasting.py
│   ├── lstm_model.py
│   ├── models.py
│   └── time_series_analysis.py
├── main.py
└── Insight_Summary_Report.docx
```

## 🚀 How to Run

1. Ensure Python 3.8+ is installed.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Place input data in `Data/` folder.
4. Run the main pipeline:
   ```
   python main.py
   ```

## 📊 Features

- **EDA and Feature Engineering**: Generates demand day-type profiles, lagged features, and visualizations.
- **Machine Learning Forecasting**: Supports XGBoost, LightGBM, CatBoost, and Random Forest across 3 scenario types.
- **SARIMAX Modeling**: Classical time series modeling for comparison.
- **LSTM + SHAP**: Neural network model with SHAP-based interpretability to evaluate forecast confidence.
- **Insight Report**: Includes summary tables and recommendations in `Insight_Summary_Report.docx`.

## 📈 Output

All charts, metrics, and model predictions are saved in the `Output/` directory and organized by model or use case.

## 📌 Contributors
- Person 1: Data Preprocessing
- Person 2: ML Model Development
- Person 3: SHAP + LSTM
- Person 4: Evaluation Pipeline & Charts
- Person 5: Insight Synthesis & Reporting

## ✅ Recommendation Summary

See `Insight_Summary_Report.docx` for detailed insights, model performance, and operational forecasting recommendations.






