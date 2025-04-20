## Course: DSS5105 — Data Science Projects in Practice  
# Semester: AY2024/25 S2


# 📊 XGBoost Demand Forecasting — Scenario Comparison

This project evaluates electricity demand forecasting performance across multiple calendar periods (e.g., COVID, Chinese New Year, Typical Days) 

Two scenarios are explored with separate datasets:
- **Scenario 1** → Combined_Demand_Data.xlsx
- **Scenario 2** → EMA_Demand Data (2015-2025).xlsx

---

## 🚀 How It Works

1. **Feature Engineering**:
   - Extracts `Hour`, `DayOfWeek`, `TreatAs_DayType_Code`, `Year`, and `Month` from raw timestamps.

2. **Period Segmentation**:
   - Splits data into:
     - 🦠 COVID Period (2020–2021)
     - 🧧 Chinese New Year (Late Jan – Early Feb)
     - 📅 Typical Weekdays (2019 non-weekends)

3. **Model Training**:
   - Trains XGBoost Regressors with fixed hyperparameters on each segment and the combined sample.

4. **Evaluation**:
   - Computes MAE, RMSE, and R².
   - Generates scatter plots and comparison bar charts.
   - Saves trained models as `.pkl` files.

---

## 📁 Project Structure

```
Output/
├── XGBoost_Scenario_1/
│   ├── xgboost_demand_evaluation_results.csv
│   ├── xgboost_all_periods_evaluation_summary.csv
│   ├── comparison_chart.png
│   ├── models/
│   └── images/
├── XGBoost_Scenario_2/
│   └── same as above
```

---

## 🧠 Scenarios

| Scenario        | Input File                          | Description                            |
|----------------|--------------------------------------|----------------------------------------|
| Scenario 1     | `Combined_Demand_Data.xlsx`          | Original demand dataset                |
| Scenario 2     | `EMA_Demand Data (2015-2025).xlsx`   | Alternative simulation or update       |

---

## 📦 How to Run

```bash
# Step into project root
cd your_project_directory/

# Run main script
python main.py
```

All results will be auto-generated in the `Output/` folder.

---

## 🛠 Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---





