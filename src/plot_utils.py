import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Box Plot of Absolute Errors ===
def plot_boxplot_absolute_errors(df_combined, output_dir="Output/Charts"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Set2", df_combined['Model'].nunique())
    sns.boxplot(data=df_combined, x="Model", y="Absolute_Error", palette=palette)
    plt.title("Box Plot of Absolute Prediction Errors Across Models")
    plt.ylabel("Absolute Error")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_absolute_errors.png"))
    plt.close()

# === 2. Residual Plot ===
def plot_residuals_vs_predicted(df_combined, output_dir="Output/Charts"):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_combined, x="Predicted", y="Residual", hue="Model", alpha=0.4)
    plt.axhline(0, linestyle="--", color="red", linewidth=1.2)
    plt.title("Residual Plot: Actual - Predicted vs Predicted Values")
    plt.ylabel("Residual")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_vs_predicted.png"))
    plt.close()

# === 3. Model Performance Comparison ===
def plot_model_performance_bar_chart(summary_df, output_dir="Output/Charts"):
    metrics = ["MAE", "RMSE", "R²"]
    os.makedirs(output_dir, exist_ok=True)
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary_df, x="Scenario", y=metric, hue="Model")
        plt.title(f"{metric} Comparison Across Models and Scenarios", fontsize=14)
        plt.ylabel(metric)
        plt.xlabel("Scenario")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric.lower()}_model_comparison.png"), dpi=300)
        plt.close()

# === NEW: Combined Metrics Plot by Model ===
def plot_model_metrics_summary(summary_df, output_dir="Output/Charts"):
    df_melted = summary_df.melt(id_vars=["Scenario", "Model"], value_vars=["MAE", "RMSE", "R²"], 
                                var_name="Metric", value_name="Value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric")
    plt.title("Model-Wise Comparison of MAE, RMSE, and R²")
    plt.ylabel("Metric Value")
    plt.xlabel("Model")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_metrics_comparison_summary.png"))
    plt.close()

# === 4. Feature Importance Plots ===
def plot_feature_importances(models, input_dir="Output", output_dir="Output/Charts"):
    os.makedirs(output_dir, exist_ok=True)
    for model in models:
        fi_path = os.path.join(input_dir, model, f"{model}_feature_importance.csv")
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path, index_col=0).squeeze()
            plt.figure(figsize=(8, 5))
            colors = sns.color_palette("viridis", n_colors=min(10, len(fi)))
            fi.head(10).plot(kind='bar', color=colors)
            plt.title(f"Top 10 Feature Importances - {model.title()}")
            plt.ylabel("Importance Score")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model}_feature_importance.png"))
            plt.close()

# === 5. Forecast Deviation on Special Days (Simulated Index Based) ===
def plot_forecast_deviation(df_combined, output_dir="Output/Charts"):
    df_special = df_combined.copy()
    df_special['Period'] = ['COVID'] * 100 + ['CNY'] * 100 + ['Typical'] * (len(df_special) - 200)
    for period in ['COVID', 'CNY']:
        period_df = df_special[df_special['Period'] == period]
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=period_df, x=period_df.index, y='Actual', label='Actual', linewidth=2)
        sns.lineplot(data=period_df, x=period_df.index, y='Predicted', label='Predicted', linestyle='--')
        plt.title(f"Forecast Deviation During {period}")
        plt.xlabel("Sample Index")
        plt.ylabel("NEM Demand")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"forecast_deviation_{period.lower()}.png"))
        plt.close()

# === Wrapper to Run All Plots ===
def generate_all_plots(df_combined, summary_df, models, output_dir="Output/Charts"):
    plot_boxplot_absolute_errors(df_combined, output_dir)
    plot_residuals_vs_predicted(df_combined, output_dir)
    plot_model_performance_bar_chart(summary_df, output_dir)
    plot_model_metrics_summary(summary_df, output_dir)
    plot_feature_importances(models, input_dir="Output", output_dir=output_dir)
    plot_forecast_deviation(df_combined, output_dir)
    print("✅ All plots generated and saved to:", output_dir)
