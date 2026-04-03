from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_actual_vs_pred_plot(df_plot: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df_plot["datetime"], df_plot["actual"], label="Actual", linewidth=1.2)
    plt.plot(df_plot["datetime"], df_plot["pred"], label="Predicted", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_feature_importance_rf(rf_model, feature_cols: List[str], out_path: str, title: str):
    importances = rf_model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="importance", y="feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _save_metric_grid(
    df: pd.DataFrame,
    x_col: str,
    split_col: str,
    metrics: List[str],
    out_path: str,
    title: str,
):
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.lineplot(data=df, x=x_col, y=metric, hue=split_col, marker="o", ax=ax)
        ax.set_title(metric)
        ax.grid(alpha=0.25)
        ax.legend(title=split_col)

    axes[-1].set_xlabel("Epochs")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_epoch_sweep_regression_metrics_plot(df: pd.DataFrame, out_path: str):
    _save_metric_grid(
        df=df,
        x_col="Epochs",
        split_col="Split",
        metrics=["MAE", "RMSE", "R2"],
        out_path=out_path,
        title="Epoch Sweep - Regression Metrics",
    )


def save_epoch_sweep_classification_metrics_plot(df: pd.DataFrame, out_path: str):
    _save_metric_grid(
        df=df,
        x_col="Epochs",
        split_col="Split",
        metrics=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        out_path=out_path,
        title="Epoch Sweep - Classification Metrics",
    )


def save_epoch_sweep_final_losses_plot(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Epochs", y="FinalTrainLoss", hue="Task", marker="o")
    plt.title("Epoch Sweep - Final Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Final train loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def save_epoch_loss_curves_plot(df: pd.DataFrame, task: str, out_path: str):
    task_df = df[df["Task"] == task]
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=task_df,
        x="EpochIndex",
        y="Loss",
        hue="Epochs",
        palette="viridis",
        legend="full",
    )
    plt.title(f"Epoch Sweep - {task} Loss Curves")
    plt.xlabel("Epoch index")
    plt.ylabel("Training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
