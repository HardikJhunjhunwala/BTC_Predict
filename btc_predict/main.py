import os

import joblib
import pandas as pd
from tqdm import tqdm

from btc_predict.config import Config
from btc_predict.data import prepare_dataset, time_split
from btc_predict.models import (
    eval_classification,
    eval_regression,
    train_classification_models,
    train_regression_models,
)
from btc_predict.plots import save_actual_vs_pred_plot, save_feature_importance_rf
from btc_predict.plots import (
    save_epoch_loss_curves_plot,
    save_epoch_sweep_classification_metrics_plot,
    save_epoch_sweep_final_losses_plot,
    save_epoch_sweep_regression_metrics_plot,
)
from btc_predict.torch_models import (
    eval_torch_classification,
    eval_torch_regression,
    train_torch_classifier,
    train_torch_regressor,
)


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("Loading and preparing dataset...")
    df, feature_cols = prepare_dataset(cfg)
    tqdm.write(f"Prepared dataset: {len(df):,} rows")

    train_df, val_df, test_df = time_split(
        df,
        test_ratio=cfg.test_size_ratio,
        val_ratio=cfg.val_size_ratio,
    )

    print(f"Rows -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train_reg = train_df["target_close_reg"]
    y_val_reg = val_df["target_close_reg"]
    y_test_reg = test_df["target_close_reg"]

    y_train_cls = train_df["target_direction_cls"]
    y_val_cls = val_df["target_direction_cls"]
    y_test_cls = test_df["target_direction_cls"]

    sp = cfg.show_progress
    if cfg.use_torch:
        reg_models = {
            "TorchMLPRegressor": train_torch_regressor(
                X_train,
                y_train_reg,
                batch_size=cfg.torch_batch_size,
                epochs=cfg.torch_epochs,
                lr=cfg.torch_lr,
                show_progress=sp,
            )
        }
        cls_models = {
            "TorchMLPClassifier": train_torch_classifier(
                X_train,
                y_train_cls,
                batch_size=cfg.torch_batch_size,
                epochs=cfg.torch_epochs,
                lr=cfg.torch_lr,
                show_progress=sp,
            )
        }
        eval_reg_fn = eval_torch_regression
        eval_cls_fn = eval_torch_classification
    else:
        reg_models = train_regression_models(
            X_train, y_train_reg, show_progress=sp,
        )
        cls_models = train_classification_models(
            X_train, y_train_cls, show_progress=sp,
        )
        eval_reg_fn = eval_regression
        eval_cls_fn = eval_classification

    reg_metrics_rows = []
    cls_metrics_rows = []

    for name, model in reg_models.items():
        epbar = tqdm(
            total=2,
            desc=f"Eval reg: {name}",
            bar_format="{l_bar}{bar}| {desc} [{elapsed}]",
            disable=not sp,
            leave=True,
        )
        m_val = eval_reg_fn(model, X_val, y_val_reg)
        epbar.update(1)
        m_test = eval_reg_fn(model, X_test, y_test_reg)
        epbar.update(1)
        epbar.close()

        reg_metrics_rows.append({
            "Model": name,
            "Split": "Validation",
            **m_val
        })
        reg_metrics_rows.append({
            "Model": name,
            "Split": "Test",
            **m_test
        })

    for name, model in cls_models.items():
        epbar = tqdm(
            total=2,
            desc=f"Eval cls: {name}",
            bar_format="{l_bar}{bar}| {desc} [{elapsed}]",
            disable=not sp,
            leave=True,
        )
        m_val = eval_cls_fn(model, X_val, y_val_cls)
        epbar.update(1)
        m_test = eval_cls_fn(model, X_test, y_test_cls)
        epbar.update(1)
        epbar.close()

        cls_metrics_rows.append({
            "Model": name,
            "Split": "Validation",
            **m_val
        })
        cls_metrics_rows.append({
            "Model": name,
            "Split": "Test",
            **m_test
        })

    reg_metrics_df = pd.DataFrame(reg_metrics_rows)
    cls_metrics_df = pd.DataFrame(cls_metrics_rows)

    reg_metrics_path = os.path.join(cfg.out_dir, "regression_metrics.csv")
    cls_metrics_path = os.path.join(cfg.out_dir, "classification_metrics.csv")
    reg_metrics_df.to_csv(reg_metrics_path, index=False)
    cls_metrics_df.to_csv(cls_metrics_path, index=False)

    print("\nRegression metrics:")
    print(reg_metrics_df)
    print("\nClassification metrics:")
    print(cls_metrics_df)

    best_reg_name = (
        reg_metrics_df[reg_metrics_df["Split"] == "Test"]
        .sort_values("RMSE", ascending=True)
        .iloc[0]["Model"]
    )
    best_reg_model = reg_models[best_reg_name]

    pred_test = best_reg_model.predict(X_test)
    plot_df = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual": y_test_reg.values,
        "pred": pred_test
    })
    save_actual_vs_pred_plot(
        plot_df,
        out_path=os.path.join(cfg.out_dir, "best_reg_actual_vs_pred.png"),
        title=f"Best Regression Model: {best_reg_name} (Actual vs Predicted)",
    )

    if "RandomForestRegressor" in reg_models:
        rf = reg_models["RandomForestRegressor"]
        save_feature_importance_rf(
            rf,
            feature_cols,
            out_path=os.path.join(cfg.out_dir, "rf_reg_feature_importance.png"),
            title="RandomForestRegressor Feature Importance",
        )

    joblib.dump(reg_models, os.path.join(cfg.out_dir, "regression_models.joblib"))
    joblib.dump(cls_models, os.path.join(cfg.out_dir, "classification_models.joblib"))
    print(f"\nArtifacts saved in: {cfg.out_dir}")

    if cfg.use_torch and cfg.run_epoch_sweep:
        run_epoch_sweep(
            cfg=cfg,
            X_train=X_train,
            y_train_reg=y_train_reg,
            y_train_cls=y_train_cls,
            X_val=X_val,
            y_val_reg=y_val_reg,
            y_val_cls=y_val_cls,
            X_test=X_test,
            y_test_reg=y_test_reg,
            y_test_cls=y_test_cls,
        )


def run_epoch_sweep(
    cfg: Config,
    X_train,
    y_train_reg,
    y_train_cls,
    X_val,
    y_val_reg,
    y_val_cls,
    X_test,
    y_test_reg,
    y_test_cls,
):
    sp = cfg.show_progress
    sweep_reg_rows = []
    sweep_cls_rows = []
    sweep_loss_rows = []

    for ep in cfg.epoch_sweep_values:
        tqdm.write(f"\n[SWEEP] Training models with epochs={ep}")
        reg_model = train_torch_regressor(
            X_train,
            y_train_reg,
            batch_size=cfg.torch_batch_size,
            epochs=ep,
            lr=cfg.torch_lr,
            show_progress=sp,
        )
        cls_model = train_torch_classifier(
            X_train,
            y_train_cls,
            batch_size=cfg.torch_batch_size,
            epochs=ep,
            lr=cfg.torch_lr,
            show_progress=sp,
        )

        reg_val = eval_torch_regression(reg_model, X_val, y_val_reg)
        reg_test = eval_torch_regression(reg_model, X_test, y_test_reg)
        cls_val = eval_torch_classification(cls_model, X_val, y_val_cls)
        cls_test = eval_torch_classification(cls_model, X_test, y_test_cls)

        sweep_reg_rows.append({"Model": "TorchMLPRegressor", "Epochs": ep, "Split": "Validation", **reg_val})
        sweep_reg_rows.append({"Model": "TorchMLPRegressor", "Epochs": ep, "Split": "Test", **reg_test})
        sweep_cls_rows.append({"Model": "TorchMLPClassifier", "Epochs": ep, "Split": "Validation", **cls_val})
        sweep_cls_rows.append({"Model": "TorchMLPClassifier", "Epochs": ep, "Split": "Test", **cls_test})

        for i, loss in enumerate(reg_model.loss_history, start=1):
            sweep_loss_rows.append(
                {"Task": "Regression", "Epochs": ep, "EpochIndex": i, "Loss": loss}
            )
        for i, loss in enumerate(cls_model.loss_history, start=1):
            sweep_loss_rows.append(
                {"Task": "Classification", "Epochs": ep, "EpochIndex": i, "Loss": loss}
            )

    reg_df = pd.DataFrame(sweep_reg_rows)
    cls_df = pd.DataFrame(sweep_cls_rows)
    loss_df = pd.DataFrame(sweep_loss_rows)
    final_loss_df = (
        loss_df.sort_values(["Task", "Epochs", "EpochIndex"])
        .groupby(["Task", "Epochs"], as_index=False)
        .tail(1)[["Task", "Epochs", "Loss"]]
        .rename(columns={"Loss": "FinalTrainLoss"})
        .sort_values(["Task", "Epochs"])
        .reset_index(drop=True)
    )

    reg_df.to_csv(os.path.join(cfg.out_dir, "epoch_sweep_regression_metrics.csv"), index=False)
    cls_df.to_csv(os.path.join(cfg.out_dir, "epoch_sweep_classification_metrics.csv"), index=False)
    loss_df.to_csv(os.path.join(cfg.out_dir, "epoch_sweep_loss_history.csv"), index=False)
    final_loss_df.to_csv(os.path.join(cfg.out_dir, "epoch_sweep_final_losses.csv"), index=False)

    save_epoch_sweep_regression_metrics_plot(
        reg_df, os.path.join(cfg.out_dir, "epoch_sweep_regression_metrics.png")
    )
    save_epoch_sweep_classification_metrics_plot(
        cls_df, os.path.join(cfg.out_dir, "epoch_sweep_classification_metrics.png")
    )
    save_epoch_sweep_final_losses_plot(
        final_loss_df, os.path.join(cfg.out_dir, "epoch_sweep_final_losses.png")
    )
    save_epoch_loss_curves_plot(
        loss_df, "Regression", os.path.join(cfg.out_dir, "epoch_sweep_regression_loss_curves.png")
    )
    save_epoch_loss_curves_plot(
        loss_df, "Classification", os.path.join(cfg.out_dir, "epoch_sweep_classification_loss_curves.png")
    )

    tqdm.write("[SWEEP] Saved CSVs and comparison plots in outputs/")


if __name__ == "__main__":
    main()
