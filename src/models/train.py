"""
Trains the credit-default model end-to-end: raw Home Credit CSVs ->
feature engineering (all six source-table modules) -> build_matrix ->
XGBoost + LightGBM blend -> serialized artifacts.

Direct port of notebook cells 20 (train/val split) and 22 (train both
models + blend-weight search), driven by the feature-engineering modules
already built (application.py, previous_app.py, installments.py,
pos_cash.py, bureau.py, credit_card.py, build_matrix.py).

Usage:
    python -m src.model.train --data-dir /path/to/home-credit-default-risk

Artifacts written to --out-dir (defaults to config.ARTIFACTS_DIR):
    xgb_final.pkl                     — trained XGBClassifier
    lgb_final.pkl                     — trained LGBMClassifier
    application_fit_artifacts.pkl     — application.py's ApplicationFitArtifacts
    build_matrix_fit_artifacts.pkl    — build_matrix.py's BuildMatrixFitArtifacts
                                         (bundles the fitted OneHotEncoder,
                                         locked column orders, and
                                         feature_names_final_v2)
    blend_weights.pkl                 — {"xgb": w_xgb, "lgb": w_lgb}

NOTE on config.py drift, flagging before predict.py gets built:
  - config.py's ENCODER_FILE / FEATURE_NAMES_FILE constants are NOT used
    here — the encoder and feature_names_final_v2 are bundled together
    inside build_matrix_fit_artifacts.pkl instead (one pickle, one source
    of truth, can't get out of sync with each other). Those two config.py
    constants are now dead and should be removed or repurposed.
  - config.py's comment on BLEND_W_XGB/BLEND_W_LGB said they'd be
    "overwritten by train.py each time it reruns" — this script does NOT
    edit config.py's source. It writes blend_weights.pkl instead.
    predict.py MUST load that pickle rather than trusting
    config.BLEND_W_XGB/BLEND_W_LGB, which will silently go stale after the
    first retrain otherwise.
  - config.DATA_DIR is hardcoded to a machine-specific path. This script
    ignores it in favor of the required --data-dir CLI argument — update
    or delete that constant in config.py so nobody trusts it by accident.

predict.py needs all four .pkl files above. The raw per-table aggregates
(prev_agg / install_agg / pos_agg / bureau_agg / cc_agg) are NOT needed at
serve time — only the locked column lists already captured inside
build_matrix_fit_artifacts, since a live request goes through
build_matrix.transform_serve_row(), not build_training_matrix().
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import lightgbm as lgb

from .. import config
from ..features.application import engineer_application_features
from ..features.previous_application import build_previous_application_features
from ..features.pos_cash import build_pos_cash_features
from ..features.bureau import build_bureau_features
from ..features.credit_card import build_cc_features
from ..features.build_features import build_training_matrix
from ..features.installments import build_installments_features
def _load_raw_tables(data_dir: Path) -> dict:
    """Reads the seven Home Credit CSVs. encoding='latin-1' matches the
    notebook — some columns contain non-UTF8 bytes that break the default."""
    return {
        "application": pd.read_csv(data_dir / "application_train.csv", encoding="latin-1"),
        "previous": pd.read_csv(data_dir / "previous_application.csv", encoding="latin-1"),
        "installments": pd.read_csv(data_dir / "installments_payments.csv", encoding="latin-1"),
        "pos": pd.read_csv(data_dir / "POS_CASH_balance.csv", encoding="latin-1"),
        "bureau": pd.read_csv(data_dir / "bureau.csv", encoding="latin-1"),
        "bureau_balance": pd.read_csv(data_dir / "bureau_balance.csv", encoding="latin-1"),
        "credit_card": pd.read_csv(data_dir / "credit_card_balance.csv", encoding="latin-1"),
    }


def train(data_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_raw_tables(data_dir)

    print("Building application-level features...")
    df_fe, application_fit_artifacts = engineer_application_features(raw["application"], fit=True)

    print("Building previous_application / installments / POS_CASH / bureau / credit_card features...")
    prev_agg = build_previous_application_features(raw["previous"])
    install_agg = build_installments_features(raw["installments"])
    pos_agg = build_pos_cash_features(raw["pos"])
    bureau_agg = build_bureau_features(raw["bureau"], raw["bureau_balance"])
    cc_agg = build_cc_features(raw["credit_card"])

    print("Assembling final feature matrix...")
    X_final, feature_names_final_v2, build_matrix_fit_artifacts = build_training_matrix(
        df_fe, prev_agg, install_agg, pos_agg, bureau_agg, cc_agg,
    )
    y = df_fe["TARGET"].astype(np.int8)
    print(f"X_final shape: {X_final.shape}")

    # ---- cell 20: train/val split ----
    train_idx, valid_idx = train_test_split(
        np.arange(X_final.shape[0]), test_size=0.20, stratify=y, random_state=config.RANDOM_STATE,
    )
    X_tr, X_va = X_final[train_idx], X_final[valid_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
    print(f"Train shape: {X_tr.shape} | Valid shape: {X_va.shape}")

    # ---- cell 22: XGBoost ----
    print("Training XGBoost...")
    xgb_final = XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.05,
        subsample=0.70, colsample_bytree=0.80, min_child_weight=10, gamma=0.10,
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=config.RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_final.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred_xgb = xgb_final.predict_proba(X_va)[:, 1]
    print(f"XGBoost  -> ROC-AUC: {roc_auc_score(y_va, pred_xgb):.5f} "
          f"| PR-AUC: {average_precision_score(y_va, pred_xgb):.5f} "
          f"| best_iter={xgb_final.best_iteration}")

    # ---- cell 22: LightGBM ----
    print("Training LightGBM...")
    lgb_final = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=4, num_leaves=15, learning_rate=0.03,
        subsample=0.70, colsample_bytree=0.80, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, objective="binary",
        random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    lgb_final.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    pred_lgb = lgb_final.predict_proba(X_va)[:, 1]
    print(f"LightGBM -> ROC-AUC: {roc_auc_score(y_va, pred_lgb):.5f} "
          f"| PR-AUC: {average_precision_score(y_va, pred_lgb):.5f} "
          f"| best_iter={lgb_final.best_iteration_}")

    # ---- cell 22: blend weight search ----
    print("Searching blend weight...")
    blend_results = []
    for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_lgb = 1 - w_xgb
        pred_blend = w_xgb * pred_xgb + w_lgb * pred_lgb
        blend_results.append({
            "w_xgb": w_xgb, "w_lgb": w_lgb,
            "ROC_AUC": roc_auc_score(y_va, pred_blend),
            "PR_AUC": average_precision_score(y_va, pred_blend),
        })
    blend_df = pd.DataFrame(blend_results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
    print(blend_df.to_string(index=False))

    best_w_xgb = float(blend_df.iloc[0]["w_xgb"])
    best_w_lgb = float(blend_df.iloc[0]["w_lgb"])
    pred_final_blend = best_w_xgb * pred_xgb + best_w_lgb * pred_lgb
    final_roc_auc = roc_auc_score(y_va, pred_final_blend)
    final_pr_auc = average_precision_score(y_va, pred_final_blend)
    print(f">>> FINAL BLEND (w_xgb={best_w_xgb}, w_lgb={best_w_lgb}) "
          f"-> ROC-AUC: {final_roc_auc:.5f} | PR-AUC: {final_pr_auc:.5f}")

    # ---- save artifacts ----
    with open(out_dir / "xgb_final.pkl", "wb") as f:
        pickle.dump(xgb_final, f)
    with open(out_dir / "lgb_final.pkl", "wb") as f:
        pickle.dump(lgb_final, f)
    with open(out_dir / "application_fit_artifacts.pkl", "wb") as f:
        pickle.dump(application_fit_artifacts, f)
    with open(out_dir / "build_matrix_fit_artifacts.pkl", "wb") as f:
        pickle.dump(build_matrix_fit_artifacts, f)
    with open(out_dir / "blend_weights.pkl", "wb") as f:
        pickle.dump({"xgb": best_w_xgb, "lgb": best_w_lgb}, f)

    print(f"\nSaved 5 artifacts to {out_dir}")
    return {
        "roc_auc": final_roc_auc, "pr_auc": final_pr_auc,
        "blend_weights": {"xgb": best_w_xgb, "lgb": best_w_lgb},
        "n_features": X_final.shape[1],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="Folder containing the 7 Home Credit CSVs")
    parser.add_argument("--out-dir", type=Path, default=config.ARTIFACTS_DIR,
                         help="Where to write trained artifacts")
    args = parser.parse_args()
    train(args.data_dir, args.out_dir)