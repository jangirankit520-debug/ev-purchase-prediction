"""EV purchase prediction - Kaggle Playground Series S6E9.

5-fold LightGBM with repeat-value (count) features and tuned parameters.
Place train.csv, test.csv and sample_submission.csv next to this file.
"""
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type",
            "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

# Found with a 60-trial Optuna search (single 80/20 split, metric = AUC)
BEST_PARAMS = {
    "n_estimators": 794, "learning_rate": 0.08170236482534843, "num_leaves": 15,
    "min_child_samples": 27, "subsample": 0.9530981251729753,
    "colsample_bytree": 0.9833747140753778, "reg_alpha": 8.664782431336526,
    "reg_lambda": 0.0011611954401190387, "random_state": 42,
}


def add_features(train, test):
    """Interaction, income-digit and count (frequency) features."""
    for df in (train, test):
        subsidy = (df["Subsidy_Available"] == "Yes").astype(int)
        df["Concern_x_Subsidy"] = df["Environmental_Concern_Level"] * subsidy
        df["Concern_x_Income"] = df["Environmental_Concern_Level"] * df["Annual_Income_USD"]
        df["Income_x_Subsidy"] = df["Annual_Income_USD"] * subsidy
        df["Income_mod100"] = df["Annual_Income_USD"] % 100
        df["Income_mod1000"] = df["Annual_Income_USD"] % 1000
        df["Income_is_round"] = (df["Annual_Income_USD"] % 1000 == 0).astype(int)

    # Count features: how often each value appears across train + test.
    # Uses no labels, so there is no target leakage.
    full = pd.concat([train, test], axis=0)
    for col in ["Annual_Income_USD", "Daily_Commute_km", "Age", "Environmental_Concern_Level"]:
        counts = full[col].value_counts()
        train[col + "_cnt"] = train[col].map(counts)
        test[col + "_cnt"] = test[col].map(counts)


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sub = pd.read_csv("sample_submission.csv")

    train["target"] = (train["Will_Buy_EV"] == "Yes").astype(int)
    add_features(train, test)

    for c in CAT_COLS:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")

    features = [c for c in train.columns if c not in ["id", "Will_Buy_EV", "target"]]
    x, y = train[features], train["target"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(x))
    test_preds = np.zeros(len(test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x, y)):
        model = lgb.LGBMClassifier(**BEST_PARAMS, verbosity=-1)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=CAT_COLS)

        oof_preds[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(test[features])[:, 1] / skf.n_splits
        print(f"Fold {fold} AUC: {roc_auc_score(y.iloc[val_idx], oof_preds[val_idx]):.5f}")

    print(f"Overall OOF AUC: {roc_auc_score(y, oof_preds):.5f}")

    os.makedirs("images", exist_ok=True)
    lgb.plot_importance(model, max_num_features=15, importance_type="gain")
    plt.tight_layout()
    plt.savefig("images/feature_importance.png")

    sub["Will_Buy_EV"] = test_preds
    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()