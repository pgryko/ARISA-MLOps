"""Run prediction on test data."""
from pathlib import Path

from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
import shap

from ARISA_DSML.config import FIGURES_DIR, MODELS_DIR, target


def plot_shap(model:CatBoostClassifier, df_plot:pd.DataFrame)->None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")


def predict(model:CatBoostClassifier, df_pred:pd.DataFrame, params:dict)->str|Path:
    """Do predictions on test data."""
    feature_columns = params.pop("feature_columns")

    preds = model.predict(df_pred[feature_columns])
    plot_shap(model, df_pred[feature_columns])
    df_pred[target] = preds
    preds_path = MODELS_DIR / "preds.csv"
    df_pred[["PassengerId", target]].to_csv(preds_path, index=False)

    return preds_path

