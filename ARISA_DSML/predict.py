"""Run prediction on test data."""
from pathlib import Path
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import shap
import joblib
import os
from ARISA_DSML.config import FIGURES_DIR, MODELS_DIR, target, PROCESSED_DATA_DIR, categorical, MODEL_NAME
from ARISA_DSML.resolve import get_model_by_alias
import mlflow
from mlflow.client import MlflowClient
import json


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


if __name__=="__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    if model_info is None:
        logger.info("No champion model, predicting using newest model")
        model_info = client.get_latest_versions(MODEL_NAME)[0]

    # extract params/metrics data for run `test_run_id` in a single dict
    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(run.data.tags['mlflow.log-model.history'])
    log_model_meta[0]['signature']


    _, artifact_folder = os.path.split(model_info.source)
    logger.info(artifact_folder)
    model_uri = "runs:/{}/{}".format(model_info.run_id, artifact_folder)
    logger.info(model_uri)
    loaded_model = mlflow.catboost.load_model(model_uri)

    params = run_data_dict["params"]
    params["feature_columns"] = [inp["name"] for inp in json.loads(log_model_meta[0]['signature']['inputs'])]
    preds_path = predict(loaded_model, df_test, params)


