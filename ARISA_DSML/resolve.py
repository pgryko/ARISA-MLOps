from loguru import logger
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

from ARISA_DSML.config import (
    MODEL_NAME,
)


def get_model_by_alias(client, model_name:str=MODEL_NAME, alias:str="champion"):
    """Get model version by alias, handling the case where model doesn't exist."""
    try:
        # First check if the registered model exists
        try:
            client.get_registered_model(model_name)
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info(f"Model {model_name} not found in registry")
                return None
            raise(e)

        # If model exists, try to get the aliased version
        try:
            alias_mv = client.get_model_version_by_alias(model_name, alias)
            return alias_mv
        except MlflowException as e:
            if f"alias {alias} not found" in str(e):
                return None
            raise(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        raise(e)

if __name__=="__main__":
    client = MlflowClient(mlflow.get_tracking_uri())
    champ_mv = get_model_by_alias(client)
    if champ_mv is None:
        chall_mv = get_model_by_alias(client, alias="challenger")
        if chall_mv is None:
            try:
                model_info = client.get_latest_versions(MODEL_NAME)[0]
                logger.info("Did not find champion or challenger, promoting newest model to champion.")
                client.set_registered_model_alias(MODEL_NAME, "champion", model_info.version)
            except MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    logger.info("No models found in registry. Please train a model first.")
                    exit(0)
                raise(e)
        else:
            logger.info("Found challenger model with no champion, promoting challenger to champion.")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)

    chall_mv = get_model_by_alias(client, alias="challenger")

    if champ_mv and chall_mv:
        champ_run = client.get_run(champ_mv.run_id)
        f1_champ = champ_run.data.metrics["f1_cv_mean"]

        chall_run = client.get_run(chall_mv.run_id)
        f1_chall = chall_run.data.metrics["f1_cv_mean"]

        if f1_chall >= f1_champ:
            logger.info("Challenger model surpassed metric of current champion, promoting challenger to champion.")
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
        else:
            challenge_failed_exc = "Challenger model does not surpass metric of current champion, ending predict workflow."
            logger.error(challenge_failed_exc)
            raise(Exception(challenge_failed_exc))
    elif champ_mv and chall_mv is None:
        logger.info("No challenger to champion, continuing with prediction.")
