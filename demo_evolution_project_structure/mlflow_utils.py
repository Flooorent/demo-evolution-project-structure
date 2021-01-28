import mlflow
from mlflow.tracking import MlflowClient


def register_best_model(model_name, experiment_name, parent_run_name, metric, order_by="ASC", model_artifact_name="model"):
    """
    Register best model obtained for model `model_name`, experiment `experiment_name` and parent run `parent_run_name`.

    :param model_name: model name in the Model registry
    :type model_name: str

    :param experiment_name: name of the experiment
    :type experiment_name: str

    :param parent_run_name: name of the parent run used when running hypeparameter optimization via Hyperopt
    :type parent_run_name: str

    :param metric: name of the metric used to optimize our models
    :type metric: str

    :param order_by: "ASC" to order metric values by ascending order, "DESC" for descending
    :type order_by: str

    :param model_artifact_name: name of the model when saved as an artifact in the Tracking Server
    :type model_artifact_name: str

    :return: ModelVersion object associated to the transitioned version
    :rtype: mlflow.entities.model_registry.ModelVersion
    """
    client = MlflowClient()

    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    parent_run = client.search_runs(experiment_id, filter_string=f"tags.mlflow.runName = '{parent_run_name}'", order_by=[f"metrics.loss {order_by}"])[0]
    parent_run_id = parent_run.info.run_id
    best_run_from_parent_run = client.search_runs(experiment_id, filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'", order_by=[f"metrics.{metric} {order_by}"])[0]

    best_model_uri = f"runs:/{best_run_from_parent_run.info.run_id}/{model_artifact_name}"
    model_details = mlflow.register_model(model_uri=best_model_uri, name=model_name)

    return model_details
