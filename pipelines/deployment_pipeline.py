import json

import numpy as np
import pandas as pd
from streamlit import columns
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from pipelines.utils import get_data_for_test
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment Trigger Config"""
    min_accuracy: float = 0.3

@step(enable_cache=False)
def dynamic_importer()->str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    """Implement a model trigger which looks at modal accuracy and decides if it is suitable to deploy"""
    print(f"MODAL ACCURACY :- {accuracy}")
    print(f"THRESHOLD ACCURACY :- {config.min_accuracy}")
    print(f"CRITERION FOR DEPLOYMENT IS MATCHING :- {accuracy >= config.min_accuracy}")
    return accuracy >= config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader( pipeline_name: str, pipeline_step_name:str, running: bool = True, modal_name: str = 'modal',) -> MLFlowDeploymentService:
    """
    Get the prediction service started by deployment pipeline
    Args:
        pipeline_name:      name of the pipeline that deployed the MLFlow prediction server
        pipeline_step_name: name of the pipeline that deployed the MLFlow prediction server
        runnning:           when this flag is set, step only returns a running service
        modal_name:         the name of the modal that is deployed

    Returns:
    """
    # Get MLFlow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch existing services with same pipeline name, step name and modal name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=modal_name,
        running=running,
    )
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment services found for pipeline {pipeline_name}",
            f"Step - {pipeline_step_name} and model {modal_name}",
        )
    return  existing_services[0]

@step()
def predictor( service: MLFlowDeploymentService, data: str) -> np.ndarray:
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    df = pd.DataFrame(data=data, columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(datapath: str, min_accuracy: float, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,):
    df = ingest_data(datapath)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2, rmse = evaluate_model(model, x_test, y_test)
    deployment_decision = deployment_trigger(r2)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout,
    )




@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name:str,):
    data = dynamic_importer()
    service = prediction_service_loader(pipeline_name,
                                        pipeline_step_name,
                                        running=False)
    prediction = predictor(service, data)
    return prediction