# from mlflow.models.cli import predict
# from pandas.conftest import datapath
# from zenml.cli import deploy
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
# from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
import click

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'
@click.command()

@click.option(
    "--config",'-c',
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Optionally you can choose to run the deployment pipeline to train and deploy the model ('deploy'), or to only run a prediction against the deployed model ('predict'), or by default both will run ('deploy_and_predict')"
)

@click.option(
    "--min-accuracy",
    # default=0.92,
    default=0.3,
    help="Minimum accuracy required to deploy the model",
)

def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            datapath = "data/heart.csv",
            min_accuracy=min_accuracy,
            workers=3,
            timeout=90,
        )
    if predict:
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )


if __name__ == "__main__":
    run_deployment()


