from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(f"Experiment tracker uri :- {Client().active_stack.experiment_tracker.get_tracking_uri()}")
    training_pipeline(datapath="data/heart.csv")

    '''
    ROUGHT WORK
    
    mlflow ui --backend-store-uri "file:C:\Users\Sumit\AppData\Roaming\zenml\local_stores\d290c6ba-ede5-4f95-b4b0-49745de5357c\mlruns"
    
    '''