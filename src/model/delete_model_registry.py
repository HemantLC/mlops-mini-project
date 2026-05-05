import dagshub
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("https://dagshub.com/HemantLC/mlops-mini-project.mlflow")
dagshub.init(repo_owner='HemantLC', repo_name='mlops-mini-project', mlflow=True)

client = MlflowClient()
model_name = "my_model"
client.delete_registered_model(name=model_name)

print(f"Successfully deleted model: {model_name}")
