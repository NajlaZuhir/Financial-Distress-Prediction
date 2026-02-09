import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up
MODEL_DIR = os.path.join(BASE_DIR, "artifacts")

BEST_PARAMS_PATH = os.path.join(MODEL_DIR, "best_params.pkl")
REPORT_DICT_PATH = os.path.join(MODEL_DIR, "report_dict.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_final_model.pkl")
TEST_PATH = os.path.join(MODEL_DIR, "test.csv")


# -----------------------------
# MLflow functions
# -----------------------------
def track_experiment(experiment_name):
    """
    Track a single experiment run in MLflow.
    Loads model, params, and metrics from disk.
    """
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri("http://localhost:5000")

    # Load artifacts
    best_params = joblib.load(BEST_PARAMS_PATH)
    report_dict = joblib.load(REPORT_DICT_PATH)
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(best_params)

        # Log key metrics
        mlflow.log_metrics({
            'accuracy': report_dict['accuracy'],
            'recall_class_0': report_dict['0']['recall'],
            'recall_class_1': report_dict['1']['recall'],
            'f1_score_macro': report_dict['macro avg']['f1-score']
        })

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="Best_XGBoost_Model")

        run_id = run.info.run_id  # ✅ Save the run_id for registration

    print(f"✅ Experiment '{experiment_name}' logged to MLflow with run_id={run_id}")
    return run_id


def register_model(model_name, run_id):
    """
    Register the logged model in MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/Best_XGBoost_Model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"✅ Model '{model_name}' registered in MLflow.")



def transition_to_production(model_name):
    client = MlflowClient()
    # Get latest version of the model
    versions = client.get_latest_versions(name=model_name)
    if not versions:
        raise Exception(f"No versions found for model {model_name}")
    
    latest_version = versions[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True  # archives old production if any
    )
    print(f"✅ Model '{model_name}' is now in Production (version {latest_version})")



def predict_with_mlflow(model_uri):
    """
    Load registered model from MLflow and predict on test set.
    """
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop("Financial Distress", axis=1)

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded_model.predict(X_test)

    print("\n✅ Predictions on test set (first 10 rows):")
    print(predictions[:10])
    return predictions


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    experiment_name = "Financial_Distress_XGB"
    model_name = "xgboost1"

    # Track experiment and get latest run_id automatically
    run_id = track_experiment(experiment_name)

    # Register model
    register_model(model_name, run_id)

    # Transition to Production
    transition_to_production(model_name)

    # Predict using production version
    model_uri_prod = f"artifacts:/{model_name}/Production"
    predict_with_mlflow(model_uri_prod)

