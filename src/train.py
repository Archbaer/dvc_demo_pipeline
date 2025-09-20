import mlflow
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse


params = yaml.safe_load(open("params.yaml"))["train"]

def hyperparameter_tuning(X_train, y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path, random_state):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    mlflow.set_experiment("RandomForest_Classifier_Experiment")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        # predict and evaluate

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_["min_samples_leaf"])

        cm=confusion_matrix(y_test, y_pred)
        cr=classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        run_id = mlflow.active_run().info.run_id
        model_artifact_path = f"model_{run_id}"

        if tracking_url_type_store == "file":
            mlflow.sklearn.log_model(best_model, model_artifact_path, signature=signature)
        else:
            mlflow.sklearn.save_model(best_model, model_artifact_path, signature=signature)
        
        filename = model_path

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(best_model, open(filename, 'wb'))

        print(f"Model saved in {filename}")

if __name__ == "__main__":
    train(
        data_path=params["data"],
        model_path=params["model"],
        random_state=params["random_state"]
    )