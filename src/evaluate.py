import mlflow
import pandas as pd 
import pickle
import yaml
from sklearn.metrics import accuracy_score, classification_report

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    mlflow.set_experiment("RandomForest_Classifier_Experiment")

    with mlflow.start_run():
        model = pickle.load(open(model_path, 'rb'))
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"Evaluation Accuracy: {accuracy}")
        report = classification_report(y, y_pred)

        mlflow.log_text(str(report), "classification_report.txt")
        mlflow.log_metric("eval_accuracy", accuracy)

if __name__ == "__main__":
    evaluate(params["data"], params["model"])