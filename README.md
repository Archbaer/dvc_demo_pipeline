# ML Pipeline with DVC and MLflow for Diabetes Prediction

This project implements an end-to-end machine learning pipeline for predicting diabetes using the Pima Indians Diabetes Dataset. It leverages DVC for data and pipeline versioning, and MLflow for experiment tracking, ensuring reproducibility and collaboration.

## Features

- **Data Version Control**: Track datasets, models, and pipeline stages with DVC for reproducible workflows
- **Experiment Tracking**: Log hyperparameters, metrics, and artifacts with MLflow for easy comparison of runs
- **Automated Pipeline**: Structured stages (preprocess, train, evaluate) that re-run automatically on dependency changes
- **Remote Storage**: Integration with DagsHub for storing data, models, and MLflow experiments
- **Hyperparameter Tuning**: Automated grid search for Random Forest optimization

## Project Structure

```
.
├── data/
│   ├── raw/data.csv          # Raw Pima Indians Diabetes Dataset
│   └── processed/data.csv    # Preprocessed data
├── models/
│   └── model.pkl             # Trained Random Forest model
├── src/
│   ├── preprocess.py         # Data preprocessing script
│   ├── train.py              # Model training with hyperparameter tuning
│   └── evaluate.py           # Model evaluation script
├── dvc.yaml                  # DVC pipeline definition
├── params.yaml               # Pipeline parameters and hyperparameters
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Archbaer/dvc_demo_pipeline
   cd dvc_demo_pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up DVC:
   ```bash
   dvc init
   ```
   

## Usage

### Running the Complete Pipeline

```bash
# Pull data from remote storage
dvc pull

# Run the entire pipeline
dvc repro

# View MLflow experiments
mlflow ui
```

### Running Individual Stages

```bash
# Preprocess data only
dvc repro preprocess

# Train model only
dvc repro train

# Evaluate model only
dvc repro evaluate
```

## Pipeline Stages

### 1. Preprocess
- **Script**: `src/preprocess.py`
- **Input**: `data/raw/data.csv`
- **Output**: `data/processed/data.csv`
- **Function**: Reads raw dataset, performs basic cleaning and preprocessing

### 2. Train
- **Script**: `src/train.py`
- **Input**: `data/processed/data.csv`
- **Output**: `models/model.pkl`
- **Function**: 
  - Trains Random Forest Classifier with grid search hyperparameter tuning
  - Logs metrics, parameters, and model to MLflow
  - Saves best model locally

### 3. Evaluate
- **Script**: `src/evaluate.py`
- **Input**: `models/model.pkl`, `data/processed/data.csv`
- **Output**: Evaluation metrics logged to MLflow
- **Function**: Evaluates trained model and logs performance metrics

## MLflow Integration

- **Experiment Tracking**: All runs are logged to DagsHub MLflow
- **Metrics Logged**: Accuracy, confusion matrix, classification report
- **Parameters Logged**: Best hyperparameters from grid search
- **Artifacts**: Model signature, confusion matrix, classification report

## Technologies Used

- **Python**: Core programming language
- **DVC**: Data and pipeline version control
- **MLflow**: Experiment tracking and model management
- **Scikit-learn**: Machine learning algorithms and metrics
- **Pandas**: Data manipulation and analysis
- **DagsHub**: Remote storage for data and MLflow experiments

## Reproducibility

This pipeline ensures reproducibility through:
- DVC tracking of data, code, and model dependencies
- Parameterized configuration in `params.yaml`
- MLflow logging of all experiments and results
- Version control of the entire pipeline

## DagsHub Repository

The full project, including the integrated DVC pipeline, MLflow experiments, and remote data storage, is hosted on **DagsHub** for a complete view of the end-to-end workflow. Visit the repository at: https://dagshub.com/Archbaer/mlpipe.

On **DagsHub**, you can:

* Explore the DVC pipeline stages and dependencies.
* View tracked data, models, and artifacts.
* Access MLflow experiment runs, metrics, and logs.

