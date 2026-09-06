# for data manipulation
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
# for model serialization
import joblib
# for creating folders and reading environment variables
import os
# for experiment tracking
import mlflow

# The workflow starts an MLflow server on localhost:5000 before running this script
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("tourism-package-prediction")

# Load the train and test splits downloaded from the workflow artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()
print("Train and test data loaded from the workflow artifact.")

# Define numeric and categorical features
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# Preprocessor: scale the numeric columns and one-hot encode the categorical columns
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# The target is imbalanced (only around 19% customers purchased the package)
# scale_pos_weight tells XGBoost to give more importance to the minority class
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
print("scale_pos_weight used for class imbalance:", round(scale_pos_weight, 2))

# Define base XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# Hyperparameter grid for tuning
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.7, 0.9]
}

# Pipeline: preprocessing + model
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Tune the model with GridSearchCV (3-fold cross validation, F1 score for imbalanced target)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    # Log all the tuned parameters to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
    mlflow.log_param("cv_folds", 3)
    print("\nBest parameters found by GridSearchCV:")
    print(grid_search.best_params_)
    print("Best cross validation F1 score:", round(grid_search.best_score_, 4))

    # Evaluate the best model on train and test data
    best_model = grid_search.best_estimator_
    ypred_train = best_model.predict(Xtrain)
    ypred_test = best_model.predict(Xtest)

    train_acc = accuracy_score(ytrain, ypred_train)
    test_acc = accuracy_score(ytest, ypred_test)
    test_precision = precision_score(ytest, ypred_test)
    test_recall = recall_score(ytest, ypred_test)
    test_f1 = f1_score(ytest, ypred_test)

    # Log the evaluation metrics to MLflow
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)

    print("\nClassification report on training data:")
    print(classification_report(ytrain, ypred_train))
    print("Classification report on test data:")
    print(classification_report(ytest, ypred_test))

    # Save the best model so the pipeline can commit it to the repository
    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/tourism_model.joblib"
    joblib.dump(best_model, model_path)
    print("Best model saved at:", model_path)
