import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--criterion")
    # parser.add_argument("--learning_rate")
    # parser.add_argument("--max_leaf_nodes")
    # parser.add_argument("--n_estimators")
    # args = parser.parse_args()

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Churn_Data_Final.csv")
    Churn_Data_01 = pd.read_csv(dataset_path)

    # Independent variable
    X = Churn_Data_01.drop(['Churn'], axis=1)
    # Dependent variables
    y = Churn_Data_01.Churn
    
    # Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.20, random_state=42)
    
    # Scalining training data sets
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # criterion = object(args.criterion)
    # learning_rate = float(args.learning_rate)
    # max_leaf_nodes = int(args.max_leaf_nodes)
    # n_estimators = int(args.n_estimators)
    
    # criterion = object(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 2 else 0.5
    max_leaf_nodes = int(sys.argv[2]) if len(sys.argv) > 3 else 0.5
    n_estimators = int(sys.argv[3]) if len(sys.argv) > 4 else 0.5
    

    with mlflow.start_run():
        GBRModel = GradientBoostingRegressor(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators)
        GBRModel.fit(X_train_scaled, y_train)

        predicted_qualities = GBRModel.predict(X_test_scaled)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Gradient Boosting Regressor model (learning_rate=%f, max_leaf_nodes=%f, n_estimators=%f):" % (learning_rate, max_leaf_nodes, n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(GBRModel, "model")
